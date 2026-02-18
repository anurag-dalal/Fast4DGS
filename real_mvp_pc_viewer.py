#!/usr/bin/env python3
"""
real_mvp_pc_viewer.py – Full GPU pipeline for live point-cloud viewing.

Every step runs on CUDA:
  1. Preprocessing + corner masking  (torch, GPU)
  2. VGGT inference
  3. Bundle adjustment               (torch, GPU)
  4. Unproject → flat points
  5. Radius filter                   (torch, GPU)
  6. Confidence filter               (torch, GPU)
  7. Intensity filter                (torch, GPU)
  8. Density filter                  (torch, GPU – voxel hashing)
  9. Downsample + push to viser
"""

import sys
import time
import json
import threading
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from dataset.read_write_model import read_model, qvec2rotmat
import viser

# Add VGGT-X to path
VGGT_PATH = Path(__file__).resolve().parent / "VGGT-X"
if str(VGGT_PATH) not in sys.path:
    sys.path.append(str(VGGT_PATH))

try:
    from vggt.models.vggt import VGGT
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from vggt.utils.geometry import unproject_depth_map_to_point_map
except ImportError as e:
    print(f"Error importing VGGT: {e}")
    sys.exit(1)

CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "stream_config.json"


# ═══════════════════════════════════════════════════════════════════════════
# Utility
# ═══════════════════════════════════════════════════════════════════════════

def load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


class StreamCapture:
    """Grabs frames from one GStreamer UDP/RTP H265 stream in a background thread."""

    def __init__(self, port: int, node_name: str, host: str, mac: str):
        self.port = port
        self.node_name = node_name
        self.host = host
        self.mac = mac
        self.frame = None
        self._lock = threading.Lock()
        self._running = False
        self._cap = None
        self._thread = None

    def start(self, timeout: float = 15.0, retry_interval: float = 3.0):
        self._running = True
        self._thread = threading.Thread(
            target=self._open_and_loop, args=(timeout, retry_interval), daemon=True
        )
        self._thread.start()

    def _open_and_loop(self, timeout: float, retry_interval: float):
        gst = (
            f'udpsrc port={self.port} buffer-size=200000000 '
            f'caps="application/x-rtp, media=video, clock-rate=90000, '
            f'encoding-name=H265, payload=96" ! '
            f'rtpjitterbuffer latency=15 drop-on-latency=true ! '
            f'rtph265depay ! h265parse config-interval=1 ! '
            f'nvh265dec enable-max-performance=1 ! '
            f'videoconvert ! video/x-raw,format=BGR ! '
            f'appsink sync=false max-buffers=1 drop=true'
        )
        deadline = time.monotonic() + timeout
        attempt = 0
        while self._running:
            if time.monotonic() > deadline:
                print(f"[FAIL] port {self.port} ({self.node_name}) – gave up after {timeout:.0f}s")
                break
            attempt += 1
            cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
            if cap.isOpened():
                self._cap = cap
                print(f"[OK]   port {self.port} ({self.node_name}) opened on attempt {attempt}")
                self._loop()
                break
            cap.release()
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(min(retry_interval, remaining))

    def _loop(self):
        while self._running and self._cap and self._cap.isOpened():
            ret, frame = self._cap.read()
            if ret and frame is not None:
                with self._lock:
                    self.frame = frame
            else:
                time.sleep(0.005)

    def read(self):
        with self._lock:
            return self.frame

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._cap:
            self._cap.release()


# ═══════════════════════════════════════════════════════════════════════════
# GPU timer
# ═══════════════════════════════════════════════════════════════════════════

class CudaTimer:
    """CUDA-event based timer for accurate GPU profiling (ms)."""

    def __init__(self):
        self._s = torch.cuda.Event(enable_timing=True)
        self._e = torch.cuda.Event(enable_timing=True)

    def start(self):
        self._s.record()

    def stop(self) -> float:
        self._e.record()
        torch.cuda.synchronize()
        return self._s.elapsed_time(self._e)


# ═══════════════════════════════════════════════════════════════════════════
# 1. GPU Preprocessing + Corner Masking
# ═══════════════════════════════════════════════════════════════════════════

def mask_corners_gpu(images: torch.Tensor, frac: float = 0.08) -> torch.Tensor:
    """Zero-out square corners on GPU in-place.  images: [B, 3, H, W]."""
    _, _, H, W = images.shape
    s = int(H * frac)
    images[:, :, :s, :s] = 0          # top-left
    images[:, :, :s, W - s:] = 0      # top-right
    images[:, :, H - s:, :s] = 0      # bottom-left
    images[:, :, H - s:, W - s:] = 0  # bottom-right
    return images


def preprocess_frames_gpu(frames, target_size: int, device, dtype) -> torch.Tensor:
    """BGR numpy list → GPU tensor [B, 3, H, W] with corners masked.

    Padding to square + bicubic resize are done entirely on GPU.
    """
    tensors = []
    for frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0).to(device)
        tensors.append(t)

    batch = torch.stack(tensors)  # [B, 3, H, W] float32 on GPU
    B, C, H, W = batch.shape

    # Pad to square
    max_dim = max(H, W)
    if H != max_dim or W != max_dim:
        pad_top    = (max_dim - H) // 2
        pad_bottom = max_dim - H - pad_top
        pad_left   = (max_dim - W) // 2
        pad_right  = max_dim - W - pad_left
        batch = F.pad(batch, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)

    # Resize
    if batch.shape[-1] != target_size:
        batch = F.interpolate(
            batch, size=(target_size, target_size),
            mode='bicubic', align_corners=False,
        ).clamp_(0.0, 1.0)

    # Corner mask (GPU, in-place)
    mask_corners_gpu(batch)

    return batch.to(dtype)


# ═══════════════════════════════════════════════════════════════════════════
# 2. Confidence Filter (GPU)
# ═══════════════════════════════════════════════════════════════════════════

def filter_by_confidence(points: torch.Tensor, colors: torch.Tensor,
                         conf: torch.Tensor, threshold: float = 0.3):
    """Keep points where depth confidence > threshold.
    points [N,3], colors [N,3], conf [N]  — all on GPU.
    """
    mask = conf > threshold
    return points[mask], colors[mask]


# ═══════════════════════════════════════════════════════════════════════════
# 3. Intensity Filter (GPU)
# ═══════════════════════════════════════════════════════════════════════════

def filter_by_intensity(points: torch.Tensor, colors: torch.Tensor,
                        threshold: float = 0.01):
    """Remove dark/black points.  colors [N, 3] in [0,1].  GPU."""
    intensity = colors.mean(dim=1)
    mask = intensity > threshold
    return points[mask], colors[mask]


# ═══════════════════════════════════════════════════════════════════════════
# 4. Density Filter (GPU – Voxel Hashing)
# ═══════════════════════════════════════════════════════════════════════════

def filter_by_density_gpu(points: torch.Tensor, colors: torch.Tensor,
                          voxel_size: float = 0.01, min_count: int = 5):
    """GPU-only density filter via voxel hashing.

    - Quantise each point to an integer voxel cell.
    - Spatial-hash the cell coordinates to a single int64.
    - Count occupancy per voxel.
    - Keep points in voxels with >= min_count members.

    O(N) on GPU — no CPU transfer, no tree building.
    """
    N = points.shape[0]
    if N < min_count:
        return points, colors

    cell = torch.floor(points / voxel_size).long()  # [N, 3]
    cell = cell - cell.min(dim=0).values              # shift to non-negative

    # prime-based spatial hash → unique int64
    P1, P2 = 73856093, 19349669
    hashes = cell[:, 0] * P1 ^ cell[:, 1] * P2 ^ cell[:, 2]

    _, inverse, counts = torch.unique(hashes, return_inverse=True, return_counts=True)
    counts_per_point = counts[inverse]

    mask = counts_per_point >= min_count
    return points[mask], colors[mask]


# ═══════════════════════════════════════════════════════════════════════════
# 5. Radius Filter (GPU)
# ═══════════════════════════════════════════════════════════════════════════

def filter_by_radius(points: torch.Tensor, colors: torch.Tensor,
                     conf: torch.Tensor, extrinsic: torch.Tensor,
                     radius: float = 0.8):
    """Keep points within *radius* of mean camera centre.  All on GPU.
    Returns (points, colors, conf) — all filtered.
    """
    R = extrinsic[:, :3, :3]
    t = extrinsic[:, :3, 3]
    centers = -torch.bmm(R.transpose(1, 2), t.unsqueeze(-1)).squeeze(-1)
    mean_center = centers.mean(dim=0)

    dists = torch.norm(points - mean_center, dim=1)
    mask = dists < radius
    return points[mask], colors[mask], conf[mask]


# ═══════════════════════════════════════════════════════════════════════════
# 6. Bundle Adjustment (GPU, differentiable)
# ═══════════════════════════════════════════════════════════════════════════

def _axis_angle_to_rotation_matrix(aa: torch.Tensor) -> torch.Tensor:
    """Rodrigues: [N, 3] → [N, 3, 3]."""
    theta = aa.norm(dim=-1, keepdim=True)
    axis = aa / (theta + 1e-8)

    zero = torch.zeros_like(axis[..., 0])
    K = torch.stack([
        zero,          -axis[..., 2],  axis[..., 1],
        axis[..., 2],  zero,          -axis[..., 0],
       -axis[..., 1],  axis[..., 0],  zero,
    ], dim=-1).reshape(*axis.shape[:-1], 3, 3)

    eye = torch.eye(3, device=aa.device, dtype=aa.dtype).expand_as(K)
    sin_t = theta.unsqueeze(-1).sin()
    cos_t = theta.unsqueeze(-1).cos()
    return eye + sin_t * K + (1 - cos_t) * (K @ K)


def bundle_adjust(extrinsic, intrinsic, depth_map, images, depth_conf,
                  n_iters=30, lr=5e-4, n_samples=4096, device='cuda'):
    """Differentiable global BA on GPU.  Refines extrinsics + per-pixel depth."""
    S, H, W, _ = depth_map.shape
    if S < 2:
        return extrinsic, depth_map

    # Clone out of inference-mode → normal float32
    R_init = extrinsic[:, :3, :3].detach().clone().float().to(device)
    t_init = extrinsic[:, :3, 3].detach().clone().float().to(device)
    K      = intrinsic.detach().clone().float().to(device)
    d_init = depth_map.detach().clone().float().to(device)
    conf   = depth_conf.detach().clone().float().to(device)
    imgs   = images.detach().clone().float().to(device)
    imgs_hwc = imgs.permute(0, 2, 3, 1)

    delta_aa = torch.nn.Parameter(torch.zeros(S, 3, device=device))
    delta_t  = torch.nn.Parameter(torch.zeros(S, 3, device=device))
    log_dc   = torch.nn.Parameter(torch.zeros(S, H, W, 1, device=device))

    optimizer = torch.optim.Adam([
        {'params': delta_aa, 'lr': lr},
        {'params': delta_t,  'lr': lr},
        {'params': log_dc,   'lr': lr * 2},
    ])

    K_inv = torch.inverse(K)
    vs, us = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij',
    )
    pix = torch.stack([us, vs, torch.ones_like(us)], dim=-1).reshape(-1, 3)

    conf_flat = conf.reshape(S, -1).clamp(min=1e-6)
    sample_w  = conf_flat / conf_flat.sum(dim=-1, keepdim=True)

    for _ in range(n_iters):
        optimizer.zero_grad()

        dR    = _axis_angle_to_rotation_matrix(delta_aa)
        R     = dR @ R_init
        t     = t_init + delta_t
        depth = d_init * torch.exp(log_dc.clamp(-0.3, 0.3))

        loss, n_pairs = torch.tensor(0.0, device=device), 0

        for i in range(S):
            rays_cam = (K_inv[i] @ pix.T).T
            pts_cam  = rays_cam * depth[i].reshape(-1, 1)
            pts_w    = (R[i].T @ (pts_cam - t[i]).T).T

            with torch.no_grad():
                idx = torch.multinomial(sample_w[i], n_samples, replacement=True)

            pts_s = pts_w[idx]
            col_s = imgs_hwc[i].reshape(-1, 3)[idx]

            for j in range(S):
                if i == j:
                    continue
                pts_cj = (R[j] @ pts_s.T).T + t[j]
                zj     = pts_cj[:, 2:3]
                proj   = (K[j] @ pts_cj.T).T
                uv     = proj[:, :2] / (zj + 1e-8)

                uv_n = torch.empty_like(uv)
                uv_n[:, 0] = 2.0 * uv[:, 0] / (W - 1) - 1.0
                uv_n[:, 1] = 2.0 * uv[:, 1] / (H - 1) - 1.0

                valid = ((uv_n[:, 0].abs() < 0.95) &
                         (uv_n[:, 1].abs() < 0.95) &
                         (zj.squeeze(-1) > 1e-3))
                if valid.sum() < 16:
                    continue

                grid = uv_n[valid].unsqueeze(0).unsqueeze(1)

                d_j_at = F.grid_sample(
                    depth[j, ..., 0][None, None], grid,
                    align_corners=True, mode='bilinear', padding_mode='zeros',
                ).reshape(-1)

                c_j_at = F.grid_sample(
                    imgs[j][None], grid,
                    align_corners=True, mode='bilinear', padding_mode='zeros',
                ).reshape(3, -1).T

                depth_loss = F.huber_loss(zj[valid].squeeze(-1), d_j_at, delta=0.05)
                photo_loss = F.l1_loss(col_s[valid], c_j_at)

                loss = loss + depth_loss + 0.1 * photo_loss
                n_pairs += 1

        if n_pairs > 0:
            loss = loss / n_pairs

        reg = (0.01 * (delta_aa.pow(2).sum() + delta_t.pow(2).sum())
               + 0.005 * log_dc.pow(2).sum() / (S * H * W))
        (loss + reg).backward()
        optimizer.step()

    with torch.no_grad():
        dR    = _axis_angle_to_rotation_matrix(delta_aa)
        R_out = dR @ R_init
        t_out = t_init + delta_t
        refined_ext   = torch.cat([R_out, t_out.unsqueeze(-1)], dim=-1)
        refined_depth = d_init * torch.exp(log_dc.clamp(-0.3, 0.3))
    return refined_ext.to(extrinsic.dtype), refined_depth.to(depth_map.dtype)


# ═══════════════════════════════════════════════════════════════════════════
# Main Loop
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--port", type=int, default=8080, help="Viser port")
    parser.add_argument("--colmap_path", type=str,
                        default="/home/anurag/stream_captures/sparse/0")
    parser.add_argument("--ba_iters", type=int, default=30,
                        help="Bundle-adjustment iterations (0 = off)")
    parser.add_argument("--target_n", type=int, default=100_000_0,
                        help="Max points for visualisation")
    args = parser.parse_args()

    cfg   = load_config(CONFIG_PATH)
    nodes = cfg.get("nodes", [])

    print(f"Loading COLMAP model from {args.colmap_path}...")
    colmap_cameras, colmap_images, _ = read_model(args.colmap_path, ext='.bin')

    captures = []
    for node in nodes:
        name = node.get("name", "")
        host = node.get("host", "")
        mac  = node.get("MAC", "")
        for port in node.get("ports", []):
            captures.append(StreamCapture(port, name, host, mac))
    print(f"Starting {len(captures)} streams...")
    for c in captures:
        c.start()

    # VGGT
    torch.set_float32_matmul_precision('high')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = (torch.bfloat16
             if device == "cuda" and torch.cuda.get_device_capability()[0] >= 8
             else torch.float16 if device == "cuda" else torch.float32)
    print(f"Loading VGGT on {device} ({dtype})...")

    model = VGGT(chunk_size=args.chunk_size)
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    try:
        model.load_state_dict(torch.hub.load_state_dict_from_url(_URL, progress=True))
    except Exception as e:
        print(f"Model load error: {e}")
    model.eval().to(device).to(dtype)
    model.track_head = None

    server = viser.ViserServer(port=args.port)
    print(f"Viser at http://localhost:{args.port}")

    timer = CudaTimer()
    count = 0

    try:
        while True:
            # ── Collect frames ───────────────────────────────────────
            frames = []
            for c in captures:
                f = c.read()
                if f is not None:
                    frames.append(f)
            if len(frames) < 2:
                time.sleep(0.1)
                continue

            wall_start = time.time()

            # ── 1  Preprocess + corner mask (GPU) ────────────────────
            timer.start()
            input_tensor = preprocess_frames_gpu(frames, 518, device, dtype)
            t_preprocess = timer.stop()

            # ── 2  VGGT Inference ────────────────────────────────────
            timer.start()
            with torch.no_grad():
                predictions = model(input_tensor)
                if count == 0:
                    extrinsic, intrinsic = pose_encoding_to_extri_intri(
                        predictions['pose_enc'], input_tensor.shape[-2:],
                    )
                    extrinsic = extrinsic.squeeze(0)
                    intrinsic = intrinsic.squeeze(0)
                depth_map  = predictions['depth'].squeeze(0)       # [S,H,W,1]
                depth_conf = predictions['depth_conf'].squeeze(0)  # [S,H,W]
            t_inference = timer.stop()

            # ── 3  Bundle Adjustment (GPU) ───────────────────────────
            if args.ba_iters > 0:
                timer.start()
                extrinsic, depth_map = bundle_adjust(
                    extrinsic, intrinsic, depth_map, input_tensor, depth_conf,
                    n_iters=args.ba_iters, n_samples=4096, device=device,
                )
                t_ba = timer.stop()
            else:
                t_ba = 0.0

            # ── 4  Unproject (GPU) ───────────────────────────────────
            timer.start()
            points   = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)
            if isinstance(points, np.ndarray):
                points = torch.from_numpy(points).to(device)
            colors4d = input_tensor.float().permute(0, 2, 3, 1)  # [S,H,W,3]
            
            points_flat = points.reshape(-1, 3).float()
            colors_flat = colors4d.reshape(-1, 3)
            conf_flat   = depth_conf.reshape(-1).float()
            t_unproject = timer.stop()

            # ── 5  Radius filter (GPU) ───────────────────────────────
            timer.start()
            points_flat, colors_flat, conf_flat = filter_by_radius(
                points_flat, colors_flat, conf_flat, extrinsic, radius=0.8,
            )
            t_radius = timer.stop()

            # ── 6  Confidence filter (GPU) ───────────────────────────
            timer.start()
            points_flat, colors_flat = filter_by_confidence(
                points_flat, colors_flat, conf_flat, threshold=0.3,
            )
            t_confidence = timer.stop()

            # ── 7  Intensity filter (GPU) ────────────────────────────
            timer.start()
            points_flat, colors_flat = filter_by_intensity(
                points_flat, colors_flat, threshold=0.01,
            )
            t_intensity = timer.stop()

            # ── 8  Density filter (GPU) ──────────────────────────────
            timer.start()
            points_flat, colors_flat = filter_by_density_gpu(
                points_flat, colors_flat, voxel_size=0.01, min_count=5,
            )
            t_density = timer.stop()

            # ── 9  Downsample + send to viser ────────────────────────
            timer.start()
            n_pts = points_flat.shape[0]
            if n_pts > args.target_n:
                idx = torch.randperm(n_pts, device=device)[:args.target_n]
                p_vis = points_flat[idx]
                c_vis = colors_flat[idx]
            else:
                p_vis = points_flat
                c_vis = colors_flat

            # Single CPU transfer at the very end
            p_np = p_vis.cpu().numpy()
            c_np = c_vis.cpu().numpy()
            t_sample = timer.stop()

            server.scene.add_point_cloud(
                "/vggt/cloud", points=p_np, colors=c_np, point_size=0.0015,
            )

            wall_dt = time.time() - wall_start
            print(
                f"[Frame {count:4d}] "
                f"pre={t_preprocess:5.1f}ms  "
                f"inf={t_inference:5.1f}ms  "
                f"BA={t_ba:5.1f}ms  "
                f"unproj={t_unproject:5.1f}ms  "
                f"rad={t_radius:4.1f}ms  "
                f"conf={t_confidence:4.1f}ms  "
                f"int={t_intensity:4.1f}ms  "
                f"dens={t_density:4.1f}ms  "
                f"samp={t_sample:4.1f}ms  "
                f"| wall={wall_dt:.3f}s  FPS={1.0/wall_dt:.1f}  "
                f"pts={n_pts}→{p_np.shape[0]}  cams={len(frames)}"
            )
            count += 1

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Stopping streams...")
        for c in captures:
            c.stop()


if __name__ == "__main__":
    main()
