#!/usr/bin/env python3
import sys
import os
import time
import json
import threading
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import viser
import viser.transforms as vtf
from torchvision import transforms as TF

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

# COLMAP helpers (from local dataset/)
DATASET_PATH = Path(__file__).resolve().parent / "dataset"
if str(DATASET_PATH) not in sys.path:
    sys.path.insert(0, str(DATASET_PATH))
from read_write_model import read_model, qvec2rotmat

# Config path
CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "stream_config.json"

def load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)

class StreamCapture:
    """Grabs frames from one GStreamer UDP/RTP H264 stream in a background thread."""
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
        self._thread = threading.Thread(target=self._open_and_loop, args=(timeout, retry_interval), daemon=True)
        self._thread.start()

    def _open_and_loop(self, timeout: float, retry_interval: float):
        # Optimized pipeline for low latency and quality
        gst = (
            f'udpsrc port={self.port} buffer-size=200000000 '
            f'caps="application/x-rtp, media=video, clock-rate=90000, encoding-name=H265, payload=96" ! '
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

def preprocess_frames(frames, device: str, target_size: int = 518) -> torch.Tensor:
    """GPU preprocessing.

    Inputs are BGR uint8 numpy arrays (H, W, 3).
    Output is RGB float32 tensor on `device` with shape [B, 3, target_size, target_size] in [0, 1].

    Note: capture/decoding from OpenCV is inherently CPU; this function ensures the *image ops*
    (pad/resize/channel swap/normalize) run on the GPU.
    """
    if len(frames) == 0:
        raise ValueError("preprocess_frames: empty frame list")

    images = []
    for frame in frames:
        if frame is None:
            continue

        # HWC uint8 (CPU) -> HWC uint8 (GPU)
        img = torch.from_numpy(frame).to(device, non_blocking=False)
        # BGR -> RGB
        img = img.flip(-1)
        # HWC -> NCHW, float32 in [0,1]
        img = img.permute(2, 0, 1).unsqueeze(0).to(torch.float32).div_(255.0)

        _, _, h, w = img.shape
        max_dim = h if h >= w else w
        pad_h = max_dim - h
        pad_w = max_dim - w
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        if pad_h != 0 or pad_w != 0:
            img = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)

        # Resize on GPU
        img = F.interpolate(img, size=(target_size, target_size), mode="bicubic", align_corners=False)
        images.append(img)

    if len(images) == 0:
        raise ValueError("preprocess_frames: no valid frames")

    return torch.cat(images, dim=0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--port", type=int, default=8080, help="Viser port")
    parser.add_argument("--scale", type=int, default=8, help="Downscale frame for VGGT (unused currently)")
    parser.add_argument("--colmap_path", type=str, default="/home/anurag/stream_captures/sparse/0",
                        help="Path to COLMAP sparse model directory")
    args = parser.parse_args()

    # Load Config
    cfg = load_config(CONFIG_PATH)
    nodes = cfg.get("nodes", [])
    
    captures = []
    print(f"Loading streams from {CONFIG_PATH}...")
    for node in nodes:
        name = node.get("name", "")
        host = node.get("host", "")
        mac = node.get("MAC", "")
        for port in node.get("ports", []):
            captures.append(StreamCapture(port, name, host, mac))

    # Start Streams
    print(f"Starting {len(captures)} streams...")
    for c in captures:
        c.start()

    # ── Load COLMAP calibration ──────────────────────────────────────────
    print(f"Loading COLMAP model from {args.colmap_path}...")
    colmap_cameras, colmap_images, _ = read_model(args.colmap_path, ext='.bin')
    name_to_image = {img.name: img for img in colmap_images.values()}

    # Precompute intrinsic / extrinsic per capture index.
    # Intrinsics are adjusted for the preprocessing pipeline:
    #   1) pad to square (max_dim × max_dim)
    #   2) resize to target_size × target_size
    TARGET_SIZE = 518
    orig_h, orig_w = cfg.get("stream_height", 1080), cfg.get("stream_width", 1920)
    max_dim = max(orig_h, orig_w)
    pad_top = (max_dim - orig_h) // 2
    pad_left = (max_dim - orig_w) // 2
    scale_factor = TARGET_SIZE / max_dim

    colmap_extrinsics = {}   # capture index -> np.ndarray (3, 4)
    colmap_intrinsics = {}   # capture index -> np.ndarray (3, 3)

    for idx in range(len(captures)):
        cam_name = f"cam{idx}.png"
        if cam_name not in name_to_image:
            print(f"  [WARN] {cam_name} not found in COLMAP model – skipping")
            continue
        col_img = name_to_image[cam_name]
        col_cam = colmap_cameras[col_img.camera_id]

        # Extrinsic: COLMAP stores world-to-camera [R_wc | t_wc].
        # VGGT uses OpenGL-ish convention (Y-down, Z-back flipped).
        # Convert: R_cw = R_wc^T,  then R_vggt = R_cw @ S,  t_vggt = -R_cw @ t_wc
        # where S = diag(1, -1, -1) flips Y and Z axes.
        S = np.diag([1.0, -1.0, -1.0])
        R_wc = qvec2rotmat(col_img.qvec)        # (3, 3)
        t_wc = col_img.tvec.reshape(3, 1)        # (3, 1)
        # R_cw = R_wc.T                            # camera-to-world rotation
        R_vggt = R_wc @ S                        # axis-flipped rotation
        t_vggt = -R_wc @ t_wc                    # camera centre in world
        colmap_extrinsics[idx] = np.hstack([R_vggt, t_vggt]).astype(np.float64)

        # Intrinsic (ignore distortion) – FULL_OPENCV: fx, fy, cx, cy, ...
        fx, fy, cx, cy = col_cam.params[0], col_cam.params[1], col_cam.params[2], col_cam.params[3]
        # Adjust for pad-then-resize
        fx_adj = fx * scale_factor
        fy_adj = fy * scale_factor
        cx_adj = (cx + pad_left) * scale_factor
        cy_adj = (cy + pad_top) * scale_factor
        colmap_intrinsics[idx] = np.array([
            [fx_adj, 0.0,    cx_adj],
            [0.0,    fy_adj, cy_adj],
            [0.0,    0.0,    1.0   ],
        ], dtype=np.float64)

    print(f"  COLMAP calibration loaded for {len(colmap_extrinsics)}/{len(captures)} cameras")

    # Load VGGT
    torch.set_float32_matmul_precision('high')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        dtype = torch.float32
    print(f"Loading VGGT on {device} ({dtype})...")
    
    model = VGGT(chunk_size=args.chunk_size)
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    try:
        model.load_state_dict(torch.hub.load_state_dict_from_url(_URL, progress=True))
    except Exception as e:
        print(f"Failed to load model from URL: {e}")
        print("Ensure internet connection or manually place model.")
        # Attempt to continue if model path cached? or fail.
    
    model.eval()
    model = model.to(device).to(dtype)
    model.track_head = None

    # Viser
    server = viser.ViserServer(port=args.port)
    print(f"Viser running at http://localhost:{args.port}")

    try:
        while True:
            # Collect frames
            frames = []
            
            # We want frames from all active streams
            # If some take too long, we might default to previous or black?
            # For now, just take what is available
            active_frames = 0
            frame_indices = []      # which capture index each frame belongs to
            for i, c in enumerate(captures):
                f = c.read()
                if f is not None and i in colmap_extrinsics:
                    frames.append(f)
                    frame_indices.append(i)
                    active_frames += 1
            
            if active_frames < 2:
                # Need at least 2 frames for meaningful structure usually?
                # Or just wait until we have something.
                time.sleep(0.1)
                continue
                
            # Preprocess (GPU)
            start_t = time.time()
            input_tensor = preprocess_frames(frames, device=device, target_size=518)
            model_input = input_tensor.to(dtype)
            
            # Inference – only depth is needed from VGGT; poses come from COLMAP
            with torch.no_grad():
                predictions = model(model_input)
                extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions['pose_enc'], model_input.shape[-2:])
                extrinsic = extrinsic.squeeze(0)
                intrinsic = intrinsic.squeeze(0)
                # print(f"Predicted extrinsics shape: {extrinsic.shape}, intrinsics shape: {intrinsic.shape}")
                # print(intrinsic)
                depth_map = predictions['depth'].squeeze(0)  # [S, H, W, 1]

            # Build COLMAP extrinsic / intrinsic tensors for the active frames
            extr_np = np.stack([colmap_extrinsics[i] for i in frame_indices])  # (S, 3, 4)
            # intr_np = np.stack([colmap_intrinsics[i] for i in frame_indices])  # (S, 3, 3)
            extrinsic = torch.tensor(extr_np, device=device, dtype=torch.float32)
            # intrinsic = torch.tensor(intr_np, device=device, dtype=torch.float32)
            # print(intrinsic)
            # print(f"COLMAP extrinsics shape: {extrinsic.shape}, intrinsics shape: {intrinsic.shape}")

            # Unproject using COLMAP calibration
            points = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)
            # points: [S, H, W, 3]
            
            # Flatten on GPU
            if not isinstance(points, torch.Tensor):
                points = torch.as_tensor(points, device=device)
            points_flat = points.reshape(-1, 3).to(torch.float32)
            colors_flat = input_tensor.permute(0, 2, 3, 1).reshape(-1, 3)  # float32 [0,1]

            # Filter by radius from mean camera position
            # extrinsic: [S, 3, 4]
            # Camera center in world frame: C = -R^T * t
            R = extrinsic[:, :3, :3]
            t = extrinsic[:, :3, 3]
            # Batched calculation of centers: (S, 3)
            centers = -torch.bmm(R.transpose(1, 2), t.unsqueeze(-1)).squeeze(-1)
            mean_center = torch.mean(centers, dim=0)  # (3,)

            radius = 0.8  # Adjustable radius

            dists = torch.linalg.norm(points_flat - mean_center.unsqueeze(0), dim=1)
            radius_mask = dists < radius
            points_flat = points_flat[radius_mask]
            colors_flat = colors_flat[radius_mask]
            
            # RAMBO: Remove all dark greyish to black points
            # We filter based on intensity (mean of RGB). 
            # Threshold 0.1 roughly corresponds to dark grey.
            intensity = colors_flat.mean(dim=1)
            valid_color_mask = intensity > 0.05
            points_flat = points_flat[valid_color_mask]
            colors_flat = colors_flat[valid_color_mask]

            # Filter invalid points (depth > 0 or not infinite)
            # depth_map usually has 0 for invalid?
            # unproject handles it.
            
            # Sample for viz
            n_points = int(points_flat.shape[0])
            target_n = 100000
            if n_points > target_n:
                idx = torch.randperm(n_points, device=points_flat.device)[:target_n]
                p_vis = points_flat[idx]
                c_vis = colors_flat[idx]
            else:
                p_vis = points_flat
                c_vis = colors_flat

            # Viser expects CPU numpy arrays
            p_vis = p_vis.detach().cpu().numpy()
            c_vis = c_vis.detach().cpu().numpy()
            
            server.scene.add_point_cloud(
                "/vggt/cloud",
                points=p_vis,
                colors=c_vis,
                point_size=0.0015
            )
            
            dt = time.time() - start_t
            print(f"Update: {dt:.3f}s, FPS: {1.0/dt:.1f}, Streams: {active_frames}")
            
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
