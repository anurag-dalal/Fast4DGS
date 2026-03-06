#!/usr/bin/env python3
"""
Infer depth & poses from a folder of images using VGGT, then display
the fused point cloud in a Viser viewer.  A mask folder (same filenames)
controls which pixels are back-projected (mask > 128 = valid).

Usage:
    python vggt_infer_on_folder.py --image_dir /path/to/images \
           --mask_dir /path/to/masks --port 8080
"""
import sys
import time
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.spatial import cKDTree
import viser
from torchvision import transforms as TF

# ── Add VGGT-X to path ──────────────────────────────────────────────────
VGGT_PATH = Path(__file__).resolve().parent / "VGGT-X"
if str(VGGT_PATH) not in sys.path:
    sys.path.append(str(VGGT_PATH))

from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

# ═════════════════════════════════════════════════════════════════════════
#  CONFIG – change these or pass via CLI
# ═════════════════════════════════════════════════════════════════════════
DEFAULT_IMAGE_DIR = "/home/anurag/stream_captures/images"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# ── Camera calibration (from dewarper config) ────────────────────────────
# projection-type=3  (rational model, 8 distortion coeffs)
CALIB_FX = 1042.00
CALIB_FY = 1042.00
CALIB_CX = 950.6477088
CALIB_CY = 557.5285168
CALIB_DIST = np.array([
    -0.0516463965,    # k1
    -0.04747710885,   # k2
    -0.0001566917679, # p1
     0.0002697267978, # p2
     0.01013947186,   # k3
     0.3133340326,    # k4
    -0.1464375728,    # k5
     0.02119491113,   # k6
], dtype=np.float64)
CALIB_W = 1920
CALIB_H = 1080


def build_undistort_maps(w: int = CALIB_W, h: int = CALIB_H):
    """Pre-compute undistortion remap tables (computed once, reused for all frames)."""
    K = np.array([
        [CALIB_FX, 0,        CALIB_CX],
        [0,        CALIB_FY, CALIB_CY],
        [0,        0,        1       ],
    ], dtype=np.float64)
    # newCameraMatrix = K  (keep same principal point / focal length)
    map1, map2 = cv2.initUndistortRectifyMap(
        K, CALIB_DIST, None, K, (w, h), cv2.CV_32FC1)
    return map1, map2


def undistort_image(img: np.ndarray, map1, map2,
                    interpolation=cv2.INTER_LINEAR) -> np.ndarray:
    """Remap an image (or mask) using pre-computed undistortion maps."""
    return cv2.remap(img, map1, map2, interpolation,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


# ═════════════════════════════════════════════════════════════════════════
#  Helpers  (adapted from live_pc_viewer.py)
# ═════════════════════════════════════════════════════════════════════════

def mask_corners(img: np.ndarray, frac: float = 0.08) -> np.ndarray:
    """Paint solid black squares in all four corners."""
    h, w = img.shape[:2]
    s = int(h * frac)
    img[:s, :s] = 0
    img[:s, w - s:] = 0
    img[h - s:, :s] = 0
    img[h - s:, w - s:] = 0
    return img


def density_filter(points: np.ndarray, colors: np.ndarray,
                   radius: float = 0.02, min_neighbors: int = 5):
    """Remove isolated / floating points using k-NN density estimation."""
    if points.shape[0] < min_neighbors + 1:
        return points, colors
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=min_neighbors + 1)
    mask = dists[:, -1] <= radius
    return points[mask], colors[mask]


def preprocess_frames(frames: list[np.ndarray],
                      target_size: int = 518) -> torch.Tensor:
    """BGR numpy frames → [B, 3, H, W] tensor, padded & resized."""
    images = []
    to_tensor = TF.ToTensor()
    for frame in frames:
        frame = mask_corners(frame)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        w, h = img.size
        max_dim = max(w, h)
        left = (max_dim - w) // 2
        top = (max_dim - h) // 2

        square = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))
        square.paste(img, (left, top))
        square = square.resize((target_size, target_size),
                               Image.Resampling.BICUBIC)
        images.append(to_tensor(square))
    return torch.stack(images)


def load_images(image_dir: str | Path) -> tuple[list[np.ndarray], list[str]]:
    """Load all images from a directory, sorted by name.
    Returns (BGR arrays, list of stems for mask matching)."""
    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    paths = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise RuntimeError(f"No images found in {image_dir}")

    frames, stems = [], []
    for p in paths:
        img = cv2.imread(str(p))
        if img is not None:
            frames.append(img)
            stems.append(p.stem)
            print(f"  loaded {p.name}  ({img.shape[1]}×{img.shape[0]})")
        else:
            print(f"  WARNING: could not read {p.name}, skipping")
    return frames, stems


def load_masks(mask_dir: str | Path, stems: list[str],
               target_size: int, orig_sizes: list[tuple[int, int]]
               ) -> np.ndarray:
    """Load masks that match image stems, apply the same square-pad + resize
    used for the images, and threshold at 128.

    Returns:
        masks_resized: bool np.ndarray [S, target_size, target_size]
    """
    mask_dir = Path(mask_dir)
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    masks = []
    for stem, (orig_h, orig_w) in zip(stems, orig_sizes):
        # Try common extensions for the mask
        mask_path = None
        for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"):
            candidate = mask_dir / f"{stem}{ext}"
            if candidate.exists():
                mask_path = candidate
                break
        if mask_path is None:
            raise FileNotFoundError(
                f"No mask found for '{stem}' in {mask_dir}")

        m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise RuntimeError(f"Could not read mask: {mask_path}")
        print(f"  mask {mask_path.name}  ({m.shape[1]}×{m.shape[0]})")

        # Replicate the same square-pad + resize that preprocess_frames does
        max_dim = max(orig_h, orig_w)
        pad_top = (max_dim - orig_h) // 2
        pad_left = (max_dim - orig_w) // 2

        # Resize mask to original image size first (in case mask resolution differs)
        m = cv2.resize(m, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        square = np.zeros((max_dim, max_dim), dtype=np.uint8)
        square[pad_top:pad_top + orig_h, pad_left:pad_left + orig_w] = m
        square = cv2.resize(square, (target_size, target_size),
                            interpolation=cv2.INTER_NEAREST)
        masks.append(square > 128)

    return np.stack(masks)  # [S, H, W]  bool


def load_vggt(device: str, dtype: torch.dtype, chunk_size: int = 512) -> VGGT:
    """Load the VGGT-1B model."""
    model = VGGT(chunk_size=chunk_size)
    url = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(url, progress=True))
    model.eval()
    model = model.to(device).to(dtype)
    model.track_head = None
    return model


# ═════════════════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="VGGT point-cloud viewer from a folder of images")
    parser.add_argument("--image_dir", default='/home/anurag/stream_captures/images_v0', type=str,
                        help="Path to folder containing images")
    parser.add_argument("--mask_dir", default='/home/anurag/stream_captures/output_sam', type=str,
                        help="Path to folder containing masks")
    parser.add_argument("--use_mask", action="store_true", default=True,
                        help="Use mask to filter pixels (default: False)")
    parser.add_argument("--no_mask", dest="use_mask", action="store_false",
                        help="Disable mask, use all pixels")
    parser.add_argument("--undistort", action="store_true", default=True,
                        help="Undistort images & masks with built-in calibration (default: True)")
    parser.add_argument("--no_undistort", dest="undistort", action="store_false",
                        help="Skip undistortion")
    parser.add_argument("--port", type=int, default=8080,
                        help="Viser server port")
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--target_size", type=int, default=518,
                        help="Resize longest side to this before inference")
    parser.add_argument("--max_points", type=int, default=200_000,
                        help="Max points to display")
    parser.add_argument("--conf_threshold", type=float, default=0.3,
                        help="Depth confidence threshold")
    parser.add_argument("--radius", type=float, default=2.0,
                        help="Keep points within this radius of mean camera center")
    parser.add_argument("--density_radius", type=float, default=0.02,
                        help="Density filter: max distance to k-th neighbor")
    parser.add_argument("--density_k", type=int, default=10,
                        help="Density filter: min neighbors within radius")
    parser.add_argument("--point_size", type=float, default=0.002,
                        help="Point size in the viewer")
    args = parser.parse_args()

    # ── 1. Load images & masks ──────────────────────────────────────────
    print(f"Loading images from {args.image_dir} ...")
    frames, stems = load_images(args.image_dir)
    n_images = len(frames)
    print(f"Loaded {n_images} images")
    if n_images < 2:
        print("Need at least 2 images for multi-view reconstruction.")
        sys.exit(1)

    # ── 1b. Undistort ───────────────────────────────────────────────────
    if args.undistort:
        print("Building undistortion maps …")
        map1, map2 = build_undistort_maps()
        print(f"Undistorting {n_images} images …")
        frames = [undistort_image(f, map1, map2, cv2.INTER_LINEAR) for f in frames]
        print("Undistortion done.")

    # Original sizes (h, w) before padding – needed to replicate padding on masks
    orig_sizes = [(f.shape[0], f.shape[1]) for f in frames]

    if args.use_mask:
        print(f"Loading masks from {args.mask_dir} ...")
        # Load raw masks, undistort, then pad+resize
        mask_dir = Path(args.mask_dir)
        raw_masks = []
        for stem in stems:
            mask_path = None
            for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"):
                candidate = mask_dir / f"{stem}{ext}"
                if candidate.exists():
                    mask_path = candidate
                    break
            if mask_path is None:
                raise FileNotFoundError(f"No mask found for '{stem}' in {mask_dir}")
            m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if m is None:
                raise RuntimeError(f"Could not read mask: {mask_path}")
            print(f"  mask {mask_path.name}  ({m.shape[1]}×{m.shape[0]})")
            raw_masks.append(m)

        # Undistort masks with nearest-neighbor interpolation
        if args.undistort:
            print("Undistorting masks …")
            raw_masks = [undistort_image(m, map1, map2, cv2.INTER_NEAREST)
                         for m in raw_masks]

        # Apply same square-pad + resize as images, then threshold
        masks_list = []
        for m, (orig_h, orig_w) in zip(raw_masks, orig_sizes):
            m = cv2.resize(m, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            max_dim = max(orig_h, orig_w)
            pad_top = (max_dim - orig_h) // 2
            pad_left = (max_dim - orig_w) // 2
            square = np.zeros((max_dim, max_dim), dtype=np.uint8)
            square[pad_top:pad_top + orig_h, pad_left:pad_left + orig_w] = m
            square = cv2.resize(square, (args.target_size, args.target_size),
                                interpolation=cv2.INTER_NEAREST)
            masks_list.append(square > 128)
        masks = np.stack(masks_list)  # [S, H, W] bool
        print(f"Masks shape: {masks.shape}  (True = valid pixel)")
    else:
        print("Mask disabled — using all pixels.")
        masks = None

    # ── 2. Load model ───────────────────────────────────────────────────
    torch.set_float32_matmul_precision("high")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        dtype = (torch.bfloat16
                 if torch.cuda.get_device_capability()[0] >= 8
                 else torch.float16)
    else:
        dtype = torch.float32
    print(f"Device: {device}  dtype: {dtype}")

    print("Loading VGGT model …")
    model = load_vggt(device, dtype, args.chunk_size)
    print("Model loaded.")

    # ── 3. Preprocess & infer ───────────────────────────────────────────
    print("Preprocessing …")
    input_tensor = preprocess_frames(frames, target_size=args.target_size)
    input_tensor = input_tensor.to(device, dtype)
    print(f"Input tensor shape: {list(input_tensor.shape)}")

    print("Running VGGT inference …")
    t0 = time.time()
    with torch.inference_mode():
        predictions = model(input_tensor)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(
            predictions["pose_enc"], input_tensor.shape[-2:])
        extrinsic = extrinsic.squeeze(0)   # [S, 3, 4]
        intrinsic = intrinsic.squeeze(0)   # [S, 3, 3]
        depth_map = predictions["depth"].squeeze(0)       # [S, H, W, 1]
        depth_conf = predictions["depth_conf"].squeeze(0)  # [S, H, W]

    print(f"Inference done in {time.time() - t0:.2f}s")

    # ── 4. Unproject to 3-D ─────────────────────────────────────────────
    points = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)
    # points: [S, H, W, 3]

    colors = input_tensor.permute(0, 2, 3, 1).cpu().float().numpy()  # [S,H,W,3]

    if isinstance(points, torch.Tensor):
        points = points.cpu().float().numpy()
    if isinstance(depth_conf, torch.Tensor):
        depth_conf = depth_conf.cpu().float().numpy()

    # ── 5. Apply mask & filter ──────────────────────────────────────────
    points_flat = points.reshape(-1, 3)
    colors_flat = colors.reshape(-1, 3)
    conf_flat = depth_conf.reshape(-1)
    total_pts = points_flat.shape[0]

    # 5a. Keep only mask-valid pixels (or all if --no_mask)
    if masks is not None:
        mask_flat = masks.reshape(-1)      # [S*H*W] bool
        points_flat = points_flat[mask_flat]
        colors_flat = colors_flat[mask_flat]
        conf_flat = conf_flat[mask_flat]
        print(f"After mask filter: {points_flat.shape[0]:,}  (from {mask_flat.sum():,} valid mask pixels)")
    else:
        print(f"No mask — using all {total_pts:,} pixels")

    # 5b. Radius from mean camera center
    with torch.inference_mode():
        R = extrinsic[:, :3, :3]
        t = extrinsic[:, :3, 3]
        centers = -torch.bmm(R.transpose(1, 2), t.unsqueeze(-1)).squeeze(-1)
        mean_center = centers.mean(dim=0).cpu().numpy()

    dists = np.linalg.norm(points_flat - mean_center, axis=1)
    keep = dists < args.radius
    points_flat = points_flat[keep]
    colors_flat = colors_flat[keep]
    conf_flat = conf_flat[keep]
    print(f"After radius filter ({args.radius}): {points_flat.shape[0]:,}")

    # 5c. Confidence threshold
    keep = conf_flat > args.conf_threshold
    points_flat = points_flat[keep]
    colors_flat = colors_flat[keep]
    print(f"After confidence filter (>{args.conf_threshold}): {points_flat.shape[0]:,}")

    # 5d. Density filter — remove isolated floaters
    points_flat, colors_flat = density_filter(
        points_flat, colors_flat,
        radius=args.density_radius,
        min_neighbors=args.density_k)
    print(f"After density filter: {points_flat.shape[0]:,}")

    # 5e. Subsample if too many
    n = points_flat.shape[0]
    if n > args.max_points:
        idx = np.random.choice(n, args.max_points, replace=False)
        points_flat = points_flat[idx]
        colors_flat = colors_flat[idx]
        print(f"Subsampled to {args.max_points:,}")

    # ── 6. Viser viewer ────────────────────────────────────────────────
    server = viser.ViserServer(port=args.port)
    print(f"\nViser running → http://localhost:{args.port}")

    server.scene.add_point_cloud(
        "/pointcloud",
        points=points_flat,
        colors=colors_flat,
        point_size=args.point_size,
    )

    # Add camera frustums for reference
    for i in range(extrinsic.shape[0]):
        ext_np = extrinsic[i].cpu().float().numpy()  # [3, 4]
        R_cam = ext_np[:3, :3]
        t_cam = ext_np[:3, 3]
        cam_center = -R_cam.T @ t_cam

        # wxyz quaternion from rotation matrix  (viser convention)
        import viser.transforms as vtf
        wxyz = vtf.SO3.from_matrix(R_cam.T).wxyz

        server.scene.add_camera_frustum(
            f"/cameras/{i}",
            fov=60.0,  # approximate
            aspect=1.0,
            scale=0.06,
            wxyz=wxyz,
            position=cam_center,
            color=(80, 180, 255),
        )

    print(f"Displaying {points_flat.shape[0]:,} points from {n_images} images.")
    print("Press Ctrl+C to exit.\n")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Shutting down.")


if __name__ == "__main__":
    main()
