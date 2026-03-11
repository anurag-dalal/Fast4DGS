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

# ── Add VGGT-X to path ──────────────────────────────────────────────────
VGGT_PATH = Path(__file__).resolve().parent / "VGGT-X"
if str(VGGT_PATH) not in sys.path:
    sys.path.append(str(VGGT_PATH))
from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

# ── ArUco utilities ──────────────────────────────────────────────────────
ARUCO_PATH = Path(__file__).resolve().parent / "ArUCo-Markers-Pose-Estimation-Generation-Python"
if str(ARUCO_PATH) not in sys.path:
    sys.path.insert(0, str(ARUCO_PATH))
from utils import ARUCO_DICT

# Config paths
CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "stream_config.json"
COLMAP_PATH = "/home/anurag/stream_captures/sparse/0"
IMAGE_FOLDER_PATH = "/home/anurag/codes_ole/pose_estimation/Images/stream_captures_churaco_v1"

# loads
def load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)
def load_vggt(device: str, dtype: torch.dtype, chunk_size: int = 512) -> VGGT:
    """Load the VGGT-1B model."""
    model = VGGT(chunk_size=chunk_size)
    url = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(url, progress=True))
    model.eval()
    model = model.to(device).to(dtype)
    model.track_head = None
    return model

# run vggt
def run_vggt(model: VGGT, input_tensor: torch.Tensor, device: str):


from dataloaders.live_stream_dataset import LiveStreamDataset
from dataloaders.static_dataset import StaticDataset
def main():

    parser = argparse.ArgumentParser(
        description="VGGT point-cloud viewer from a folder of images")
    parser.add_argument("--image_dir", default='/home/anurag/codes_ole/pose_estimation/Images/stream_captures_churaco_v1', type=str,
                        help="Path to folder containing images")
    parser.add_argument("--mask_dir", default='/home/anurag/stream_captures/output_sam', type=str,
                        help="Path to folder containing masks")
    parser.add_argument("--use_mask", action="store_true", default=False,
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
    parser.add_argument("--aruco_type", type=str, default="DICT_5X5_100",
                        help="ArUco dictionary type (e.g. DICT_5X5_100)")
    parser.add_argument("--marker_size", type=float, default=0.05,
                        help="Sphere size for ArUco markers in viewer")
    parser.add_argument("--no_aruco", action="store_true", default=False,
                        help="Skip ArUco marker detection")
    args = parser.parse_args()


    config = load_config(CONFIG_PATH)
    dataset = LiveStreamDataset(config, COLMAP_PATH)

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

    while True:
        try:
            input_tensor = dataset.get_processed_frames()
            
        except Exception as e:
            print(f"Error processing frames: {e}")
            time.sleep(1)  # Avoid tight loop on error

    # dataset = StaticDataset(IMAGE_FOLDER_PATH, COLMAP_PATH)
    # input_tensor = dataset.get_processed_frames()
    # print(f"Processed input tensor shape: {input_tensor.shape}")





if __name__ == "__main__":
    main()