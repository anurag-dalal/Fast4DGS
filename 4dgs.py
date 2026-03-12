## ID 0 5 4 -> world coordinate system (origin at first camera)
## ID 1 2 3 -> ROBOT frame

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
from utilities.vggt_utils import VGGTUtils
from utilities.aruco_utils import ArUcoUtils
from utilities.pp_pointcloud import PostProcessPointCloud
from utilities.mesh_utils import pointcloud_to_mesh
from utilities.ros_utils import ROSUtils

import viser
import viser.transforms as vtf

import json
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import viser
import yaml
from gsplat.color_correct import color_correct_affine, color_correct_quadratic
from datasets.colmap import Dataset, Parser
from datasets.traj import (
    generate_ellipse_path_z,
    generate_interpolated_path,
    generate_spiral_path,
)
from fused_ssim import fused_ssim
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal, assert_never
from utils import AppearanceOptModule, CameraOptModule, knn, rgb_to_sh, set_random_seed

from gsplat import export_splats
from gsplat.compression import PngCompression
from gsplat.distributed import cli
from gsplat.optimizers import SelectiveAdam
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from gsplat_viewer import GsplatViewer, GsplatRenderTabState
from nerfview import CameraState, RenderTabState, apply_float_colormap


# Config paths
CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "stream_config.json"
PP_POINTCLOUD_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "pp_pointcloud.json"
COLMAP_PATH = "/home/anurag/stream_captures/sparse/0"
IMAGE_FOLDER_PATH = "/home/anurag/codes_ole/pose_estimation/Images/stream_captures_churaco_v1"

# loads
def load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


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
    parser.add_argument("--viser", action="store_true", default=False,
                        help="Skip ArUco marker detection")    
    parser.add_argument("--port", type=int, default=8080,
                        help="Viser server port")
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--target_size", type=int, default=518,
                        help="Resize longest side to this before inference")
    parser.add_argument("--point_size", type=float, default=0.002,
                        help="Point size in the viewer")
    parser.add_argument("--aruco_type", type=str, default="DICT_5X5_100",
                        help="ArUco dictionary type (e.g. DICT_5X5_100)")
    parser.add_argument("--marker_size", type=float, default=0.05,
                        help="Sphere size for ArUco markers in viewer")
    parser.add_argument("--aruco", action="store_true", default=True,
                        help="Enable ArUco marker detection")
    parser.add_argument("--ros", action="store_true", default=False,
                        help="Enable ROS point cloud publishing")
    args = parser.parse_args()

    dtype = (torch.bfloat16
            if torch.cuda.get_device_capability()[0] >= 8
            else torch.float16)

    config = load_config(CONFIG_PATH)
    pp_pointcloud_config = load_config(PP_POINTCLOUD_CONFIG_PATH)
    dataset = LiveStreamDataset(config, COLMAP_PATH, device="cuda", dtype=dtype)
    vggt_utils = VGGTUtils(device="cuda", dtype=dtype, chunk_size=args.chunk_size, colmap_path=COLMAP_PATH)
    vggt_utils.load_vggt()

    aruco_utils = ArUcoUtils(args.marker_size)
    pp = PostProcessPointCloud(config, pp_pointcloud_config, args.mask_dir if args.use_mask else None)
    input_tensor = None
    while True:
        try:
            input_tensor, cam_names = dataset.get_processed_frames()
            
        except Exception as e:
            print(f"Error processing frames: {e}")
            time.sleep(1)  # Avoid tight loop on error
        if input_tensor is not None and input_tensor.shape[0] == 24:
            intrinsic, extrinsic, depth_map, depth_conf, points, colors, colmap_intrinsic, colmap_extrinsic = vggt_utils.run_vggt(input_tensor, cam_names=cam_names)
            marker_poses = aruco_utils.aruco_3d_poses_from_pointmap(input_tensor, points, patch_radius=2)
            points, colors = pp.filter_points(intrinsic, extrinsic, depth_map, depth_conf, points, colors)
        




if __name__ == "__main__":
    main()