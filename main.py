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
    parser.add_argument("--no_aruco", action="store_true", default=False,
                        help="Skip ArUco marker detection")
    parser.add_argument("--ros", action="store_false", default=True,
                        help="Enable ArUco marker detection")
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
    if args.viser:
        server = viser.ViserServer(port=args.port)
    if args.ros:
        ros_utils = ROSUtils(node_name="pointcloud_publisher", topic="/vggt/pointcloud")
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
            print(points.shape)
            if args.ros:
                ros_utils.create_point_cloud2(points, colors, frame_id="map")


            # construct mesh from point cloud and display in viser
            # tri_mesh = pointcloud_to_mesh(
            #     points, colors,
            #     poisson_depth=8,
            #     density_quantile=0.05,
            #     estimate_normals_radius=0.05,
            #     estimate_normals_max_nn=30,
            # )
            # server.scene.add_mesh_trimesh("/mesh", tri_mesh)
            # print(f"Mesh: {len(tri_mesh.vertices)} verts, {len(tri_mesh.faces)} faces")

            # display point cloud
            if args.viser:
                server.scene.add_point_cloud(
                "/pointcloud",
                points=points,
                colors=colors,
                point_size=args.point_size)

            # display camera frustums
            # Add camera frustums for reference
            if args.viser:
                for i in range(extrinsic.shape[0]):
                    ext_np = extrinsic[i].cpu().float().numpy()  # [3, 4]
                    R_cam = ext_np[:3, :3]
                    t_cam = ext_np[:3, 3]
                    cam_center = -R_cam.T @ t_cam

                    # wxyz quaternion from rotation matrix  (viser convention)
                    wxyz = vtf.SO3.from_matrix(R_cam.T).wxyz

                    server.scene.add_camera_frustum(
                        f"/cameras/{i}",
                        fov=1.74533,  # approximate
                        aspect=16/9,
                        scale=0.06,
                        wxyz=wxyz,
                        position=cam_center,
                        color=(80, 180, 255),
                        image=input_tensor[i].permute(1, 2, 0).cpu().float().numpy()
                    )

            if args.viser:
                # ── 7b. Add ArUco marker axes + labels ─────────────────────────────
                MARKER_COLORS = [
                    (255,  50,  50),   # red
                    ( 50, 255,  50),   # green
                    ( 50,  50, 255),   # blue
                    (255, 255,  50),   # yellow
                    (255,  50, 255),   # magenta
                    ( 50, 255, 255),   # cyan
                    (255, 128,   0),   # orange
                    (128,   0, 255),   # purple
                ]

                axis_length = args.marker_size * 2.0   # length of each RGB axis line

                for idx, (mid, (pos, rot)) in enumerate(sorted(marker_poses.items())):
                    color = MARKER_COLORS[idx % len(MARKER_COLORS)]

                    # ── RGB coordinate frame (orientation) ──────────────────────────
                    # rot columns: [X, Y, Z]  →  draw as Red, Green, Blue lines
                    wxyz = vtf.SO3.from_matrix(rot).wxyz
                    server.scene.add_frame(
                        f"/markers/id_{mid}/axes",
                        wxyz=wxyz,
                        position=pos,
                        axes_length=axis_length,
                        axes_radius=axis_length * 0.06,
                    )

                    # Small sphere at center for visibility
                    server.scene.add_icosphere(
                        f"/markers/id_{mid}/sphere",
                        radius=args.marker_size * 0.3,
                        color=color,
                        position=pos,
                    )

                    # Text label floating above
                    server.scene.add_label(
                        f"/markers/id_{mid}/label",
                        text=f"ID {mid}",
                        position=pos + np.array([0.0, 0.0, args.marker_size * 2.5]),
                    )

    # dataset = StaticDataset(IMAGE_FOLDER_PATH, COLMAP_PATH)
    # input_tensor = dataset.get_processed_frames()
    # print(f"Processed input tensor shape: {input_tensor.shape}")





if __name__ == "__main__":
    main()