#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from dataloaders.live_stream_dataset import LiveStreamDataset, StreamFrameBatch
from utilities.aruco_utils import ArUcoUtils
from utilities.pp_pointcloud import PostProcessPointCloud
from utilities.sam_utils import SAMVideoMaskGenerator, overlay_mask_on_frame
from utilities.target_utils import (
    BoundingBox3D,
    collect_points_from_processed_masks,
    compute_axis_aligned_bbox,
    resize_mask_to_processed,
)
from utilities.vggt_utils import VGGTUtils


ROOT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = ROOT_DIR / "configs" / "stream_config.json"
PP_POINTCLOUD_CONFIG_PATH = ROOT_DIR / "configs" / "pp_pointcloud.json"
DEFAULT_MODEL_CFG = ROOT_DIR / "configs" / "sam2.1_hiera_l.yaml"
DEFAULT_CHECKPOINT = Path("/home/anurag/codes_ole/segmentation_tool/sam2/checkpoints/sam2.1_hiera_large.pt")
DEFAULT_COLMAP_PATH = Path("/home/anurag/stream_captures/sparse/0")
DEFAULT_OUTPUT_DIR = Path("/home/anurag/stream_captures/output_sam")
DEFAULT_VISER_PORT = 8080


@dataclass(slots=True)
class AppPaths:
    stream_config_path: Path = CONFIG_PATH
    pp_pointcloud_config_path: Path = PP_POINTCLOUD_CONFIG_PATH
    colmap_path: Path = DEFAULT_COLMAP_PATH
    sam_checkpoint: Path = DEFAULT_CHECKPOINT
    sam_model_cfg: Path = DEFAULT_MODEL_CFG


@dataclass(slots=True)
class RuntimeOptions:
    skip_seconds: float = 10.0
    startup_timeout: float = 25.0
    chunk_size: int = 512
    point_size: float = 0.002
    marker_size: float = 0.05


@dataclass(slots=True)
class ViewerOptions:
    port: int = DEFAULT_VISER_PORT
    show_aruco_markers: bool = True
    show_target_bbox: bool = True
    show_camera_positions: bool = True
    point_size: float = 0.002
    marker_size: float = 0.05


@dataclass(slots=True)
class TargetSelectionState:
    selected_display_name: str
    raw_box: tuple[int, int, int, int]
    display_names: list[str]
    cam_names: list[str]
    source_rgb_frames_by_display_name: dict[str, np.ndarray]
    source_processed_tensor: torch.Tensor | None
    masks_by_display_name: dict[str, np.ndarray]
    processed_masks_by_display_name: dict[str, np.ndarray]
    mask_grid: np.ndarray
    overlay_grid: np.ndarray
    selected_overlay: np.ndarray


@dataclass(slots=True)
class SavedTargetMemory:
    display_names: list[str]
    cam_names: list[str]
    source_rgb_frames_by_display_name: dict[str, np.ndarray]
    source_processed_tensor: torch.Tensor
    masks_by_display_name: dict[str, np.ndarray]
    processed_masks_by_display_name: dict[str, np.ndarray]


@dataclass(slots=True)
class PointCloudComputationResult:
    point_count: int
    aruco_count: int
    viewer_running: bool
    viewer_url: str | None
    target_bbox: BoundingBox3D | None
    status_message: str


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_config(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=4)
        handle.write("\n")


def json_to_text(data: dict) -> str:
    return json.dumps(data, indent=4) + "\n"


def parse_json_text(text: str) -> dict:
    return json.loads(text)


def get_device() -> tuple[torch.device, torch.dtype | None]:
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        dtype = torch.bfloat16 if major >= 8 else torch.float16
        return torch.device("cuda"), dtype
    return torch.device("cpu"), None


def expected_stream_count(config: dict) -> int:
    return sum(len(node.get("ports", [])) for node in config.get("nodes", []))


def wait_for_stable_batch(dataset: LiveStreamDataset, min_frames: int, timeout_s: float) -> StreamFrameBatch:
    deadline = time.monotonic() + timeout_s
    last_count = 0
    while time.monotonic() < deadline:
        batch = dataset.get_stream_batch(min_frames=min_frames, wait=False)
        last_count = len(batch.display_names)
        if last_count >= min_frames:
            return batch
        time.sleep(0.2)
    raise RuntimeError(f"Timed out waiting for {min_frames} streams; only received {last_count}.")


def resize_for_display(image: np.ndarray, max_width: int = 1600, max_height: int = 900) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    if scale == 1.0:
        return image.copy(), 1.0
    new_size = (int(round(width * scale)), int(round(height * scale)))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA), scale


def select_bounding_box(image: np.ndarray, window_name: str = "Draw bounding box") -> tuple[int, int, int, int]:
    display_image, scale = resize_for_display(image)
    roi = cv2.selectROI(window_name, display_image, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(window_name)

    x, y, w, h = roi
    if w <= 0 or h <= 0:
        raise RuntimeError("No bounding box was selected.")

    x1 = int(round(x / scale))
    y1 = int(round(y / scale))
    x2 = int(round((x + w) / scale))
    y2 = int(round((y + h) / scale))
    return x1, y1, x2, y2


def save_masks(mask_by_index: dict[int, np.ndarray], frame_names: list[str], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for frame_index, frame_name in enumerate(frame_names):
        if frame_index not in mask_by_index:
            continue
        mask_path = output_dir / f"{frame_name}.png"
        Image.fromarray(mask_by_index[frame_index], mode="L").save(mask_path)
        print(f"Saved {mask_path}")


def _draw_tile_label(image: np.ndarray, label: str) -> np.ndarray:
    labeled = image.copy()
    cv2.putText(labeled, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    return labeled


def make_frame_tile(frame: np.ndarray, label: str, tile_size: tuple[int, int] = (320, 240)) -> np.ndarray:
    tile = cv2.resize(frame, tile_size, interpolation=cv2.INTER_AREA)
    return _draw_tile_label(tile, label)


def make_mask_tile(mask: np.ndarray, label: str, tile_size: tuple[int, int] = (320, 240)) -> np.ndarray:
    tile = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    tile = cv2.resize(tile, tile_size, interpolation=cv2.INTER_NEAREST)
    return _draw_tile_label(tile, label)


def make_overlay_tile(frame: np.ndarray, mask: np.ndarray, label: str, tile_size: tuple[int, int] = (320, 240)) -> np.ndarray:
    tile = cv2.resize(overlay_mask_on_frame(frame, mask), tile_size, interpolation=cv2.INTER_AREA)
    return _draw_tile_label(tile, label)


def _build_image_grid(tiles: list[np.ndarray], columns: int | None = None) -> np.ndarray:
    if not tiles:
        raise RuntimeError("No images were available to build a grid.")

    cols = columns or min(4, len(tiles))
    rows = math.ceil(len(tiles) / cols)
    tile_h, tile_w = tiles[0].shape[:2]
    black_tile = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)

    padded_tiles = tiles + [black_tile] * (rows * cols - len(tiles))
    row_images = []
    for row_idx in range(rows):
        row_tiles = padded_tiles[row_idx * cols:(row_idx + 1) * cols]
        row_images.append(np.hstack(row_tiles))
    return np.vstack(row_images)


def build_frame_grid(frames_by_name: dict[str, np.ndarray], frame_names: list[str]) -> np.ndarray:
    tiles = [make_frame_tile(frames_by_name[name], name) for name in frame_names if name in frames_by_name]
    return _build_image_grid(tiles)


def build_mask_grid(mask_by_name: dict[str, np.ndarray], frame_names: list[str]) -> np.ndarray:
    tiles = [make_mask_tile(mask_by_name[name], name) for name in frame_names if name in mask_by_name]
    return _build_image_grid(tiles)


def build_overlay_grid(mask_by_name: dict[str, np.ndarray], frames_by_name: dict[str, np.ndarray], frame_names: list[str]) -> np.ndarray:
    tiles = [
        make_overlay_tile(frames_by_name[name], mask_by_name[name], name)
        for name in frame_names
        if name in mask_by_name and name in frames_by_name
    ]
    return _build_image_grid(tiles)


def show_mask_grid(mask_grid: np.ndarray, output_dir: Path) -> None:
    overview_path = output_dir / "mask_overview.png"
    cv2.imwrite(str(overview_path), mask_grid)
    print(f"Saved {overview_path}")

    display_image, _ = resize_for_display(mask_grid, max_width=1800, max_height=1000)
    window_name = "Masks for all frames"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, display_image)
    print("Press any key in the masks window to close it.")
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)


def show_overlay_grid(overlay_grid: np.ndarray, output_dir: Path) -> None:
    overview_path = output_dir / "overlay_overview.png"
    cv2.imwrite(str(overview_path), overlay_grid)
    print(f"Saved {overview_path}")

    display_image, _ = resize_for_display(overlay_grid, max_width=1800, max_height=1000)
    window_name = "Masks overlaid on RGB frames"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, display_image)
    print("Press any key in the overlay window to close it.")
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return value


class Fast4DGSController:
    def __init__(self, paths: AppPaths | None = None, options: RuntimeOptions | None = None) -> None:
        self.paths = paths or AppPaths()
        self.options = options or RuntimeOptions()
        self.device, self.dtype = get_device()
        self.config = load_config(self.paths.stream_config_path)
        self.pp_pointcloud_config = load_config(self.paths.pp_pointcloud_config_path)

        self.dataset: LiveStreamDataset | None = None
        self.latest_batch: StreamFrameBatch | None = None
        self.target_state: TargetSelectionState | None = None
        self.saved_target_memory: SavedTargetMemory | None = None
        self.target_bbox: BoundingBox3D | None = None
        self.vggt_utils: VGGTUtils | None = None
        self.viewer_server = None
        self.viewer_port: int | None = None
        self.ros_utils = None

    def reload_configs(self) -> None:
        self.config = load_config(self.paths.stream_config_path)
        self.pp_pointcloud_config = load_config(self.paths.pp_pointcloud_config_path)

    def read_json_text(self, path: Path) -> str:
        return path.read_text(encoding="utf-8")

    def save_json_text(self, path: Path, text: str) -> dict:
        data = parse_json_text(text)
        save_config(path, data)
        if path == self.paths.stream_config_path or path == self.paths.pp_pointcloud_config_path:
            self.reload_configs()
        return data

    def start_session(self) -> StreamFrameBatch:
        self.stop_session(clear_target=False, stop_viewer=False)
        self.reload_configs()

        dataset_dtype = self.dtype or torch.float32
        self.dataset = LiveStreamDataset(
            self.config,
            str(self.paths.colmap_path),
            device=str(self.device),
            dtype=dataset_dtype,
        )
        try:
            time.sleep(self.options.skip_seconds)
            self.latest_batch = wait_for_stable_batch(
                self.dataset,
                min_frames=expected_stream_count(self.config),
                timeout_s=self.options.startup_timeout,
            )
            self.target_state = None
            self.saved_target_memory = None
            self.target_bbox = None
            return self.latest_batch
        except Exception:
            self.stop_session(clear_target=False, stop_viewer=False)
            raise

    def stop_session(self, clear_target: bool = False, stop_viewer: bool = False) -> None:
        if self.dataset is not None:
            self.dataset.stop()
            self.dataset = None
        self.latest_batch = None
        if clear_target:
            self.target_state = None
            self.saved_target_memory = None
            self.target_bbox = None
        if stop_viewer:
            self.stop_viewer()

    def shutdown(self) -> None:
        self.stop_session(clear_target=True, stop_viewer=True)
        if self.ros_utils is not None:
            try:
                self.ros_utils.shutdown()
            finally:
                self.ros_utils = None

    def require_session(self) -> LiveStreamDataset:
        if self.dataset is None:
            raise RuntimeError("Session has not been started yet.")
        return self.dataset

    def refresh_snapshot(self, require_all_streams: bool = False) -> StreamFrameBatch:
        dataset = self.require_session()
        if require_all_streams:
            batch = wait_for_stable_batch(
                dataset,
                min_frames=expected_stream_count(self.config),
                timeout_s=self.options.startup_timeout,
            )
        else:
            batch = dataset.get_stream_batch(min_frames=1, wait=False)
            if not batch.display_names:
                batch = wait_for_stable_batch(dataset, min_frames=1, timeout_s=5.0)
        self.latest_batch = batch
        return batch

    def list_display_names(self) -> list[str]:
        batch = self.latest_batch or self.refresh_snapshot(require_all_streams=False)
        return list(batch.display_names)

    def build_all_streams_grid(self) -> np.ndarray:
        batch = self.refresh_snapshot(require_all_streams=False)
        return build_frame_grid(batch.undistorted_frames, batch.display_names)

    def get_single_stream_frame(self, display_name: str) -> np.ndarray:
        batch = self.refresh_snapshot(require_all_streams=False)
        if display_name not in batch.undistorted_frames:
            raise KeyError(f"Unknown stream: {display_name}")
        return batch.undistorted_frames[display_name]

    def segment_target(self, prompt_display_name: str, raw_box: tuple[int, int, int, int]) -> TargetSelectionState:
        batch = self.refresh_snapshot(require_all_streams=True)
        if prompt_display_name not in batch.display_names:
            raise KeyError(f"Unknown stream: {prompt_display_name}")

        display_names = list(batch.display_names)
        selected_index = display_names.index(prompt_display_name)
        ordered_indices = [selected_index] + [index for index in range(len(display_names)) if index != selected_index]
        ordered_frames = [batch.undistorted_frames[display_names[index]] for index in ordered_indices]

        sam_mask_generator = SAMVideoMaskGenerator(
            model_cfg=self.paths.sam_model_cfg,
            checkpoint=self.paths.sam_checkpoint,
            device=self.device,
        )

        try:
            ordered_masks = sam_mask_generator.segment_frames(ordered_frames, raw_box, frame_index=0)
        finally:
            sam_mask_generator.release()

        masks_by_display_name: dict[str, np.ndarray] = {}
        for ordered_position, original_index in enumerate(ordered_indices):
            masks_by_display_name[display_names[original_index]] = ordered_masks[ordered_position]

        processed_target_size = 518
        if batch.processed_tensor is not None:
            processed_target_size = int(batch.processed_tensor.shape[-1])
        processed_masks_by_display_name = {
            name: resize_mask_to_processed(mask, target_size=processed_target_size)
            for name, mask in masks_by_display_name.items()
        }

        overlay_grid = build_overlay_grid(masks_by_display_name, batch.undistorted_frames, display_names)
        mask_grid = build_mask_grid(masks_by_display_name, display_names)
        selected_overlay = overlay_mask_on_frame(batch.undistorted_frames[prompt_display_name], masks_by_display_name[prompt_display_name])

        self.target_state = TargetSelectionState(
            selected_display_name=prompt_display_name,
            raw_box=raw_box,
            display_names=list(display_names),
            cam_names=list(batch.cam_names),
            source_rgb_frames_by_display_name={name: batch.undistorted_frames[name].copy() for name in display_names},
            source_processed_tensor=batch.processed_tensor.detach().cpu().clone() if batch.processed_tensor is not None else None,
            masks_by_display_name=masks_by_display_name,
            processed_masks_by_display_name=processed_masks_by_display_name,
            mask_grid=mask_grid,
            overlay_grid=overlay_grid,
            selected_overlay=selected_overlay,
        )
        return self.target_state

    def save_target_to_memory(self) -> None:
        if self.target_state is None:
            raise RuntimeError("No target selection is available to save.")
        if self.target_state.source_processed_tensor is None:
            raise RuntimeError("Target selection does not contain processed RGB snapshots.")

        self.saved_target_memory = SavedTargetMemory(
            display_names=list(self.target_state.display_names),
            cam_names=list(self.target_state.cam_names),
            source_rgb_frames_by_display_name={
                name: frame.copy() for name, frame in self.target_state.source_rgb_frames_by_display_name.items()
            },
            source_processed_tensor=self.target_state.source_processed_tensor.detach().cpu().clone(),
            masks_by_display_name={name: mask.copy() for name, mask in self.target_state.masks_by_display_name.items()},
            processed_masks_by_display_name={
                name: mask.copy() for name, mask in self.target_state.processed_masks_by_display_name.items()
            },
        )
        self.target_bbox = None

    def clear_target(self) -> None:
        self.target_state = None
        self.saved_target_memory = None
        self.target_bbox = None

    def _compute_target_bbox_from_saved_memory(self) -> BoundingBox3D | None:
        if self.saved_target_memory is None:
            return self.target_bbox

        saved = self.saved_target_memory
        processed_tensor = saved.source_processed_tensor.to(self.device, self.dtype or torch.float32)
        vggt_utils = self._ensure_vggt()
        vggt_utils.reinit()
        _, _, _, _, saved_points, _, _, _ = vggt_utils.run_vggt(
            processed_tensor,
            cam_names=saved.cam_names,
        )

        processed_masks = [saved.processed_masks_by_display_name.get(name) for name in saved.display_names]
        target_points = collect_points_from_processed_masks(_to_numpy(saved_points), processed_masks)
        if target_points.size == 0:
            self.saved_target_memory = None
            return None

        self.target_bbox = compute_axis_aligned_bbox(target_points)
        self.saved_target_memory = None
        return self.target_bbox

    def is_viewer_running(self) -> bool:
        return self.viewer_server is not None

    def viewer_url(self) -> str | None:
        if self.viewer_port is None:
            return None
        return f"http://127.0.0.1:{self.viewer_port}"

    def ensure_viewer(self, port: int):
        if self.viewer_server is not None and self.viewer_port == port:
            return self.viewer_server
        self.stop_viewer()
        import viser

        self.viewer_server = viser.ViserServer(port=port)
        self.viewer_port = port
        return self.viewer_server

    def stop_viewer(self) -> None:
        if self.viewer_server is not None:
            self.viewer_server.stop()
            self.viewer_server = None
            self.viewer_port = None

    def _ensure_vggt(self) -> VGGTUtils:
        if self.vggt_utils is None:
            dataset_dtype = self.dtype or torch.float32
            self.vggt_utils = VGGTUtils(
                device=str(self.device),
                dtype=dataset_dtype,
                chunk_size=self.options.chunk_size,
                colmap_path=str(self.paths.colmap_path),
            )
        return self.vggt_utils

    def _ensure_ros_utils(self):
        if self.ros_utils is None:
            from utilities.ros_utils import ROSUtils

            self.ros_utils = ROSUtils(node_name="fast4dgs_gui")
        return self.ros_utils

    def _populate_viewer_scene(
        self,
        server,
        processed_tensor: torch.Tensor,
        points: np.ndarray,
        colors: np.ndarray,
        extrinsic: torch.Tensor | np.ndarray,
        aruco_poses: dict[int, tuple[np.ndarray, np.ndarray]],
        target_bbox: BoundingBox3D | None,
        options: ViewerOptions,
    ) -> None:
        import viser.transforms as vtf

        server.scene.reset()
        server.scene.world_axes.visible = True
        server.scene.add_point_cloud(
            "/pointcloud",
            points=points,
            colors=colors,
            point_size=options.point_size,
        )

        if options.show_camera_positions:
            extrinsic_np = _to_numpy(extrinsic)
            for index in range(extrinsic_np.shape[0]):
                ext_np = extrinsic_np[index]
                rotation = ext_np[:3, :3]
                translation = ext_np[:3, 3]
                camera_center = -rotation.T @ translation
                wxyz = vtf.SO3.from_matrix(rotation.T).wxyz
                server.scene.add_camera_frustum(
                    f"/cameras/{index}",
                    fov=1.74533,
                    aspect=16 / 9,
                    scale=0.06,
                    wxyz=wxyz,
                    position=camera_center,
                    color=(80, 180, 255),
                    image=processed_tensor[index].permute(1, 2, 0).detach().cpu().float().numpy(),
                )

        if options.show_aruco_markers:
            marker_colors = [
                (255, 50, 50),
                (50, 255, 50),
                (50, 50, 255),
                (255, 255, 50),
                (255, 50, 255),
                (50, 255, 255),
                (255, 128, 0),
                (128, 0, 255),
            ]
            axis_length = options.marker_size * 2.0
            for index, (marker_id, (position, rotation)) in enumerate(sorted(aruco_poses.items())):
                color = marker_colors[index % len(marker_colors)]
                wxyz = vtf.SO3.from_matrix(rotation).wxyz
                server.scene.add_frame(
                    f"/markers/id_{marker_id}/axes",
                    wxyz=wxyz,
                    position=position,
                    axes_length=axis_length,
                    axes_radius=axis_length * 0.06,
                )
                server.scene.add_icosphere(
                    f"/markers/id_{marker_id}/sphere",
                    radius=options.marker_size * 0.3,
                    color=color,
                    position=position,
                )
                server.scene.add_label(
                    f"/markers/id_{marker_id}/label",
                    text=f"ID {marker_id}",
                    position=position + np.array([0.0, 0.0, options.marker_size * 2.5]),
                )

        if options.show_target_bbox and target_bbox is not None:
            server.scene.add_box(
                "/init_target/bbox",
                dimensions=tuple(np.maximum(target_bbox.size, 1e-4)),
                color=(64, 255, 96),
                position=target_bbox.center,
                # wireframe=True,
                # opacity=0.35,
            )
            server.scene.add_label(
                "/init_target/label",
                text="init_target",
                position=target_bbox.center + np.array([0.0, 0.0, max(float(target_bbox.size[2]), 0.03)]),
            )

    def compute_point_cloud(
        self,
        *,
        publish_ros_pointcloud: bool = False,
        publish_aruco_markers: bool = False,
        publish_target_position: bool = False,
        viewer_options: ViewerOptions | None = None,
    ) -> PointCloudComputationResult:
        target_bbox_message = None
        if self.target_bbox is None and self.saved_target_memory is not None:
            computed_bbox = self._compute_target_bbox_from_saved_memory()
            if computed_bbox is not None:
                target_bbox_message = f"target bbox locked from {computed_bbox.point_count} saved samples"

        batch = self.refresh_snapshot(require_all_streams=True)
        if batch.processed_tensor is None:
            raise RuntimeError("No processed tensor was available for VGGT inference.")

        viewer_options = viewer_options or ViewerOptions(
            point_size=self.options.point_size,
            marker_size=self.options.marker_size,
        )

        vggt_utils = self._ensure_vggt()
        vggt_utils.reinit()
        intrinsic, extrinsic, depth_map, depth_conf, points, colors, _, _ = vggt_utils.run_vggt(
            batch.processed_tensor,
            cam_names=batch.cam_names,
        )

        pp = PostProcessPointCloud(self.config, self.pp_pointcloud_config, None)
        filtered_points, filtered_colors = pp.filter_points(intrinsic, extrinsic, depth_map, depth_conf, points, colors)

        aruco_poses: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        if viewer_options.show_aruco_markers or publish_aruco_markers:
            aruco_utils = ArUcoUtils(viewer_options.marker_size)
            aruco_poses = aruco_utils.aruco_3d_poses_from_pointmap(batch.processed_tensor, _to_numpy(points), patch_radius=2)

        target_bbox = self.target_bbox

        if publish_ros_pointcloud or publish_aruco_markers or publish_target_position:
            ros_utils = self._ensure_ros_utils()
            if publish_ros_pointcloud:
                ros_utils.create_point_cloud2(filtered_points, filtered_colors, frame_id="map")
            if publish_aruco_markers and aruco_poses:
                ros_utils.publish_aruco_marker_poses(aruco_poses, frame_id="map")
            if publish_target_position and target_bbox is not None:
                ros_utils.publish_target_pose(target_bbox.center, frame_id="map")
                ros_utils.publish_target_bounding_box(target_bbox.center, target_bbox.size, frame_id="map")

        viewer_running = self.is_viewer_running()
        if viewer_running:
            server = self.ensure_viewer(viewer_options.port)
            self._populate_viewer_scene(
                server,
                batch.processed_tensor,
                filtered_points,
                filtered_colors,
                extrinsic,
                aruco_poses,
                target_bbox,
                viewer_options,
            )

        viewer_url = self.viewer_url()
        status_parts = [f"Point cloud updated with {filtered_points.shape[0]} points"]
        if aruco_poses:
            status_parts.append(f"{len(aruco_poses)} ArUco markers")
        if target_bbox_message is not None:
            status_parts.append(target_bbox_message.replace("target bbox", "init_target bbox"))
        if target_bbox is not None:
            status_parts.append(f"init_target bbox kept at center {np.round(target_bbox.center, 4).tolist()}")
        if viewer_running and viewer_url:
            status_parts.append(f"viewer: {viewer_url}")

        return PointCloudComputationResult(
            point_count=int(filtered_points.shape[0]),
            aruco_count=len(aruco_poses),
            viewer_running=viewer_running,
            viewer_url=viewer_url,
            target_bbox=target_bbox,
            status_message=" | ".join(status_parts),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture live streams, draw one box, and save SAM2 masks.")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH, help="Path to stream config JSON")
    parser.add_argument("--pp-config", type=Path, default=PP_POINTCLOUD_CONFIG_PATH, help="Path to point-cloud post-processing JSON")
    parser.add_argument("--colmap-path", type=Path, default=DEFAULT_COLMAP_PATH, help="Path to COLMAP sparse model")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="Path to SAM2 checkpoint")
    parser.add_argument("--model-cfg", type=Path, default=DEFAULT_MODEL_CFG, help="Path to SAM2 config YAML")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Folder where masks are saved")
    parser.add_argument("--skip-seconds", type=float, default=10.0, help="Seconds to wait before capturing frames")
    parser.add_argument("--startup-timeout", type=float, default=25.0, help="Seconds to wait for all streams to become available")
    parser.add_argument("--prompt-stream", type=str, default=None, help="Display name of the stream used to draw the bounding box")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    controller = Fast4DGSController(
        paths=AppPaths(
            stream_config_path=args.config,
            pp_pointcloud_config_path=args.pp_config,
            colmap_path=args.colmap_path,
            sam_checkpoint=args.checkpoint,
            sam_model_cfg=args.model_cfg,
        ),
        options=RuntimeOptions(
            skip_seconds=args.skip_seconds,
            startup_timeout=args.startup_timeout,
        ),
    )

    try:
        batch = controller.start_session()
        display_names = list(batch.display_names)
        if not display_names:
            raise RuntimeError("No streams were available.")

        prompt_display_name = args.prompt_stream or display_names[0]
        if prompt_display_name not in batch.undistorted_frames:
            raise RuntimeError(f"Prompt stream '{prompt_display_name}' is not available.")

        prompt_frame = batch.undistorted_frames[prompt_display_name].copy()
        cv2.putText(
            prompt_frame,
            f"Draw box on {prompt_display_name} and press ENTER/SPACE",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        prompt_box = select_bounding_box(prompt_frame)
        target_state = controller.segment_target(prompt_display_name, prompt_box)

        mask_by_index = {
            index: target_state.masks_by_display_name[name]
            for index, name in enumerate(batch.display_names)
            if name in target_state.masks_by_display_name
        }
        save_masks(mask_by_index, batch.cam_names, args.output_dir)
        show_overlay_grid(target_state.overlay_grid, args.output_dir)
        show_mask_grid(target_state.mask_grid, args.output_dir)
        print("Done.")
        return 0
    finally:
        controller.shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())