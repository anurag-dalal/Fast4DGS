from pathlib import Path
import sys


# ── ArUco utilities ──────────────────────────────────────────────────────
ARUCO_PATH = Path(__file__).resolve().parent.parent / "ArUCo-Markers-Pose-Estimation-Generation-Python"
if str(ARUCO_PATH) not in sys.path:
    sys.path.insert(0, str(ARUCO_PATH))
from utils import ARUCO_DICT
import cv2
import numpy as np
from collections import defaultdict


class ArUcoUtils:
    def __init__(self, marker_size: float = 0.1):
        self.marker_size = marker_size
        self.aruco_dict = ARUCO_DICT

    # ── Detection ────────────────────────────────────────────────────────

    def detect_aruco_centers(self, frames, aruco_dict_type=None):
        """Detect ArUco markers in every frame and return their pixel centers.

        Returns
        -------
        marker_observations : dict  id -> list of (frame_idx, center_xy_px)
        """
        if aruco_dict_type is None:
            aruco_dict_type = ARUCO_DICT["DICT_5X5_100"]
        aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

        marker_obs: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)

        for fi, frame in enumerate(frames):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)
            if ids is None or len(corners) == 0:
                continue
            for i, marker_id in enumerate(ids.flatten()):
                c = corners[i].reshape(4, 2)
                center = c.mean(axis=0)
                marker_obs[int(marker_id)].append((fi, center))
                # print(f"  frame {fi}: ArUco ID {marker_id}  center px = "
                #       f"({center[0]:.1f}, {center[1]:.1f})")

        return dict(marker_obs)

    def detect_aruco_with_corners(self, frames, aruco_dict_type=None):
        """Detect ArUco markers and return pixel centers + all 4 corners.

        Returns
        -------
        marker_observations : dict  id -> list of (frame_idx, center_xy_px, corners_4x2)
            corners_4x2 rows are [TL, TR, BR, BL].
        """
        if aruco_dict_type is None:
            aruco_dict_type = ARUCO_DICT["DICT_5X5_100"]
        aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

        marker_obs: dict[int, list[tuple[int, np.ndarray, np.ndarray]]] = defaultdict(list)

        for fi, frame in enumerate(frames):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)
            if ids is None or len(corners) == 0:
                continue
            for i, marker_id in enumerate(ids.flatten()):
                c = corners[i].reshape(4, 2)   # [TL, TR, BR, BL]
                center = c.mean(axis=0)
                marker_obs[int(marker_id)].append((fi, center, c.copy()))
                # print(f"  frame {fi}: ArUco ID {marker_id}  center px = "
                #       f"({center[0]:.1f}, {center[1]:.1f})")

        return dict(marker_obs)

    # ── Coordinate mapping ───────────────────────────────────────────────

    def pixel_to_vggt_coords(self, px, orig_h, orig_w, target_size=518):
        """Map an (x, y) pixel in the original image to (row, col) in the
        VGGT preprocessed tensor (square-padded then resized)."""
        max_dim = max(orig_h, orig_w)
        pad_left = (max_dim - orig_w) // 2
        pad_top  = (max_dim - orig_h) // 2
        scale = target_size / max_dim
        col = int((px[0] + pad_left) * scale)
        row = int((px[1] + pad_top)  * scale)
        col = min(max(col, 0), target_size - 1)
        row = min(max(row, 0), target_size - 1)
        return row, col

    # ── 3-D lookup helpers ───────────────────────────────────────────────

    def _lookup_3d_at_pixel(
        self,
        px: np.ndarray,
        fi: int,
        points: np.ndarray,
        orig_h: int,
        orig_w: int,
        target_size: int = 518,
        patch_radius: int = 2,
    ) -> np.ndarray | None:
        """Look up a single pixel in the VGGT point map and return 3-D."""
        _, H, W, _ = points.shape
        row, col = self.pixel_to_vggt_coords(px, orig_h, orig_w, target_size)
        r_lo, r_hi = max(row - patch_radius, 0), min(row + patch_radius + 1, H)
        c_lo, c_hi = max(col - patch_radius, 0), min(col + patch_radius + 1, W)
        patch = points[fi, r_lo:r_hi, c_lo:c_hi].reshape(-1, 3)
        valid = np.isfinite(patch).all(axis=1) & (np.linalg.norm(patch, axis=1) > 1e-6)
        if valid.any():
            return patch[valid].mean(axis=0)
        return None

    # ── Center-only 3-D positions ────────────────────────────────────────

    def aruco_3d_from_pointmap(
        self,
        marker_obs: dict[int, list[tuple[int, np.ndarray]]],
        points: np.ndarray,          # [S, H, W, 3]
        orig_sizes: list[tuple[int, int]],
        target_size: int = 518,
        patch_radius: int = 2,
    ) -> dict[int, np.ndarray]:
        """Look up the VGGT 3-D point at each marker center and average by ID.

        Returns
        -------
        avg_positions : dict  marker_id -> (3,) world-space 3-D position
        """
        avg_positions: dict[int, np.ndarray] = {}
        _, H, W, _ = points.shape

        for mid, obs_list in marker_obs.items():
            pts_3d = []
            for fi, center_px in obs_list:
                orig_h, orig_w = orig_sizes[fi]
                row, col = self.pixel_to_vggt_coords(center_px, orig_h, orig_w,
                                                     target_size)
                r_lo = max(row - patch_radius, 0)
                r_hi = min(row + patch_radius + 1, H)
                c_lo = max(col - patch_radius, 0)
                c_hi = min(col + patch_radius + 1, W)
                patch = points[fi, r_lo:r_hi, c_lo:c_hi].reshape(-1, 3)

                valid = (np.isfinite(patch).all(axis=1)
                         & (np.linalg.norm(patch, axis=1) > 1e-6))
                if valid.any():
                    pts_3d.append(patch[valid].mean(axis=0))

            if pts_3d:
                avg_positions[mid] = np.stack(pts_3d).mean(axis=0)
                print(f"  ArUco ID {mid}: averaged from {len(pts_3d)} views → "
                      f"{avg_positions[mid]}")
            else:
                print(f"  ArUco ID {mid}: no valid 3-D samples found")

        return avg_positions

    # ── Tensor → BGR uint8 frames for detection ─────────────────────────

    @staticmethod
    def _tensor_to_bgr_frames(input_tensor) -> list[np.ndarray]:
        """Convert a [B, 3, H, W] float RGB tensor (0-1) to a list of BGR uint8 images."""
        import torch
        if isinstance(input_tensor, torch.Tensor):
            imgs = input_tensor.cpu().float().permute(0, 2, 3, 1).numpy()  # [B,H,W,3] RGB 0-1
        else:
            imgs = input_tensor  # assume already [B,H,W,3] numpy
        frames = []
        for i in range(imgs.shape[0]):
            rgb = np.clip(imgs[i] * 255, 0, 255).astype(np.uint8)
            frames.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        return frames

    # ── Full 3-D pose (center + orientation) ─────────────────────────────

    def aruco_3d_poses_from_pointmap(
        self,
        input_tensor,                  # [B, 3, H, W] torch tensor or [B, H, W, 3] numpy
        points: np.ndarray,            # [S, H, W, 3]
        aruco_dict_type=None,
        patch_radius: int = 2,
    ) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """Detect ArUco markers in the preprocessed images and compute
        3-D center + orientation for each marker from the VGGT point map.

        Since `input_tensor` is already square-padded & resized to the same
        resolution as `points`, pixel coordinates from ArUco detection map
        directly into the point map (no orig_sizes remapping needed).

        Parameters
        ----------
        input_tensor : preprocessed image tensor [B, 3, H, W] (RGB, 0-1)
        points       : VGGT unprojected point map [S, H, W, 3]
        aruco_dict_type : ArUco dictionary constant (default DICT_5X5_100)
        patch_radius : half-size of pixel neighbourhood for robust 3-D lookup

        Returns
        -------
        poses : dict  marker_id -> (center_3d (3,), rotation_3x3 (3,3))
            rotation columns are [X, Y, Z] axes of the marker.
        """
        # ── 1. Detect markers on the preprocessed frames ─────────────────
        bgr_frames = self._tensor_to_bgr_frames(input_tensor)
        marker_obs = self.detect_aruco_with_corners(bgr_frames, aruco_dict_type)
        if not marker_obs:
            print("  No ArUco markers detected in any frame.")
            return {}

        # ── 2. Since images are already at point-map resolution,
        #       detected pixel coords map directly (orig == target). ──────
        _, H, W, _ = points.shape
        target_size = H  # should equal W for square-padded input
        orig_sizes = [(H, W)] * len(bgr_frames)

        # ── 3. Compute 3-D pose per marker ───────────────────────────────
        poses: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        # points = points.reshape(-1, 3)
        for mid, obs_list in marker_obs.items():
            all_centers = []
            all_rotations = []

            for fi, center_px, corners_px in obs_list:
                orig_h, orig_w = orig_sizes[fi]

                # Look up center in 3-D
                center_3d = self._lookup_3d_at_pixel(
                    center_px, fi, points, orig_h, orig_w,
                    target_size, patch_radius)
                if center_3d is None:
                    continue

                # Look up all 4 corners in 3-D  (TL, TR, BR, BL)
                corners_3d = []
                for ci in range(4):
                    c3d = self._lookup_3d_at_pixel(
                        corners_px[ci], fi, points, orig_h, orig_w,
                        target_size, patch_radius)
                    if c3d is not None:
                        corners_3d.append(c3d)
                    else:
                        break

                all_centers.append(center_3d)

                if len(corners_3d) == 4:
                    tl, tr, br, bl = [np.asarray(c) for c in corners_3d]
                    # X axis: right  (midpoint of right edge − midpoint of left edge)
                    x_axis = ((tr + br) / 2.0) - ((tl + bl) / 2.0)
                    # Y axis: down  (midpoint of bottom edge − midpoint of top edge)
                    y_axis = ((bl + br) / 2.0) - ((tl + tr) / 2.0)
                    # Z axis: outward normal
                    z_axis = np.cross(x_axis, y_axis)

                    xn = np.linalg.norm(x_axis)
                    yn = np.linalg.norm(y_axis)
                    zn = np.linalg.norm(z_axis)

                    if xn > 1e-8 and yn > 1e-8 and zn > 1e-8:
                        x_axis /= xn
                        z_axis /= zn
                        # Re-orthogonalise Y from Z×X
                        y_axis = np.cross(z_axis, x_axis)
                        y_axis /= np.linalg.norm(y_axis)
                        rot = np.column_stack([x_axis, y_axis, z_axis])  # 3×3
                        all_rotations.append(rot)

            if all_centers:
                avg_center = np.stack(all_centers).mean(axis=0)
                if all_rotations:
                    # Average rotations via mean + SVD re-projection to SO(3)
                    avg_rot = np.stack(all_rotations).mean(axis=0)
                    U, _, Vt = np.linalg.svd(avg_rot)
                    avg_rot = U @ Vt
                    if np.linalg.det(avg_rot) < 0:       # fix reflection
                        U[:, -1] *= -1
                        avg_rot = U @ Vt
                else:
                    avg_rot = np.eye(3)

                poses[mid] = (avg_center, avg_rot)
                # print(f"  ArUco ID {mid}: center={avg_center}  "
                #       f"orientation={'YES' if all_rotations else 'identity (no corners)'}")
            else:
                print(f"  ArUco ID {mid}: no valid 3-D samples found")

        return poses