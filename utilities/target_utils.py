from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import cv2
import numpy as np


@dataclass(slots=True)
class BoundingBox3D:
    minimum: np.ndarray
    maximum: np.ndarray
    center: np.ndarray
    size: np.ndarray
    point_count: int


def resize_mask_to_processed(mask: np.ndarray, target_size: int = 518) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError("mask must be a single-channel image")

    height, width = mask.shape[:2]
    max_dim = max(height, width)
    top = (max_dim - height) // 2
    left = (max_dim - width) // 2

    square_mask = np.zeros((max_dim, max_dim), dtype=np.uint8)
    square_mask[top:top + height, left:left + width] = (mask > 0).astype(np.uint8) * 255
    return cv2.resize(square_mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST)


def collect_points_from_processed_masks(
    points: np.ndarray,
    processed_masks: Sequence[np.ndarray | None],
) -> np.ndarray:
    if points.ndim != 4 or points.shape[-1] != 3:
        raise ValueError("points must have shape [B, H, W, 3]")
    if len(processed_masks) != points.shape[0]:
        raise ValueError("processed_masks must have the same length as the batch dimension in points")

    collected: list[np.ndarray] = []
    for frame_index, mask in enumerate(processed_masks):
        if mask is None:
            continue
        if mask.shape != points.shape[1:3]:
            raise ValueError(
                f"mask shape {mask.shape} does not match point-map shape {points.shape[1:3]}"
            )

        mask_bool = mask > 0
        if not np.any(mask_bool):
            continue

        frame_points = points[frame_index][mask_bool]
        valid = np.isfinite(frame_points).all(axis=1) & (np.linalg.norm(frame_points, axis=1) > 1e-6)
        if np.any(valid):
            collected.append(frame_points[valid])

    if not collected:
        return np.empty((0, 3), dtype=np.float32)
    return np.concatenate(collected, axis=0).astype(np.float32, copy=False)


def compute_axis_aligned_bbox(points: np.ndarray) -> BoundingBox3D:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape [N, 3]")
    if points.shape[0] == 0:
        raise ValueError("points must not be empty")

    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    center = (minimum + maximum) / 2.0
    size = maximum - minimum
    return BoundingBox3D(
        minimum=minimum.astype(np.float32, copy=False),
        maximum=maximum.astype(np.float32, copy=False),
        center=center.astype(np.float32, copy=False),
        size=size.astype(np.float32, copy=False),
        point_count=int(points.shape[0]),
    )
