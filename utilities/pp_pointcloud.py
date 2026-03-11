import numpy as np
import torch


class PostProcessPointCloud:
    def __init__(self, stream_config, pp_pointcloud_config, mask_path):
        self.stream_config = stream_config
        self.pp_pointcloud_config = pp_pointcloud_config
        self.mask_path = mask_path

    def _to_numpy(self, x):
        """Convert torch tensors to numpy, pass numpy through."""
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy()
        return x

    def filter_points(self, intrinsic, extrinsic, depth_map, depth_conf, points, colors):
        """Filter a batched point cloud.

        Parameters
        ----------
        intrinsic  : [B, 3, 3]
        extrinsic  : [B, 3, 4]
        depth_map  : [B, H, W, 1]
        depth_conf : [B, H, W]
        points     : [B, H, W, 3]
        colors     : [B, H, W, 3]

        Returns
        -------
        points_flat, colors_flat : [N, 3] each
        """
        intrinsic = self._to_numpy(intrinsic)
        extrinsic = self._to_numpy(extrinsic)
        depth_map = self._to_numpy(depth_map)
        depth_conf = self._to_numpy(depth_conf)
        points = self._to_numpy(points)
        colors = self._to_numpy(colors)

        B, H, W, _ = points.shape

        # Start with an all-True mask [B, H, W]
        valid = np.ones((B, H, W), dtype=bool)

        # 1. Remove corner regions
        if self.pp_pointcloud_config["remove_corners"]["enabled"]:
            corner_thresh = self.pp_pointcloud_config["remove_corners"]["corner_threshold"]
            corner_mask = np.ones((H, W), dtype=bool)
            ch, cw = int(H * corner_thresh), int(W * corner_thresh)
            corner_mask[:ch, :] = False
            corner_mask[-ch:, :] = False
            corner_mask[:, :cw] = False
            corner_mask[:, -cw:] = False
            # broadcast [H, W] -> [B, H, W]
            valid &= (depth_conf > 0) & corner_mask[None, :, :]

        # 2. Flatten everything for point-level filters
        points_flat = points[valid]       # [N, 3]
        colors_flat = colors[valid]       # [N, 3]
        conf_flat = depth_conf[valid]     # [N]

        # 3. Working-area radius from mean camera center
        if self.pp_pointcloud_config["working_area_based"]["enabled"]:
            radius = self.pp_pointcloud_config["working_area_based"]["radius_from_center"]
            # Camera centers: c = -R^T @ t  for each view
            R = extrinsic[:, :3, :3]                       # [B, 3, 3]
            t = extrinsic[:, :3, 3]                        # [B, 3]
            cam_centers = -np.einsum('bij,bi->bj', R.transpose(0, 2, 1), t)  # [B, 3]
            mean_center = t.mean(axis=0)         # [3]
            dists = np.linalg.norm(points_flat - mean_center, axis=1)
            keep = dists < radius
            points_flat = points_flat[keep]
            colors_flat = colors_flat[keep]
            conf_flat = conf_flat[keep]

        # 4. Confidence threshold (absolute fraction of max)
        if self.pp_pointcloud_config["confidence_based_filtering"]["enabled"]:
            frac = self.pp_pointcloud_config["confidence_based_filtering"]["confidence_threshold"]
            conf_thresh = frac * conf_flat.max()
            keep = conf_flat > conf_thresh
            points_flat = points_flat[keep]
            colors_flat = colors_flat[keep]
            conf_flat = conf_flat[keep]

        # 5. Delete lowest-confidence percentile
        if self.pp_pointcloud_config["delete_points_by_confience_by_percentage"]["enabled"]:
            percentage = self.pp_pointcloud_config["delete_points_by_confience_by_percentage"]["percentage_to_delete"]
            if conf_flat.size > 0:
                conf_thresh = np.percentile(conf_flat, percentage * 100)
                keep = conf_flat > conf_thresh
                points_flat = points_flat[keep]
                colors_flat = colors_flat[keep]
                conf_flat = conf_flat[keep]

        if self.pp_pointcloud_config["outlier_removal"]["enabled"]:
            from sklearn.neighbors import NearestNeighbors
            nbrs = NearestNeighbors(n_neighbors=self.pp_pointcloud_config["outlier_removal"]["num_neighbors"] + 1).fit(points_flat)
            distances, _ = nbrs.kneighbors(points_flat)
            # distances includes the point itself at index 0, so take the k-th neighbor at index k
            kth_distances = distances[:, self.pp_pointcloud_config["outlier_removal"]["num_neighbors"]]
            keep = kth_distances < self.pp_pointcloud_config["outlier_removal"]["distance_threshold"]
            points_flat = points_flat[keep]
            colors_flat = colors_flat[keep]

        final_num_points = self.pp_pointcloud_config["final_target_num_points"]
        if points_flat.shape[0] > final_num_points:
            # Randomly sample points to reduce to final_num_points
            indices = np.random.choice(points_flat.shape[0], final_num_points, replace=False)
            points_flat = points_flat[indices]
            colors_flat = colors_flat[indices]

        return points_flat, colors_flat
