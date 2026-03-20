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

from fused_ssim import fused_ssim
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal, assert_never
from utilities.gsplat_utils import AppearanceOptModule, CameraOptModule, knn, rgb_to_sh, set_random_seed
from utilities.vggt_utils import VGGTUtils

from gsplat import export_splats
from gsplat.compression import PngCompression
from gsplat.distributed import cli
from gsplat.optimizers import SelectiveAdam
from gsplat.rendering import rasterization, RasterizeMode
from gsplat.cuda._wrapper import CameraModel
from gsplat.strategy import DefaultStrategy, MCMCStrategy
# from gsplat.viewer import GsplatViewer, GsplatRenderTabState
from nerfview import CameraState, RenderTabState, apply_float_colormap

from dataloaders.live_stream_dataset import LiveStreamDataset, StreamFrameBatch


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_STREAM_CONFIG_PATH = ROOT_DIR / "configs" / "stream_config.json"
DEFAULT_COLMAP_PATH = Path("/home/anurag/stream_captures/sparse/0")


@dataclass
class LiveTrainingSnapshot:
    processed_tensor: Tensor
    pixels: Tensor
    camtoworlds: Tensor
    Ks: Tensor
    init_points: Tensor
    init_colors: Tensor
    display_names: List[str]
    cam_names: List[str]
    width: int
    height: int

@dataclass
class Config:
    # Disable viewer
    disable_viewer: bool = False
    # Path to the .pt files. If provide, it will skip training and run evaluation only.
    ckpt: Optional[List[str]] = None
    # Name of compression strategy to use
    compression: Optional[Literal["png"]] = None
    # Render trajectory path
    render_traj_path: str = "interp"

    # Path to the Mip-NeRF 360 dataset
    data_dir: str = "data/360_v2/garden"
    # Live-stream JSON config used by LiveStreamDataset
    stream_config_path: str = str(DEFAULT_STREAM_CONFIG_PATH)
    # COLMAP sparse model path used by VGGT / stream loader
    colmap_path: str = str(DEFAULT_COLMAP_PATH)
    # Downsample factor for the dataset
    data_factor: int = 4
    # Directory to save results
    result_dir: str = "results/garden"
    # Every N images there is a test image
    test_every: int = 8
    # Random crop size for training  (experimental)
    patch_size: Optional[int] = None
    # A global scaler that applies to the scene size related parameters
    global_scale: float = 1.0
    # Normalize the world space
    normalize_world_space: bool = True
    # Camera model
    camera_model: CameraModel = "pinhole"
    # Load EXIF exposure metadata from images (if available)
    load_exposure: bool = True

    # Port for the viewer server
    port: int = 8080

    # Live-stream startup delay before requesting frames
    skip_seconds: float = 10.0
    # Timeout while waiting for the required number of streams
    startup_timeout: float = 25.0
    # Override required stream count. Set to 0 to use all configured streams.
    min_streams: int = 0
    # Target square resolution produced by LiveStreamDataset preprocessing
    target_size: int = 518
    # VGGT chunk size
    vggt_chunk_size: int = 512
    # Minimum depth confidence used when building the initial point cloud
    depth_confidence_threshold: float = 0.1
    # Maximum number of points used to initialize the Gaussian model
    max_initial_points: int = 200_000

    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 1
    # A global factor to scale the number of training steps
    steps_scaler: float = 1.0

    # Number of training steps
    max_steps: int = 30_000
    # Steps to evaluate the model
    eval_steps: List[int] = field(default_factory=lambda: [2000, 8000])
    # Steps to save the model
    save_steps: List[int] = field(default_factory=lambda: [2000, 8000])
    # Whether to save ply file (storage size can be large)
    save_ply: bool = False
    # Steps to save the model as ply
    ply_steps: List[int] = field(default_factory=lambda: [2000, 8000])
    # Whether to disable video generation during training and evaluation
    disable_video: bool = False

    # Initialization strategy
    init_type: str = "sfm"
    # Initial number of GSs. Ignored if using sfm
    init_num_pts: int = 100_000
    # Initial extent of GSs as a multiple of the camera extent. Ignored if using sfm
    init_extent: float = 3.0
    # Degree of spherical harmonics
    sh_degree: int = 3
    # Turn on another SH degree every this steps
    sh_degree_interval: int = 1000
    # Initial opacity of GS
    init_opa: float = 0.1
    # Initial scale of GS
    init_scale: float = 1.0
    # Weight for SSIM loss
    ssim_lambda: float = 0.2

    # Near plane clipping distance
    near_plane: float = 0.01
    # Far plane clipping distance
    far_plane: float = 1e10

    # Strategy for GS densification
    strategy: Union[DefaultStrategy, MCMCStrategy] = field(
        default_factory=DefaultStrategy
    )
    # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    packed: bool = False
    # Use sparse gradients for optimization. (experimental)
    sparse_grad: bool = False
    # Use visible adam from Taming 3DGS. (experimental)
    visible_adam: bool = False
    # Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.
    antialiased: bool = False

    # Use random background for training to discourage transparency
    random_bkgd: bool = False

    # LR for 3D point positions
    means_lr: float = 1.6e-4
    # LR for Gaussian scale factors
    scales_lr: float = 5e-3
    # LR for alpha blending weights
    opacities_lr: float = 5e-2
    # LR for orientation (quaternions)
    quats_lr: float = 1e-3
    # LR for SH band 0 (brightness)
    sh0_lr: float = 2.5e-3
    # LR for higher-order SH (detail)
    shN_lr: float = 2.5e-3 / 20

    # Opacity regularization
    opacity_reg: float = 0.0
    # Scale regularization
    scale_reg: float = 0.0

    # Enable camera optimization.
    pose_opt: bool = False
    # Learning rate for camera optimization
    pose_opt_lr: float = 1e-5
    # Regularization for camera optimization as weight decay
    pose_opt_reg: float = 1e-6
    # Add noise to camera extrinsics. This is only to test the camera pose optimization.
    pose_noise: float = 0.0

    # Enable appearance optimization. (experimental)
    app_opt: bool = False
    # Feature dimension used when appearance optimization is enabled
    app_feature_dim: int = 32
    # Appearance embedding dimension
    app_embed_dim: int = 16
    # Learning rate for appearance optimization
    app_opt_lr: float = 1e-3
    # Regularization for appearance optimization as weight decay
    app_opt_reg: float = 1e-6

    # Post-processing method for appearance correction (experimental)
    post_processing: Optional[Literal["bilateral_grid", "ppisp"]] = None
    # Use fused implementation for bilateral grid (only applies when post_processing="bilateral_grid")
    bilateral_grid_fused: bool = False
    # Shape of the bilateral grid (X, Y, W)
    bilateral_grid_shape: Tuple[int, int, int] = (16, 16, 8)
    # Enable PPISP controller
    ppisp_use_controller: bool = True
    # Use controller distillation in PPISP (only applies when post_processing="ppisp" and ppisp_use_controller=True)
    ppisp_controller_distillation: bool = True
    # Controller activation ratio for PPISP (only applies when post_processing="ppisp" and ppisp_use_controller=True)
    ppisp_controller_activation_num_steps: int = 25_000
    # Color correction method for cc_* metrics (only applies when post_processing is set)
    color_correct_method: Literal["affine", "quadratic"] = "affine"
    # Compute color-corrected metrics (cc_psnr, cc_ssim, cc_lpips) during evaluation
    use_color_correction_metric: bool = False

    # Enable depth loss. (experimental)
    depth_loss: bool = False
    # Weight for depth loss
    depth_lambda: float = 1e-2

    # Dump information to tensorboard every this steps
    tb_every: int = 100
    # Save training images to tensorboard
    tb_save_image: bool = False

    lpips_net: Literal["vgg", "alex"] = "alex"

    # 3DGUT (uncented transform + eval 3D)
    with_ut: bool = False
    with_eval3d: bool = False

    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.ply_steps = [int(i * factor) for i in self.ply_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)

        strategy = self.strategy
        if isinstance(strategy, DefaultStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.reset_every = int(strategy.reset_every * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        elif isinstance(strategy, MCMCStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
            if strategy.noise_injection_stop_iter >= 0:
                strategy.noise_injection_stop_iter = int(
                    strategy.noise_injection_stop_iter * factor
                )
        else:
            assert_never(strategy)


def create_splats_with_optimizers(
    points: Tensor,
    rgbs: Tensor,
    init_type: str = "sfm",
    init_num_pts: int = 100_000,
    init_extent: float = 3.0,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    means_lr: float = 1.6e-4,
    scales_lr: float = 5e-3,
    opacities_lr: float = 5e-2,
    quats_lr: float = 1e-3,
    sh0_lr: float = 2.5e-3,
    shN_lr: float = 2.5e-3 / 20,
    scene_scale: float = 1.0,
    sh_degree: int = 3,
    sparse_grad: bool = False,
    visible_adam: bool = False,
    batch_size: int = 1,
    feature_dim: Optional[int] = None,
    device: str = "cuda",
    world_rank: int = 0,
    world_size: int = 1,
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    if init_type == "sfm":
        points = points
        rgbs = rgbs
    elif init_type == "random":
        points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
        rgbs = torch.rand((init_num_pts, 3))
    else:
        raise ValueError("Please specify a correct init_type: sfm or random")

    # Initialize the GS size to be the average dist of the 3 nearest neighbors
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]

    # Distribute the GSs to different ranks (also works for single rank)
    points = points[world_rank::world_size]
    rgbs = rgbs[world_rank::world_size]
    scales = scales[world_rank::world_size]

    N = points.shape[0]
    quats = torch.rand((N, 4))  # [N, 4]
    opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]

    params = [
        # name, value, lr
        ("means", torch.nn.Parameter(points), means_lr * scene_scale),
        ("scales", torch.nn.Parameter(scales), scales_lr),
        ("quats", torch.nn.Parameter(quats), quats_lr),
        ("opacities", torch.nn.Parameter(opacities), opacities_lr),
    ]

    if feature_dim is None:
        # color is SH coefficients.
        colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
        colors[:, 0, :] = rgb_to_sh(rgbs)
        params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), sh0_lr))
        params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), shN_lr))
    else:
        # features will be used for appearance and view-dependent shading
        features = torch.rand(N, feature_dim)  # [N, feature_dim]
        params.append(("features", torch.nn.Parameter(features), sh0_lr))
        colors = torch.logit(rgbs)  # [N, 3]
        params.append(("colors", torch.nn.Parameter(colors), sh0_lr))

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    # Scale learning rate based on batch size, reference:
    # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    # Note that this would not make the training exactly equivalent, see
    # https://arxiv.org/pdf/2402.18824v1
    BS = batch_size * world_size
    optimizer_class = None
    if sparse_grad:
        optimizer_class = torch.optim.SparseAdam
    elif visible_adam:
        optimizer_class = SelectiveAdam
    else:
        optimizer_class = torch.optim.Adam
    optimizers = {
        name: optimizer_class(
            [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
            eps=1e-15 / math.sqrt(BS),
            # TODO: check betas logic when BS is larger than 10 betas[0] will be zero.
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
            fused=str(device).startswith("cuda"),
        )
        for name, _, lr in params
    }
    return splats, optimizers


def load_json(path: Union[str, Path]) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def get_device() -> Tuple[torch.device, torch.dtype]:
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        dtype = torch.bfloat16 if major >= 8 else torch.float16
        return torch.device("cuda"), dtype
    return torch.device("cpu"), torch.float32


def expected_stream_count(config: dict) -> int:
    return sum(len(node.get("ports", [])) for node in config.get("nodes", []))


def wait_for_stable_batch(
    dataset: LiveStreamDataset,
    min_frames: int,
    timeout_s: float,
    target_size: int,
) -> StreamFrameBatch:
    deadline = time.monotonic() + timeout_s
    last_count = 0
    while time.monotonic() < deadline:
        batch = dataset.get_stream_batch(target_size=target_size, min_frames=min_frames, wait=False)
        last_count = len(batch.display_names)
        if batch.processed_tensor is not None and last_count >= min_frames:
            return batch
        time.sleep(0.2)
    raise RuntimeError(f"Timed out waiting for {min_frames} streams; only received {last_count}.")


def extrinsics_to_camtoworlds(extrinsic: Tensor) -> Tensor:
    if extrinsic.shape[-2:] == (3, 4):
        bottom = torch.zeros((*extrinsic.shape[:-2], 1, 4), dtype=extrinsic.dtype, device=extrinsic.device)
        bottom[..., 0, 3] = 1.0
        extrinsic = torch.cat([extrinsic, bottom], dim=-2)
    return torch.linalg.inv(extrinsic)


def build_initial_point_cloud(
    points: Union[np.ndarray, Tensor],
    colors: Tensor,
    depth_conf: Union[np.ndarray, Tensor],
    depth_confidence_threshold: float,
    max_initial_points: int,
    device: torch.device,
) -> Tuple[Tensor, Tensor]:
    if isinstance(points, torch.Tensor):
        point_tensor = points.detach().cpu().float().reshape(-1, 3)
    else:
        point_tensor = torch.as_tensor(points, dtype=torch.float32).reshape(-1, 3)

    color_tensor = colors.detach().cpu().float().reshape(-1, 3)

    if isinstance(depth_conf, torch.Tensor):
        conf_tensor = depth_conf.detach().cpu().float().reshape(-1)
    else:
        conf_tensor = torch.as_tensor(depth_conf, dtype=torch.float32).reshape(-1)

    valid = torch.isfinite(point_tensor).all(dim=-1)
    valid &= torch.isfinite(color_tensor).all(dim=-1)
    valid &= conf_tensor >= depth_confidence_threshold

    point_tensor = point_tensor[valid]
    color_tensor = color_tensor[valid].clamp(0.0, 1.0)

    if point_tensor.numel() == 0:
        raise RuntimeError("VGGT did not produce any valid initialization points.")

    if max_initial_points > 0 and point_tensor.shape[0] > max_initial_points:
        keep = torch.randperm(point_tensor.shape[0])[:max_initial_points]
        point_tensor = point_tensor[keep]
        color_tensor = color_tensor[keep]

    return point_tensor.to(device), color_tensor.to(device)


def estimate_scene_scale(points: Tensor, camtoworlds: Tensor, global_scale: float) -> float:
    cam_centers = camtoworlds[:, :3, 3]
    all_xyz = torch.cat([points, cam_centers], dim=0)
    center = all_xyz.mean(dim=0, keepdim=True)
    radius = torch.linalg.norm(all_xyz - center, dim=-1)
    scene_radius = float(torch.quantile(radius, 0.95).item())
    return max(scene_radius * 1.1 * global_scale, 1e-3)


class Runner:
    """Minimal single-process 3DGS trainer driven by LiveStreamDataset + VGGT."""

    def __init__(self, cfg: Config) -> None:
        set_random_seed(42)

        self.cfg = cfg
        self.device, self.model_dtype = get_device()
        if self.device.type != "cuda":
            raise RuntimeError("This runner requires CUDA because gsplat rasterization is GPU-only.")

        os.makedirs(cfg.result_dir, exist_ok=True)
        self.ckpt_dir = Path(cfg.result_dir) / "ckpts"
        self.stats_dir = Path(cfg.result_dir) / "stats"
        self.render_dir = Path(cfg.result_dir) / "renders"
        self.video_dir = Path(cfg.result_dir) / "videos"
        self.ply_dir = Path(cfg.result_dir) / "ply"
        for path in [self.ckpt_dir, self.stats_dir, self.render_dir, self.video_dir, self.ply_dir]:
            path.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(log_dir=str(Path(cfg.result_dir) / "tb"))

        self.stream_config = load_json(cfg.stream_config_path)
        self.dataset = LiveStreamDataset(
            self.stream_config,
            cfg.colmap_path,
            device=str(self.device),
            dtype=self.model_dtype,
        )
        self.vggt = VGGTUtils(
            device=str(self.device),
            dtype=self.model_dtype,
            chunk_size=cfg.vggt_chunk_size,
            colmap_path=cfg.colmap_path,
        )

        self.required_streams = cfg.min_streams or expected_stream_count(self.stream_config)
        if self.required_streams <= 0:
            raise RuntimeError("No streams were configured in the live stream configuration.")

        if cfg.skip_seconds > 0:
            time.sleep(cfg.skip_seconds)

        self.snapshot = self._capture_snapshot()
        self.scene_scale = estimate_scene_scale(self.snapshot.init_points, self.snapshot.camtoworlds, cfg.global_scale)
        self.num_views = self.snapshot.pixels.shape[0]

        feature_dim = cfg.app_feature_dim if cfg.app_opt else None
        self.splats, self.optimizers = create_splats_with_optimizers(
            points=self.snapshot.init_points,
            rgbs=self.snapshot.init_colors,
            init_type=cfg.init_type,
            init_num_pts=cfg.init_num_pts,
            init_extent=cfg.init_extent,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            means_lr=cfg.means_lr,
            scales_lr=cfg.scales_lr,
            opacities_lr=cfg.opacities_lr,
            quats_lr=cfg.quats_lr,
            sh0_lr=cfg.sh0_lr,
            shN_lr=cfg.shN_lr,
            scene_scale=self.scene_scale,
            sh_degree=cfg.sh_degree,
            sparse_grad=cfg.sparse_grad,
            visible_adam=cfg.visible_adam,
            batch_size=cfg.batch_size,
            feature_dim=feature_dim,
            device=str(self.device),
        )
        print("Model initialized. Number of GS:", len(self.splats["means"]))

        self.cfg.strategy.check_sanity(self.splats, self.optimizers)
        if isinstance(self.cfg.strategy, DefaultStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state(scene_scale=self.scene_scale)
        elif isinstance(self.cfg.strategy, MCMCStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state()
        else:
            assert_never(self.cfg.strategy)

        self.pose_optimizers: List[torch.optim.Optimizer] = []
        if cfg.pose_opt:
            self.pose_adjust = CameraOptModule(self.num_views).to(self.device)
            self.pose_adjust.zero_init()
            self.pose_optimizers = [
                torch.optim.Adam(
                    self.pose_adjust.parameters(),
                    lr=cfg.pose_opt_lr * math.sqrt(cfg.batch_size),
                    weight_decay=cfg.pose_opt_reg,
                )
            ]
        else:
            self.pose_adjust = None

        self.pose_perturb = None
        if cfg.pose_noise > 0.0:
            self.pose_perturb = CameraOptModule(self.num_views).to(self.device)
            self.pose_perturb.random_init(cfg.pose_noise)

        self.app_optimizers: List[torch.optim.Optimizer] = []
        self.app_module = None
        if cfg.app_opt:
            self.app_module = AppearanceOptModule(
                n=self.num_views,
                feature_dim=cfg.app_feature_dim,
                embed_dim=cfg.app_embed_dim,
                sh_degree=cfg.sh_degree,
            ).to(self.device)
            self.app_optimizers = [
                torch.optim.Adam(
                    self.app_module.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size),
                    weight_decay=cfg.app_opt_reg,
                )
            ]

        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        if cfg.lpips_net == "alex":
            self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(self.device)
        elif cfg.lpips_net == "vgg":
            self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=False).to(self.device)
        else:
            raise ValueError(f"Unknown LPIPS network: {cfg.lpips_net}")

    def _capture_snapshot(self) -> LiveTrainingSnapshot:
        batch = wait_for_stable_batch(
            self.dataset,
            min_frames=self.required_streams,
            timeout_s=self.cfg.startup_timeout,
            target_size=self.cfg.target_size,
        )
        if batch.processed_tensor is None:
            raise RuntimeError("LiveStreamDataset returned no processed tensor.")

        processed_tensor = batch.processed_tensor.to(self.device, self.model_dtype)
        self.vggt.reinit()
        intrinsic, extrinsic, _depth_map, depth_conf, points, _colors, _, _ = self.vggt.run_vggt(
            processed_tensor,
            cam_names=batch.cam_names,
        )

        pixels = processed_tensor.float().permute(0, 2, 3, 1).contiguous()
        Ks = torch.as_tensor(intrinsic, dtype=torch.float32, device=self.device)
        w2c = torch.as_tensor(extrinsic, dtype=torch.float32, device=self.device)
        camtoworlds = extrinsics_to_camtoworlds(w2c)
        init_points, init_colors = build_initial_point_cloud(
            points=points,
            colors=processed_tensor.permute(0, 2, 3, 1),
            depth_conf=depth_conf,
            depth_confidence_threshold=self.cfg.depth_confidence_threshold,
            max_initial_points=self.cfg.max_initial_points,
            device=self.device,
        )

        return LiveTrainingSnapshot(
            processed_tensor=processed_tensor.float(),
            pixels=pixels,
            camtoworlds=camtoworlds,
            Ks=Ks,
            init_points=init_points,
            init_colors=init_colors,
            display_names=list(batch.display_names),
            cam_names=list(batch.cam_names),
            width=int(processed_tensor.shape[-1]),
            height=int(processed_tensor.shape[-2]),
        )

    def _sample_image_ids(self) -> Tensor:
        if self.cfg.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        replace = self.cfg.batch_size > self.num_views
        if replace:
            return torch.randint(0, self.num_views, (self.cfg.batch_size,), device=self.device)
        return torch.randperm(self.num_views, device=self.device)[: self.cfg.batch_size]

    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        masks: Optional[Tensor] = None,
        rasterize_mode: Optional[RasterizeMode] = None,
        camera_model: Optional[CameraModel] = None,
        image_ids: Optional[Tensor] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        means = self.splats["means"]
        quats = self.splats["quats"]
        scales = torch.exp(self.splats["scales"])
        opacities = torch.sigmoid(self.splats["opacities"])

        sh_degree = kwargs.pop("sh_degree", self.cfg.sh_degree)
        if self.cfg.app_opt:
            if self.app_module is None:
                raise RuntimeError("Appearance optimization is enabled but the module is missing.")
            colors = self.app_module(
                features=self.splats["features"],
                embed_ids=image_ids,
                dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
                sh_degree=sh_degree,
            )
            colors = torch.sigmoid(colors + self.splats["colors"])
        else:
            colors = torch.cat([self.splats["sh0"], self.splats["shN"]], dim=1)

        if rasterize_mode is None:
            rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"
        if camera_model is None:
            camera_model = self.cfg.camera_model

        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            sh_degree=sh_degree,
            viewmats=torch.linalg.inv(camtoworlds),
            Ks=Ks,
            width=width,
            height=height,
            packed=self.cfg.packed,
            absgrad=self.cfg.strategy.absgrad if isinstance(self.cfg.strategy, DefaultStrategy) else False,
            sparse_grad=self.cfg.sparse_grad,
            rasterize_mode=rasterize_mode,
            camera_model=camera_model,
            distributed=False,
            **kwargs,
        )
        if masks is not None:
            render_colors = render_colors.clone()
            render_colors[~masks] = 0
        return render_colors, render_alphas, info

    def _zero_grad_all(self) -> None:
        for optimizer in list(self.optimizers.values()) + self.pose_optimizers + self.app_optimizers:
            optimizer.zero_grad(set_to_none=True)

    def _step_all(self) -> None:
        for optimizer in list(self.optimizers.values()) + self.pose_optimizers + self.app_optimizers:
            optimizer.step()

    def train(self) -> None:
        cfg = self.cfg

        with open(Path(cfg.result_dir) / "cfg.yml", "w", encoding="utf-8") as handle:
            yaml.dump(vars(cfg), handle)

        max_steps = cfg.max_steps
        schedulers = [
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"],
                gamma=0.01 ** (1.0 / max_steps),
            )
        ]
        if cfg.pose_opt:
            schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.pose_optimizers[0],
                    gamma=0.01 ** (1.0 / max_steps),
                )
            )

        global_tic = time.time()
        pbar = tqdm.tqdm(range(max_steps), desc="training")
        for step in pbar:
            image_ids = self._sample_image_ids()
            pixels = self.snapshot.pixels[image_ids]
            Ks = self.snapshot.Ks[image_ids]
            camtoworlds = camtoworlds_gt = self.snapshot.camtoworlds[image_ids]

            if self.pose_perturb is not None:
                camtoworlds = self.pose_perturb(camtoworlds, image_ids)
            if self.pose_adjust is not None:
                camtoworlds = self.pose_adjust(camtoworlds, image_ids)

            sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)
            renders, alphas, info = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=self.snapshot.width,
                height=self.snapshot.height,
                image_ids=image_ids if cfg.app_opt else None,
                sh_degree=sh_degree_to_use,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB",
            )
            colors = renders[..., :3].clamp(0.0, 1.0)
            if cfg.random_bkgd:
                bkgd = torch.rand((colors.shape[0], 1, 1, 3), device=self.device)
                colors = colors + bkgd * (1.0 - alphas)

            self.cfg.strategy.step_pre_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
            )

            l1loss = F.l1_loss(colors, pixels)
            ssimloss = 1.0 - fused_ssim(
                colors.permute(0, 3, 1, 2),
                pixels.permute(0, 3, 1, 2),
                padding="valid",
            )
            loss = torch.lerp(l1loss, ssimloss, cfg.ssim_lambda)
            if cfg.opacity_reg > 0.0:
                loss = loss + cfg.opacity_reg * torch.sigmoid(self.splats["opacities"]).mean()
            if cfg.scale_reg > 0.0:
                loss = loss + cfg.scale_reg * torch.exp(self.splats["scales"]).mean()

            self._zero_grad_all()
            loss.backward()

            if cfg.sparse_grad:
                if not cfg.packed:
                    raise ValueError("sparse_grad=True requires packed=True.")
                gaussian_ids = info["gaussian_ids"]
                for key in self.splats.keys():
                    grad = self.splats[key].grad
                    if grad is None or grad.is_sparse:
                        continue
                    self.splats[key].grad = torch.sparse_coo_tensor(
                        indices=gaussian_ids[None],
                        values=grad[gaussian_ids],
                        size=self.splats[key].size(),
                        device=grad.device,
                    ).coalesce()

            self._step_all()
            for scheduler in schedulers:
                scheduler.step()

            if isinstance(self.cfg.strategy, DefaultStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    packed=cfg.packed,
                )
            elif isinstance(self.cfg.strategy, MCMCStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    lr=schedulers[0].get_last_lr()[0],
                )
            else:
                assert_never(self.cfg.strategy)

            desc = f"loss={loss.item():.4f} | l1={l1loss.item():.4f} | ssim={1.0 - ssimloss.item():.4f} | GS={len(self.splats['means'])}"
            if cfg.pose_opt and cfg.pose_noise > 0.0:
                pose_err = F.l1_loss(camtoworlds_gt, camtoworlds)
                desc += f" | pose_err={pose_err.item():.6f}"
            pbar.set_description(desc)

            if step % max(cfg.tb_every, 1) == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/l1loss", l1loss.item(), step)
                self.writer.add_scalar("train/ssim", 1.0 - ssimloss.item(), step)
                self.writer.add_scalar("train/num_GS", len(self.splats["means"]), step)
                self.writer.add_scalar("train/mem_gb", mem, step)
                if cfg.tb_save_image:
                    canvas = torch.cat([pixels[:1], colors[:1]], dim=2).squeeze(0).permute(2, 0, 1)
                    self.writer.add_image("train/render", canvas, step)
                self.writer.flush()

            if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
                self.save_checkpoint(step, time.time() - global_tic)

            if step in [i - 1 for i in cfg.eval_steps] or step == max_steps - 1:
                self.eval(step)
                self.render_traj(step)

    @torch.no_grad()
    def eval(self, step: int) -> None:
        print("Running evaluation...")
        metrics = defaultdict(list)
        render_time = 0.0
        first_canvas = None

        for index in range(self.num_views):
            pixels = self.snapshot.pixels[index : index + 1]
            camtoworlds = self.snapshot.camtoworlds[index : index + 1]
            Ks = self.snapshot.Ks[index : index + 1]

            torch.cuda.synchronize()
            tic = time.time()
            renders, _alphas, _info = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=self.snapshot.width,
                height=self.snapshot.height,
                sh_degree=self.cfg.sh_degree,
                near_plane=self.cfg.near_plane,
                far_plane=self.cfg.far_plane,
                render_mode="RGB",
            )
            torch.cuda.synchronize()
            render_time += max(time.time() - tic, 1e-10)

            colors = renders[..., :3].clamp(0.0, 1.0)
            metrics["psnr"].append(self.psnr(colors, pixels))
            metrics["ssim"].append(self.ssim(colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2)))
            metrics["lpips"].append(self.lpips(colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2)))

            if first_canvas is None:
                first_canvas = torch.cat([pixels, colors], dim=2).squeeze(0).cpu().numpy()

        psnr = torch.stack(metrics["psnr"]).mean()
        ssim = torch.stack(metrics["ssim"]).mean()
        lpips = torch.stack(metrics["lpips"]).mean()
        stats = {
            "step": step,
            "psnr": float(psnr.item()),
            "ssim": float(ssim.item()),
            "lpips": float(lpips.item()),
            "render_time_per_image": render_time / max(self.num_views, 1),
            "num_gs": int(len(self.splats["means"])),
        }
        print(stats)
        with open(self.stats_dir / f"eval_step{step:05d}.json", "w", encoding="utf-8") as handle:
            json.dump(stats, handle, indent=2)

        self.writer.add_scalar("eval/psnr", stats["psnr"], step)
        self.writer.add_scalar("eval/ssim", stats["ssim"], step)
        self.writer.add_scalar("eval/lpips", stats["lpips"], step)
        if first_canvas is not None:
            imageio.imwrite(self.render_dir / f"eval_step{step:05d}.png", (first_canvas * 255.0).astype(np.uint8))
        self.writer.flush()

    @torch.no_grad()
    def render_traj(self, step: int) -> None:
        if self.cfg.disable_video:
            return
        video_path = self.video_dir / f"traj_{step:05d}.mp4"
        writer = imageio.get_writer(video_path, fps=8)
        for index in range(self.num_views):
            renders, _alphas, _info = self.rasterize_splats(
                camtoworlds=self.snapshot.camtoworlds[index : index + 1],
                Ks=self.snapshot.Ks[index : index + 1],
                width=self.snapshot.width,
                height=self.snapshot.height,
                sh_degree=self.cfg.sh_degree,
                near_plane=self.cfg.near_plane,
                far_plane=self.cfg.far_plane,
                render_mode="RGB",
            )
            colors = renders[..., :3].clamp(0.0, 1.0)
            canvas = torch.cat([self.snapshot.pixels[index : index + 1], colors], dim=2)
            writer.append_data((canvas.squeeze(0).cpu().numpy() * 255.0).astype(np.uint8))
        writer.close()
        print(f"Video saved to {video_path}")

    def save_checkpoint(self, step: int, elapsed: float) -> None:
        stats = {
            "step": step,
            "elapsed": elapsed,
            "num_gs": int(len(self.splats["means"])),
            "mem_gb": torch.cuda.max_memory_allocated() / 1024**3,
        }
        with open(self.stats_dir / f"train_step{step:05d}.json", "w", encoding="utf-8") as handle:
            json.dump(stats, handle, indent=2)

        ckpt = {
            "step": step,
            "cfg": vars(self.cfg),
            "splats": self.splats.state_dict(),
            "snapshot_display_names": self.snapshot.display_names,
            "snapshot_cam_names": self.snapshot.cam_names,
        }
        if self.pose_adjust is not None:
            ckpt["pose_adjust"] = self.pose_adjust.state_dict()
        if self.app_module is not None:
            ckpt["app_module"] = self.app_module.state_dict()
        torch.save(ckpt, self.ckpt_dir / f"ckpt_{step:05d}.pt")

        if self.cfg.save_ply:
            if self.cfg.app_opt:
                rgb = self.app_module(
                    features=self.splats["features"],
                    embed_ids=None,
                    dirs=torch.zeros_like(self.splats["means"][None, :, :]),
                    sh_degree=self.cfg.sh_degree,
                )
                rgb = torch.sigmoid(rgb + self.splats["colors"]).squeeze(0).unsqueeze(1)
                sh0 = rgb_to_sh(rgb)
                shN = torch.empty((sh0.shape[0], 0, 3), device=sh0.device)
            else:
                sh0 = self.splats["sh0"]
                shN = self.splats["shN"]
            export_splats(
                means=self.splats["means"],
                scales=torch.exp(self.splats["scales"]),
                quats=F.normalize(self.splats["quats"], dim=-1),
                opacities=torch.sigmoid(self.splats["opacities"]),
                sh0=sh0,
                shN=shN,
                format="ply",
                save_to=str(self.ply_dir / f"point_cloud_{step:05d}.ply"),
            )

    def load_checkpoint(self, ckpt_path: Union[str, Path]) -> int:
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.splats.load_state_dict(ckpt["splats"])
        if self.pose_adjust is not None and "pose_adjust" in ckpt:
            self.pose_adjust.load_state_dict(ckpt["pose_adjust"])
        if self.app_module is not None and "app_module" in ckpt:
            self.app_module.load_state_dict(ckpt["app_module"])
        return int(ckpt["step"])

    @torch.no_grad()
    def _viewer_render_fn(self, camera_state: CameraState, render_tab_state: RenderTabState):
        width = getattr(render_tab_state, "viewer_width", self.snapshot.width)
        height = getattr(render_tab_state, "viewer_height", self.snapshot.height)
        c2w = torch.from_numpy(camera_state.c2w).float().to(self.device)
        K = torch.from_numpy(camera_state.get_K((width, height))).float().to(self.device)
        renders, alphas, _info = self.rasterize_splats(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=width,
            height=height,
            sh_degree=self.cfg.sh_degree,
            near_plane=self.cfg.near_plane,
            far_plane=self.cfg.far_plane,
            render_mode="RGB",
        )
        mode = getattr(render_tab_state, "render_mode", "rgb")
        if mode == "alpha":
            return alphas[0, ..., 0].detach().cpu().numpy()
        if mode.startswith("depth") and renders.shape[-1] > 3:
            return apply_float_colormap(renders[0, ..., 3].detach().cpu().numpy())
        return renders[0, ..., :3].clamp(0.0, 1.0).detach().cpu().numpy()

    def close(self) -> None:
        try:
            self.writer.close()
        finally:
            if self.dataset is not None:
                self.dataset.stop()


def main(cfg: Config) -> None:
    runner = Runner(cfg)
    try:
        if cfg.ckpt is not None:
            ckpt_path = cfg.ckpt[0] if isinstance(cfg.ckpt, list) else cfg.ckpt
            step = runner.load_checkpoint(ckpt_path)
            runner.eval(step)
            runner.render_traj(step)
        else:
            runner.train()
    finally:
        runner.close()


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    cfg.adjust_steps(cfg.steps_scaler)
    main(cfg)

