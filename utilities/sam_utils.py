from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_CFG = ROOT_DIR / "configs" / "sam2.1_hiera_l.yaml"
DEFAULT_CHECKPOINT = Path("/home/anurag/codes_ole/segmentation_tool/sam2/checkpoints/sam2.1_hiera_large.pt")


def get_sam_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_sam_autocast_dtype(device: torch.device) -> torch.dtype | None:
    if device.type != "cuda":
        return None
    major, _ = torch.cuda.get_device_capability()
    return torch.bfloat16 if major >= 8 else torch.float16


def _write_video_predictor_frames(frames: Sequence[np.ndarray], temp_dir: Path) -> None:
    for index, frame in enumerate(frames):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        Image.fromarray(rgb_frame).save(temp_dir / f"{index:05d}.jpg", format="JPEG", quality=95)


def _load_checkpoint(model, checkpoint: Path, device: torch.device) -> None:
    state_dict = torch.load(str(checkpoint), map_location="cpu", weights_only=True)["model"]
    missing_keys, unexpected_keys = model.load_state_dict(state_dict)
    if missing_keys:
        raise RuntimeError(f"Missing checkpoint keys: {missing_keys}")
    if unexpected_keys:
        raise RuntimeError(f"Unexpected checkpoint keys: {unexpected_keys}")
    model.to(device)
    model.eval()


def _build_video_predictor(model_cfg: Path | str, checkpoint: Path, device: torch.device):
    hydra_overrides = [
        "++model._target_=sam2.sam2_video_predictor.SAM2VideoPredictor",
        "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
        "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
        "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
        "++model.binarize_mask_from_pts_for_mem_enc=true",
        "++model.fill_hole_area=8",
    ]

    config_path = Path(model_cfg)
    if config_path.exists():
        GlobalHydra.instance().clear()
        with initialize_config_dir(version_base=None, config_dir=str(config_path.parent.resolve())):
            cfg = compose(config_name=config_path.name, overrides=hydra_overrides)
        OmegaConf.resolve(cfg)
        model = instantiate(cfg.model, _recursive_=True)
        _load_checkpoint(model, checkpoint, device)
        return model

    return build_sam2_video_predictor(str(model_cfg), str(checkpoint), device=device)


def _extract_mask(mask_tensor: torch.Tensor, obj_ids, object_id: int) -> np.ndarray:
    object_index = 0
    for index, current_id in enumerate(obj_ids):
        if int(current_id) == object_id:
            object_index = index
            break
    return (mask_tensor[object_index] > 0.0).detach().cpu().numpy().astype(np.uint8) * 255


def overlay_mask_on_frame(frame: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    if frame.shape[:2] != mask.shape[:2]:
        raise ValueError("frame and mask must have matching height and width")

    overlay = frame.copy()
    overlay[mask > 0] = (0, 0, 255)
    blended = cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0)

    mask_outline = cv2.Canny(mask, 50, 150)
    blended[mask_outline > 0] = (0, 255, 255)
    return blended


class SAMVideoMaskGenerator:
    def __init__(
        self,
        model_cfg: Path | str = DEFAULT_MODEL_CFG,
        checkpoint: Path | str = DEFAULT_CHECKPOINT,
        device: torch.device | str | None = None,
    ) -> None:
        self.model_cfg = Path(model_cfg) if isinstance(model_cfg, str) else model_cfg
        self.checkpoint = Path(checkpoint) if isinstance(checkpoint, str) else checkpoint
        self.device = torch.device(device) if device is not None else get_sam_device()
        self.autocast_dtype = get_sam_autocast_dtype(self.device)
        self._predictor = None

    def _get_predictor(self):
        if self._predictor is None:
            self._predictor = _build_video_predictor(self.model_cfg, self.checkpoint, self.device)
        return self._predictor

    def release(self) -> None:
        self._predictor = None
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def segment_frames(
        self,
        frames: Sequence[np.ndarray],
        prompt_box: Sequence[int | float],
        *,
        frame_index: int = 0,
        object_id: int = 1,
    ) -> list[np.ndarray]:
        if not frames:
            raise ValueError("frames must not be empty")
        if len(prompt_box) != 4:
            raise ValueError("prompt_box must contain 4 values: x1, y1, x2, y2")

        predictor = self._get_predictor()
        box = np.array(prompt_box, dtype=np.float32)

        def _run() -> list[np.ndarray]:
            with tempfile.TemporaryDirectory(prefix="sam2_frames_") as temp_dir_name:
                temp_dir = Path(temp_dir_name)
                _write_video_predictor_frames(frames, temp_dir)

                inference_state = predictor.init_state(video_path=str(temp_dir))
                _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=frame_index,
                    obj_id=object_id,
                    box=box,
                )

                mask_by_index: dict[int, np.ndarray] = {
                    frame_index: _extract_mask(out_mask_logits[0], out_obj_ids, object_id)
                }
                for current_frame_idx, obj_ids, video_res_masks in predictor.propagate_in_video(inference_state):
                    mask_by_index[current_frame_idx] = _extract_mask(video_res_masks[0], obj_ids, object_id)

                return [mask_by_index[index] for index in range(len(frames))]

        if self.device.type == "cuda" and self.autocast_dtype is not None:
            with torch.inference_mode(), torch.autocast("cuda", dtype=self.autocast_dtype):
                return _run()
        with torch.inference_mode():
            return _run()