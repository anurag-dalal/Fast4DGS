from pathlib import Path
import sys
import torch

# ── Add VGGT-X to path ──────────────────────────────────────────────────
VGGT_PATH = Path(__file__).resolve().parent.parent / "VGGT-X"
if str(VGGT_PATH) not in sys.path:
    sys.path.append(str(VGGT_PATH))

from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from dataset.read_write_model import read_model
import numpy as np
class VGGTUtils:
    def __init__(self, device: str, dtype: torch.dtype, chunk_size: int = 512, colmap_path: str = None):
        self.device = device
        self.dtype = dtype
        self.chunk_size = chunk_size
        self.colmap_path = colmap_path
        self.model = self.load_vggt()
    
    def load_vggt(self):
        """Load the VGGT-1B model."""
        self.model = VGGT(chunk_size=self.chunk_size)
        url = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        self.model.load_state_dict(torch.hub.load_state_dict_from_url(url, progress=True))
        self.model.eval()
        self.model = self.model.to(self.device).to(self.dtype)
        self.model.track_head = None

        self.colmap_cameras, self.colmap_images, _ = read_model(self.colmap_path, ext='.bin')
        self.name_to_image = {img.name[0:-4]: img for img in self.colmap_images.values()}

        # print(self.colmap_cameras)

        self.first_init_done = False

    def change_colmap_to_vggt(self):
        colmap_h, colmap_w = self.colmap_intrinsics[0][0,2]*2, self.colmap_intrinsics[0][1,2]*2
        vggt_h, vggt_w = 518, 518
        scale_h, scale_w = vggt_h / colmap_h, vggt_w / colmap_w
        self.colmap_intrinsics[:, 0, 0] = self.colmap_intrinsics[:, 0, 0] * scale_w
        self.colmap_intrinsics[:, 1, 1] = self.colmap_intrinsics[:, 1, 1] * scale_h
        self.colmap_intrinsics[:, 0, 2] = self.colmap_intrinsics[:, 0, 2] * scale_w
        self.colmap_intrinsics[:, 1, 2] = self.colmap_intrinsics[:, 1, 2] * scale_h


    def bundle_adjust(self, input_tensor, num_steps=1000, lr=1e-4):
        # Simple bundle adjustment to align COLMAP extrinsics to VGGt's
        # Only optimizes extrinsics, not intrinsics
        if self.intrinsic is None or self.extrinsic is None:
            print("Cannot run bundle adjustment before first init")
            return
        predictions = self.model(input_tensor)
        depth_map = predictions["depth"].squeeze(0)       # [S, H, W, 1]
        extrinsic_opt = torch.tensor(self.extrinsic, device=self.device, dtype=self.dtype, requires_grad=True)
        optimizer = torch.optim.Adam([extrinsic_opt], lr=lr)

        
        

    def run_vggt(self, input_tensor, cam_names=None):
        with torch.inference_mode():
            predictions = self.model(input_tensor)
            if not self.first_init_done:
                self.extrinsic, self.intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], input_tensor.shape[-2:])
                self.extrinsic = self.extrinsic.squeeze(0)   # [S, 3, 4]
                self.intrinsic = self.intrinsic.squeeze(0)   # [S, 3, 3]
                self.cam_names = cam_names
                self.colmap_extrinsics = []
                self.colmap_intrinsics = []
                for cam_name in self.cam_names:
                    img = self.name_to_image[cam_name]
                    cam = self.colmap_cameras[img.camera_id]
                    self.colmap_intrinsics.append(np.array([[cam.params[0], 0, cam.params[2]], [0, cam.params[1], cam.params[3]], [0, 0, 1]]))
                    R = np.array(img.qvec2rotmat())
                    t = np.array(img.tvec).reshape(3, 1)
                    ee = np.concatenate([R, t], axis=1)  # [3, 4]
                    self.colmap_extrinsics.append(ee)
                self.colmap_intrinsics = np.stack(self.colmap_intrinsics, axis=0)  # [S, 3, 3]
                self.colmap_extrinsics = np.stack(self.colmap_extrinsics, axis=0)  # [S, 3, 4]
                self.change_colmap_to_vggt()
                self.first_init_done = True
            depth_map = predictions["depth"].squeeze(0)       # [S, H, W, 1]
            depth_conf = predictions["depth_conf"].squeeze(0)  # [S, H, W]
            
        # ── 4. Unproject to 3-D ─────────────────────────────────────────────
        points = unproject_depth_map_to_point_map(depth_map, self.extrinsic, self.intrinsic)
        # points: [S, H, W, 3]

        colors = input_tensor.permute(0, 2, 3, 1).cpu().float().numpy()  # [S,H,W,3]

        return self.intrinsic, self.extrinsic, depth_map, depth_conf, points, colors, self.colmap_intrinsics, self.colmap_extrinsics
    
    def reinit(self):
        self.first_init_done = False
        self.intrinsic = None
        self.extrinsic = None
        
        