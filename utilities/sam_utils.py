#Example from SAM2, script is based on this:   https://github.com/facebookresearch/sam2/blob/main/notebooks/video_predictor_example.ipynb

#Boolean configs

prewiew = False  # Find points for prompt box

#Imports
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "sam2")) # Add SAM2 code to path, must be in same directory as this script

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

from sam2.build_sam import build_sam2_video_predictor


# Configuration for folders.
base_dir = os.path.dirname(os.path.abspath(__file__))
source_images = os.path.join(base_dir, "images/images_v0")
output = os.path.join(base_dir, "output")
temp_folder = os.path.join(base_dir, "jpeg_frames")

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using cuda")
else:
    device = torch.device("cpu")
    print("Using cpu")

if device.type == "cuda":
    # use bfloat16 from dockumentation
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()


png_files = [f for f in os.listdir(source_images) if f.endswith(".png")]
png_files.sort(key=lambda f: int(f.replace("cam", "").replace(".png", "")))

os.makedirs(temp_folder, exist_ok=True)

for idx, fname in enumerate(png_files):
    img = Image.open(os.path.join(source_images, fname)).convert("RGB")
    img.save(os.path.join(temp_folder, f"{idx:05d}.jpg"))


sam2_checkpoint = os.path.join(base_dir, "sam2/checkpoints/sam2.1_hiera_large.pt")
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

inference_state = predictor.init_state(video_path=temp_folder)


# Prompt object to find: 
prompt_box = np.array([600, 50, 1200, 600], dtype=np.float32)  # Box around object of intrest [x_min, y_min, x_max, y_max] 

frame_index = 0
frame_begin = 0
frame_end = 24
object_id = 1


if prewiew:
    print("Preview mode: adjust prompt_box as needed, then set prewiew=False and re-run.")
    image = Image.open(os.path.join(source_images, f"cam{frame_index}.png")).convert("RGB")
    plt.imshow(np.array(image))
    plt.show()
    sys.exit(0)


_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=frame_index,
    obj_id=object_id,
    box=prompt_box,
)


for frame_idx, obj_ids, video_res_masks in predictor.propagate_in_video(inference_state):
    mask = (video_res_masks[0, 0] > 0.0).cpu().numpy().astype(np.uint8) * 255

    output_path = os.path.join(output, f"cam{frame_idx}.png")
    mask_image = Image.fromarray(mask, mode="L")
    mask_image.save(output_path)
    print(f"Saved {output_path}")

print("Done!")