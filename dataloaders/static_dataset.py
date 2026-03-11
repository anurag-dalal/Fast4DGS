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
from PIL import Image
from torchvision import transforms as TF

from dataset.read_write_model import read_model




# ═════════════════════════════════════════════════════════════════════════
#  CONFIG – change these or pass via CLI
# ═════════════════════════════════════════════════════════════════════════
DEFAULT_IMAGE_DIR = "/home/anurag/stream_captures/images"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# ── Camera calibration (from dewarper config) ────────────────────────────
# projection-type=3  (rational model, 8 distortion coeffs)
CALIB_FX = 1042.00
CALIB_FY = 1042.00
CALIB_CX = 950.6477088
CALIB_CY = 557.5285168
CALIB_DIST = np.array([
    -0.0516463965,    # k1
    -0.04747710885,   # k2
    -0.0001566917679, # p1
     0.0002697267978, # p2
     0.01013947186,   # k3
     0.3133340326,    # k4
    -0.1464375728,    # k5
     0.02119491113,   # k6
], dtype=np.float64)
CALIB_W = 1920
CALIB_H = 1080



class StaticDataset:
    def __init__(self, image_folder_path, colmap_path, device='cuda', dtype=torch.float16, undistort=True):
        self.image_folder_path = image_folder_path
        self.colmap_path = colmap_path
        self.device = device
        self.dtype = dtype
        # ── Load COLMAP calibration ──────────────────────────────────────────
        print(f"Loading COLMAP model from {self.colmap_path}...")
        colmap_cameras, colmap_images, _ = read_model(self.colmap_path, ext='.bin')
        name_to_image = {img.name: img for img in colmap_images.values()}
        self.undistort = undistort
        self.build_frames(target_size=518)


        

    def preprocess_frames(self, frames, target_size=518):
        # Frames are list of BGR numpy arrays
        # Return: tensor [B, 3, H, W]
        images = []
        to_tensor = TF.ToTensor()
        
        for frame in frames:
            # Black out corners so intensity filter removes them from point cloud
            # BGR to RGB
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            
            width, height = img.size
            max_dim = max(width, height)
            
            left = (max_dim - width) // 2
            top = (max_dim - height) // 2
            
            square_img = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))
            square_img.paste(img, (left, top))
            
            square_img = square_img.resize((target_size, target_size), Image.Resampling.BICUBIC)
            
            img_tensor = to_tensor(square_img)
            images.append(img_tensor)
            
        return torch.stack(images)
    
    def build_frames(self, target_size=518):
        frames = []
        for filename in sorted(os.listdir(self.image_folder_path)):
            if any(filename.lower().endswith(ext) for ext in IMAGE_EXTENSIONS):
                img_path = os.path.join(self.image_folder_path, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    frames.append(img)
                else:
                    print(f"Warning: Failed to read image {img_path}")
        if self.undistort:
            # Undistort frames using COLMAP calibration
            K = np.array([[CALIB_FX, 0, CALIB_CX],
                          [0, CALIB_FY, CALIB_CY],
                          [0, 0, 1]], dtype=np.float64)
            D = CALIB_DIST
            frames = [cv2.undistort(frame, K, D) for frame in frames]
        if not frames:
            raise ValueError(f"No valid images found in {self.image_folder_path}")
        
        # Preprocess
        self.input_tensor = self.preprocess_frames(frames, target_size=target_size).to(self.device, self.dtype)

    def get_processed_frames(self):
        return self.input_tensor