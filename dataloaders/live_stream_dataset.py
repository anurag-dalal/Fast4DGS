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



class StreamCapture:
    """Grabs frames from one GStreamer UDP/RTP H264 stream in a background thread."""
    def __init__(self, port: int, node_name: str, host: str, mac: str):
        self.port = port
        self.node_name = node_name
        self.host = host
        self.mac = mac
        self.frame = None
        self._lock = threading.Lock()
        self._running = False
        self._cap = None
        self._thread = None

    def start(self, timeout: float = 15.0, retry_interval: float = 3.0):
        self._running = True
        self._thread = threading.Thread(target=self._open_and_loop, args=(timeout, retry_interval), daemon=True)
        self._thread.start()

    def _open_and_loop(self, timeout: float, retry_interval: float):
        # Optimized pipeline for low latency and quality
        gst = (
            f'udpsrc port={self.port} buffer-size=200000000 '
            f'caps="application/x-rtp, media=video, clock-rate=90000, encoding-name=H265, payload=96" ! '
            f'rtpjitterbuffer latency=15 drop-on-latency=true ! '
            f'rtph265depay ! h265parse config-interval=1 ! '
            f'nvh265dec enable-max-performance=1 ! '          
            f'videoconvert ! video/x-raw,format=BGR ! '
            f'appsink sync=false max-buffers=1 drop=true'
        )
        deadline = time.monotonic() + timeout
        attempt = 0
        while self._running:
            if time.monotonic() > deadline:
                 print(f"[FAIL] port {self.port} ({self.node_name}) – gave up after {timeout:.0f}s")
                 break

            attempt += 1
            cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
            if cap.isOpened():
                self._cap = cap
                print(f"[OK]   port {self.port} ({self.node_name}) opened on attempt {attempt}")
                self._loop()
                break
            
            cap.release()
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(min(retry_interval, remaining))

    def _loop(self):
        while self._running and self._cap and self._cap.isOpened():
            ret, frame = self._cap.read()
            if ret and frame is not None:
                with self._lock:
                    self.frame = frame
            else:
                time.sleep(0.005)

    def read(self):
        with self._lock:
            return self.frame

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._cap:
            self._cap.release()

class LiveStreamDataset:
    def __init__(self, config, colmap_path, device='cuda', dtype=torch.float16):
        self.config = config
        self.captures = []
        self.colmap_path = colmap_path
        self.device = device
        self.dtype = dtype
        # ── Load COLMAP calibration ──────────────────────────────────────────
        print(f"Loading COLMAP model from {self.colmap_path}...")
        colmap_cameras, colmap_images, _ = read_model(self.colmap_path, ext='.bin')
        name_to_image = {img.name: img for img in colmap_images.values()}
        self._setup_captures()
        

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
    
    def _setup_captures(self):
        nodes = self.config.get("nodes", [])
        for node in nodes:
            name = node.get("name", "")
            host = node.get("host", "")
            mac = node.get("MAC", "")
            for port in node.get("ports", []):
                self.captures.append(StreamCapture(port, name, host, mac))
        for c in self.captures:
            c.start()
    def get_processed_frames(self, target_size=518):
        count = 0
        try:
            while True:
                # Collect frames
                frames = []
                
                # We want frames from all active streams
                # If some take too long, we might default to previous or black?
                # For now, just take what is available
                active_frames = 0
                for c in self.captures:
                    f = c.read()
                    if f is not None:
                        frames.append(f)
                        active_frames += 1
                    else:
                        # Provide dummy frame or skip? 
                        # VGGT might need consistent count if we wanted consistent poses?
                        # For now skip.
                        pass
                
                if active_frames < 2:
                    # Need at least 2 frames for meaningful structure usually?
                    # Or just wait until we have something.
                    time.sleep(0.1)
                    continue
                    
                # Preprocess
                start_t = time.time()
                input_tensor = self.preprocess_frames(frames, target_size=518).to(self.device, self.dtype)
                return input_tensor
                
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
                
    def stop(self):            
        print("Stopping streams...")
        for c in self.captures:
            c.stop()

