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
import viser
import viser.transforms as vtf
from torchvision import transforms as TF

# Add VGGT-X to path
VGGT_PATH = Path(__file__).resolve().parent / "VGGT-X"
if str(VGGT_PATH) not in sys.path:
    sys.path.append(str(VGGT_PATH))

try:
    from vggt.models.vggt import VGGT
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from vggt.utils.geometry import unproject_depth_map_to_point_map
except ImportError as e:
    print(f"Error importing VGGT: {e}")
    sys.exit(1)

# Config path
CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "stream_config.json"

def load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)

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

def preprocess_frames(frames, target_size=518):
    # Frames are list of BGR numpy arrays
    # Return: tensor [B, 3, H, W]
    images = []
    to_tensor = TF.ToTensor()
    
    for frame in frames:
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--port", type=int, default=8080, help="Viser port")
    parser.add_argument("--scale", type=int, default=8, help="Downscale frame for VGGT (unused currently)")
    args = parser.parse_args()

    # Load Config
    cfg = load_config(CONFIG_PATH)
    nodes = cfg.get("nodes", [])
    
    captures = []
    print(f"Loading streams from {CONFIG_PATH}...")
    for node in nodes:
        name = node.get("name", "")
        host = node.get("host", "")
        mac = node.get("MAC", "")
        for port in node.get("ports", []):
            captures.append(StreamCapture(port, name, host, mac))

    # Start Streams
    print(f"Starting {len(captures)} streams...")
    for c in captures:
        c.start()
        
    # Load VGGT
    torch.set_float32_matmul_precision('high')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        dtype = torch.float32
    print(f"Loading VGGT on {device} ({dtype})...")
    
    model = VGGT(chunk_size=args.chunk_size)
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    try:
        model.load_state_dict(torch.hub.load_state_dict_from_url(_URL, progress=True))
    except Exception as e:
        print(f"Failed to load model from URL: {e}")
        print("Ensure internet connection or manually place model.")
        # Attempt to continue if model path cached? or fail.
    
    model.eval()
    model = model.to(device).to(dtype)
    model.track_head = None

    # Viser
    server = viser.ViserServer(port=args.port)
    print(f"Viser running at http://localhost:{args.port}")

    try:
        while True:
            # Collect frames
            frames = []
            
            # We want frames from all active streams
            # If some take too long, we might default to previous or black?
            # For now, just take what is available
            active_frames = 0
            for c in captures:
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
            input_tensor = preprocess_frames(frames, target_size=518).to(device, dtype)
            
            # Inference
            with torch.no_grad():
                predictions = model(input_tensor)
                # input_tensor is [B, 3, H, W]. Model treats B as Sequence (S) with Batch=1 -> [1, S, 3, H, W]
                # Outputs are [1, S, ...]. We squeeze the first dimension.
                extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions['pose_enc'], input_tensor.shape[-2:])
                extrinsic = extrinsic.squeeze(0)
                intrinsic = intrinsic.squeeze(0)
                depth_map = predictions['depth'].squeeze(0) # [S, H, W, 1]
            
            # Unproject
            points = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic) 
            # points: [B, H, W, 3]
            
            # Colorize
            colors = input_tensor.permute(0, 2, 3, 1).cpu().float().numpy() # [B, H, W, 3]
            
            if isinstance(points, torch.Tensor):
                points = points.cpu().float().numpy()

            points_flat = points.reshape(-1, 3)
            colors_flat = colors.reshape(-1, 3)
            
            # RAMBO: Remove all dark greyish to black points
            # We filter based on intensity (mean of RGB). 
            # Threshold 0.1 roughly corresponds to dark grey.
            intensity = np.mean(colors_flat, axis=1)
            valid_color_mask = intensity > 0.01
            points_flat = points_flat[valid_color_mask]
            colors_flat = colors_flat[valid_color_mask]

            # Filter invalid points (depth > 0 or not infinite)
            # depth_map usually has 0 for invalid?
            # unproject handles it.
            
            # Sample for viz
            n_points = points_flat.shape[0]
            target_n = 100000
            if n_points > target_n:
                mask = np.random.choice(n_points, target_n, replace=False)
                p_vis = points_flat[mask]
                c_vis = colors_flat[mask]
            else:
                p_vis = points_flat
                c_vis = colors_flat
            
            server.scene.add_point_cloud(
                "/vggt/cloud",
                points=p_vis,
                colors=c_vis,
                point_size=0.0015
            )
            
            dt = time.time() - start_t
            print(f"Update: {dt:.3f}s, FPS: {1.0/dt:.1f}, Streams: {active_frames}")
            
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Stopping streams...")
        for c in captures:
            c.stop()

if __name__ == "__main__":
    main()
