#!/usr/bin/env python3
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as TF

from dataset.read_write_model import read_model

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


@dataclass(slots=True)
class StreamFrameBatch:
    processed_tensor: torch.Tensor | None
    cam_names: list[str]
    display_names: list[str]
    raw_frames: dict[str, np.ndarray]
    undistorted_frames: dict[str, np.ndarray]
    processed_frames: dict[str, np.ndarray]
    processed_index_by_display_name: dict[str, int]


class StreamCapture:
    """Grabs frames from one GStreamer UDP/RTP H264 stream in a background thread."""
    def __init__(self, port: int, node_name: str, host: str, mac: str, cam_name: str):
        self.port = port
        self.node_name = node_name
        self.host = host
        self.mac = mac
        self.cam_name = cam_name
        self.frame = None
        self._lock = threading.Lock()
        self._running = False
        self._cap = None
        self._thread = None

    @property
    def display_name(self) -> str:
        suffix = f" ({self.cam_name})" if self.cam_name else ""
        return f"{self.node_name}:{self.port}{suffix}"

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
            if self.frame is None:
                return None
            return self.frame.copy()

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
        self.build_undistort_maps()
        self._setup_captures()
    
    def _preprocess_single_frame(self, frame, target_size=518):
        to_tensor = TF.ToTensor()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)

        width, height = img.size
        max_dim = max(width, height)
        left = (max_dim - width) // 2
        top = (max_dim - height) // 2

        square_img = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))
        square_img.paste(img, (left, top))
        square_img = square_img.resize((target_size, target_size), Image.Resampling.BICUBIC)

        processed_rgb = np.array(square_img)
        processed_bgr = cv2.cvtColor(processed_rgb, cv2.COLOR_RGB2BGR)
        img_tensor = to_tensor(square_img)
        return processed_bgr, img_tensor

    def preprocess_frames(self, frames, target_size=518):
        # Frames are list of BGR numpy arrays
        # Return: tensor [B, 3, H, W]
        images = []
        
        for frame in frames:
            _, img_tensor = self._preprocess_single_frame(frame, target_size=target_size)
            images.append(img_tensor)
            
        return torch.stack(images)
    
    def build_undistort_maps(self, w: int = CALIB_W, h: int = CALIB_H):
        """Pre-compute undistortion remap tables (computed once, reused for all frames)."""
        K = np.array([
            [CALIB_FX, 0,        CALIB_CX],
            [0,        CALIB_FY, CALIB_CY],
            [0,        0,        1       ],
        ], dtype=np.float64)
        # newCameraMatrix = K  (keep same principal point / focal length)
        self.undistort_map1, self.undistort_map2 = cv2.initUndistortRectifyMap(
            K, CALIB_DIST, None, K, (w, h), cv2.CV_32FC1)
        


    def undistort_image(self, img: np.ndarray, map1, map2,
                        interpolation=cv2.INTER_LINEAR) -> np.ndarray:
        """Remap an image (or mask) using pre-computed undistortion maps."""
        return cv2.remap(img, map1, map2, interpolation,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    def _setup_captures(self):
        nodes = self.config.get("nodes", [])
        for node in nodes:
            name = node.get("name", "")
            host = node.get("host", "")
            mac = node.get("MAC", "")
            cam_names = node.get("names", [])
            for port, cam_name in zip(node.get("ports", []), cam_names):
                self.captures.append(StreamCapture(port, name, host, mac, cam_name))
        for c in self.captures:
            c.start()

    def get_stream_batch(self, target_size=518, min_frames=2, wait=True):
        try:
            while True:
                raw_frames = {}
                undistorted_frames = {}
                processed_frames = {}
                tensors = []
                cam_names = []
                display_names = []

                for c in self.captures:
                    f = c.read()
                    if f is not None:
                        raw_frames[c.display_name] = f
                        undistorted = self.undistort_image(f, self.undistort_map1, self.undistort_map2)
                        undistorted_frames[c.display_name] = undistorted
                        processed_bgr, img_tensor = self._preprocess_single_frame(undistorted, target_size=target_size)
                        processed_frames[c.display_name] = processed_bgr
                        tensors.append(img_tensor)
                        cam_names.append(c.cam_name)
                        display_names.append(c.display_name)

                if len(display_names) < min_frames:
                    if not wait:
                        return StreamFrameBatch(
                            processed_tensor=None,
                            cam_names=cam_names,
                            display_names=display_names,
                            raw_frames=raw_frames,
                            undistorted_frames=undistorted_frames,
                            processed_frames=processed_frames,
                            processed_index_by_display_name={name: index for index, name in enumerate(display_names)},
                        )
                    time.sleep(0.1)
                    continue

                input_tensor = torch.stack(tensors).to(self.device, self.dtype)
                return StreamFrameBatch(
                    processed_tensor=input_tensor,
                    cam_names=cam_names,
                    display_names=display_names,
                    raw_frames=raw_frames,
                    undistorted_frames=undistorted_frames,
                    processed_frames=processed_frames,
                    processed_index_by_display_name={name: index for index, name in enumerate(display_names)},
                )
                
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

    def get_processed_frames(self, target_size=518):
        batch = self.get_stream_batch(target_size=target_size, min_frames=2, wait=True)
        return batch.processed_tensor, batch.cam_names

    def get_unprocessed_frames(self, min_frames=1, wait=False):
        batch = self.get_stream_batch(target_size=518, min_frames=min_frames, wait=wait)
        return batch.raw_frames, batch.display_names

    def map_raw_box_to_processed(self, raw_frame: np.ndarray, raw_box, target_size=518):
        x1, y1, x2, y2 = [int(v) for v in raw_box]
        h, w = raw_frame.shape[:2]
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))

        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255
        undistorted_mask = self.undistort_image(mask, self.undistort_map1, self.undistort_map2, interpolation=cv2.INTER_NEAREST)
        ys, xs = np.where(undistorted_mask > 0)

        if len(xs) == 0 or len(ys) == 0:
            und_x1, und_y1, und_x2, und_y2 = x1, y1, x2, y2
        else:
            und_x1 = int(xs.min())
            und_y1 = int(ys.min())
            und_x2 = int(xs.max()) + 1
            und_y2 = int(ys.max()) + 1

        max_dim = max(w, h)
        left = (max_dim - w) // 2
        top = (max_dim - h) // 2
        scale = target_size / max_dim
        return (
            int(round((und_x1 + left) * scale)),
            int(round((und_y1 + top) * scale)),
            int(round((und_x2 + left) * scale)),
            int(round((und_y2 + top) * scale)),
        )
                
    def stop(self):            
        print("Stopping streams...")
        for c in self.captures:
            c.stop()

