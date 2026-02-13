#!/usr/bin/env python3
"""
MAXIMUM PERFORMANCE VERSION - Pre-scaled frames, parallel processing, performance monitoring

Key improvements:
- Frames are resized in capture threads (parallel, not in main loop)
- Minimal locking in display loop
- Performance monitoring to identify bottlenecks
- Optimized numpy operations
"""

import argparse
import json
import threading
import time
from pathlib import Path
from collections import deque

import cv2
import numpy as np
CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "stream_config.json"


def load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


class StreamCapture:
    """Captures and PRE-SCALES frames in background thread."""

    def __init__(self, port: int, node_name: str, host: str, mac: str, 
                 target_width: int, target_height: int):
        self.port = port
        self.node_name = node_name
        self.host = host
        self.mac = mac
        self.target_width = target_width
        self.target_height = target_height

        # Store PRE-SCALED frame (no resize in main loop!)
        self.scaled_frame = None
        self._lock = threading.Lock()
        self._running = False
        self._cap = None
        self._thread = None
        
        # Performance monitoring
        self.frame_count = 0
        self.last_fps_check = time.time()
        self.current_fps = 0.0

    def start(self, timeout: float = 15.0, retry_interval: float = 3.0):
        self._running = True
        self._thread = threading.Thread(target=self._open_and_loop,
                                        args=(timeout, retry_interval),
                                        daemon=True)
        self._thread.start()

    def _open_and_loop(self, timeout: float, retry_interval: float):
        gst = (
            f'udpsrc port={self.port} buffer-size=200000000 '
            f'caps="application/x-rtp, media=video, clock-rate=90000, encoding-name=H265, payload=96" ! '
            f'rtpjitterbuffer latency=100 drop-on-latency=false ! '
            f'queue max-size-buffers=4 max-size-time=0 max-size-bytes=0 ! '
            f'rtph265depay ! h265parse config-interval=1 ! '
            f'queue max-size-buffers=4 max-size-time=0 max-size-bytes=0 ! '
            f'nvh265dec ! '
            f'queue max-size-buffers=4 max-size-time=0 max-size-bytes=0 ! '
            f'videoconvert ! video/x-raw,format=BGR ! '
            f'appsink sync=false max-buffers=4 drop=true emit-signals=false'
        )
        
        deadline = time.monotonic() + timeout
        attempt = 0
        while self._running and time.monotonic() < deadline:
            attempt += 1
            cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
            if cap.isOpened():
                self._cap = cap
                print(f"[OK]   port {self.port} ({self.node_name}) opened on attempt {attempt}")
                self._loop()
                return
            cap.release()
            remaining = deadline - time.monotonic()
            if remaining > 0:
                print(f"[RETRY] port {self.port} ({self.node_name}) attempt {attempt} failed, "
                      f"{remaining:.1f}s left")
                time.sleep(min(retry_interval, remaining))

        print(f"[FAIL] port {self.port} ({self.node_name}) – gave up after {timeout:.0f}s")

    def _loop(self):
        """Capture loop - RESIZES FRAMES HERE in parallel thread"""
        while self._running and self._cap and self._cap.isOpened():
            ret, frame = self._cap.read()
            if ret and frame is not None:
                # RESIZE IN THIS THREAD (parallel across all streams)
                scaled = cv2.resize(frame, (self.target_width, self.target_height), 
                                   interpolation=cv2.INTER_AREA)
                
                # Atomic swap - minimal lock time
                with self._lock:
                    self.scaled_frame = scaled
                
                # FPS tracking
                self.frame_count += 1
                now = time.time()
                if now - self.last_fps_check >= 1.0:
                    self.current_fps = self.frame_count / (now - self.last_fps_check)
                    self.frame_count = 0
                    self.last_fps_check = now
            else:
                time.sleep(0.001)

    def read_scaled(self):
        """Returns pre-scaled frame - NO RESIZE NEEDED"""
        with self._lock:
            return self.scaled_frame

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._cap:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None


def draw_overlay(img, text_lines, scale_factor):
    """Draw overlay - optimized version"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.35, 0.6 / (scale_factor ** 0.3))
    thickness = 1
    line_h = int(20 / (scale_factor ** 0.3))
    pad = 4

    max_w = 0
    for line in text_lines:
        (w, _), _ = cv2.getTextSize(line, font, font_scale, thickness)
        max_w = max(max_w, w)
    bg_h = line_h * len(text_lines) + pad * 2
    bg_w = max_w + pad * 2

    # Optimized overlay drawing
    overlay = img[:bg_h, :bg_w].copy()
    cv2.rectangle(overlay, (0, 0), (bg_w, bg_h), (0, 0, 0), -1)
    img[:bg_h, :bg_w] = cv2.addWeighted(overlay, 0.55, img[:bg_h, :bg_w], 0.45, 0)

    y = pad + line_h - 4
    for line in text_lines:
        cv2.putText(img, line, (pad, y), font, font_scale, (255, 255, 255), 
                   thickness, cv2.LINE_AA)
        y += line_h


def main():
    parser = argparse.ArgumentParser(description="High-performance multi-stream viewer")
    parser.add_argument("--scale", type=int, default=4,
                        help="Downscale factor (default: 4)")
    parser.add_argument("--fps-monitor", action="store_true",
                        help="Show per-stream FPS in overlay")
    args = parser.parse_args()

    cfg = load_config(CONFIG_PATH)
    nodes = cfg.get("nodes", [])
    stream_w = cfg.get("stream_width", 1920)
    stream_h = cfg.get("stream_height", 1080)

    if not nodes:
        print("No nodes found in config – exiting.")
        return

    n_rows = len(nodes)
    n_cols = max(len(n.get("ports", [])) for n in nodes)

    scale = args.scale
    cell_w = stream_w // scale
    cell_h = stream_h // scale

    # Pre-allocate black cell
    black_cell = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
    cv2.putText(black_cell, "waiting...",
                (cell_w // 4, cell_h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, max(0.4, 0.7 / (scale ** 0.3)),
                (80, 80, 80), 1, cv2.LINE_AA)

    # Create captures - they will PRE-SCALE frames
    grid: list[list[StreamCapture | None]] = []
    all_captures: list[StreamCapture] = []

    for node in nodes:
        row = []
        name = node.get("name", "")
        host = node.get("host", "")
        mac = node.get("MAC", node.get("mac", ""))
        for port in node.get("ports", []):
            # Pass target dimensions - frames will be scaled in capture thread
            sc = StreamCapture(port, name, host, mac, cell_w, cell_h)
            row.append(sc)
            all_captures.append(sc)
        while len(row) < n_cols:
            row.append(None)
        grid.append(row)

    print(f"Starting {len(all_captures)} streams ({n_rows}×{n_cols}) at {cell_w}×{cell_h} each")
    for sc in all_captures:
        sc.start()

    show_text = True
    show_fps = args.fps_monitor

    win_name = "All Streams - High Performance"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, cell_w * n_cols, cell_h * n_rows)

    # Performance monitoring for main loop
    frame_times = deque(maxlen=30)
    last_time = time.time()

    try:
        while True:
            loop_start = time.time()
            
            # Build grid - frames are ALREADY SCALED
            rows_imgs = []
            for r in range(n_rows):
                cols_imgs = []
                for c in range(n_cols):
                    sc = grid[r][c]
                    
                    if sc:
                        # NO RESIZE - frame is already scaled!
                        cell = sc.read_scaled()
                        if cell is None:
                            cell = black_cell.copy()
                    else:
                        cell = black_cell.copy()

                    # Add overlay if enabled
                    if show_text and sc is not None and cell is not black_cell:
                        text_lines = [f"{sc.node_name}", f"port {sc.port}", f"{sc.mac}"]
                        if show_fps:
                            text_lines.append(f"FPS: {sc.current_fps:.1f}")
                        draw_overlay(cell, text_lines, scale)

                    cols_imgs.append(cell)
                
                # Stack horizontally - pre-allocated size
                rows_imgs.append(np.hstack(cols_imgs))
            
            # Stack vertically - single operation
            canvas = np.vstack(rows_imgs)
            
            cv2.imshow(win_name, canvas)

            # Track main loop performance
            loop_time = time.time() - loop_start
            frame_times.append(loop_time)
            
            # Show stats every second
            now = time.time()
            if now - last_time >= 1.0:
                avg_time = sum(frame_times) / len(frame_times)
                display_fps = 1.0 / avg_time if avg_time > 0 else 0
                print(f"Display: {display_fps:.1f} FPS (loop time: {avg_time*1000:.1f}ms)")
                last_time = now

            # Minimal wait
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('t'):
                show_text = not show_text
                print(f"Overlay text: {'ON' if show_text else 'OFF'}")
            elif key == ord('f'):
                show_fps = not show_fps
                print(f"FPS monitor: {'ON' if show_fps else 'OFF'}")
            elif key in (ord('+'), ord('=')):
                if scale > 1:
                    scale -= 1
                    cell_w = stream_w // scale
                    cell_h = stream_h // scale
                    # Update all captures with new target size
                    for sc in all_captures:
                        sc.target_width = cell_w
                        sc.target_height = cell_h
                    black_cell = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
                    cv2.resizeWindow(win_name, cell_w * n_cols, cell_h * n_rows)
                    print(f"Scale: 1/{scale} ({cell_w}×{cell_h} per stream)")
            elif key in (ord('-'), ord('_')):
                scale += 1
                cell_w = stream_w // scale
                cell_h = stream_h // scale
                for sc in all_captures:
                    sc.target_width = cell_w
                    sc.target_height = cell_h
                black_cell = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
                cv2.resizeWindow(win_name, cell_w * n_cols, cell_h * n_rows)
                print(f"Scale: 1/{scale} ({cell_w}×{cell_h} per stream)")

    except KeyboardInterrupt:
        pass
    finally:
        print("\nStopping all streams...")
        for sc in all_captures:
            sc.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()