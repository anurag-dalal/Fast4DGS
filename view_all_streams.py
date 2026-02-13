#!/usr/bin/env python3
"""
View all H264 UDP RTP streams in a single window.

Reads nodes/ports from configs/stream_config.json and opens a GStreamer
capture for every stream.  Frames are grabbed in parallel threads and
composited into a 6-column × 4-row (ports × nodes) grid with OpenCV.

Controls:
    t  – toggle overlay text (node name / port / MAC)
    +  – decrease downscale factor (zoom in)
    -  – increase downscale factor (zoom out)
    q  – quit

Usage:
    python3 view_all_streams.py              # default downscale=4
    python3 view_all_streams.py --scale 2    # less downscale (bigger)
    python3 view_all_streams.py --scale 8    # more downscale (smaller)
"""

import argparse
import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np


# ── config ──────────────────────────────────────────────────────────────────

CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "stream_config.json"


def load_config(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


# ── per-stream capture thread ──────────────────────────────────────────────

class StreamCapture:
    """Grabs frames from one GStreamer UDP/RTP H264 stream in a background thread."""

    def __init__(self, port: int, node_name: str, host: str, mac: str):
        self.port = port
        self.node_name = node_name
        self.host = host
        self.mac = mac

        self.frame = None          # latest BGR frame (or None)
        self._lock = threading.Lock()
        self._running = False
        self._cap = None
        self._thread = None

    def start(self, timeout: float = 15.0, retry_interval: float = 3.0):
        """Try to open the GStreamer capture with retries up to *timeout* seconds.

        If it still can't open after the deadline, mark the stream as failed
        (frame stays None → shown as black cell).
        """
        self._running = True
        self._thread = threading.Thread(target=self._open_and_loop,
                                        args=(timeout, retry_interval),
                                        daemon=True)
        self._thread.start()

    def _open_and_loop(self, timeout: float, retry_interval: float):
        gst = (
            f'udpsrc port={self.port} buffer-size=100000000 ' 
            f'caps="application/x-rtp, media=video, clock-rate=90000, encoding-name=H264, payload=96" ! '
            f'rtpjitterbuffer latency=50 drop-on-latency=true ! '
            f'rtph264depay ! h264parse config-interval=1 ! '
            f'nvh264dec disable-dpb=true ! '
            # f'cudaconvert ! '
            # f'cudadownload ! '
            f'videoconvert ! video/x-raw,format=BGR ! ' 
            f'appsink sync=false max-buffers=1 drop=true'
        )
        deadline = time.monotonic() + timeout
        attempt = 0
        while self._running and time.monotonic() < deadline:
            attempt += 1
            cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
            if cap.isOpened():
                self._cap = cap
                print(f"[OK]   port {self.port} ({self.node_name}) opened on attempt {attempt}")
                self._loop()          # blocks until self._running is False
                return
            cap.release()
            remaining = deadline - time.monotonic()
            if remaining > 0:
                print(f"[RETRY] port {self.port} ({self.node_name}) attempt {attempt} failed, "
                      f"{remaining:.1f}s left")
                time.sleep(min(retry_interval, remaining))

        print(f"[FAIL] port {self.port} ({self.node_name}) – gave up after {timeout:.0f}s")

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
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None


# ── text overlay ────────────────────────────────────────────────────────────

def draw_overlay(img, text_lines, scale_factor):
    """Draw semi-transparent text lines at top-left of *img* (mutates in place)."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.35, 0.6 / (scale_factor ** 0.3))
    thickness = 1
    line_h = int(20 / (scale_factor ** 0.3))
    pad = 4

    # compute background rectangle
    max_w = 0
    for line in text_lines:
        (w, _), _ = cv2.getTextSize(line, font, font_scale, thickness)
        max_w = max(max_w, w)
    bg_h = line_h * len(text_lines) + pad * 2
    bg_w = max_w + pad * 2

    # draw translucent black box
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (bg_w, bg_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

    # draw text
    y = pad + line_h - 4
    for line in text_lines:
        cv2.putText(img, line, (pad, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        y += line_h


# ── main loop ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="View all UDP H264 streams in a grid")
    parser.add_argument("--scale", type=int, default=4,
                        help="Downscale factor for each stream (default: 4)")
    args = parser.parse_args()

    cfg = load_config(CONFIG_PATH)
    nodes = cfg.get("nodes", [])
    stream_w = cfg.get("stream_width", 1920)
    stream_h = cfg.get("stream_height", 1080)

    if not nodes:
        print("No nodes found in config – exiting.")
        return

    # determine grid size: rows = nodes, cols = max ports across all nodes
    n_rows = len(nodes)
    n_cols = max(len(n.get("ports", [])) for n in nodes)

    # create & start all captures
    # grid[row][col] = StreamCapture | None
    grid: list[list[StreamCapture | None]] = []
    all_captures: list[StreamCapture] = []

    for node in nodes:
        row = []
        name = node.get("name", "")
        host = node.get("host", "")
        mac = node.get("MAC", node.get("mac", ""))
        for port in node.get("ports", []):
            sc = StreamCapture(port, name, host, mac)
            row.append(sc)
            all_captures.append(sc)
        # pad row to n_cols
        while len(row) < n_cols:
            row.append(None)
        grid.append(row)

    print(f"Starting {len(all_captures)} streams  ({n_rows} nodes × {n_cols} ports)  scale=1/{args.scale}")
    for sc in all_captures:
        sc.start()

    scale = args.scale
    show_text = True

    cell_w = stream_w // scale
    cell_h = stream_h // scale
    black_cell = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)

    win_name = "All Streams"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, cell_w * n_cols, cell_h * n_rows)

    try:
        while True:
            rows_imgs = []
            for r in range(n_rows):
                cols_imgs = []
                for c in range(n_cols):
                    sc = grid[r][c]
                    frame = sc.read() if sc else None
                    if frame is not None:
                        cell = cv2.resize(frame, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
                    else:
                        cell = black_cell.copy()
                        if sc is not None:
                            # show "waiting" text on black cell
                            cv2.putText(cell, "waiting...",
                                        (cell_w // 4, cell_h // 2),
                                        cv2.FONT_HERSHEY_SIMPLEX, max(0.4, 0.7 / (scale ** 0.3)),
                                        (80, 80, 80), 1, cv2.LINE_AA)

                    if show_text and sc is not None:
                        draw_overlay(cell, [
                            f"{sc.node_name}",
                            f"port {sc.port}",
                            f"{sc.mac}",
                        ], scale)

                    cols_imgs.append(cell)
                rows_imgs.append(np.hstack(cols_imgs))
            canvas = np.vstack(rows_imgs)

            cv2.imshow(win_name, canvas)

            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('t'):
                show_text = not show_text
                print(f"Overlay text: {'ON' if show_text else 'OFF'}")
            elif key in (ord('+'), ord('=')):
                if scale > 1:
                    scale -= 1
                    cell_w = stream_w // scale
                    cell_h = stream_h // scale
                    black_cell = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
                    cv2.resizeWindow(win_name, cell_w * n_cols, cell_h * n_rows)
                    print(f"Scale: 1/{scale}")
            elif key in (ord('-'), ord('_')):
                scale += 1
                cell_w = stream_w // scale
                cell_h = stream_h // scale
                black_cell = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
                cv2.resizeWindow(win_name, cell_w * n_cols, cell_h * n_rows)
                print(f"Scale: 1/{scale}")

    except KeyboardInterrupt:
        pass
    finally:
        print("Stopping all streams...")
        for sc in all_captures:
            sc.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
