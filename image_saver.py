#!/usr/bin/env python3
"""
Simple Tkinter GUI to view an H264 UDP RTP stream via GStreamer + OpenCV

Features:
- Enter UDP port and click `Start Stream` to connect
- `Select Folder` to choose a base folder; the program creates a subfolder named by the port
- `Save Frames` checkbox enables saving frames as 0001.png, 0002.png... in that folder
- Buffer kept minimal by using GStreamer appsink: `max-buffers=1 drop=true`

Requirements:
- OpenCV with GStreamer support
- Pillow (for Tkinter image display)

Example pipeline used internally (appsink):
 udpsrc port=5000 caps="application/x-rtp, media=video, encoding-name=H264, payload=96" ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! appsink sync=false max-buffers=1 drop=true

Run: python3 image_saver_gui.py
"""
import threading
import time
import os
import cv2
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk


class StreamSaverApp:
    def __init__(self, root):
        self.root = root
        self.root.title("UDP H264 Stream Viewer & Saver")

        self.port_var = tk.StringVar(value="5000")
        self.save_var = tk.BooleanVar(value=True)
        self.base_folder = Path.cwd()

        self.latest_frame = None

        self._build_ui()

        self.cap = None
        self.thread = None
        self.running = False
        self.frame_img = None
        self.save_index = 1

    def _build_ui(self):
        frm = tk.Frame(self.root)
        frm.pack(padx=8, pady=8)

        tk.Label(frm, text="Port:").grid(row=0, column=0, sticky="w")
        tk.Entry(frm, textvariable=self.port_var, width=8).grid(row=0, column=1, sticky="w")
        tk.Button(frm, text="Start Stream", command=self.start_stream).grid(row=0, column=2, padx=6)
        tk.Button(frm, text="Stop Stream", command=self.stop_stream).grid(row=0, column=3)

        tk.Button(frm, text="Select Folder", command=self.select_folder).grid(row=1, column=0, pady=6)
        self.folder_label = tk.Label(frm, text=str(self.base_folder), anchor="w")
        self.folder_label.grid(row=1, column=1, columnspan=3, sticky="w")

        # Button to save only the currently displayed frame
        self.save_button = tk.Button(frm, text="Save Frame", command=self.save_current_frame, state="disabled")
        self.save_button.grid(row=2, column=0, sticky="w")

        self.video_label = tk.Label(self.root)
        self.video_label.pack(padx=6, pady=6)

    def select_folder(self):
        folder = filedialog.askdirectory(initialdir=str(self.base_folder))
        if folder:
            self.base_folder = Path(folder)
            self.folder_label.config(text=str(self.base_folder))

    def start_stream(self):
        if self.running:
            return
        try:
            port = int(self.port_var.get())
        except ValueError:
            messagebox.showerror("Invalid port", "Port must be an integer")
            return

        self.port = port
        self.stream_folder = self.base_folder / f"port_{self.port}"
        self.stream_folder.mkdir(parents=True, exist_ok=True)

        # build GStreamer pipeline for appsink with minimal buffering
        gst_pipeline = (
            f'udpsrc port={self.port} caps="application/x-rtp, media=video, encoding-name=H264, payload=96" '
            "! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! "
            "appsink sync=false max-buffers=1 drop=true"
        )

        self.cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            messagebox.showerror("Stream error", "Unable to open stream. Ensure GStreamer is installed and the pipeline is correct.")
            self.cap.release()
            self.cap = None
            return

        self.running = True
        self.save_index = self._initial_save_index()
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        # enable save button
        try:
            self.save_button.config(state="normal")
        except Exception:
            pass

    def _initial_save_index(self):
        existing = sorted(self.stream_folder.glob("*.png"))
        if not existing:
            return 1
        # find highest existing index
        max_idx = 0
        for p in existing:
            try:
                name = p.stem
                idx = int(name)
                if idx > max_idx:
                    max_idx = idx
            except Exception:
                continue
        return max_idx + 1

    def _capture_loop(self):
        last_t = 0
        while self.running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            # display every frame (convert BGR->RGB)
            # keep a copy of the latest frame for manual saving
            try:
                self.latest_frame = frame.copy()
            except Exception:
                self.latest_frame = frame

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)

            # update UI (must be done in main thread via .after)
            self.root.after(0, self._update_image, imgtk)
            # small sleep to yield — we rely on appsink drop to keep buffer minimal
            time.sleep(0.01)

        # release capture when loop ends
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    def _update_image(self, imgtk):
        # keep reference
        self.frame_img = imgtk
        self.video_label.config(image=self.frame_img)

    def stop_stream(self):
        if not self.running:
            return
        self.running = False
        # wait a bit for thread to finish
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        try:
            self.save_button.config(state="disabled")
        except Exception:
            pass

    def on_close(self):
        self.stop_stream()
        self.root.destroy()

    def save_current_frame(self):
        """Save the most recent frame to the stream folder when the button is clicked."""
        if not hasattr(self, "stream_folder") or self.stream_folder is None:
            messagebox.showwarning("No folder", "Start the stream and select a folder first")
            return

        if self.latest_frame is None:
            messagebox.showwarning("No frame", "No frame available to save yet")
            return

        filename = self.stream_folder / f"{self.save_index:04d}.png"
        try:
            cv2.imwrite(str(filename), self.latest_frame)
            self.save_index += 1
        except Exception as e:
            messagebox.showerror("Save error", f"Failed to save frame: {e}")


def main():
    root = tk.Tk()
    app = StreamSaverApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
