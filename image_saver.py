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
import json
import cv2
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk


class StreamSaverApp:
    def __init__(self, root):
        self.root = root
        self.root.title("UDP H264 Stream Viewer & Saver")

        # load config nodes (if available)
        self.config_path = Path(__file__).resolve().parent / "configs" / "stream_config.json"
        self.nodes = self._load_nodes_from_config()

        # UI variables
        default_port = "5000"
        if self.nodes:
            default_port = str(self.nodes[0].get("ports", [5000])[0])

        self.port_var = tk.StringVar(value=default_port)
        self.node_var = tk.StringVar(value=self.nodes[0]["name"] if self.nodes else "")
        self.save_var = tk.BooleanVar(value=True)
        # default save location per user request
        self.base_folder = Path("/home/anurag/Codes/Fast4DGS/dataset")

        self.latest_frame = None

        self._build_ui()

        self.cap = None
        self.thread = None
        self.running = False
        self.frame_img = None
        self.save_index = 1

    def _build_ui(self):
        self.root.minsize(1920, 1200)

        # ── controls frame ──
        frm = tk.Frame(self.root, padx=12, pady=8)
        frm.pack(fill="x")
        frm.columnconfigure(1, weight=1)
        frm.columnconfigure(3, weight=1)

        lbl_font = ("Helvetica", 10)
        val_font = ("Helvetica", 10, "bold")

        # Row 0 – Node selector + Port selector
        tk.Label(frm, text="Node:", font=lbl_font).grid(row=0, column=0, sticky="w", padx=(0, 4))
        if self.nodes:
            node_names = [n.get("name", "") for n in self.nodes]
            self.node_menu = tk.OptionMenu(frm, self.node_var, *node_names, command=self._on_node_change)
            self.node_menu.config(width=14, font=lbl_font)
            self.node_menu.grid(row=0, column=1, sticky="w", padx=(0, 12))
        else:
            tk.Label(frm, text="(no config)", font=lbl_font).grid(row=0, column=1, sticky="w")

        tk.Label(frm, text="Port:", font=lbl_font).grid(row=0, column=2, sticky="w", padx=(0, 4))
        if self.nodes:
            self.ports_menu = tk.OptionMenu(frm, self.port_var, "")
            self.ports_menu.config(width=8, font=lbl_font)
            self.ports_menu.grid(row=0, column=3, sticky="w")
        else:
            tk.Entry(frm, textvariable=self.port_var, width=10, font=lbl_font).grid(row=0, column=3, sticky="w")

        # Row 1 – Host (readonly)
        tk.Label(frm, text="Host:", font=lbl_font).grid(row=1, column=0, sticky="w", pady=(4, 0))
        self.host_label = tk.Label(frm, text="—", anchor="w", font=val_font)
        self.host_label.grid(row=1, column=1, sticky="w", pady=(4, 0))

        # MAC (readonly)
        tk.Label(frm, text="MAC:", font=lbl_font).grid(row=1, column=2, sticky="w", pady=(4, 0))
        self.mac_label = tk.Label(frm, text="—", anchor="w", font=val_font)
        self.mac_label.grid(row=1, column=3, sticky="w", pady=(4, 0))

        # ── button bar ──
        btn_frm = tk.Frame(self.root, padx=12, pady=6)
        btn_frm.pack(fill="x")

        tk.Button(btn_frm, text="▶  Start Stream", width=16, command=self.start_stream, font=lbl_font,
                  bg="#4CAF50", fg="white", activebackground="#45a049").pack(side="left", padx=(0, 6))
        tk.Button(btn_frm, text="■  Stop Stream", width=16, command=self.stop_stream, font=lbl_font,
                  bg="#f44336", fg="white", activebackground="#d32f2f").pack(side="left", padx=(0, 6))
        self.save_button = tk.Button(btn_frm, text="💾  Save Frame", width=16,
                                     command=self.save_current_frame, state="disabled", font=lbl_font)
        self.save_button.pack(side="left", padx=(0, 6))
        tk.Button(btn_frm, text="📁  Select Folder", width=16, command=self.select_folder,
                  font=lbl_font).pack(side="left")

        # ── folder path display ──
        path_frm = tk.Frame(self.root, padx=12)
        path_frm.pack(fill="x")
        tk.Label(path_frm, text="Save to:", font=lbl_font).pack(side="left")
        self.folder_label = tk.Label(path_frm, text=str(self.base_folder), anchor="w",
                                     font=("Courier", 9), fg="#555")
        self.folder_label.pack(side="left", padx=4)

        # ── video display ──
        self.video_label = tk.Label(self.root, bg="#222")
        self.video_label.pack(padx=10, pady=10, fill="both", expand=True)

        # populate initial node info (ports, host, mac)
        if self.nodes:
            self._on_node_change()

    def select_folder(self):
        folder = filedialog.askdirectory(initialdir=str(self.base_folder))
        if folder:
            self.base_folder = Path(folder)
            self.folder_label.config(text=str(self.base_folder))

    def _load_nodes_from_config(self):
        try:
            if not self.config_path.exists():
                return []
            with open(self.config_path, "r") as f:
                cfg = json.load(f)
            nodes = cfg.get("nodes", [])
            # normalize entries: ensure keys name, host, ports, MAC exist
            normalized = []
            for n in nodes:
                normalized.append({
                    "name": n.get("name", ""),
                    "host": n.get("host", ""),
                    "ports": n.get("ports", []),
                    "MAC": n.get("MAC", n.get("mac", "")),
                })
            return normalized
        except Exception:
            return []

    def _on_node_change(self, _=None):
        # update ports menu and host/mac labels based on selected node
        name = self.node_var.get()
        node = next((n for n in self.nodes if n.get("name") == name), None)
        if not node:
            return
        # update host/mac
        self.host_label.config(text=node.get("host", ""))
        self.mac_label.config(text=node.get("MAC", ""))
        # update ports dropdown
        ports = [str(p) for p in node.get("ports", [])]
        if ports:
            menu = self.ports_menu["menu"]
            menu.delete(0, "end")
            for p in ports:
                menu.add_command(label=p, command=lambda v=p: self.port_var.set(v))
            # set default port
            self.port_var.set(ports[0])

    def start_stream(self):
        # if a stream is already running, stop it first so we can switch
        if self.running:
            self.stop_stream()
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
            f'udpsrc port={self.port} buffer-size=200000000 '
            f'caps="application/x-rtp, media=video, clock-rate=90000, encoding-name=H265, payload=96" ! '
            f'rtpjitterbuffer latency=15 drop-on-latency=true ! '
            f'rtph265depay ! h265parse config-interval=1 ! '
            f'nvh265dec ! '          
            f'videoconvert ! video/x-raw,format=BGR ! '
            f'appsink sync=false max-buffers=1 drop=true'
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
            print(frame.shape)
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
