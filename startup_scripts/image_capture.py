# gst-launch-1.0 v4l2src device=/dev/video0 num-buffers=1 ! videoconvert ! pngenc ! filesink location=capture.png

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import subprocess
from PIL import Image, ImageTk
import glob
import cv2

class CameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Multi-Camera Capture Tool")
        self.root.geometry("900x700")

        # Variables
        self.base_dir = tk.StringVar(value=os.path.expanduser("~"))
        self.start_index = tk.IntVar(value=0)
        self.loaded_images_path = tk.StringVar()
        
        self.current_stream_idx = None # Which video device is streaming
        self.cap = None # OpenCV VideoCapture object
        self.is_streaming = False

        self.setup_ui()

    def setup_ui(self):
        # Tabs
        tab_control = ttk.Notebook(self.root)
        
        self.tab_capture = ttk.Frame(tab_control)
        self.tab_viewer = ttk.Frame(tab_control)
        
        tab_control.add(self.tab_capture, text='Capture')
        tab_control.add(self.tab_viewer, text='Gallery Viewer')
        tab_control.pack(expand=1, fill="both")

        # --- Tab 1: Capture ---
        self.setup_capture_tab()

        # --- Tab 2: Viewer ---
        self.setup_viewer_tab()

    def setup_capture_tab(self):
        # Configuration Frame
        control_frame = ttk.LabelFrame(self.tab_capture, text="Configuration", padding="10")
        control_frame.pack(fill="x", padx=10, pady=5)

        # Base Directory Selection
        ttk.Label(control_frame, text="Save Location:").grid(row=0, column=0, sticky="w")
        ttk.Entry(control_frame, textvariable=self.base_dir, width=40).grid(row=0, column=1, padx=5)
        ttk.Button(control_frame, text="Browse", command=self.browse_folder).grid(row=0, column=2)

        # Starting Index
        ttk.Label(control_frame, text="Start Cam Index (e.g., cam0):").grid(row=0, column=3, padx=10)
        ttk.Spinbox(control_frame, from_=0, to=100, textvariable=self.start_index, width=5).grid(row=0, column=4)

        # Video Source Selection (Radio Buttons for single selection)
        src_frame = ttk.LabelFrame(self.tab_capture, text="Select Active Camera", padding="10")
        src_frame.pack(fill="x", padx=10, pady=5)
        
        self.selected_video_device = tk.IntVar(value=-1)
        
        for i in range(6):
            rb = ttk.Radiobutton(src_frame, text=f"/dev/video{i}", variable=self.selected_video_device, value=i, command=self.on_camera_select)
            rb.pack(side="left", padx=10)

        # Live View & Actions
        live_frame = ttk.Frame(self.tab_capture)
        live_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # Camera Display
        self.video_label = ttk.Label(live_frame, text="No Camera Selected", background="black", foreground="white")
        self.video_label.pack(fill="both", expand=True, pady=5)

        # Control Buttons
        btn_frame = ttk.Frame(live_frame)
        btn_frame.pack(fill="x", pady=5)

        self.btn_stop = ttk.Button(btn_frame, text="Stop Stream", command=self.stop_stream, state="disabled")
        self.btn_stop.pack(side="left", padx=5)
        
        self.btn_capture = ttk.Button(btn_frame, text="Capture Frame", command=self.capture_single_frame, state="disabled")
        self.btn_capture.pack(side="left", fill="x", expand=True, padx=5)

    def setup_viewer_tab(self):
        viewer_frame = ttk.LabelFrame(self.tab_viewer, text="Image Viewer", padding="10")
        viewer_frame.pack(fill="both", expand=True, padx=10, pady=5)

        v_ctrl_frame = ttk.Frame(viewer_frame)
        v_ctrl_frame.pack(fill="x", pady=5)
        
        ttk.Label(v_ctrl_frame, text="Load Path:").pack(side="left")
        ttk.Entry(v_ctrl_frame, textvariable=self.loaded_images_path, width=40).pack(side="left", padx=5)
        ttk.Button(v_ctrl_frame, text="Browse & Load", command=self.load_images_view).pack(side="left")

        # Scrollable Canvas for Grid of Images
        self.canvas = tk.Canvas(viewer_frame)
        scrollbar = ttk.Scrollbar(viewer_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def on_camera_select(self):
        new_dev_idx = self.selected_video_device.get()
        
        if self.is_streaming:
            if self.current_stream_idx == new_dev_idx:
                return # Already streaming this one
            else:
                # User tried to switch while streaming
                response = messagebox.askyesno("Switch Camera", "Stop current stream and switch?")
                if response:
                    self.stop_stream()
                    self.start_stream(new_dev_idx)
                else:
                    # Revert radio button selection
                    self.selected_video_device.set(self.current_stream_idx)
        else:
            self.start_stream(new_dev_idx)

    def start_stream(self, dev_idx):
        if self.is_streaming:
            self.stop_stream()
            
        try:
            # OpenCV capture
            self.cap = cv2.VideoCapture(dev_idx)
            if not self.cap.isOpened():
                messagebox.showerror("Error", f"Could not open /dev/video{dev_idx}")
                return

            self.current_stream_idx = dev_idx
            self.is_streaming = True
            self.btn_stop.config(state="normal")
            self.btn_capture.config(state="normal")
            self.update_video_feed()
            
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def stop_stream(self):
        self.is_streaming = False
        if self.cap:
            self.cap.release()
        self.cap = None
        self.video_label.config(image='', text="Stream Stopped")
        self.btn_stop.config(state="disabled")
        self.btn_capture.config(state="disabled")
        self.current_stream_idx = None
        self.selected_video_device.set(-1) # Deselect radio buttons

    def update_video_feed(self):
        if self.is_streaming and self.cap:
            ret, frame = self.cap.read()
            if ret:
                # Resize for display
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                img.thumbnail((640, 480))
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk # Keep reference
                self.video_label.configure(image=imgtk)
            
            self.root.after(30, self.update_video_feed) # ~30 fps

    def browse_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.base_dir.set(folder)

    def capture_single_frame(self):
        if not self.is_streaming or not self.cap:
            return

        base_path = self.base_dir.get()
        if not os.path.exists(base_path):
            messagebox.showerror("Error", "Base directory does not exist.")
            return

        # Determine target folder: cam[startIndex + activeDevIndex]
        # BUT user requirement: "like cam0 to cam6(this number cam0 cam1, the starting number can be selected by user)"
        # Assuming mapping: internal /dev/videoX -> cam(start_index + X)
        # Or does the user want strictly sequential assignment based on selection?
        # User prompt: "create folder ... like cam0 to cam6 ... starting number can be selected"
        # Since we are selecting ONE camera, we need to know WHICH cam folder it corresponds to.
        # Logic: If I select /dev/video2, and start_index is 0, is it cam2? Or cam0?
        # Typically in multi-cam setups, video0 -> cam0. So videoX -> cam(start + X) is logical.
        
        target_cam_idx = self.start_index.get() + self.current_stream_idx
        folder_name = f"cam{target_cam_idx}"
        full_path = os.path.join(base_path, folder_name)
        
        os.makedirs(full_path, exist_ok=True)
        
        # Determine filename
        existing_files = glob.glob(os.path.join(full_path, "*.png"))
        indices = []
        for f in existing_files:
            try:
                name = os.path.basename(f)
                idx = int(os.path.splitext(name)[0])
                indices.append(idx)
            except ValueError:
                pass
        next_idx = max(indices) + 1 if indices else 0
        output_filename = f"{next_idx:04d}.png"
        output_file = os.path.join(full_path, output_filename)
        
        # Capture current frame from OpenCV
        ret, frame = self.cap.read()
        if ret:
            cv2.imwrite(output_file, frame)
            print(f"Saved {output_file}")
            # Flash success on GUI potentially, or just console
        else:
            messagebox.showerror("Error", "Failed to grab frame")

    # Removed old capture_frames logic as we are now single-stream interactive

    def load_images_view(self):
        path = self.loaded_images_path.get()
        if not path:
            path = filedialog.askdirectory()
            if path:
                self.loaded_images_path.set(path)
        
        if not path or not os.path.exists(path):
            return

        # Clear existing
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        # Find folders like cam*
        cam_folders = sorted(glob.glob(os.path.join(path, "cam*")))
        
        if not cam_folders:
            ttk.Label(self.scrollable_frame, text="No 'cam*' folders found.").pack()
            return

        row = 0
        col = 0
        max_cols = 4 # Adjust grid width
        
        self.image_refs = [] # Keep references to avoid garbage collection

        for folder in cam_folders:
            folder_name = os.path.basename(folder)
            
            # Find png or jpg images in the folder
            images = glob.glob(os.path.join(folder, "*.png")) + glob.glob(os.path.join(folder, "*.jpg"))
            
            # Sort by modification time (newest first)
            images.sort(key=os.path.getmtime, reverse=True)
            
            if images:
                img_path = images[0] # Take newest image
                try:
                    pil_img = Image.open(img_path)
                    pil_img.thumbnail((200, 150)) # Resize for thumbnail
                    tk_img = ImageTk.PhotoImage(pil_img)
                    self.image_refs.append(tk_img)

                    # Create Container
                    frame = ttk.Frame(self.scrollable_frame, borderwidth=1, relief="solid")
                    frame.grid(row=row, column=col, padx=5, pady=5)
                    
                    ttk.Label(frame, text=folder_name).pack()
                    ttk.Label(frame, image=tk_img).pack()
                    
                    col += 1
                    if col >= max_cols:
                        col = 0
                        row += 1
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = CameraApp(root)
    root.mainloop()