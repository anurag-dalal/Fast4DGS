#!/usr/bin/env python3
import sys
import os
import time
import json
import threading
import argparse
from pathlib import Path
import struct

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as TF

# ROS 2 Import Fix ---
# Since we are in a conda environment, we need to add the ROS system paths manually.
# ROS Humble uses Python 3.10.
ros_path = "/opt/ros/humble/lib/python3.10/site-packages"
if ros_path not in sys.path:
    sys.path.append(ros_path)
    # Also add the local lib path if it exists (sometimes needed for dist-packages)
    sys.path.append("/opt/ros/humble/local/lib/python3.10/dist-packages")

# Hack to add ROS dll paths for LD_LIBRARY_PATH if not present (only works if done before import)
# If this fails, we might need to re-exec the script with the correct env.
if "LD_LIBRARY_PATH" not in os.environ or "/opt/ros/humble/lib" not in os.environ["LD_LIBRARY_PATH"]:
    # We can't easily update LD_LIBRARY_PATH within a running process for C-extensions
    # So we re-execute ourselves with the new environment
    print("Restarting script with ROS LD_LIBRARY_PATH...")
    new_env = os.environ.copy()
    current_ld = new_env.get("LD_LIBRARY_PATH", "")
    new_env["LD_LIBRARY_PATH"] = f"/opt/ros/humble/lib:{current_ld}"
    # Also ensure PYTHONPATH includes ROS
    current_python = new_env.get("PYTHONPATH", "")
    new_env["PYTHONPATH"] = f"/opt/ros/humble/lib/python3.10/site-packages:/opt/ros/humble/local/lib/python3.10/dist-packages:{current_python}"
    
    os.execve(sys.executable, [sys.executable] + sys.argv, new_env)
# ------------------------

# ROS 2 Imports
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import PointCloud2, PointField
    from std_msgs.msg import Header
except ImportError as e:
    print(f"Error importing ROS modules: {e}")
    # print("Error: Could not import rclpy or ROS messages. Ensure you have sourced ROS setup files.")
    sys.exit(1)

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

def create_point_cloud2(points, colors, frame_id="map"):
    """
    Creates a sensor_msgs/PointCloud2 message from points and colors.
    points: Nx3 float32 numpy array (x, y, z)
    colors: Nx3 float32 numpy array (r, g, b) in range [0, 1]
    """
    header = Header()
    header.frame_id = frame_id
    header.stamp = rclpy.clock.Clock().now().to_msg()
    
    fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
    ]

    # Convert colors to packed float (ROS PointCloud2 standard for RGB)
    # Pack r, g, b (0-255) into a float
    r = (colors[:, 0] * 255).astype(np.uint32)
    g = (colors[:, 1] * 255).astype(np.uint32)
    b = (colors[:, 2] * 255).astype(np.uint32)
    rgb = ((r << 16) | (g << 8) | b)

    # Reinterpret cast uint32 to float32
    rgb_float = np.frombuffer(rgb.tobytes(), dtype=np.float32)

    # Combine x, y, z, rgb
    data = np.column_stack((points, rgb_float)).astype(np.float32)
    
    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width = points.shape[0]
    msg.fields = fields
    msg.is_bigendian = False
    msg.point_step = 16 # 4 float32 * 4 bytes
    msg.row_step = msg.point_step * points.shape[0]
    msg.is_dense = True # filtered invalid points
    msg.data = data.tobytes()
    
    return msg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--scale", type=int, default=8, help="Downscale frame for VGGT (unused currently)")
    parser.add_argument("--topic", type=str, default="/vggt/pointcloud", help="ROS topic to publish to")
    parser.add_argument("--frame_id", type=str, default="map", help="Frame ID for point cloud")
    args = parser.parse_args()

    # Initialize ROS
    rclpy.init()
    node = Node('vggt_pointcloud_publisher')
    
    # Use RELIABLE QoS to match RViz2 default subscriber
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
    qos_profile = QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        durability=DurabilityPolicy.VOLATILE
    )
    publisher = node.create_publisher(PointCloud2, args.topic, qos_profile)
    print(f"Publishing to topic: {args.topic} with Reliable QoS")

    # Load Config
    cfg = load_config(CONFIG_PATH)
    nodes = cfg.get("nodes", [])
    
    captures = []
    print(f"Loading streams from {CONFIG_PATH}...")
    for n in nodes:
        name = n.get("name", "")
        host = n.get("host", "")
        mac = n.get("MAC", "")
        for port in n.get("ports", []):
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
    
    model.eval()
    model = model.to(device).to(dtype)
    model.track_head = None

    try:
        while rclpy.ok():
            # Collect frames
            frames = []
            
            active_frames = 0
            for c in captures:
                f = c.read()
                if f is not None:
                    frames.append(f)
                    active_frames += 1
                else:
                    pass
            
            if active_frames < 2:
                time.sleep(0.01)
                continue
                
            # Preprocess
            start_t = time.time()
            input_tensor = preprocess_frames(frames, target_size=518).to(device, dtype)
            
            # Inference
            with torch.no_grad():
                predictions = model(input_tensor)
                extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions['pose_enc'], input_tensor.shape[-2:])
                extrinsic = extrinsic.squeeze(0)
                intrinsic = intrinsic.squeeze(0)
                depth_map = predictions['depth'].squeeze(0)
            
            # Unproject
            points = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic) 
            
            # Colorize
            colors = input_tensor.permute(0, 2, 3, 1).cpu().float().numpy()
            
            if isinstance(points, torch.Tensor):
                points = points.cpu().float().numpy()

            points_flat = points.reshape(-1, 3)
            colors_flat = colors.reshape(-1, 3)

            # Filter by radius
            R = extrinsic[:, :3, :3]
            t = extrinsic[:, :3, 3]
            centers = -torch.bmm(R.transpose(1, 2), t.unsqueeze(-1)).squeeze(-1)
            mean_center = torch.mean(centers, dim=0).cpu().numpy()
            
            radius = .8 
            dists = np.linalg.norm(points_flat - mean_center, axis=1)
            radius_mask = dists < radius
            points_flat = points_flat[radius_mask]
            colors_flat = colors_flat[radius_mask]
            
            # Filter darker points
            intensity = np.mean(colors_flat, axis=1)
            valid_color_mask = intensity > 0.2
            points_flat = points_flat[valid_color_mask]
            colors_flat = colors_flat[valid_color_mask]

            # Downsample if too many points for ROS bandwidth (required for DDS transport)
            n_points = points_flat.shape[0]
            target_n = 50000  # ~800KB per message - safe for DDS
            if n_points > target_n:
                mask = np.random.choice(n_points, target_n, replace=False)
                points_flat = points_flat[mask]
                colors_flat = colors_flat[mask]

            # Publish
            msg = create_point_cloud2(points_flat, colors_flat, frame_id=args.frame_id)
            publisher.publish(msg)
            
            # Spin node to process callbacks
            rclpy.spin_once(node, timeout_sec=0.0)
            
            dt = time.time() - start_t
            print(f"Published Cloud with {points_flat.shape[0]} points. Update: {dt:.3f}s, FPS: {1.0/dt:.1f}, Streams: {active_frames}")
            
            # Small delay to prevent overwhelming DDS transport
            time.sleep(0.01)
            
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
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
