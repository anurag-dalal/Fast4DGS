import os
import sys
import numpy as np
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



class ROSUtils:
    def __init__(self, node_name="pointcloud_publisher", topic="/vggt/pointcloud"):
        # Initialize ROS
        rclpy.init()
        self.node = Node(node_name)
        
        # Use RELIABLE QoS to match RViz2 default subscriber
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE
        )
        self.publisher = self.node.create_publisher(PointCloud2, topic, qos_profile)
        


    def create_point_cloud2(self, points, colors, frame_id="map"):
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
        
        self.publisher.publish(msg)