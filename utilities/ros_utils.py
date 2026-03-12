from __future__ import annotations

import os
import sys

import numpy as np

ros_path = "/opt/ros/humble/lib/python3.10/site-packages"
if ros_path not in sys.path:
    sys.path.append(ros_path)
    sys.path.append("/opt/ros/humble/local/lib/python3.10/dist-packages")

if "LD_LIBRARY_PATH" not in os.environ or "/opt/ros/humble/lib" not in os.environ["LD_LIBRARY_PATH"]:
    print("Restarting script with ROS LD_LIBRARY_PATH...")
    new_env = os.environ.copy()
    current_ld = new_env.get("LD_LIBRARY_PATH", "")
    new_env["LD_LIBRARY_PATH"] = f"/opt/ros/humble/lib:{current_ld}"
    current_python = new_env.get("PYTHONPATH", "")
    new_env["PYTHONPATH"] = (
        f"/opt/ros/humble/lib/python3.10/site-packages:"
        f"/opt/ros/humble/local/lib/python3.10/dist-packages:{current_python}"
    )
    os.execve(sys.executable, [sys.executable] + sys.argv, new_env)

try:
    import rclpy
    from geometry_msgs.msg import Pose, PoseArray, PoseStamped
    from rclpy.node import Node
    from sensor_msgs.msg import PointCloud2, PointField
    from std_msgs.msg import Header, Int32MultiArray
    from visualization_msgs.msg import Marker
except ImportError as exc:
    print(f"Error importing ROS modules: {exc}")
    sys.exit(1)


def rotation_matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = np.trace(matrix)
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (matrix[2, 1] - matrix[1, 2]) / s
        y = (matrix[0, 2] - matrix[2, 0]) / s
        z = (matrix[1, 0] - matrix[0, 1]) / s
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        s = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        w = (matrix[2, 1] - matrix[1, 2]) / s
        x = 0.25 * s
        y = (matrix[0, 1] + matrix[1, 0]) / s
        z = (matrix[0, 2] + matrix[2, 0]) / s
    elif matrix[1, 1] > matrix[2, 2]:
        s = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        w = (matrix[0, 2] - matrix[2, 0]) / s
        x = (matrix[0, 1] + matrix[1, 0]) / s
        y = 0.25 * s
        z = (matrix[1, 2] + matrix[2, 1]) / s
    else:
        s = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        w = (matrix[1, 0] - matrix[0, 1]) / s
        x = (matrix[0, 2] + matrix[2, 0]) / s
        y = (matrix[1, 2] + matrix[2, 1]) / s
        z = 0.25 * s
    quaternion = np.asarray([x, y, z, w], dtype=np.float64)
    norm = np.linalg.norm(quaternion)
    if norm == 0.0:
        return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return quaternion / norm


class ROSUtils:
    def __init__(
        self,
        node_name: str = "pointcloud_publisher",
        topic: str = "/vggt/pointcloud",
        aruco_topic: str = "/vggt/aruco_poses",
        aruco_ids_topic: str = "/vggt/aruco_ids",
        target_topic: str = "/vggt/target_pose",
        target_bbox_topic: str = "/vggt/target_bbox",
    ):
        if not rclpy.ok():
            rclpy.init()
        self.node = Node(node_name)

        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.pointcloud_publisher = self.node.create_publisher(PointCloud2, topic, qos_profile)
        self.aruco_pose_publisher = self.node.create_publisher(PoseArray, aruco_topic, qos_profile)
        self.aruco_id_publisher = self.node.create_publisher(Int32MultiArray, aruco_ids_topic, qos_profile)
        self.target_pose_publisher = self.node.create_publisher(PoseStamped, target_topic, qos_profile)
        self.target_bbox_publisher = self.node.create_publisher(Marker, target_bbox_topic, qos_profile)

    def create_point_cloud2(self, points, colors, frame_id: str = "map"):
        header = Header()
        header.frame_id = frame_id
        header.stamp = self.node.get_clock().now().to_msg()

        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        r = (colors[:, 0] * 255).astype(np.uint32)
        g = (colors[:, 1] * 255).astype(np.uint32)
        b = (colors[:, 2] * 255).astype(np.uint32)
        rgb = (r << 16) | (g << 8) | b
        rgb_float = np.frombuffer(rgb.tobytes(), dtype=np.float32)
        data = np.column_stack((points, rgb_float)).astype(np.float32)

        msg = PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width = points.shape[0]
        msg.fields = fields
        msg.is_bigendian = False
        msg.point_step = 16
        msg.row_step = msg.point_step * points.shape[0]
        msg.is_dense = True
        msg.data = data.tobytes()

        self.pointcloud_publisher.publish(msg)
        self.spin_once()

    def publish_aruco_marker_poses(self, marker_poses: dict[int, tuple[np.ndarray, np.ndarray]], frame_id: str = "map"):
        pose_array = PoseArray()
        pose_array.header.frame_id = frame_id
        pose_array.header.stamp = self.node.get_clock().now().to_msg()

        ids_msg = Int32MultiArray()
        ids_msg.data = []

        for marker_id, (position, rotation) in sorted(marker_poses.items()):
            quaternion = rotation_matrix_to_quaternion(rotation)
            pose = Pose()
            pose.position.x = float(position[0])
            pose.position.y = float(position[1])
            pose.position.z = float(position[2])
            pose.orientation.x = float(quaternion[0])
            pose.orientation.y = float(quaternion[1])
            pose.orientation.z = float(quaternion[2])
            pose.orientation.w = float(quaternion[3])
            pose_array.poses.append(pose)
            ids_msg.data.append(int(marker_id))

        self.aruco_pose_publisher.publish(pose_array)
        self.aruco_id_publisher.publish(ids_msg)
        self.spin_once()

    def publish_target_pose(
        self,
        position: np.ndarray,
        rotation: np.ndarray | None = None,
        frame_id: str = "map",
    ):
        pose_msg = PoseStamped()
        pose_msg.header.frame_id = frame_id
        pose_msg.header.stamp = self.node.get_clock().now().to_msg()
        pose_msg.pose.position.x = float(position[0])
        pose_msg.pose.position.y = float(position[1])
        pose_msg.pose.position.z = float(position[2])

        quaternion = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        if rotation is not None:
            quaternion = rotation_matrix_to_quaternion(rotation)
        pose_msg.pose.orientation.x = float(quaternion[0])
        pose_msg.pose.orientation.y = float(quaternion[1])
        pose_msg.pose.orientation.z = float(quaternion[2])
        pose_msg.pose.orientation.w = float(quaternion[3])
        self.target_pose_publisher.publish(pose_msg)
        self.spin_once()

    def publish_target_bounding_box(
        self,
        center: np.ndarray,
        dimensions: np.ndarray,
        rotation: np.ndarray | None = None,
        frame_id: str = "map",
        marker_id: int = 0,
        rgba: tuple[float, float, float, float] = (0.1, 0.9, 0.2, 0.35),
    ):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = self.node.get_clock().now().to_msg()
        marker.ns = "target_bbox"
        marker.id = int(marker_id)
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position.x = float(center[0])
        marker.pose.position.y = float(center[1])
        marker.pose.position.z = float(center[2])

        quaternion = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        if rotation is not None:
            quaternion = rotation_matrix_to_quaternion(rotation)
        marker.pose.orientation.x = float(quaternion[0])
        marker.pose.orientation.y = float(quaternion[1])
        marker.pose.orientation.z = float(quaternion[2])
        marker.pose.orientation.w = float(quaternion[3])

        marker.scale.x = max(float(dimensions[0]), 1e-4)
        marker.scale.y = max(float(dimensions[1]), 1e-4)
        marker.scale.z = max(float(dimensions[2]), 1e-4)
        marker.color.r = float(rgba[0])
        marker.color.g = float(rgba[1])
        marker.color.b = float(rgba[2])
        marker.color.a = float(rgba[3])
        self.target_bbox_publisher.publish(marker)
        self.spin_once()

    def spin_once(self, timeout_sec: float = 0.0):
        rclpy.spin_once(self.node, timeout_sec=timeout_sec)

    def shutdown(self):
        try:
            self.node.destroy_node()
        finally:
            if rclpy.ok():
                rclpy.shutdown()