#!/usr/bin/env python3
import csv
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
import tf2_ros
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

EE_LINK = "gripper"
BASE_LINK = "base_link"
X_THRESHOLD = 0.12
MONITOR_HZ = 30.0
CSV_FILE = "point.csv"
POINT_TIMES = [0.7, 0.5, 0.3, 0.5, 1.0]


class EEMonitor(Node):
    def __init__(self):
        super().__init__("ee_monitor")
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.triggered = False
        self.action_client = ActionClient(
            self, FollowJointTrajectory, "/arm_controller/follow_joint_trajectory"
        )

    def get_x(self):
        try:
            t = self.tf_buffer.lookup_transform(BASE_LINK, EE_LINK, rclpy.time.Time())
            return t.transform.translation.x
        except Exception:
            return None

    def load_csv(self):
        with open(CSV_FILE, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = []
            for row in reader:
                if row:
                    rows.append([float(v) for v in row])
        return header, rows

    def on_threshold_exceeded(self, x):
        self.get_logger().warn(f"X = {x:.4f} exceeded {X_THRESHOLD}, executing trajectory...")

        joint_names, points = self.load_csv()
        self.get_logger().info(f"Loaded {len(points)} points from {CSV_FILE}")

        if not self.action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("arm_controller action server not available")
            return

        traj = JointTrajectory()
        traj.joint_names = joint_names

        cumulative = 0.0
        for i, positions in enumerate(points):
            pt = JointTrajectoryPoint()
            pt.positions = positions
            cumulative += POINT_TIMES[i] if i < len(POINT_TIMES) else POINT_TIMES[-1]
            secs = int(cumulative)
            nsecs = int((cumulative - secs) * 1e9)
            pt.time_from_start = Duration(sec=secs, nanosec=nsecs)
            traj.points.append(pt)
            self.get_logger().info(f"  Point {i+1} @ {cumulative:.1f}s: {[f'{p:.4f}' for p in positions]}")

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj

        future = self.action_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().error("Trajectory rejected")
            return

        self.get_logger().info("Trajectory accepted, executing...")
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        self.get_logger().info("Trajectory execution done.")

    def run(self):
        for _ in range(100):
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.get_x() is not None:
                break

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=1.0 / MONITOR_HZ)
            x = self.get_x()
            if x is None:
                continue
            if x > X_THRESHOLD and not self.triggered:
                self.triggered = True
                self.on_threshold_exceeded(x)
            elif x <= X_THRESHOLD:
                self.triggered = False


def main():
    rclpy.init()
    node = EEMonitor()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
