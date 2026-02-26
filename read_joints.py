#!/usr/bin/env python3
import csv
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

CSV_FILE = "point.csv"


class JointReader(Node):
    def __init__(self):
        super().__init__("joint_reader")
        self.received = False
        self.sub = self.create_subscription(JointState, "/joint_states", self.cb, 10)

    def cb(self, msg):
        if self.received:
            return
        self.received = True

        names = list(msg.name)
        positions = list(msg.position)

        for name, pos in zip(names, positions):
            self.get_logger().info(f"{name}: {pos:.4f}")

        import os
        file_exists = os.path.exists(CSV_FILE) and os.path.getsize(CSV_FILE) > 0
        with open(CSV_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(names)
            writer.writerow([f"{p:.4f}" for p in positions])

        self.get_logger().info(f"Saved to {CSV_FILE}")


def main():
    rclpy.init()
    node = JointReader()
    while rclpy.ok() and not node.received:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
