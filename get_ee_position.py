#!/usr/bin/env python3
"""
读取 SO-ARM 末端执行器 (EE) 的当前位置 (x, y, z)

用法:
    python3 get_ee_position.py           # 读取一次
    python3 get_ee_position.py --loop    # 持续读取
    python3 get_ee_position.py --hz 10   # 指定刷新频率 (默认 5Hz)
"""
import sys
import argparse
import rclpy
from rclpy.node import Node
import tf2_ros

EE_LINK = "gripper"
BASE_LINK = "base_link"


class EEPositionReader(Node):
    def __init__(self, loop=False, hz=5.0):
        super().__init__("ee_position_reader")
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.loop = loop
        self.hz = hz

    def get_position(self):
        """查询 EE 位置，返回 (x, y, z) 或 None"""
        try:
            t = self.tf_buffer.lookup_transform(BASE_LINK, EE_LINK, rclpy.time.Time())
            p = t.transform.translation
            return (p.x, p.y, p.z)
        except Exception:
            return None

    def run(self):
        # 等待 TF 数据就绪
        self.get_logger().info(f"Waiting for TF: {BASE_LINK} → {EE_LINK} ...")
        pos = None
        for _ in range(100):
            rclpy.spin_once(self, timeout_sec=0.05)
            pos = self.get_position()
            if pos is not None:
                break

        if pos is None:
            self.get_logger().error("TF lookup failed. Is ros2_control running?")
            return

        if not self.loop:
            # 单次读取
            print(f"{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}")
        else:
            # 持续读取
            self.get_logger().info(f"Streaming EE position at {self.hz} Hz (Ctrl+C to stop)")
            print(f"{'x':>10s} {'y':>10s} {'z':>10s}")
            print("-" * 32)
            try:
                while rclpy.ok():
                    rclpy.spin_once(self, timeout_sec=1.0 / self.hz)
                    pos = self.get_position()
                    if pos is not None:
                        print(f"{pos[0]:10.4f} {pos[1]:10.4f} {pos[2]:10.4f}", end="\r")
            except KeyboardInterrupt:
                print()


def main():
    parser = argparse.ArgumentParser(description="Read SO-ARM EE position")
    parser.add_argument("--loop", action="store_true", help="Continuously read position")
    parser.add_argument("--hz", type=float, default=5.0, help="Update rate in Hz (default: 5)")
    args = parser.parse_args()

    rclpy.init()
    node = EEPositionReader(loop=args.loop, hz=args.hz)
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
