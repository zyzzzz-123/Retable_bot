#!/usr/bin/env python3
"""Real-time Joint Position Display Script.

Connects to the SO101 robot and displays current joint positions in real-time.
Press Ctrl+C to stop.

Usage:
    python show_joints.py
"""

import time
import sys

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

# ════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ════════════════════════════════════════════════════════════════════════

ROBOT_PORT = "/dev/ttyACM0"
ROBOT_ID = "follower_hope"
REFRESH_RATE = 10  # Hz (updates per second)

# ════════════════════════════════════════════════════════════════════════


def main():
    print("\n" + "=" * 60)
    print("  🤖 Real-time Joint Position Display")
    print("=" * 60)
    print(f"  Robot: {ROBOT_ID} @ {ROBOT_PORT}")
    print(f"  Refresh: {REFRESH_RATE} Hz")
    print("-" * 60)
    print("  Press Ctrl+C to stop")
    print("=" * 60 + "\n")

    # Connect to robot (no cameras needed)
    robot_cfg = SO101FollowerConfig(port=ROBOT_PORT, id=ROBOT_ID, cameras={})
    robot = SO101Follower(robot_cfg)

    print("Connecting to robot...")
    robot.connect()
    print("Robot connected ✓\n")

    # Joint names for display
    joint_names = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    ]

    try:
        while True:
            start_t = time.perf_counter()

            # Read current positions
            obs = robot.get_observation()

            # Clear screen and move cursor to top
            sys.stdout.write("\033[2J\033[H")
            sys.stdout.flush()

            # Print header
            print("=" * 60)
            print("  🤖 Joint Positions (LeRobot normalized)")
            print("=" * 60)
            print(f"  Robot: {ROBOT_ID} @ {ROBOT_PORT}")
            print("-" * 60)
            print(f"  {'Joint':<20} {'Value':>12}")
            print("-" * 60)

            # Print each joint position
            for name in joint_names:
                key = f"{name}.pos"
                if key in obs:
                    value = obs[key]
                    print(f"  {name:<20} {value:>12.2f}")
                else:
                    print(f"  {name:<20} {'N/A':>12}")

            print("-" * 60)
            print("  Press Ctrl+C to stop")
            print("=" * 60)

            # Maintain refresh rate
            dt = time.perf_counter() - start_t
            sleep_time = max(1.0 / REFRESH_RATE - dt, 0.0)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n\nStopping...")

    finally:
        robot.disconnect()
        print("Robot disconnected.")


if __name__ == "__main__":
    main()
