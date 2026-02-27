#!/usr/bin/env python3
"""Disable torque on robot arm (make it limp/compliant)."""

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

ROBOT_PORT = "/dev/ttyACM0"
ROBOT_ID = "follower_hope"

robot_cfg = SO101FollowerConfig(port=ROBOT_PORT, id=ROBOT_ID, cameras={})
robot = SO101Follower(robot_cfg)

print("Connecting...")
robot.connect()
print("Disabling torque...")
robot.bus.disable_torque()
print("✓ Torque disabled - arm is now limp")
robot.disconnect()
