#!/usr/bin/env python3
"""Save current robot arm position to saved_positions.json"""

import json
import sys
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

ROBOT_PORT = "/dev/ttyACM0"
ROBOT_ID = "follower_hope"
JSON_FILE = "/home/robotlab/lerobot/saved_positions.json"

JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

def main():
    # Get optional name for position
    name = sys.argv[1] if len(sys.argv) > 1 else None
    
    # Connect to robot
    robot_cfg = SO101FollowerConfig(port=ROBOT_PORT, id=ROBOT_ID, cameras={})
    robot = SO101Follower(robot_cfg)
    robot.connect()
    
    # Read current position
    obs = robot.get_observation()
    position = {name: obs[f"{name}.pos"] for name in JOINT_NAMES}
    
    robot.disconnect()
    
    # Load existing JSON
    with open(JSON_FILE, "r") as f:
        data = json.load(f)
    
    # Add new position
    entry = {"values": position}
    if name:
        entry["name"] = name
    entry["index"] = len(data["positions"]) + 1
    
    data["positions"].append(entry)
    
    # Save
    with open(JSON_FILE, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"✓ Position {entry['index']} saved" + (f" as '{name}'" if name else ""))
    print(f"  {position}")

if __name__ == "__main__":
    main()
