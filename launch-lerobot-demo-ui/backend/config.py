"""
Robot Configuration for eval_act_safe.py Integration

Parameters match eval_act_andy_tube.sh exactly.
"""

import os

# Control file path — shared between backend and eval_act_safe.py
CONTROL_FILE = "/tmp/lerobot_cmd"

# ── Configuration (identical to eval_act_andy_tube.sh) ──
ROBOT_CONFIG = {
    "model": "FrankYuzhe/act_merged_tissue_spoon_0203_0204_2202",
    "robot_port": "/dev/ttyACM0",
    "robot_id": "hope",
    "cameras": "front:/dev/video0,wrist:/dev/video4",
    "fps": 30,
    "episode_time": 200,
    "num_episodes": 10,
    "device": "cuda",
    "rest_duration": 2.0,
}

# Path to the lerobot repo (where eval_act_safe.py lives)
LEROBOT_DIR = os.path.expanduser("~/lerobot")


def build_inference_command(task: str = None) -> list[str]:
    """Build the eval_act_safe.py command — same params as eval_act_andy_tube.sh."""
    cfg = ROBOT_CONFIG
    cmd = [
        "python", os.path.join(LEROBOT_DIR, "eval_act_safe.py"),
        "--model", cfg["model"],
        "--robot-port", cfg["robot_port"],
        "--robot-id", cfg["robot_id"],
        "--cameras", cfg["cameras"],
        "--fps", str(cfg["fps"]),
        "--episode-time", str(cfg["episode_time"]),
        "--num-episodes", str(cfg["num_episodes"]),
        "--device", cfg["device"],
        "--rest-duration", str(cfg["rest_duration"]),
        "--control-file", CONTROL_FILE,
        "--wait-for-start",
    ]
    return cmd
