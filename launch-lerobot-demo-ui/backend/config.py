"""
Robot Configuration for eval_pipeline.py Integration

Multi-model pipeline: each stage runs a model, monitors a trigger condition,
then moves through waypoints before loading the next model.
Easily extensible — just add entries to PIPELINE_STAGES.
"""

import os

# ── Load .env file if present ──
_env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.isfile(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

# Control file path — shared between backend and eval_pipeline.py
CONTROL_FILE = "/tmp/lerobot_cmd"

# Directory where eval_pipeline.py writes JPEG frames for UI camera streaming
FRAME_DIR = "/tmp/lerobot_frames"

# Path to the lerobot repo
LEROBOT_DIR = os.path.expanduser("~/lerobot")

# ── Robot Hardware ──
ROBOT_PORT = "/dev/ttyACM0"
ROBOT_ID = "follower_hope"
CAMERAS = "front:/dev/video4,wrist:/dev/video8"
FPS = 30
DEVICE = "cuda"

# ── Hand Safety Detection ──
HAND_DETECT_ENABLED = True
HAND_DETECT_CAMERA = "front"
HAND_DETECT_INTERVAL = 0.25
HAND_DETECT_COOLDOWN = 8

# ── Waypoints (saved_positions.json) — LeRobot normalized values ──
WAYPOINTS_JSON = os.path.join(LEROBOT_DIR, "saved_positions.json")

# ── Points CSV — joint positions for 16-grid prepositions (ROS2 radians) ──
POINTS_CSV = os.path.join(LEROBOT_DIR, "point.csv")

# ── Points CSV — joint positions for 16-grid prepositions (ROS2 radians) ──
# (kept for GOTO point mover only, not used for waypoints anymore)

# ── JSON waypoint files (LeRobot normalized, from save_position.py) ──
LEMON_JSON = os.path.join(LEROBOT_DIR, "lemon.json")
TISSUE_JSON = os.path.join(LEROBOT_DIR, "tissue.json")
CUP_JSON = os.path.join(LEROBOT_DIR, "cup.json")
CLOTH_JSON = os.path.join(LEROBOT_DIR, "cloth.json")

# ════════════════════════════════════════════════════════════════════════
#  PIPELINE STAGES — Each stage: run a model, trigger → waypoints → next
#
#  To add a new object/model, just append another dict to this list.
#  Fields:
#    name              — human-readable stage name
#    model             — HuggingFace model ID
#    trigger_joint     — joint name to monitor (without ".pos" suffix)
#    trigger_op        — "lt" (less than) or "gt" (greater than)
#    trigger_value     — threshold value (LeRobot normalized)
#    waypoints         — "all" / "none" / list of names (from saved_positions.json)
#    waypoint_json     — JSON file path (from save_position.py, LeRobot normalized)
#                        If set, overrides "waypoints" field
#    waypoint_duration — seconds per waypoint movement (default for all steps)
#    waypoint_timings  — (optional) per-step [(move_time, hold_time), ...]
#                        Overrides waypoint_duration for each step
#    episode_time      — max seconds per episode
#    num_episodes      — number of episodes to run
# ════════════════════════════════════════════════════════════════════════

PIPELINE_STAGES = [
    {
        "name": "Lemon",
        "model": "FrankYuzhe/act_lemon_box_0226_merged_160_ckpt_040000",
        "trigger_joint": "shoulder_pan",
        "trigger_op": "lt",
        "trigger_value": -25.0,
        "waypoint_json": LEMON_JSON,
        "waypoint_duration": 0.8,
        "episode_time": 200,
        "num_episodes": 10,
    },
    {
        "name": "Tissue",
        "model": "FrankYuzhe/act_tissue_box_0226_merged_80_0226_221249",
        "trigger_joint": "shoulder_pan",
        "trigger_op": "lt",
        "trigger_value": -25.0,
        "waypoint_json": TISSUE_JSON,
        "waypoint_duration": 0.8,
        "episode_time": 200,
        "num_episodes": 10,
    },
    {
        "name": "Cup",
        "model": "FrankYuzhe/act_cup_box_0301_merged_80",
        "trigger_joint": "shoulder_pan",
        "trigger_op": "lt",
        "trigger_value": -25.0,
        "waypoint_json": CUP_JSON,
        "waypoint_duration": 0.8,
        "episode_time": 200,
        "num_episodes": 10,
    },
    {
        "name": "Cloth",
        "model": "FrankYuzhe/act_cloth_0301_merged_80_0301_200931",
        "trigger_joint": "shoulder_pan",
        "trigger_op": "gt",            # greater than -25 → 接管
        "trigger_value": 0.0,
        "waypoint_csv": os.path.join(LEROBOT_DIR, "cloth.csv"),
        "waypoint_duration": 0.8,
        "waypoint_timings": [
            (0.3, 0.0), (0.8, 0.0), (0.3, 0.0), (0.8, 0.0), (0.3, 0.0),
            (0.8, 0.0), (0.5, 0.0), (0.5, 0.0), (0.4, 0.5), (0.4, 0.5),
        ],
        "episode_time": 200,
        "num_episodes": 10,
    },
]

# ════════════════════════════════════════════════════════════════════════
#  LLM VISION PLANNER — OpenRouter / Gemini
# ════════════════════════════════════════════════════════════════════════

_LLM_API_KEY_RAW = os.environ.get("OPENROUTER_API_KEY", "")
LLM_PLANNER_ENABLED = bool(_LLM_API_KEY_RAW)   # auto-disable if no key
LLM_API_BASE = "https://openrouter.ai/api/v1"
LLM_API_KEY = _LLM_API_KEY_RAW
LLM_MODEL = "google/gemini-3-flash-preview"
LLM_PLANNABLE_OBJECTS = ["Lemon", "Tissue", "Cup", "Cloth"]

# ── Legacy single-model config (for backward compatibility) ──
ROBOT_CONFIG = {
    "model": PIPELINE_STAGES[0]["model"] if PIPELINE_STAGES else "",
    "robot_port": ROBOT_PORT,
    "robot_id": ROBOT_ID,
    "cameras": CAMERAS,
    "fps": FPS,
    "episode_time": PIPELINE_STAGES[0].get("episode_time", 200) if PIPELINE_STAGES else 200,
    "num_episodes": PIPELINE_STAGES[0].get("num_episodes", 10) if PIPELINE_STAGES else 10,
    "device": DEVICE,
    "rest_duration": 2.0,
}


def build_inference_command(task: str = None) -> list[str]:
    """Build the eval_pipeline.py command for multi-model pipeline."""
    cmd = [
        "python", os.path.join(LEROBOT_DIR, "eval_pipeline.py"),
        "--robot-port", ROBOT_PORT,
        "--robot-id", ROBOT_ID,
        "--cameras", CAMERAS,
        "--fps", str(FPS),
        "--device", DEVICE,
        "--control-file", CONTROL_FILE,
        "--wait-for-start",
        "--frame-dir", FRAME_DIR,
        "--pipeline-config", os.path.join(LEROBOT_DIR, "launch-lerobot-demo-ui", "backend", "config.py"),
    ]
    # Hand safety detection
    if HAND_DETECT_ENABLED:
        cmd.extend([
            "--hand-detect",
            "--hand-detect-camera", HAND_DETECT_CAMERA,
            "--hand-detect-interval", str(HAND_DETECT_INTERVAL),
            "--hand-detect-cooldown", str(HAND_DETECT_COOLDOWN),
        ])
    # GOTO points (prepositions)
    if os.path.exists(POINTS_CSV):
        cmd.extend(["--points-csv", POINTS_CSV])
    return cmd
