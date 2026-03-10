#!/usr/bin/env python3
"""Multi-model pipeline evaluation script.

Orchestrates multiple ACT models in sequence. Each stage:
  1. Loads a model
  2. Runs inference
  3. Monitors a trigger condition (e.g. shoulder_pan < -25)
  4. When triggered → stops inference → moves through waypoints
  5. Loads the next model and repeats

The pipeline is configured via PIPELINE_STAGES in config.py.
Supports all the same safety features as eval_act_safe.py:
  - Emergency stop, go-to-home, hand detection
  - Control file IPC for UI backend
  - Camera frame streaming

Usage:
    python eval_pipeline.py \\
        --robot-port /dev/ttyACM0 \\
        --robot-id follower_hope \\
        --cameras "front:/dev/video4,wrist:/dev/video6" \\
        --fps 30 --device cuda \\
        --control-file /tmp/lerobot_cmd \\
        --wait-for-start \\
        --frame-dir /tmp/lerobot_frames \\
        --pipeline-config launch-lerobot-demo-ui/backend/config.py

Keyboard controls:
    Spacebar → Emergency Stop
    Enter    → Resume inference
    r        → Go to home/rest position
    →(right) → Skip current episode
    Esc      → Quit
"""

import argparse
import csv
import importlib.util
import json
import logging
import os
import sys
import threading
import time

import cv2
import numpy as np
import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import build_inference_frame, make_robot_action
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import go_to_rest_position, init_keyboard_listener
from lerobot.utils.robot_utils import precise_sleep

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
#  Hand Detection (reused from eval_act_safe.py)
# ═══════════════════════════════════════════════════════════════════════

class HandDetector:
    """Lightweight hand detector running in a background thread."""

    MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "hand_landmarker.task")

    def __init__(self, frame_path, check_interval=0.25, cooldown_frames=8, min_confidence=0.5):
        self.frame_path = frame_path
        self.check_interval = check_interval
        self.cooldown_frames = cooldown_frames
        self.min_confidence = min_confidence
        self.hand_detected = False
        self.enabled = True
        self._no_hand_count = 0
        self._stop_event = threading.Event()
        self._thread = None
        self._detector = None

    def start(self):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        if not os.path.exists(self.MODEL_PATH):
            logger.error(f"Hand detection model not found: {self.MODEL_PATH}")
            return

        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.MODEL_PATH),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=self.min_confidence,
        )
        self._detector = vision.HandLandmarker.create_from_options(options)
        logger.info("HandDetector: model loaded, starting background thread")
        self._thread = threading.Thread(target=self._run, daemon=True, name="hand-detector")
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        if self._detector:
            self._detector.close()
            self._detector = None

    def _run(self):
        import mediapipe as mp
        while not self._stop_event.is_set():
            if self.enabled and self._detector:
                try:
                    self._check_frame(mp)
                except Exception:
                    pass
            self._stop_event.wait(self.check_interval)

    def _check_frame(self, mp):
        if not os.path.exists(self.frame_path):
            return
        frame_bgr = cv2.imread(self.frame_path)
        if frame_bgr is None:
            return
        h, w = frame_bgr.shape[:2]
        if w > 320:
            scale = 320.0 / w
            frame_bgr = cv2.resize(frame_bgr, (320, int(h * scale)), interpolation=cv2.INTER_AREA)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._detector.detect(mp_image)
        if result.hand_landmarks:
            if not self.hand_detected:
                logger.info("🖐️  Hand detected in front camera!")
            self.hand_detected = True
            self._no_hand_count = 0
        else:
            self._no_hand_count += 1
            if self._no_hand_count >= self.cooldown_frames:
                if self.hand_detected:
                    logger.info("✅  Hand cleared from front camera.")
                self.hand_detected = False


# ═══════════════════════════════════════════════════════════════════════
#  Utility functions
# ═══════════════════════════════════════════════════════════════════════

def parse_cameras(cameras_str, width=640, height=480, fps=30):
    cameras = {}
    for item in cameras_str.split(","):
        name, path = item.strip().split(":", 1)
        cameras[name.strip()] = OpenCVCameraConfig(
            index_or_path=path.strip(), width=width, height=height, fps=fps
        )
    return cameras


def save_camera_frames(frame_dir, camera_names, obs=None, robot=None):
    if not frame_dir:
        return
    for name in camera_names:
        frame = None
        if obs is not None and name in obs:
            frame = obs[name]
        elif robot is not None and hasattr(robot, "cameras") and name in robot.cameras:
            try:
                frame = robot.cameras[name].async_read()
            except Exception:
                continue
        if frame is None or not isinstance(frame, np.ndarray) or frame.ndim != 3:
            continue
        try:
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ok:
                tmp = os.path.join(frame_dir, f".{name}.jpg.tmp")
                dst = os.path.join(frame_dir, f"{name}.jpg")
                with open(tmp, "wb") as f:
                    f.write(buf.tobytes())
                os.replace(tmp, dst)
        except Exception:
            pass


def read_control_command(control_file):
    if not control_file or not os.path.exists(control_file):
        return ""
    try:
        with open(control_file, "r") as f:
            raw = f.read().strip()
        os.remove(control_file)
        # Commands with stage names preserve case; all others are uppercased
        upper = raw.upper()
        if upper.startswith("PLAN_RESTART:"):
            return "PLAN_RESTART:" + raw[13:]
        if upper.startswith("PLAN_START:"):
            return "PLAN_START:" + raw[11:]
        if upper.startswith("PLAN:"):
            return "PLAN:" + raw[5:]
        return upper
    except Exception:
        return ""


def check_control_file(control_file, events, hand_detector=None):
    cmd = read_control_command(control_file)
    if not cmd:
        return
    logger.info(f"Control command received: {cmd}")
    if cmd == "ESTOP":
        events["emergency_stop"] = True
        events["exit_early"] = True
    elif cmd == "RESUME":
        events["emergency_stop"] = False
        events["auto_stopped"] = False
    elif cmd == "HOME":
        events["go_to_rest"] = True
        events["exit_early"] = True
    elif cmd.startswith("GOTO:"):
        try:
            point_idx = int(cmd.split(":")[1])
            events["go_to_rest"] = True
            events["exit_early"] = True
            events["_goto_point_idx"] = point_idx
        except (ValueError, IndexError):
            logger.error(f"Invalid GOTO command: {cmd}")
    elif cmd == "RESTART":
        events["emergency_stop"] = True
        events["exit_early"] = True
        events["_restart"] = True
    elif cmd.startswith("PLAN_RESTART:"):
        # Combined PLAN + RESTART in a single command (avoids race condition)
        stage_names = [s.strip() for s in cmd[13:].split(",") if s.strip()]
        events["_plan_stages"] = stage_names
        events["emergency_stop"] = True
        events["exit_early"] = True
        events["_restart"] = True
        logger.info(f"PLAN_RESTART received: run stages {stage_names}")
        print(f"PLAN_RECEIVED:{','.join(stage_names)}", flush=True)
    elif cmd.startswith("PLAN_START:"):
        # Combined PLAN + START — triggers restart so new plan takes effect
        stage_names = [s.strip() for s in cmd[11:].split(",") if s.strip()]
        events["_plan_stages"] = stage_names
        events["emergency_stop"] = True
        events["exit_early"] = True
        events["_restart"] = True
        logger.info(f"PLAN_START received: run stages {stage_names}")
        print(f"PLAN_RECEIVED:{','.join(stage_names)}", flush=True)
    elif cmd == "RETRY":
        events["emergency_stop"] = True
        events["exit_early"] = True
        events["_retry"] = True
    elif cmd == "QUIT":
        events["stop_recording"] = True
        events["exit_early"] = True
    elif cmd.startswith("PLAN:"):
        # LLM planner sends list of stage names to execute
        stage_names = [s.strip() for s in cmd[5:].split(",") if s.strip()]
        events["_plan_stages"] = stage_names
        logger.info(f"LLM plan received: run stages {stage_names}")
        print(f"PLAN_RECEIVED:{','.join(stage_names)}", flush=True)
    elif cmd == "START":
        events["emergency_stop"] = False
        events["auto_stopped"] = False
    elif cmd == "HAND_ON" and hand_detector:
        hand_detector.enabled = True
        print("HAND_DETECT_ON", flush=True)
    elif cmd == "HAND_OFF" and hand_detector:
        hand_detector.enabled = False
        hand_detector.hand_detected = False
        events["auto_stopped"] = False
        print("HAND_DETECT_OFF", flush=True)


# ── GOTO point support (ROS2 radians → lerobot normalized) ──

_ROS2_SCALE = 0.00153398
_ROS2_OFFSET = 2048.0

_CSV_TO_MOTOR = {
    "Rotation": "shoulder_pan",
    "Pitch": "shoulder_lift",
    "Elbow": "elbow_flex",
    "Wrist_Pitch": "wrist_flex",
    "Wrist_Roll": "wrist_roll",
    "Jaw": "gripper",
}


def _load_goto_points(csv_path):
    points, joint_names = [], []
    if not csv_path or not os.path.exists(csv_path):
        return points, joint_names
    try:
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i == 0:
                    joint_names = [c.strip() for c in row]
                else:
                    if row and any(c.strip() for c in row):
                        points.append([float(x) for x in row])
    except Exception as e:
        logger.error(f"Failed to load GOTO points: {e}")
    return points, joint_names


def _radians_to_lerobot(radians_vals, csv_names, robot):
    from lerobot.motors import MotorNormMode
    target = {}
    for rad, cname in zip(radians_vals, csv_names):
        motor = _CSV_TO_MOTOR.get(cname.strip())
        if motor is None:
            continue
        raw = rad / _ROS2_SCALE + _ROS2_OFFSET
        cal = robot.bus.calibration[motor]
        bounded = min(cal.range_max, max(cal.range_min, int(round(raw))))
        m = robot.bus.motors[motor]
        dm = robot.bus.apply_drive_mode and cal.drive_mode
        if m.norm_mode is MotorNormMode.RANGE_M100_100:
            n = (((bounded - cal.range_min) / (cal.range_max - cal.range_min)) * 200) - 100
            target[f"{motor}.pos"] = -n if dm else n
        elif m.norm_mode is MotorNormMode.RANGE_0_100:
            n = ((bounded - cal.range_min) / (cal.range_max - cal.range_min)) * 100
            target[f"{motor}.pos"] = (100 - n) if dm else n
    return target


# ═══════════════════════════════════════════════════════════════════════
#  Waypoint loading from saved_positions.json
# ═══════════════════════════════════════════════════════════════════════

def load_waypoints(json_path, names=None):
    """Load waypoints from saved_positions.json.

    Args:
        json_path: Path to the JSON file.
        names: List of waypoint names to load, or "all" / None for all.

    Returns:
        List of dicts: [{"shoulder_pan.pos": val, ...}, ...]
    """
    if not os.path.exists(json_path):
        logger.warning(f"Waypoints file not found: {json_path}")
        return []

    with open(json_path, "r") as f:
        data = json.load(f)

    positions = data.get("positions", [])
    if not positions:
        return []

    waypoints = []
    for pos in positions:
        values = pos.get("values", {})
        # Convert to lerobot key format: "joint_name" → "joint_name.pos"
        wp = {f"{k}.pos": v for k, v in values.items()}
        wp["_name"] = pos.get("name", f"p{pos.get('index', '?')}")
        waypoints.append(wp)

    # Filter by names if specified
    if names and names != "all":
        waypoints = [wp for wp in waypoints if wp.get("_name") in names]

    return waypoints


# ═══════════════════════════════════════════════════════════════════════
#  Trigger condition checking
# ═══════════════════════════════════════════════════════════════════════

def check_trigger(obs, trigger_joint, trigger_op, trigger_value):
    """Check if the trigger condition is met.

    Args:
        obs: Robot observation dict.
        trigger_joint: Joint name (e.g. "shoulder_pan").
        trigger_op: "lt" or "gt".
        trigger_value: Threshold value.

    Returns:
        (triggered: bool, current_value: float)
    """
    key = f"{trigger_joint}.pos"
    if key not in obs:
        return False, 0.0

    current = obs[key]
    if trigger_op == "lt":
        return current < trigger_value, current
    elif trigger_op == "gt":
        return current > trigger_value, current
    return False, current


# ═══════════════════════════════════════════════════════════════════════
#  Pipeline config loading
# ═══════════════════════════════════════════════════════════════════════

def load_pipeline_config(config_path):
    """Dynamically load PIPELINE_STAGES and WAYPOINTS_JSON from config.py."""
    spec = importlib.util.spec_from_file_location("pipeline_config", config_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    stages = getattr(mod, "PIPELINE_STAGES", [])
    waypoints_json = getattr(mod, "WAYPOINTS_JSON", "")
    return stages, waypoints_json


# ═══════════════════════════════════════════════════════════════════════
#  Wait for command (same as eval_act_safe.py)
# ═══════════════════════════════════════════════════════════════════════

def wait_for_command(control_file, events, target_cmd="START",
                     robot=None, frame_dir="", camera_names=None,
                     hand_detector=None,
                     goto_points=None, goto_joint_names=None):
    if goto_points is None:
        goto_points = []
    if goto_joint_names is None:
        goto_joint_names = []

    _last_ft = 0.0
    while True:
        cmd = read_control_command(control_file)
        if cmd == target_cmd:
            return cmd
        if cmd == "RESUME":
            return target_cmd
        if cmd == "QUIT":
            return "QUIT"
        if cmd == "RESTART":
            # Restart handled by main loop, treat as HOME for wait_for_command
            return target_cmd
        if cmd == "RETRY":
            # Retry handled by main loop, treat as HOME for wait_for_command
            return target_cmd
        if cmd.startswith("PLAN_START:"):
            # Combined PLAN + START — store plan and return immediately
            stage_names = [s.strip() for s in cmd[11:].split(",") if s.strip()]
            events["_plan_stages"] = stage_names
            logger.info(f"PLAN_START received during wait: run stages {stage_names}")
            print(f"PLAN_RECEIVED:{','.join(stage_names)}", flush=True)
            return target_cmd
        elif cmd.startswith("PLAN_RESTART:"):
            # Combined PLAN + RESTART — store plan and return
            stage_names = [s.strip() for s in cmd[13:].split(",") if s.strip()]
            events["_plan_stages"] = stage_names
            events["_restart"] = True
            logger.info(f"PLAN_RESTART received during wait: run stages {stage_names}")
            print(f"PLAN_RECEIVED:{','.join(stage_names)}", flush=True)
            return target_cmd
        elif cmd.startswith("PLAN:"):
            # LLM planner sends stage list — store in events for main loop
            stage_names = [s.strip() for s in cmd[5:].split(",") if s.strip()]
            events["_plan_stages"] = stage_names
            logger.info(f"LLM plan received during wait: run stages {stage_names}")
            print(f"PLAN_RECEIVED:{','.join(stage_names)}", flush=True)
        elif cmd == "HAND_ON" and hand_detector:
            hand_detector.enabled = True
            print("HAND_DETECT_ON", flush=True)
        elif cmd == "HAND_OFF" and hand_detector:
            hand_detector.enabled = False
            hand_detector.hand_detected = False
            print("HAND_DETECT_OFF", flush=True)
        elif cmd == "HOME" and robot:
            print("GOTO_STARTED:HOME", flush=True)
            go_to_rest_position(robot, rest_position=robot.rest_position,
                                fps=30, duration_s=2.0, events=events)
            print("GOTO_DONE:HOME", flush=True)
        elif cmd.startswith("GOTO:") and robot and goto_points:
            try:
                point_idx = int(cmd.split(":")[1])
                if 1 <= point_idx <= len(goto_points):
                    print(f"GOTO_STARTED:{point_idx}", flush=True)
                    target = _radians_to_lerobot(goto_points[point_idx - 1],
                                                  goto_joint_names, robot)
                    go_to_rest_position(robot, rest_position=target,
                                        fps=30, duration_s=2.0, events=events)
                    print(f"GOTO_DONE:{point_idx}", flush=True)
            except Exception as e:
                logger.error(f"GOTO command failed: {cmd}: {e}", exc_info=True)

        if events.get("stop_recording"):
            return "QUIT"

        if robot and frame_dir and camera_names:
            _now = time.perf_counter()
            if _now - _last_ft > 0.2:
                save_camera_frames(frame_dir, camera_names, robot=robot)
                _last_ft = _now

        time.sleep(0.1)


# ═══════════════════════════════════════════════════════════════════════
#  Emergency stop / hold loop (handles GOTO, HOME, RESUME during pause)
# ═══════════════════════════════════════════════════════════════════════

def handle_estop_loop(robot, events, args, hand_detector, frame_dir, camera_names,
                      goto_points, goto_joint_names):
    """Hold position during emergency stop.

    Returns:
        "restart" if RESTART command was received (pipeline should restart from stage 1)
        "retry"   if RETRY command was received (go home + retry current stage)
        "continue" otherwise (resume current stage)
    """
    hold_pos = robot.bus.sync_read("Present_Position")
    robot.bus.sync_write("Goal_Position", hold_pos)
    _action = "continue"  # track what action to take

    if events.get("auto_stopped"):
        print("🖐️  AUTO E-STOP — holding position. Remove hand to auto-resume...", flush=True)
    else:
        print("⚠️  EMERGENCY STOP — holding position. Press [Enter] to resume...", flush=True)

    _last_ft = 0.0
    while events.get("emergency_stop"):
        check_control_file(args.control_file, events, hand_detector)

        # ── RESTART command: go home → restart pipeline from stage 1 ──
        if events.get("_restart"):
            events["_restart"] = False
            events["emergency_stop"] = False
            events["auto_stopped"] = False
            print("GOTO_STARTED:HOME", flush=True)
            go_to_rest_position(robot, rest_position=robot.rest_position,
                                fps=args.fps, duration_s=2.0, events=events)
            print("GOTO_DONE:HOME", flush=True)
            _action = "restart"
            break

        # ── RETRY command: go home → retry current stage ──
        if events.get("_retry"):
            events["_retry"] = False
            events["emergency_stop"] = False
            events["auto_stopped"] = False
            print("GOTO_STARTED:HOME", flush=True)
            go_to_rest_position(robot, rest_position=robot.rest_position,
                                fps=args.fps, duration_s=2.0, events=events)
            print("GOTO_DONE:HOME", flush=True)
            _action = "retry"
            break

        # Auto-resume from hand detection
        if (hand_detector and events.get("auto_stopped")
                and not hand_detector.hand_detected):
            events["emergency_stop"] = False
            events["auto_stopped"] = False
            print("✅  Hand cleared — auto resuming inference...", flush=True)
            break

        if events.get("go_to_rest"):
            goto_idx = events.pop("_goto_point_idx", None)
            if goto_idx is not None and goto_points and 1 <= goto_idx <= len(goto_points):
                target_pos = _radians_to_lerobot(goto_points[goto_idx - 1], goto_joint_names, robot)
                print(f"GOTO_STARTED:{goto_idx}", flush=True)
            else:
                target_pos = robot.rest_position
                goto_idx = None
                print("GOTO_STARTED:HOME", flush=True)
            events["go_to_rest"] = False
            events["emergency_stop"] = False
            events["auto_stopped"] = False
            go_to_rest_position(robot, rest_position=target_pos,
                                fps=args.fps, duration_s=2.0, events=events)
            if goto_idx is not None:
                print(f"GOTO_DONE:{goto_idx}", flush=True)
            else:
                print("GOTO_DONE:HOME", flush=True)
            # After GOTO/HOME, hold at new position
            events["emergency_stop"] = True
            hold_pos = robot.bus.sync_read("Present_Position")
            continue

        robot.bus.sync_write("Goal_Position", hold_pos)
        _now = time.perf_counter()
        if _now - _last_ft > 0.2:
            save_camera_frames(frame_dir, camera_names, robot=robot)
            _last_ft = _now
        time.sleep(0.05)
        if events.get("stop_recording"):
            break

    return _action


def handle_goto_rest_loop(robot, events, args, hand_detector, frame_dir, camera_names,
                          goto_points, goto_joint_names):
    """Handle go-to-rest/home/goto-point, then hold until resumed.

    Returns:
        "restart" if RESTART command was received (pipeline should restart from stage 1)
        "retry"   if RETRY command was received (go home + retry current stage)
        "continue" otherwise (resume current stage)
    """
    _action = "continue"
    goto_idx = events.pop("_goto_point_idx", None)
    if goto_idx is not None and goto_points and 1 <= goto_idx <= len(goto_points):
        target_pos = _radians_to_lerobot(goto_points[goto_idx - 1], goto_joint_names, robot)
        label = f"📍 point {goto_idx}"
        print(f"GOTO_STARTED:{goto_idx}", flush=True)
    else:
        target_pos = robot.rest_position
        label = "🏠 rest"
        goto_idx = None
        print("GOTO_STARTED:HOME", flush=True)

    events["go_to_rest"] = False
    events["exit_early"] = False
    go_to_rest_position(robot, rest_position=target_pos,
                        fps=args.fps, duration_s=2.0, events=events)

    if goto_idx is not None:
        print(f"GOTO_DONE:{goto_idx}", flush=True)
    else:
        print("GOTO_DONE:HOME", flush=True)
    print(f"  {label} reached. Inference PAUSED.", flush=True)

    # Hold at position until resumed
    events["emergency_stop"] = True
    hold_pos = robot.bus.sync_read("Present_Position")
    _last_ft = 0.0
    while events.get("emergency_stop"):
        check_control_file(args.control_file, events, hand_detector)

        # ── RESTART command: go home → restart pipeline from stage 1 ──
        if events.get("_restart"):
            events["_restart"] = False
            events["emergency_stop"] = False
            print("GOTO_STARTED:HOME", flush=True)
            go_to_rest_position(robot, rest_position=robot.rest_position,
                                fps=args.fps, duration_s=2.0, events=events)
            print("GOTO_DONE:HOME", flush=True)
            _action = "restart"
            break

        # ── RETRY command: go home → retry current stage ──
        if events.get("_retry"):
            events["_retry"] = False
            events["emergency_stop"] = False
            print("GOTO_STARTED:HOME", flush=True)
            go_to_rest_position(robot, rest_position=robot.rest_position,
                                fps=args.fps, duration_s=2.0, events=events)
            print("GOTO_DONE:HOME", flush=True)
            _action = "retry"
            break

        if events.get("go_to_rest"):
            _gi = events.pop("_goto_point_idx", None)
            if _gi is not None and goto_points and 1 <= _gi <= len(goto_points):
                _tp = _radians_to_lerobot(goto_points[_gi - 1], goto_joint_names, robot)
                print(f"GOTO_STARTED:{_gi}", flush=True)
            else:
                _tp = robot.rest_position
                _gi = None
            events["go_to_rest"] = False
            events["exit_early"] = False
            events["emergency_stop"] = False
            go_to_rest_position(robot, rest_position=_tp,
                                fps=args.fps, duration_s=2.0, events=events)
            if _gi is not None:
                print(f"GOTO_DONE:{_gi}", flush=True)
            else:
                print("GOTO_DONE:HOME", flush=True)
            events["emergency_stop"] = True
            hold_pos = robot.bus.sync_read("Present_Position")
            continue
        robot.bus.sync_write("Goal_Position", hold_pos)
        _now = time.perf_counter()
        if _now - _last_ft > 0.2:
            save_camera_frames(frame_dir, camera_names, robot=robot)
            _last_ft = _now
        time.sleep(0.05)
        if events.get("stop_recording"):
            break

    return _action


# ═══════════════════════════════════════════════════════════════════════
#  Run one stage of the pipeline
# ═══════════════════════════════════════════════════════════════════════

def run_stage_inference(stage, model, preprocess, postprocess, robot, ds_features,
                        device, events, args, frame_dir, camera_names,
                        hand_detector, goto_points, goto_joint_names):
    """Run inference for one pipeline stage.

    Returns:
        "triggered" — trigger condition met, proceed to waypoints
        "quit"      — user requested quit
        "done"      — all episodes finished without trigger
        "restart"   — user pressed HOME then RESUME → restart pipeline from stage 1
    """
    fps = args.fps
    max_steps = int(stage.get("episode_time", 200) * fps)
    num_episodes = stage.get("num_episodes", 10)
    trigger_joint = stage.get("trigger_joint", "")
    trigger_op = stage.get("trigger_op", "lt")
    trigger_value = stage.get("trigger_value", -25.0)

    # Track consecutive trigger frames to avoid false positives
    TRIGGER_CONFIRM_FRAMES = 5
    trigger_count = 0

    for ep in range(num_episodes):
        print(f"\n{'─'*50}", flush=True)
        print(f"  [{stage['name']}] Episode {ep + 1}/{num_episodes}", flush=True)
        print(f"  Trigger: {trigger_joint} {trigger_op} {trigger_value}", flush=True)
        print(f"{'─'*50}", flush=True)

        model.reset()
        events["exit_early"] = False
        events["emergency_stop"] = False
        events["go_to_rest"] = False
        events["stop_recording"] = False
        events["auto_stopped"] = False
        trigger_count = 0

        for step in range(max_steps):
            start_t = time.perf_counter()

            # ── Check control file ──
            check_control_file(args.control_file, events, hand_detector)

            # ── Hand safety ──
            if (hand_detector and hand_detector.enabled and hand_detector.hand_detected
                    and not events.get("emergency_stop")):
                events["emergency_stop"] = True
                events["auto_stopped"] = True
                events["exit_early"] = True
                print("🖐️  HAND DETECTED — auto emergency stop!", flush=True)

            # ── Emergency stop loop ──
            if events.get("emergency_stop"):
                model.reset()
                trigger_count = 0
                estop_result = handle_estop_loop(robot, events, args, hand_detector,
                                  frame_dir, camera_names, goto_points, goto_joint_names)
                if estop_result == "restart":
                    print("🔄  Restarting pipeline from stage 1...", flush=True)
                    return "restart"
                if estop_result == "retry":
                    print("🔄  Retrying current stage...", flush=True)
                    return "retry"
                print("▶️  Resumed. Continuing inference...", flush=True)
                continue

            # ── Go to rest/home/goto ──
            if events.get("go_to_rest"):
                model.reset()
                trigger_count = 0
                rest_result = handle_goto_rest_loop(robot, events, args, hand_detector,
                                      frame_dir, camera_names, goto_points, goto_joint_names)
                if rest_result == "restart":
                    print("🔄  Restarting pipeline from stage 1...", flush=True)
                    return "restart"
                if rest_result == "retry":
                    print("🔄  Retrying current stage...", flush=True)
                    return "retry"
                print("▶️  Resumed from position. Continuing inference...", flush=True)
                continue

            # ── Quit / Skip ──
            if events.get("stop_recording"):
                return "quit"
            if events.get("exit_early"):
                events["exit_early"] = False
                print("⏭️  Skipping episode.", flush=True)
                break

            # ── Policy inference ──
            obs = robot.get_observation()
            save_camera_frames(frame_dir, camera_names, obs=obs)
            obs_frame = build_inference_frame(observation=obs, ds_features=ds_features, device=device)
            obs_processed = preprocess(obs_frame)
            action = model.select_action(obs_processed)
            action = postprocess(action)
            action_dict = make_robot_action(action, ds_features)
            robot.send_action(action_dict)

            # ── Check trigger condition ──
            if trigger_joint:
                triggered, current_val = check_trigger(obs, trigger_joint, trigger_op, trigger_value)
                if triggered:
                    trigger_count += 1
                    if trigger_count >= TRIGGER_CONFIRM_FRAMES:
                        print(f"\n🎯 TRIGGER: {trigger_joint}={current_val:.2f} {trigger_op} {trigger_value} "
                              f"(confirmed {trigger_count} frames)", flush=True)
                        print(f"STAGE_TRIGGERED:{stage['name']}", flush=True)
                        return "triggered"
                else:
                    trigger_count = 0

            # ── Maintain FPS ──
            dt = time.perf_counter() - start_t
            precise_sleep(max(1.0 / fps - dt, 0.0))

            if step % (fps * 10) == 0 and step > 0:
                print(f"    Step {step}/{max_steps} ({step / fps:.0f}s)", flush=True)

        if events.get("stop_recording"):
            return "quit"
        print(f"  [{stage['name']}] Episode {ep + 1} finished.", flush=True)

    return "done"


# ═══════════════════════════════════════════════════════════════════════
#  Move through waypoints
# ═══════════════════════════════════════════════════════════════════════

def load_csv_waypoints(csv_path, robot):
    """Load waypoints from a CSV file (ROS2 radians) and convert to lerobot format.

    Args:
        csv_path: Path to CSV file with header row and ROS2 radian values.
        robot: Connected robot instance (needed for calibration conversion).

    Returns:
        List of dicts: [{"shoulder_pan.pos": val, ..., "_name": "csv_1"}, ...]
    """
    if not csv_path or not os.path.exists(csv_path):
        logger.warning(f"CSV waypoint file not found: {csv_path}")
        return []

    points, csv_names = _load_goto_points(csv_path)
    if not points:
        return []

    waypoints = []
    for i, pt in enumerate(points):
        wp = _radians_to_lerobot(pt, csv_names, robot)
        wp["_name"] = f"csv_{i+1}"
        waypoints.append(wp)

    logger.info(f"Loaded {len(waypoints)} CSV waypoints from {csv_path}")
    return waypoints


def move_through_waypoints(robot, waypoints, duration_per_wp, fps, events,
                           frame_dir="", camera_names=None, timings=None):
    """Smoothly move the robot through a sequence of waypoints.

    Args:
        robot: Connected robot instance.
        waypoints: List of position dicts [{"shoulder_pan.pos": val, ...}, ...].
        duration_per_wp: Default seconds to spend moving to each waypoint.
        fps: Control loop frequency.
        events: Event dict for emergency stop.
        frame_dir: Frame directory for camera streaming.
        camera_names: List of camera names.
        timings: Optional per-step list of (move_time, hold_time) tuples.
                 Overrides duration_per_wp for each step.
    """
    if not waypoints:
        logger.warning("No waypoints to move through.")
        return

    for i, wp in enumerate(waypoints):
        name = wp.pop("_name", f"p{i+1}")

        # Per-step timing or default
        if timings and i < len(timings):
            move_t, hold_t = timings[i]
        else:
            move_t, hold_t = duration_per_wp, 0.1

        print(f"WAYPOINT_STARTED:{i+1}:{name}", flush=True)
        logger.info(f"Moving to waypoint {i+1}/{len(waypoints)}: {name} "
                     f"(move={move_t:.2f}s, hold={hold_t:.2f}s)")

        go_to_rest_position(
            robot, rest_position=wp,
            fps=fps, duration_s=move_t, events=events,
        )

        if events.get("emergency_stop") or events.get("stop_recording"):
            logger.warning("Waypoint movement interrupted.")
            return

        print(f"WAYPOINT_DONE:{i+1}:{name}", flush=True)
        logger.info(f"Reached waypoint {name}")

        # Hold at waypoint
        if hold_t > 0:
            time.sleep(hold_t)

        # Save camera frames while at waypoint
        if frame_dir and camera_names:
            save_camera_frames(frame_dir, camera_names, robot=robot)


# ═══════════════════════════════════════════════════════════════════════
#  Parse arguments
# ═══════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-model pipeline evaluation")

    parser.add_argument("--robot-port", type=str, default="/dev/ttyACM0")
    parser.add_argument("--robot-id", type=str, default="follower_hope")
    parser.add_argument("--cameras", type=str, default="front:/dev/video4,wrist:/dev/video6")
    parser.add_argument("--cam-width", type=int, default=640)
    parser.add_argument("--cam-height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--rest-duration", type=float, default=2.0)

    parser.add_argument("--control-file", type=str, default="")
    parser.add_argument("--wait-for-start", action="store_true")
    parser.add_argument("--frame-dir", type=str, default="/tmp/lerobot_frames")

    parser.add_argument("--hand-detect", action="store_true")
    parser.add_argument("--hand-detect-camera", type=str, default="front")
    parser.add_argument("--hand-detect-interval", type=float, default=0.25)
    parser.add_argument("--hand-detect-cooldown", type=int, default=8)

    parser.add_argument("--points-csv", type=str, default="")

    parser.add_argument("--pipeline-config", type=str, required=True,
                        help="Path to config.py containing PIPELINE_STAGES")

    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    device = torch.device(args.device)

    # ── Load pipeline config ──
    pipeline_stages, waypoints_json = load_pipeline_config(args.pipeline_config)
    if not pipeline_stages:
        logger.error("No PIPELINE_STAGES found in config!")
        sys.exit(1)

    print(f"PIPELINE_LOADED:{len(pipeline_stages)}", flush=True)
    logger.info(f"Pipeline: {len(pipeline_stages)} stages")
    for i, s in enumerate(pipeline_stages):
        logger.info(f"  Stage {i+1}: {s['name']} → {s['model']}")

    # ── Load waypoints ──
    all_waypoints = load_waypoints(waypoints_json)
    logger.info(f"Loaded {len(all_waypoints)} waypoints from {waypoints_json}")

    # ── Phase 1: Connect robot (once for entire pipeline) ──
    print("WARMUP_PHASE: connecting_robot", flush=True)
    camera_config = parse_cameras(args.cameras, args.cam_width, args.cam_height, args.fps)
    camera_names = list(camera_config.keys())
    robot_cfg = SO101FollowerConfig(port=args.robot_port, id=args.robot_id, cameras=camera_config)

    if args.frame_dir:
        os.makedirs(args.frame_dir, exist_ok=True)

    robot = SO101Follower(robot_cfg)
    logger.info(f"Connecting to robot '{args.robot_id}' on {args.robot_port}...")
    robot.connect()
    logger.info("Robot connected.")
    print("WARMUP_PHASE: robot_connected", flush=True)

    # ── Build features (same for all ACT models) ──
    ds_features = {}
    ds_features.update(hw_to_dataset_features(robot.observation_features, OBS_STR, use_video=False))
    ds_features.update(hw_to_dataset_features(robot.action_features, ACTION, use_video=False))

    # ── Load GOTO points ──
    goto_points, goto_joint_names = _load_goto_points(args.points_csv)
    if goto_points:
        logger.info(f"Loaded {len(goto_points)} GOTO points from {args.points_csv}")

    # ── Keyboard listener — disabled for UI-only control ──
    listener = None
    events = {
        "exit_early": False,
        "emergency_stop": False,
        "stop_recording": False,
        "go_to_rest": False,
    }
    events["auto_stopped"] = False

    # ── Hand detector ──
    hand_detector = None
    if args.hand_detect:
        frame_path = os.path.join(args.frame_dir, f"{args.hand_detect_camera}.jpg")
        hand_detector = HandDetector(
            frame_path=frame_path,
            check_interval=args.hand_detect_interval,
            cooldown_frames=args.hand_detect_cooldown,
        )
        hand_detector.start()
        print("HAND_DETECT_ON", flush=True)

    # ── Phase 2: Skip model preload — LLM planner decides which model to load first ──
    model = None
    preprocess = postprocess = None
    print("WARMUP_PHASE: model_loaded", flush=True)
    logger.info("Skipping model preload — will load after LLM plan")

    # ── Print summary ──
    print("\n" + "=" * 60, flush=True)
    print("  🤖 Multi-Model Pipeline Evaluation", flush=True)
    print("=" * 60, flush=True)
    print(f"  Robot   : {args.robot_id} @ {args.robot_port}", flush=True)
    print(f"  Camera  : {args.cameras}", flush=True)
    print(f"  FPS     : {args.fps}", flush=True)
    print(f"  Device  : {args.device}", flush=True)
    print(f"  Stages  : {len(pipeline_stages)}", flush=True)
    for i, s in enumerate(pipeline_stages):
        print(f"    {i+1}. [{s['name']}] {s['model']}", flush=True)
        print(f"       Trigger: {s.get('trigger_joint','-')} {s.get('trigger_op','-')} {s.get('trigger_value','-')}", flush=True)
    print(f"  Waypoints: {len(all_waypoints)}", flush=True)
    if hand_detector:
        print(f"  Hand    : ENABLED", flush=True)
    print("=" * 60 + "\n", flush=True)

    print("WARMUP_COMPLETE", flush=True)

    try:
        # ═══════════════════════════════════════════════════════════
        #  Main pipeline loop
        # ═══════════════════════════════════════════════════════════
        _skip_wait = False  # Set True after pipeline restart to skip wait_for_start
        while True:
            if args.wait_for_start and not _skip_wait:
                print("READY_FOR_START", flush=True)
                logger.info("Waiting for START command...")
                cmd = wait_for_command(args.control_file, events, "START",
                                       robot=robot, frame_dir=args.frame_dir,
                                       camera_names=camera_names,
                                       hand_detector=hand_detector,
                                       goto_points=goto_points,
                                       goto_joint_names=goto_joint_names)
                if cmd == "QUIT":
                    break
            _skip_wait = False  # Reset for next iteration

            print("INFERENCE_STARTED", flush=True)
            events["stop_recording"] = False
            events["exit_early"] = False
            events["emergency_stop"] = False
            events["go_to_rest"] = False
            events["auto_stopped"] = False
            events["_restart"] = False
            events["_retry"] = False

            quit_requested = False
            restart_requested = False
            retry_requested = False

            # ── LLM Plan: filter stages if PLAN command was received ──
            plan_stages = events.pop("_plan_stages", None)
            if plan_stages:
                active_stages = [s for s in pipeline_stages if s["name"] in plan_stages]
                logger.info(f"LLM plan active: running {len(active_stages)}/{len(pipeline_stages)} stages: "
                            f"{[s['name'] for s in active_stages]}")
            else:
                active_stages = pipeline_stages

            stage_idx = 0
            while stage_idx < len(active_stages):
                stage = active_stages[stage_idx]
                stage_name = stage["name"]
                stage_model_id = stage["model"]

                # ── Load model for this stage ──
                print(f"STAGE_LOADING:{stage_name}", flush=True)
                logger.info(f"Loading model for stage [{stage_name}]: {stage_model_id}")
                model = ACTPolicy.from_pretrained(stage_model_id)
                model.eval()
                preprocess, postprocess = make_pre_post_processors(
                    model.config, pretrained_path=stage_model_id
                )
                print(f"STAGE_LOADED:{stage_name}", flush=True)
                logger.info(f"Model loaded for [{stage_name}]")
                retry_requested = False

                print(f"STAGE_STARTED:{stage_name}", flush=True)
                logger.info(f"═══ Starting stage [{stage_name}] ═══")

                # ── Run inference for this stage ──
                result = run_stage_inference(
                    stage, model, preprocess, postprocess, robot, ds_features,
                    device, events, args, args.frame_dir, camera_names,
                    hand_detector, goto_points, goto_joint_names,
                )

                if result == "quit":
                    quit_requested = True
                    break

                if result == "restart":
                    restart_requested = True
                    print("PIPELINE_RESTART", flush=True)
                    logger.info("═══ Pipeline restart requested ═══")
                    break

                if result == "retry":
                    retry_requested = True
                    print(f"PIPELINE_RETRY:{stage_name}", flush=True)
                    logger.info(f"═══ Retry requested for stage [{stage_name}] ═══")
                    # Don't increment stage_idx — re-run the same stage
                    events["stop_recording"] = False
                    events["exit_early"] = False
                    events["emergency_stop"] = False
                    events["go_to_rest"] = False
                    events["auto_stopped"] = False
                    events["_restart"] = False
                    events["_retry"] = False
                    continue

                if result == "triggered":
                    print(f"STAGE_COMPLETE:{stage_name}:triggered", flush=True)
                    logger.info(f"Stage [{stage_name}] triggered → moving through waypoints")

                    # ── Load waypoints (CSV → per-stage JSON → global JSON) ──
                    wp_csv = stage.get("waypoint_csv", "")
                    wp_json = stage.get("waypoint_json", "")
                    if wp_csv:
                        # CSV waypoints (ROS2 radians → lerobot normalized)
                        stage_waypoints = load_csv_waypoints(wp_csv, robot)
                    elif wp_json:
                        # Per-stage JSON waypoints (LeRobot normalized, from save_position.py)
                        stage_waypoints = load_waypoints(wp_json)
                    else:
                        # Global JSON waypoints (saved_positions.json)
                        wp_names = stage.get("waypoints", "all")
                        if wp_names == "all":
                            stage_waypoints = load_waypoints(waypoints_json)
                        elif wp_names == "none":
                            stage_waypoints = []
                        else:
                            stage_waypoints = load_waypoints(waypoints_json, names=wp_names)

                    wp_duration = stage.get("waypoint_duration", 3.0)
                    wp_timings = stage.get("waypoint_timings", None)

                    if stage_waypoints:
                        print(f"WAYPOINTS_STARTED:{len(stage_waypoints)}", flush=True)
                        move_through_waypoints(
                            robot, stage_waypoints, wp_duration, args.fps, events,
                            frame_dir=args.frame_dir, camera_names=camera_names,
                            timings=wp_timings,
                        )
                        print(f"WAYPOINTS_DONE:{len(stage_waypoints)}", flush=True)

                    if events.get("stop_recording"):
                        quit_requested = True
                        break

                elif result == "done":
                    print(f"STAGE_COMPLETE:{stage_name}:done", flush=True)
                    logger.info(f"Stage [{stage_name}] completed all episodes")

                stage_idx += 1  # Move to next stage

            # ── Restart pipeline from stage 1 ──
            if restart_requested:
                # Check if a new PLAN command arrived (from LLM planner)
                check_control_file(args.control_file, events, hand_detector)

                # Determine which stages to reload for
                restart_plan = events.pop("_plan_stages", None)
                if restart_plan:
                    restart_stages = [s for s in pipeline_stages if s["name"] in restart_plan]
                    logger.info(f"Restart with LLM plan: {[s['name'] for s in restart_stages]}")
                else:
                    restart_stages = pipeline_stages

                # No need to preload — stage loop will load the right model
                model = None
                preprocess = postprocess = None
                events["stop_recording"] = False
                events["exit_early"] = False
                events["emergency_stop"] = False
                events["go_to_rest"] = False
                events["auto_stopped"] = False
                events["_restart"] = False
                events["_retry"] = False
                # Preserve plan stages for the next iteration
                if restart_plan:
                    events["_plan_stages"] = restart_plan
                # Go directly back to stage loop (skip wait_for_start)
                _skip_wait = True
                continue

            # ── Pipeline complete ──
            if quit_requested:
                print("INFERENCE_DONE", flush=True)
                if not args.wait_for_start:
                    break
                events["stop_recording"] = False
                continue

            # ── Go home after pipeline complete ──
            logger.info("Pipeline done — returning to home position")
            print("GOTO_STARTED:HOME", flush=True)
            go_to_rest_position(robot, rest_position=robot.rest_position,
                                fps=args.fps, duration_s=2.0, events=events)
            print("GOTO_DONE:HOME", flush=True)

            print("PIPELINE_COMPLETE", flush=True)
            print("INFERENCE_DONE", flush=True)
            logger.info("═══ Pipeline complete ═══")

            if not args.wait_for_start:
                break

            # No need to preload — next run will load after LLM plan
            model = None
            preprocess = postprocess = None

    except KeyboardInterrupt:
        print("\n\nCtrl+C detected. Shutting down...", flush=True)

    finally:
        if hand_detector:
            hand_detector.stop()
        if robot.is_connected:
            print("Disconnecting robot...", flush=True)
            robot.disconnect()
        if listener is not None:
            listener.stop()
        print("Done. ✅", flush=True)


if __name__ == "__main__":
    main()
