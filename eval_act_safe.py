#!/usr/bin/env python3
"""ACT policy safe evaluation script with Emergency Stop, Go-to-Home, and Hand Safety.

Control via keyboard (terminal must have focus) OR control file (for UI backend):
    Spacebar / ESTOP   → Emergency Stop (hold current position, pause inference)
    Enter    / RESUME  → Resume inference (from e-stop or after go-to-home)
    r        / HOME    → Go to rest/home position (smooth 2s interpolation), then pause
    →(right)           → Skip current episode
    Esc      / QUIT    → Quit all episodes

Hand Safety (--hand-detect):
    Automatically detects human hands in the front camera using MediaPipe.
    Triggers emergency stop when a hand is detected; auto-resumes when clear.
    Runs in a separate lightweight CPU thread (~15ms/frame, ~3-5fps).

Pre-warm mode (--wait-for-start):
    Loads model + connects robot on startup, then waits for START command.
    User presses Start in UI → inference begins immediately (zero delay).

Usage:
    python eval_act_safe.py \\
        --model FrankYuzhe/act_merged_tissue_spoon_0203_0204_2202 \\
        --robot-port /dev/ttyACM0 \\
        --robot-id hope \\
        --cameras "front:/dev/video4,wrist:/dev/video0" \\
        --fps 30 \\
        --episode-time 200 \\
        --num-episodes 10 \\
        --device cuda \\
        --hand-detect

    # Pre-warm mode (for UI):
    python eval_act_safe.py --wait-for-start --control-file /tmp/lerobot_cmd --hand-detect ...
"""

import argparse
import csv
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


# ── Hand Detection Thread (MediaPipe, CPU-only, ~15ms/frame) ──

class HandDetector:
    """Lightweight hand detector running in a background thread.

    Reads the latest front camera frame from the shared JPEG file
    (already written by save_camera_frames). Zero extra camera I/O.
    Uses MediaPipe HandLandmarker (float16, CPU) — ~15ms per frame.

    Attributes:
        hand_detected (bool): True if a hand is currently visible.
        enabled (bool): Toggle detection on/off at runtime.
    """

    MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "hand_landmarker.task")

    def __init__(
        self,
        frame_path: str,
        check_interval: float = 0.25,   # seconds between checks (~4 fps)
        cooldown_frames: int = 8,        # consecutive no-hand frames before clearing
        min_confidence: float = 0.5,
    ):
        self.frame_path = frame_path
        self.check_interval = check_interval
        self.cooldown_frames = cooldown_frames
        self.min_confidence = min_confidence

        self.hand_detected = False
        self.enabled = True
        self._no_hand_count = 0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._detector = None

    def start(self):
        """Start the background detection thread."""
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        if not os.path.exists(self.MODEL_PATH):
            logger.error(f"Hand detection model not found: {self.MODEL_PATH}")
            logger.error("Download: https://storage.googleapis.com/mediapipe-models/"
                         "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")
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
        """Stop the background thread and release resources."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        if self._detector:
            self._detector.close()
            self._detector = None
        logger.info("HandDetector: stopped")

    def _run(self):
        """Background loop: read JPEG → detect → update flag."""
        import mediapipe as mp

        while not self._stop_event.is_set():
            if self.enabled and self._detector:
                try:
                    self._check_frame(mp)
                except Exception as e:
                    logger.debug(f"HandDetector frame check error: {e}")
            self._stop_event.wait(self.check_interval)

    def _check_frame(self, mp):
        """Read latest front camera JPEG and run hand detection."""
        if not os.path.exists(self.frame_path):
            return

        # Read JPEG (already saved by save_camera_frames)
        frame_bgr = cv2.imread(self.frame_path)
        if frame_bgr is None:
            return

        # Downscale for speed (detection doesn't need full res)
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


def parse_cameras(cameras_str: str, width: int = 640, height: int = 480, fps: int = 30) -> dict:
    """Parse camera string like 'front:/dev/video4,wrist:/dev/video0' into config dict."""
    cameras = {}
    for item in cameras_str.split(","):
        name, path = item.strip().split(":", 1)
        cameras[name.strip()] = OpenCVCameraConfig(
            index_or_path=path.strip(), width=width, height=height, fps=fps
        )
    return cameras


# ── Camera frame saving for UI display ──

def save_camera_frames(frame_dir: str, camera_names: list, obs: dict = None, robot=None) -> None:
    """Save camera frames to JPEG files for UI display.

    Uses frames from `obs` dict if provided (zero extra I/O, preferred during inference).
    Falls back to calling async_read() on camera objects (used in hold/wait loops).
    async_read() lazily starts the background thread on first call.
    """
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
            # async_read() returns RGB; cv2.imencode expects BGR
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


# ── GOTO point support: convert ROS2 radians → lerobot normalized ──

# ROS2 hardware interface params (from so101.ros2_control.hardware.xacro)
_ROS2_SCALE = 0.00153398   # radians per servo step
_ROS2_OFFSET = 2048.0      # raw servo midpoint

# point.csv column name → lerobot motor name
_CSV_TO_MOTOR = {
    "Rotation": "shoulder_pan",
    "Pitch": "shoulder_lift",
    "Elbow": "elbow_flex",
    "Wrist_Pitch": "wrist_flex",
    "Wrist_Roll": "wrist_roll",
    "Jaw": "gripper",
}


def _load_goto_points(csv_path: str):
    """Load joint positions from point.csv (header row + N data rows in radians)."""
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
        logger.error(f"Failed to load GOTO points from {csv_path}: {e}")
    return points, joint_names


def _radians_to_lerobot(radians_vals, csv_names, robot):
    """Convert ROS2 radian positions (from point.csv) to a lerobot normalized position dict.

    The conversion chain is:
        radians  →  raw servo value  →  lerobot normalized [-100..100] or [0..100]
    """
    from lerobot.motors import MotorNormMode

    target = {}
    for rad, cname in zip(radians_vals, csv_names):
        motor = _CSV_TO_MOTOR.get(cname.strip())
        if motor is None:
            continue
        # Radians → raw servo value (same formula the ROS2 hardware interface uses)
        raw = rad / _ROS2_SCALE + _ROS2_OFFSET
        # Raw → lerobot normalized (same logic as MotorsBus._normalize)
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


def parse_args():
    parser = argparse.ArgumentParser(description="ACT policy evaluation with safety features")

    # Model
    parser.add_argument(
        "--model", type=str,
        default="FrankYuzhe/act_merged_tissue_spoon_0203_0204_2202",
        help="HuggingFace model ID or local path",
    )

    # Robot
    parser.add_argument("--robot-port", type=str, default="/dev/ttyACM0")
    parser.add_argument("--robot-id", type=str, default="hope")

    # Cameras: "name1:path1,name2:path2"
    parser.add_argument(
        "--cameras", type=str,
        default="front:/dev/video4,wrist:/dev/video0",
        help='Camera config, e.g. "front:/dev/video4,wrist:/dev/video0"',
    )
    parser.add_argument("--cam-width", type=int, default=640)
    parser.add_argument("--cam-height", type=int, default=480)

    # Evaluation
    parser.add_argument("--fps", type=int, default=30, help="Control loop frequency")
    parser.add_argument("--episode-time", type=float, default=200, help="Max seconds per episode")
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda", help="cuda / cpu / mps")

    # Safety
    parser.add_argument("--rest-duration", type=float, default=2.0, help="Go-to-home duration in seconds")

    # External control (for UI backend)
    parser.add_argument(
        "--control-file", type=str, default="",
        help="Path to a control file for receiving commands from an external UI (e.g. /tmp/lerobot_cmd)",
    )

    # Pre-warm mode
    parser.add_argument(
        "--wait-for-start", action="store_true",
        help="Pre-load model + connect robot, then wait for START command before inference",
    )

    # Camera frame streaming for UI
    parser.add_argument(
        "--frame-dir", type=str, default="/tmp/lerobot_frames",
        help="Directory to save camera frames as JPEG for UI streaming",
    )

    # Hand safety detection
    parser.add_argument(
        "--hand-detect", action="store_true",
        help="Enable automatic hand detection safety: auto e-stop when a hand is seen in the front camera",
    )
    parser.add_argument(
        "--hand-detect-camera", type=str, default="front",
        help="Which camera to use for hand detection (default: front)",
    )
    parser.add_argument(
        "--hand-detect-interval", type=float, default=0.25,
        help="Seconds between hand detection checks (default: 0.25 = ~4fps)",
    )
    parser.add_argument(
        "--hand-detect-cooldown", type=int, default=8,
        help="Consecutive no-hand frames before auto-resume (default: 8 = ~2s)",
    )

    # GOTO points (prepositions in ROS2 radians)
    parser.add_argument(
        "--points-csv", type=str, default="",
        help="Path to point.csv with joint positions in ROS2 radians for GOTO:N commands",
    )

    return parser.parse_args()


def read_control_command(control_file: str) -> str:
    """Read and consume a command from the control file. Returns empty string if none."""
    if not control_file or not os.path.exists(control_file):
        return ""
    try:
        with open(control_file, "r") as f:
            cmd = f.read().strip().upper()
        os.remove(control_file)
        return cmd
    except Exception:
        return ""


def check_control_file(control_file: str, events: dict, hand_detector: "HandDetector | None" = None) -> None:
    """Read a command from the control file and map it to events."""
    cmd = read_control_command(control_file)
    if not cmd:
        return
    if cmd == "ESTOP":
        events["emergency_stop"] = True
        events["exit_early"] = True
    elif cmd == "RESUME":
        events["emergency_stop"] = False
        events["auto_stopped"] = False  # Clear auto-stop flag on manual resume
    elif cmd == "HOME":
        events["go_to_rest"] = True
        events["exit_early"] = True
    elif cmd.startswith("GOTO:"):
        # GOTO:N — move arm to point N from points CSV, then pause
        try:
            point_idx = int(cmd.split(":")[1])
            events["go_to_rest"] = True
            events["exit_early"] = True
            events["_goto_point_idx"] = point_idx
        except (ValueError, IndexError):
            logger.error(f"Invalid GOTO command: {cmd}")
    elif cmd == "QUIT":
        events["stop_recording"] = True
        events["exit_early"] = True
    elif cmd == "START":
        # Used in wait-for-start mode; also works as resume
        events["emergency_stop"] = False
        events["auto_stopped"] = False
    elif cmd == "HAND_ON":
        if hand_detector:
            hand_detector.enabled = True
            print("HAND_DETECT_ON", flush=True)
            logger.info("Hand detection ENABLED")
    elif cmd == "HAND_OFF":
        if hand_detector:
            hand_detector.enabled = False
            hand_detector.hand_detected = False
            events["auto_stopped"] = False
            print("HAND_DETECT_OFF", flush=True)
            logger.info("Hand detection DISABLED")


def wait_for_command(control_file: str, events: dict, target_cmd: str = "START",
                     robot=None, frame_dir: str = "", camera_names: list = None,
                     hand_detector: "HandDetector | None" = None,
                     goto_points: list = None, goto_joint_names: list = None) -> str:
    """Block until a specific command (or QUIT) arrives via control file or keyboard.

    While waiting, captures camera frames for UI display (~5 fps).
    Also handles GOTO:N commands (move arm to preposition, then continue waiting).
    Returns the command received ("START" or "QUIT").
    """
    if goto_points is None:
        goto_points = []
    if goto_joint_names is None:
        goto_joint_names = []

    _last_ft = 0.0
    while True:
        # Check control file (including HAND_ON/OFF and GOTO commands)
        cmd = read_control_command(control_file)
        if cmd == target_cmd:
            return cmd
        if cmd == "QUIT":
            return "QUIT"
        # Handle HAND_ON/OFF even while waiting
        if cmd == "HAND_ON" and hand_detector:
            hand_detector.enabled = True
            print("HAND_DETECT_ON", flush=True)
        elif cmd == "HAND_OFF" and hand_detector:
            hand_detector.enabled = False
            hand_detector.hand_detected = False
            print("HAND_DETECT_OFF", flush=True)
        elif cmd == "HOME" and robot:
            print("GOTO_STARTED:HOME", flush=True)
            logger.info("Moving to HOME position while waiting...")
            go_to_rest_position(robot, rest_position=robot.rest_position,
                                fps=30, duration_s=2.0, events=events)
            print("GOTO_DONE:HOME", flush=True)
            logger.info("HOME reached (still waiting for START).")
        elif cmd.startswith("GOTO:") and robot and goto_points:
            try:
                point_idx = int(cmd.split(":")[1])
                if 1 <= point_idx <= len(goto_points):
                    target = _radians_to_lerobot(goto_points[point_idx - 1],
                                                  goto_joint_names, robot)
                    print(f"GOTO_STARTED:{point_idx}", flush=True)
                    logger.info(f"Moving to point {point_idx} while waiting...")
                    go_to_rest_position(robot, rest_position=target,
                                        fps=30, duration_s=2.0, events=events)
                    print(f"GOTO_DONE:{point_idx}", flush=True)
                    logger.info(f"Point {point_idx} reached (still waiting for START).")
                else:
                    logger.error(f"GOTO point {point_idx} out of range 1-{len(goto_points)}")
            except (ValueError, IndexError) as e:
                logger.error(f"Invalid GOTO command: {cmd}: {e}")

        # Check keyboard events (Esc → quit)
        if events.get("stop_recording"):
            return "QUIT"

        # Save camera frames for UI while waiting (~5fps)
        if robot and frame_dir and camera_names:
            _now = time.perf_counter()
            if _now - _last_ft > 0.2:
                save_camera_frames(frame_dir, camera_names, robot=robot)
                _last_ft = _now

        time.sleep(0.1)


def run_episodes(args, model, preprocess, postprocess, robot, ds_features, device, events, max_steps,
                 frame_dir: str = "", camera_names: list = None,
                 hand_detector: "HandDetector | None" = None,
                 goto_points: list = None, goto_joint_names: list = None):
    """Run the inference episode loop. Returns True if should continue, False to quit."""
    if camera_names is None:
        camera_names = []
    if goto_points is None:
        goto_points = []
    if goto_joint_names is None:
        goto_joint_names = []
    for ep in range(args.num_episodes):
        print(f"\n{'─'*50}", flush=True)
        print(f"  Episode {ep + 1}/{args.num_episodes}", flush=True)
        print(f"{'─'*50}", flush=True)

        model.reset()
        events["exit_early"] = False
        events["emergency_stop"] = False
        events["go_to_rest"] = False
        events["stop_recording"] = False
        events["auto_stopped"] = False

        for step in range(max_steps):
            start_t = time.perf_counter()

            # ── Check external control file for commands ──
            check_control_file(args.control_file, events, hand_detector)

            # ── Hand Safety: auto e-stop when hand detected ──
            if (hand_detector and hand_detector.enabled and hand_detector.hand_detected
                    and not events.get("emergency_stop")):
                events["emergency_stop"] = True
                events["auto_stopped"] = True
                events["exit_early"] = True
                print("🖐️  HAND DETECTED — auto emergency stop!", flush=True)

            # ── Emergency Stop: hold current position, pause inference ──
            if events.get("emergency_stop"):
                hold_pos = robot.bus.sync_read("Present_Position")
                robot.bus.sync_write("Goal_Position", hold_pos)
                model.reset()
                if events.get("auto_stopped"):
                    print("🖐️  AUTO E-STOP — holding position. Remove hand to auto-resume...", flush=True)
                else:
                    print("⚠️  EMERGENCY STOP — holding position. Press [Enter] to resume, [r] to go home...", flush=True)
                _last_ft = 0.0
                while events.get("emergency_stop"):
                    check_control_file(args.control_file, events, hand_detector)

                    # Auto-resume: hand detector says clear + this was an auto stop
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
                            label = f"📍 point {goto_idx}"
                            print(f"GOTO_STARTED:{goto_idx}", flush=True)
                        else:
                            target_pos = robot.rest_position
                            label = "🏠 rest"
                        events["go_to_rest"] = False
                        events["emergency_stop"] = False
                        events["auto_stopped"] = False
                        print(f"  Moving to {label} position...", flush=True)
                        go_to_rest_position(
                            robot, rest_position=target_pos,
                            fps=args.fps, duration_s=args.rest_duration, events=events,
                        )
                        model.reset()
                        if goto_idx is not None:
                            print(f"GOTO_DONE:{goto_idx}", flush=True)
                        print(f"  {label} reached. Inference PAUSED. Press [Enter] to resume, [Esc] to quit.", flush=True)
                        events["emergency_stop"] = True
                        home_pos = robot.bus.sync_read("Present_Position")
                        while events.get("emergency_stop"):
                            check_control_file(args.control_file, events, hand_detector)
                            robot.bus.sync_write("Goal_Position", home_pos)
                            _now = time.perf_counter()
                            if _now - _last_ft > 0.2:
                                save_camera_frames(frame_dir, camera_names, robot=robot)
                                _last_ft = _now
                            time.sleep(0.05)
                            if events.get("stop_recording"):
                                break
                        break
                    robot.bus.sync_write("Goal_Position", hold_pos)
                    _now = time.perf_counter()
                    if _now - _last_ft > 0.2:
                        save_camera_frames(frame_dir, camera_names, robot=robot)
                        _last_ft = _now
                    time.sleep(0.05)
                print("▶️  Resumed. Continuing inference...", flush=True)
                continue

            # ── Go to rest / home / goto point, then PAUSE ──
            if events.get("go_to_rest"):
                goto_idx = events.pop("_goto_point_idx", None)
                if goto_idx is not None and goto_points and 1 <= goto_idx <= len(goto_points):
                    target_pos = _radians_to_lerobot(goto_points[goto_idx - 1], goto_joint_names, robot)
                    label = f"📍 point {goto_idx}"
                    print(f"GOTO_STARTED:{goto_idx}", flush=True)
                else:
                    target_pos = robot.rest_position
                    label = "🏠 rest"
                events["go_to_rest"] = False
                events["exit_early"] = False
                print(f"  Moving to {label} position...", flush=True)
                go_to_rest_position(
                    robot, rest_position=target_pos,
                    fps=args.fps, duration_s=args.rest_duration, events=events,
                )
                model.reset()
                if goto_idx is not None:
                    print(f"GOTO_DONE:{goto_idx}", flush=True)
                print(f"  {label} reached. Inference PAUSED. Press [Enter] to resume, [Esc] to quit.", flush=True)
                events["emergency_stop"] = True
                home_pos = robot.bus.sync_read("Present_Position")
                _last_ft2 = 0.0
                while events.get("emergency_stop"):
                    check_control_file(args.control_file, events, hand_detector)
                    robot.bus.sync_write("Goal_Position", home_pos)
                    _now = time.perf_counter()
                    if _now - _last_ft2 > 0.2:
                        save_camera_frames(frame_dir, camera_names, robot=robot)
                        _last_ft2 = _now
                    time.sleep(0.05)
                    if events.get("stop_recording"):
                        break
                print("▶️  Resumed from home. Continuing inference...", flush=True)
                continue

            # ── Quit / Skip ──
            if events.get("stop_recording"):
                break
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

            # ── Maintain target FPS ──
            dt = time.perf_counter() - start_t
            precise_sleep(max(1.0 / args.fps - dt, 0.0))

            if step % (args.fps * 10) == 0 and step > 0:
                print(f"    Step {step}/{max_steps} ({step / args.fps:.0f}s)", flush=True)

        if events.get("stop_recording"):
            print("\n[Esc/Quit] Quit requested. Stopping.", flush=True)
            return False  # signal to quit

        print(f"  Episode {ep + 1} finished.", flush=True)

    return True  # all episodes done, can continue


def main():
    args = parse_args()

    device = torch.device(args.device)
    max_steps = int(args.episode_time * args.fps)

    # ── Phase 1: Load model ──
    print("WARMUP_PHASE: loading_model", flush=True)
    logger.info(f"Loading model: {args.model}")
    model = ACTPolicy.from_pretrained(args.model)
    model.eval()
    print("WARMUP_PHASE: model_loaded", flush=True)

    # ── Phase 2: Create pre/post processors ──
    preprocess, postprocess = make_pre_post_processors(model.config, pretrained_path=args.model)

    # ── Phase 3: Connect robot ──
    print("WARMUP_PHASE: connecting_robot", flush=True)
    camera_config = parse_cameras(args.cameras, args.cam_width, args.cam_height, args.fps)
    camera_names = list(camera_config.keys())
    robot_cfg = SO101FollowerConfig(port=args.robot_port, id=args.robot_id, cameras=camera_config)

    # Set up frame directory for UI camera display
    if args.frame_dir:
        os.makedirs(args.frame_dir, exist_ok=True)
    robot = SO101Follower(robot_cfg)

    logger.info(f"Connecting to robot '{args.robot_id}' on {args.robot_port}...")
    robot.connect()
    logger.info("Robot connected.")
    print("WARMUP_PHASE: robot_connected", flush=True)

    # ── Phase 4: Build features ──
    ds_features = {}
    ds_features.update(hw_to_dataset_features(robot.observation_features, OBS_STR, use_video=False))
    ds_features.update(hw_to_dataset_features(robot.action_features, ACTION, use_video=False))
    logger.info(f"Dataset features keys: {list(ds_features.keys())}")

    # ── Phase 5: Load GOTO points (if configured) ──
    goto_points, goto_joint_names = _load_goto_points(args.points_csv)
    if goto_points:
        logger.info(f"Loaded {len(goto_points)} GOTO points from {args.points_csv}")
    elif args.points_csv:
        logger.warning(f"No GOTO points loaded from {args.points_csv}")

    # ── Initialize keyboard listener ──
    listener, events = init_keyboard_listener()
    events["auto_stopped"] = False  # Track auto e-stop from hand detection

    # ── Initialize hand detector (if enabled) ──
    hand_detector = None
    if args.hand_detect:
        hand_cam = args.hand_detect_camera
        frame_path = os.path.join(args.frame_dir, f"{hand_cam}.jpg")
        hand_detector = HandDetector(
            frame_path=frame_path,
            check_interval=args.hand_detect_interval,
            cooldown_frames=args.hand_detect_cooldown,
        )
        hand_detector.start()
        print(f"HAND_DETECT_ON", flush=True)

    print("\n" + "=" * 60)
    print("  ACT Safe Evaluation")
    print("  Model : " + args.model)
    print("  Robot : " + f"{args.robot_id} @ {args.robot_port}")
    print("  Camera: " + args.cameras)
    print("  FPS   : " + str(args.fps))
    print("  Episodes: " + f"{args.num_episodes} × {args.episode_time}s ({max_steps} steps)")
    print("  Mode  : " + ("WAIT-FOR-START (pre-warm)" if args.wait_for_start else "IMMEDIATE"))
    if hand_detector:
        print(f"  Hand  : ENABLED (camera={args.hand_detect_camera}, interval={args.hand_detect_interval}s, cooldown={args.hand_detect_cooldown})")
    else:
        print("  Hand  : DISABLED (use --hand-detect to enable)")
    if goto_points:
        print(f"  GOTO  : {len(goto_points)} points loaded from {args.points_csv}")
    print("-" * 60)
    print("  Keyboard shortcuts:")
    print("    [Space]  Emergency Stop (hold position, pause inference)")
    print("    [Enter]  Resume inference")
    print("    [r]      Go to Home / rest position (2s), then pause")
    print("    [→]      Skip current episode")
    print("    [Esc]    Quit")
    if hand_detector:
        print("  Auto safety:")
        print("    🖐️  Hand in front camera → auto e-stop")
        print("    ✅  Hand removed → auto resume (~2s delay)")
    print("=" * 60 + "\n")

    # Signal warmup complete
    print("WARMUP_COMPLETE", flush=True)

    try:
        if args.wait_for_start:
            # ── Pre-warm mode: wait for START, run, loop ──
            while True:
                print("READY_FOR_START", flush=True)
                logger.info("Waiting for START command...")

                cmd = wait_for_command(args.control_file, events, "START",
                                       robot=robot, frame_dir=args.frame_dir,
                                       camera_names=camera_names,
                                       hand_detector=hand_detector,
                                       goto_points=goto_points,
                                       goto_joint_names=goto_joint_names)
                if cmd == "QUIT":
                    print("Quit received while waiting.", flush=True)
                    break

                print("INFERENCE_STARTED", flush=True)
                logger.info("START received — beginning inference!")

                # Reset events for fresh run
                events["stop_recording"] = False
                events["exit_early"] = False
                events["emergency_stop"] = False
                events["go_to_rest"] = False
                events["auto_stopped"] = False

                should_continue = run_episodes(
                    args, model, preprocess, postprocess,
                    robot, ds_features, device, events, max_steps,
                    frame_dir=args.frame_dir, camera_names=camera_names,
                    hand_detector=hand_detector,
                    goto_points=goto_points, goto_joint_names=goto_joint_names,
                )

                print("INFERENCE_DONE", flush=True)

                if not should_continue:
                    events["stop_recording"] = False
                    continue

        else:
            # ── Immediate mode (original behavior) ──
            run_episodes(
                args, model, preprocess, postprocess,
                robot, ds_features, device, events, max_steps,
                frame_dir=args.frame_dir, camera_names=camera_names,
                hand_detector=hand_detector,
                goto_points=goto_points, goto_joint_names=goto_joint_names,
            )

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
