#!/usr/bin/env python3
"""ACT policy safe evaluation script with Emergency Stop and Go-to-Home.

Control via keyboard (terminal must have focus) OR control file (for UI backend):
    Spacebar / ESTOP   → Emergency Stop (hold current position, pause inference)
    Enter    / RESUME  → Resume inference (from e-stop or after go-to-home)
    r        / HOME    → Go to rest/home position (smooth 2s interpolation), then pause
    →(right)           → Skip current episode
    Esc      / QUIT    → Quit all episodes

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
        --device cuda

    # Pre-warm mode (for UI):
    python eval_act_safe.py --wait-for-start --control-file /tmp/lerobot_cmd ...
"""

import argparse
import logging
import os
import sys
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


def check_control_file(control_file: str, events: dict) -> None:
    """Read a command from the control file and map it to events."""
    cmd = read_control_command(control_file)
    if not cmd:
        return
    if cmd == "ESTOP":
        events["emergency_stop"] = True
        events["exit_early"] = True
    elif cmd == "RESUME":
        events["emergency_stop"] = False
    elif cmd == "HOME":
        events["go_to_rest"] = True
        events["exit_early"] = True
    elif cmd == "QUIT":
        events["stop_recording"] = True
        events["exit_early"] = True
    elif cmd == "START":
        # Used in wait-for-start mode; also works as resume
        events["emergency_stop"] = False


def wait_for_command(control_file: str, events: dict, target_cmd: str = "START",
                     robot=None, frame_dir: str = "", camera_names: list = None) -> str:
    """Block until a specific command (or QUIT) arrives via control file or keyboard.

    While waiting, captures camera frames for UI display (~5 fps).
    Returns the command received ("START" or "QUIT").
    """
    _last_ft = 0.0
    while True:
        # Check control file
        cmd = read_control_command(control_file)
        if cmd == target_cmd:
            return cmd
        if cmd == "QUIT":
            return "QUIT"

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
                 frame_dir: str = "", camera_names: list = None):
    """Run the inference episode loop. Returns True if should continue, False to quit."""
    if camera_names is None:
        camera_names = []
    for ep in range(args.num_episodes):
        print(f"\n{'─'*50}", flush=True)
        print(f"  Episode {ep + 1}/{args.num_episodes}", flush=True)
        print(f"{'─'*50}", flush=True)

        model.reset()
        events["exit_early"] = False
        events["emergency_stop"] = False
        events["go_to_rest"] = False
        events["stop_recording"] = False

        for step in range(max_steps):
            start_t = time.perf_counter()

            # ── Check external control file for commands ──
            check_control_file(args.control_file, events)

            # ── Emergency Stop: hold current position, pause inference ──
            if events.get("emergency_stop"):
                hold_pos = robot.bus.sync_read("Present_Position")
                robot.bus.sync_write("Goal_Position", hold_pos)
                model.reset()
                print("⚠️  EMERGENCY STOP — holding position. Press [Enter] to resume, [r] to go home...", flush=True)
                _last_ft = 0.0
                while events.get("emergency_stop"):
                    check_control_file(args.control_file, events)
                    if events.get("go_to_rest"):
                        events["go_to_rest"] = False
                        events["emergency_stop"] = False
                        print("🏠  Moving to rest position...", flush=True)
                        go_to_rest_position(
                            robot, rest_position=robot.rest_position,
                            fps=args.fps, duration_s=args.rest_duration, events=events,
                        )
                        model.reset()
                        print("🏠  Home reached. Inference PAUSED. Press [Enter] to resume, [Esc] to quit.", flush=True)
                        events["emergency_stop"] = True
                        home_pos = robot.bus.sync_read("Present_Position")
                        while events.get("emergency_stop"):
                            check_control_file(args.control_file, events)
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

            # ── Go to rest / home, then PAUSE ──
            if events.get("go_to_rest"):
                events["go_to_rest"] = False
                events["exit_early"] = False
                print("🏠  Moving to rest position...", flush=True)
                go_to_rest_position(
                    robot, rest_position=robot.rest_position,
                    fps=args.fps, duration_s=args.rest_duration, events=events,
                )
                model.reset()
                print("🏠  Home reached. Inference PAUSED. Press [Enter] to resume, [Esc] to quit.", flush=True)
                events["emergency_stop"] = True
                home_pos = robot.bus.sync_read("Present_Position")
                _last_ft2 = 0.0
                while events.get("emergency_stop"):
                    check_control_file(args.control_file, events)
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

    # ── Initialize keyboard listener ──
    listener, events = init_keyboard_listener()

    print("\n" + "=" * 60)
    print("  ACT Safe Evaluation")
    print("  Model : " + args.model)
    print("  Robot : " + f"{args.robot_id} @ {args.robot_port}")
    print("  Camera: " + args.cameras)
    print("  FPS   : " + str(args.fps))
    print("  Episodes: " + f"{args.num_episodes} × {args.episode_time}s ({max_steps} steps)")
    print("  Mode  : " + ("WAIT-FOR-START (pre-warm)" if args.wait_for_start else "IMMEDIATE"))
    print("-" * 60)
    print("  Keyboard shortcuts:")
    print("    [Space]  Emergency Stop (hold position, pause inference)")
    print("    [Enter]  Resume inference")
    print("    [r]      Go to Home / rest position (2s), then pause")
    print("    [→]      Skip current episode")
    print("    [Esc]    Quit")
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
                                       camera_names=camera_names)
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

                should_continue = run_episodes(
                    args, model, preprocess, postprocess,
                    robot, ds_features, device, events, max_steps,
                    frame_dir=args.frame_dir, camera_names=camera_names,
                )

                print("INFERENCE_DONE", flush=True)

                if not should_continue:
                    # User pressed Quit during episodes — but in wait mode, go back to waiting
                    # unless stop_recording was from a QUIT command
                    events["stop_recording"] = False
                    continue

        else:
            # ── Immediate mode (original behavior) ──
            run_episodes(
                args, model, preprocess, postprocess,
                robot, ds_features, device, events, max_steps,
                frame_dir=args.frame_dir, camera_names=camera_names,
            )

    except KeyboardInterrupt:
        print("\n\nCtrl+C detected. Shutting down...", flush=True)

    finally:
        if robot.is_connected:
            print("Disconnecting robot...", flush=True)
            robot.disconnect()
        if listener is not None:
            listener.stop()
        print("Done. ✅", flush=True)


if __name__ == "__main__":
    main()
