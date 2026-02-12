#!/usr/bin/env python3
"""ACT policy safe evaluation script with Emergency Stop and Go-to-Home.

Safety keyboard shortcuts (terminal must have focus):
    Spacebar  → Emergency Stop (hold current position, pause inference)
    Enter     → Resume inference (from e-stop or after go-to-home)
    r         → Go to rest/home position (smooth 2s interpolation), then pause
    →(right)  → Skip current episode
    Esc       → Quit all episodes

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

    # Or simply run with defaults:
    python eval_act_safe.py
"""

import argparse
import logging
import time

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

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device)
    max_steps = int(args.episode_time * args.fps)

    # ── Load model ──
    logger.info(f"Loading model: {args.model}")
    model = ACTPolicy.from_pretrained(args.model)
    model.eval()

    # ── Create pre/post processors (load from pretrained path) ──
    preprocess, postprocess = make_pre_post_processors(model.config, pretrained_path=args.model)

    # ── Configure robot ──
    camera_config = parse_cameras(args.cameras, args.cam_width, args.cam_height, args.fps)
    robot_cfg = SO101FollowerConfig(port=args.robot_port, id=args.robot_id, cameras=camera_config)
    robot = SO101Follower(robot_cfg)

    logger.info(f"Connecting to robot '{args.robot_id}' on {args.robot_port}...")
    robot.connect()
    logger.info("Robot connected.")

    # ── Build dataset features from robot hardware (needed by build_inference_frame / make_robot_action) ──
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
    print("-" * 60)
    print("  Keyboard shortcuts:")
    print("    [Space]  Emergency Stop (hold position, pause inference)")
    print("    [Enter]  Resume inference")
    print("    [r]      Go to Home / rest position (2s), then pause")
    print("    [→]      Skip current episode")
    print("    [Esc]    Quit")
    print("=" * 60 + "\n")

    try:
        for ep in range(args.num_episodes):
            print(f"\n{'─'*50}")
            print(f"  Episode {ep + 1}/{args.num_episodes}")
            print(f"{'─'*50}")

            model.reset()
            events["exit_early"] = False

            for step in range(max_steps):
                start_t = time.perf_counter()

                # ── Emergency Stop: hold current position, pause inference ──
                if events.get("emergency_stop"):
                    # Read current position and command robot to hold there
                    hold_pos = robot.bus.sync_read("Present_Position")
                    robot.bus.sync_write("Goal_Position", hold_pos)
                    model.reset()
                    print("⚠️  EMERGENCY STOP — holding position. Press [Enter] to resume inference...")
                    while events.get("emergency_stop"):
                        # Keep commanding hold position to resist external forces
                        robot.bus.sync_write("Goal_Position", hold_pos)
                        time.sleep(0.05)
                    print("▶️  Resumed. Continuing inference...")
                    continue

                # ── Go to rest / home, then PAUSE (wait for Enter) ──
                if events.get("go_to_rest"):
                    events["go_to_rest"] = False
                    events["exit_early"] = False
                    print("🏠  Moving to rest position...")
                    go_to_rest_position(
                        robot,
                        rest_position=robot.rest_position,
                        fps=args.fps,
                        duration_s=args.rest_duration,
                        events=events,
                    )
                    model.reset()
                    # Hold at home and wait for Enter to resume inference
                    print("🏠  Home reached. Inference PAUSED. Press [Enter] to resume, [Esc] to quit.")
                    events["emergency_stop"] = True  # reuse e-stop flag to block
                    home_pos = robot.bus.sync_read("Present_Position")
                    while events.get("emergency_stop"):
                        robot.bus.sync_write("Goal_Position", home_pos)
                        time.sleep(0.05)
                        if events.get("stop_recording"):
                            break
                    print("▶️  Resumed from home. Continuing inference...")
                    continue

                # ── Quit / Skip ──
                if events.get("stop_recording"):
                    break
                if events.get("exit_early"):
                    events["exit_early"] = False
                    print("⏭️  Skipping episode.")
                    break

                # ── Policy inference ──
                obs = robot.get_observation()
                obs_frame = build_inference_frame(
                    observation=obs, ds_features=ds_features, device=device
                )

                obs_processed = preprocess(obs_frame)
                action = model.select_action(obs_processed)
                action = postprocess(action)
                action_dict = make_robot_action(action, ds_features)

                robot.send_action(action_dict)

                # ── Maintain target FPS ──
                dt = time.perf_counter() - start_t
                precise_sleep(max(1.0 / args.fps - dt, 0.0))

                if step % (args.fps * 10) == 0 and step > 0:
                    print(f"    Step {step}/{max_steps} ({step / args.fps:.0f}s)")

            if events.get("stop_recording"):
                print("\n[Esc] Quit requested. Stopping.")
                break

            print(f"  Episode {ep + 1} finished.")

    except KeyboardInterrupt:
        print("\n\nCtrl+C detected. Shutting down...")

    finally:
        # ── Cleanup ──
        if robot.is_connected:
            print("Disconnecting robot...")
            robot.disconnect()
        if listener is not None:
            listener.stop()
        print("Done. ✅")


if __name__ == "__main__":
    main()
