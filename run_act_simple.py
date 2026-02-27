#!/usr/bin/env python3
"""Simple ACT Policy Inference Script.

A minimal standalone script to run an ACT policy on the SO101 robot.
Based on lerobot patterns, without UI/backend integration.

Usage:
    python run_act_simple.py

Keyboard controls:
    Spacebar → Emergency Stop (hold position)
    Enter    → Resume inference
    r        → Go to home/rest position
    Esc      → Quit

Model: FrankYuzhe/act_lemon_box_0226_merged_40_0226_140554
"""

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

# ════════════════════════════════════════════════════════════════════════
#  CONFIGURATION — Edit these values as needed
# ════════════════════════════════════════════════════════════════════════

MODEL = "FrankYuzhe/act_lemon_box_0226_merged_40_0226_140554"
ROBOT_PORT = "/dev/ttyACM1"
ROBOT_ID = "follower_hope"
CAMERAS = {
    "wrist": "/dev/video0",
    "front": "/dev/video6",
}
FPS = 30
EPISODE_TIME = 200  # seconds
NUM_EPISODES = 10
DEVICE = "cuda"
REST_DURATION = 2.0  # seconds for go-to-home

# ════════════════════════════════════════════════════════════════════════


def parse_cameras(camera_dict: dict, width=640, height=480, fps=30) -> dict:
    """Convert camera dict to OpenCVCameraConfig objects."""
    return {
        name: OpenCVCameraConfig(index_or_path=path, width=width, height=height, fps=fps)
        for name, path in camera_dict.items()
    }


def main():
    device = torch.device(DEVICE)
    max_steps = int(EPISODE_TIME * FPS)

    # ── Load model ──
    print("\n" + "=" * 60)
    print("  🤖 Simple ACT Inference")
    print("=" * 60)
    print(f"  Model : {MODEL}")
    print(f"  Robot : {ROBOT_ID} @ {ROBOT_PORT}")
    print(f"  FPS   : {FPS}")
    print(f"  Device: {DEVICE}")
    print("-" * 60)

    logger.info(f"Loading model: {MODEL}")
    model = ACTPolicy.from_pretrained(MODEL)
    model.eval()
    logger.info("Model loaded ✓")

    # ── Create processors ──
    preprocess, postprocess = make_pre_post_processors(model.config, pretrained_path=MODEL)

    # ── Connect robot ──
    camera_config = parse_cameras(CAMERAS, fps=FPS)
    robot_cfg = SO101FollowerConfig(port=ROBOT_PORT, id=ROBOT_ID, cameras=camera_config)
    robot = SO101Follower(robot_cfg)

    logger.info(f"Connecting to robot...")
    robot.connect()
    logger.info("Robot connected ✓")

    # ── Build features ──
    ds_features = {}
    ds_features.update(hw_to_dataset_features(robot.observation_features, OBS_STR, use_video=False))
    ds_features.update(hw_to_dataset_features(robot.action_features, ACTION, use_video=False))

    # ── Keyboard listener ──
    listener, events = init_keyboard_listener()

    print("-" * 60)
    print("  Keyboard shortcuts:")
    print("    [Space]  Emergency Stop (hold position)")
    print("    [Enter]  Resume inference")
    print("    [r]      Go to Home / rest position")
    print("    [→]      Skip current episode")
    print("    [Esc]    Quit")
    print("=" * 60 + "\n")

    try:
        for ep in range(NUM_EPISODES):
            print(f"\n{'─'*50}")
            print(f"  Episode {ep + 1}/{NUM_EPISODES}")
            print(f"{'─'*50}")

            model.reset()
            events["exit_early"] = False
            events["emergency_stop"] = False
            events["go_to_rest"] = False
            events["stop_recording"] = False

            for step in range(max_steps):
                start_t = time.perf_counter()

                # ── Emergency Stop ──
                if events.get("emergency_stop"):
                    hold_pos = robot.bus.sync_read("Present_Position")
                    robot.bus.sync_write("Goal_Position", hold_pos)
                    model.reset()
                    print("⚠️  EMERGENCY STOP — press [Enter] to resume, [r] to go home...", flush=True)
                    while events.get("emergency_stop"):
                        if events.get("go_to_rest"):
                            events["go_to_rest"] = False
                            events["emergency_stop"] = False
                            print("🏠  Moving to home position...", flush=True)
                            go_to_rest_position(robot, rest_position=robot.rest_position,
                                                fps=FPS, duration_s=REST_DURATION, events=events)
                            model.reset()
                            print("🏠  Home reached. Press [Enter] to resume.", flush=True)
                            events["emergency_stop"] = True
                            hold_pos = robot.bus.sync_read("Present_Position")
                        robot.bus.sync_write("Goal_Position", hold_pos)
                        time.sleep(0.05)
                        if events.get("stop_recording"):
                            break
                    print("▶️  Resumed.", flush=True)
                    continue

                # ── Go to rest ──
                if events.get("go_to_rest"):
                    events["go_to_rest"] = False
                    events["exit_early"] = False
                    print("🏠  Moving to home position...", flush=True)
                    go_to_rest_position(robot, rest_position=robot.rest_position,
                                        fps=FPS, duration_s=REST_DURATION, events=events)
                    model.reset()
                    print("🏠  Home reached. Paused. Press [Enter] to resume.", flush=True)
                    events["emergency_stop"] = True
                    hold_pos = robot.bus.sync_read("Present_Position")
                    while events.get("emergency_stop"):
                        robot.bus.sync_write("Goal_Position", hold_pos)
                        time.sleep(0.05)
                        if events.get("stop_recording"):
                            break
                    print("▶️  Resumed.", flush=True)
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
                obs_frame = build_inference_frame(observation=obs, ds_features=ds_features, device=device)
                obs_processed = preprocess(obs_frame)
                action = model.select_action(obs_processed)
                action = postprocess(action)
                action_dict = make_robot_action(action, ds_features)
                robot.send_action(action_dict)

                # ── Maintain target FPS ──
                dt = time.perf_counter() - start_t
                precise_sleep(max(1.0 / FPS - dt, 0.0))

                if step % (FPS * 10) == 0 and step > 0:
                    print(f"    Step {step}/{max_steps} ({step / FPS:.0f}s)", flush=True)

            if events.get("stop_recording"):
                print("\n[Esc] Quit requested. Stopping.", flush=True)
                break

            print(f"  Episode {ep + 1} finished.", flush=True)

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
