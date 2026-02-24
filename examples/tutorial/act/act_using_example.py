"""ACT policy inference example with Emergency Stop and Go-to-Home functionality.

This example demonstrates how to deploy a trained ACT policy on a real robot
with two safety features:

- **Emergency Stop (Spacebar)**: Immediately disables torque on all motors,
  pausing execution. Press Enter to resume.
- **Go to Rest ('r' key)**: Smoothly interpolates the arm back to its
  home/rest position over ~2 seconds.

Key bindings (active when the terminal window has focus):
    Spacebar  → Emergency stop (toggle)
    Enter     → Resume from emergency stop
    r         → Go to rest / home position
    Esc       → Quit
"""

import logging
import time

import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import build_inference_frame, make_robot_action
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from lerobot.utils.control_utils import go_to_rest_position, init_keyboard_listener
from lerobot.utils.robot_utils import precise_sleep

logging.basicConfig(level=logging.INFO)

MAX_EPISODES = 5
MAX_STEPS_PER_EPISODE = 20


def main():
    device = torch.device("mps")  # or "cuda" or "cpu"
    model_id = "<user>/robot_learning_tutorial_act"
    model = ACTPolicy.from_pretrained(model_id)

    dataset_id = "lerobot/svla_so101_pickplace"
    # This only downloads the metadata for the dataset, ~10s of MB even for large-scale datasets
    dataset_metadata = LeRobotDatasetMetadata(dataset_id)
    preprocess, postprocess = make_pre_post_processors(model.config, dataset_stats=dataset_metadata.stats)

    # # find ports using lerobot-find-port
    follower_port = ...  # something like "/dev/tty.usbmodem58760431631"

    # # the robot ids are used the load the right calibration files
    follower_id = ...  # something like "follower_so100"

    # Robot and environment configuration
    # Camera keys must match the name and resolutions of the ones used for training!
    # You can check the camera keys expected by a model in the info.json card on the model card on the Hub
    camera_config = {
        "side": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=30),
        "up": OpenCVCameraConfig(index_or_path=1, width=640, height=480, fps=30),
    }

    robot_cfg = SO100FollowerConfig(port=follower_port, id=follower_id, cameras=camera_config)
    robot = SO100Follower(robot_cfg)
    robot.connect()

    # ── Initialize keyboard listener for e-stop and go-to-rest ──
    listener, events = init_keyboard_listener()

    fps = 30  # Target control frequency

    try:
        for ep in range(MAX_EPISODES):
            print(f"\n{'='*50}")
            print(f"Episode {ep + 1}/{MAX_EPISODES}")
            print(f"  Spacebar = Emergency Stop | Enter = Resume | r = Go Home | Esc = Quit")
            print(f"{'='*50}")

            model.reset()

            for step in range(MAX_STEPS_PER_EPISODE):
                start_t = time.perf_counter()

                # ── Emergency Stop check ──
                if events.get("emergency_stop"):
                    robot.emergency_stop()
                    model.reset()  # Clear buffered action chunks
                    print("⚠️  EMERGENCY STOP — press Enter to resume...")
                    while events.get("emergency_stop"):
                        time.sleep(0.1)
                    robot.resume()
                    print("▶️  Resumed.")
                    continue

                # ── Go to rest check ──
                if events.get("go_to_rest"):
                    events["go_to_rest"] = False
                    events["exit_early"] = False
                    print("🏠  Moving to rest position...")
                    go_to_rest_position(
                        robot,
                        rest_position=robot.rest_position,
                        fps=fps,
                        duration_s=2.0,
                        events=events,
                    )
                    print("🏠  Reached rest position.")
                    break  # End this episode

                # ── Quit check ──
                if events.get("stop_recording") or events.get("exit_early"):
                    events["exit_early"] = False
                    break

                # ── Normal policy inference ──
                obs = robot.get_observation()
                obs_frame = build_inference_frame(
                    observation=obs, ds_features=dataset_metadata.features, device=device
                )

                obs_processed = preprocess(obs_frame)
                action = model.select_action(obs_processed)
                action = postprocess(action)
                action_dict = make_robot_action(action, dataset_metadata.features)

                robot.send_action(action_dict)

                # ── Maintain target fps ──
                dt = time.perf_counter() - start_t
                precise_sleep(max(1 / fps - dt, 0.0))

            if events.get("stop_recording"):
                print("\nEsc pressed — stopping.")
                break

            print("Episode finished! Starting new episode...")

    finally:
        if robot.is_connected:
            robot.disconnect()
        if listener is not None:
            listener.stop()
        print("Done.")


if __name__ == "__main__":
    main()
