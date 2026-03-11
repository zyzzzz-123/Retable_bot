# ReTable Bot — Autonomous Table-Clearing Robot

> A multi-model robotic pipeline that autonomously clears a table by sequentially picking up objects (ex: lemon, tissue box, cup, cloth) using trained ACT policies, with a web-based control UI and safety features.

Built on top of [HuggingFace LeRobot](https://github.com/huggingface/lerobot)

---

## Table of Contents

- [Overview](#overview)
- [Hardware Setup](#hardware-setup)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [How to Run](#how-to-run)
  - [Option A: Web UI (Recommended)](#option-a-web-ui-recommended)
  - [Option B: Command-Line Pipeline](#option-b-command-line-pipeline)
- [What to Expect](#what-to-expect)
- [Custom Components](#custom-components)
- [Trained Models](#trained-models)
- [Troubleshooting](#troubleshooting)
- [Acknowledgments](#acknowledgments)

---

## Overview

**ReTable Bot** is an end-to-end autonomous table-clearing system. A SO-101 robot arm uses four separately-trained [ACT (Action Chunking with Transformers)](https://huggingface.co/docs/lerobot/act) policies, one per object class, orchestrated in a pipeline that:

1. **Plans** the clearing order (optionally via an LLM vision planner using Gemini)
2. **Picks** each object using the corresponding ACT model
3. **Moves** through predefined waypoints to place the object in a bin
4. **Transitions** to the next stage automatically via joint-angle trigger conditions

Safety features include **MediaPipe hand detection** (auto emergency-stop when a human hand enters the workspace) and keyboard/UI controls for E-stop, resume, and go-to-home.

A **React + FastAPI web UI** provides a dashboard to start, stop, and monitor the robot in real time with live camera feeds and state broadcasting via WebSocket.

### System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Web UI (React/TS)                    │
│                    control dashboard                    │
│         Camera feeds · State display · Controls         │
└──────────────────────┬──────────────────────────────────┘
                       │ REST + WebSocket (port 8000)
┌──────────────────────▼──────────────────────────────────┐
│                 FastAPI Backend (Python)                │
│   main_robot.py — subprocess mgmt, IPC, state machine   │
│   config.py    — pipeline stages, hardware config       │
│   llm_planner  — Gemini vision-based task planning      │
└──────────────────────┬──────────────────────────────────┘
                       │ subprocess + /tmp/lerobot_cmd IPC
┌──────────────────────▼──────────────────────────────────┐
│              eval_pipeline.py (Core Engine)             │
│   Loads ACT models sequentially · Runs inference loop   │
│   Monitors trigger conditions · Executes waypoints      │
│   Hand safety detection · Camera frame streaming        │
└──────────────────────┬──────────────────────────────────┘
                       │ LeRobot API
┌──────────────────────▼──────────────────────────────────┐
│          HuggingFace LeRobot Library                    │
│   Robot control · Policy inference · Dataset tools      │
│   Modified: e-stop, go-to-home in control_utils/robot   │
└──────────────────────┬──────────────────────────────────┘
                       │ Serial (Feetech bus)
┌──────────────────────▼──────────────────────────────────┐
│              SO-101 Robot Arm Hardware                  │
│         6-DOF · Feetech STS3215 servos · Gripper        │
│         USB cameras (front + wrist)                     │
└─────────────────────────────────────────────────────────┘
```

---

## Hardware Setup

| Component | Details |
|-----------|---------|
| Robot Arm | SO-101 (6-DOF, Feetech STS3215 servos) |
| Cameras | 2× USB cameras (front view + wrist-mounted) |
| Compute | Ubuntu PC with NVIDIA GPU (CUDA) |
| Connection | Robot via `/dev/ttyACM0`, cameras via `/dev/video4`, `/dev/video8` |

---

## Project Structure

```
lerobot/                          # Root (forked from huggingface/lerobot)
│
├── README.md                     # ← You are here
│
├── ── Core Pipeline Scripts ──
├── eval_pipeline.py              # Multi-model pipeline engine (main inference loop)
├── eval_act_safe.py              # Single-model inference with full safety controls
├── run_act_simple.py             # Minimal single ACT model inference
├── run_smolvla_simple.py         # Minimal SmolVLA model inference
│
├── ── Web UI + Backend ──
├── launch-lerobot-demo-ui/
│   ├── start.sh                  # One-click launcher (bash start.sh)
│   ├── backend/
│   │   ├── main_robot.py         # FastAPI backend — subprocess mgmt, WebSocket
│   │   ├── config.py             # Pipeline stages, hardware config, model paths
│   │   ├── preflight_server.py   # Camera/hardware preflight checker
│   │   ├── llm_planner.py        # LLM vision planner (Gemini via OpenRouter)
│   │   └── _camera_worker.py     # Camera frame streaming worker
│   └── ui/
│       └── src/
│           ├── App.tsx            # Main React control dashboard
│           ├── PreflightCheck.tsx # Camera setup wizard
│           └── RobotArm3D.tsx     # 3D robot visualization
│
├── ── Robot Utility Scripts ──
├── disable_torque.py             # Disable all servo torques (safe manual posing)
├── read_joints.py                # Print current joint positions (normalized)
├── show_joints.py                # Live joint position display
├── save_position.py              # Save current arm pose to JSON waypoint file
├── get_ee_position.py            # Print end-effector position
├── move_to_points.py             # Move arm through a CSV of waypoints
├── view_cameras.py               # Capture snapshots from all cameras
├── view_cameras_live.py          # Live camera feed display
├── takeover.py                   # Record demonstrations via teleoperation
│
├── ── Shell Scripts ──
├── record_dataset.sh             # Record training data
├── eval_act_andy_tube.sh         # Evaluate ACT on specific task
├── eval_pi05_andy_tube.sh        # Evaluate Pi0.5 on specific task
│
├── ── Waypoint / Config Files ──
├── lemon.json                    # Waypoints for lemon pick-and-place
├── tissue.json                   # Waypoints for tissue box
├── cup.json                      # Waypoints for cup
├── cloth.csv                     # Waypoints for cloth (CSV format)
├── saved_positions.json          # Named arm positions
├── point.csv                     # 16-grid prepositions (ROS2 radians)
│
├── ── LeRobot Library (modified fork) ──
├── src/lerobot/                  # Core library with custom safety patches
│   ├── common/robot_devices/     # Modified: e-stop, go-to-home support
│   └── scripts/                  # Modified control utilities
│
├── ── Documentation ──
├── CUSTOM_CHANGES.md             # Detailed log of all modifications to base lerobot
├── runbook.md                    # Operational runbook
├── plan.md / progress.md         # Project planning documents
│
├── ── Original LeRobot ──
├── src/lerobot/                  # Full LeRobot library
├── tests/                        # LeRobot test suite
├── examples/                     # LeRobot examples
├── docs/                         # LeRobot documentation
└── pyproject.toml                # Python package config
```

---

## Installation

### Prerequisites

| Requirement | Details |
|---|---|
| **OS** | Ubuntu Linux |
| **GPU** | NVIDIA GPU with CUDA support (tested on RTX 4090 Laptop, 16 GB VRAM, CUDA 13.0) |
| **Conda env** | Python 3.10+ |
| **Node.js** | 18+ with npm (for the web UI frontend) |
| **Robot** | SO-101 arm connected via USB serial (`/dev/ttyACM*`) |
| **Cameras** | 2× USB cameras (front workspace + wrist-mounted) |

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/zyzzzz-123/Retable_bot.git
cd retable-bot

# 2. Create and activate the conda environment
#    (following the official LeRobot installation guide)
conda create -y -n lerobot python=3.10
conda activate lerobot
conda install ffmpeg -c conda-forge

# 3. Install system build dependencies (Linux)
sudo apt-get install cmake build-essential python3-dev pkg-config \
  libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev \
  libswscale-dev libswresample-dev libavfilter-dev

# 4. Install LeRobot with Feetech motor support (editable mode)
pip install -e ".[feetech]"

# 5. Install backend dependencies for the web UI
pip install fastapi uvicorn[standard] pydantic websockets opencv-python pyserial

# 6. Install the web UI frontend dependencies
cd launch-lerobot-demo-ui/ui
npm install
cd ../..

# 7. Verify hardware connections
lerobot-find-cameras opencv      # List detected cameras
lerobot-find-port                # Find robot serial port
python view_cameras.py           # Visual camera check
python read_joints.py            # Check robot arm
```

> **Note:** The conda environment name `lerobot` matches the official [LeRobot installation guide](https://huggingface.co/docs/lerobot/installation) and is also referenced in `launch-lerobot-demo-ui/start.sh`. If you use a different env name, update `CONDA_ENV` in `start.sh` accordingly.

---

## How to Run

### Option A: Web UI (Recommended)

The web UI provides a full control dashboard with live camera feeds, start/stop controls, and real-time status.

#### Step 1 — Preflight: Configure Cameras & Robot Port

USB cameras get assigned different `/dev/video*` paths on each boot or re-plug. Run preflight first to verify and save the mapping:

```bash
conda activate lerobot
bash ~/lerobot/launch-lerobot-demo-ui/start.sh preflight
```

Open **http://localhost:5173** in a browser. The preflight UI will:
1. Auto-detect all connected cameras and serial ports
2. Show live previews from each camera so you can visually identify them
3. Let you assign roles (click "Front" or "Wrist" on each camera card)
4. Select the robot port (e.g., `/dev/ttyACM0`)
5. Save the mapping to `launch-lerobot-demo-ui/backend/config.py`

After saving, stop the preflight server:
```bash
bash ~/lerobot/launch-lerobot-demo-ui/start.sh stop
```

#### Step 2 — Launch the Main Control UI

```bash
bash ~/lerobot/launch-lerobot-demo-ui/start.sh
```

This starts:
- **Backend** (`main_robot.py`) on `http://localhost:8000` — spawns `eval_pipeline.py` with model pre-warming
- **Frontend** on `http://localhost:5173` — the robot control dashboard

Open **http://localhost:5173** in a browser.

#### Step 3 — Wait for Model Warmup

The backend automatically loads the ACT models, connects to the robot, and initializes cameras. The UI shows a **WARMUP** state with a progress indicator. This takes ~30 seconds.

#### Step 4 — Run Inference

Once the state changes to **READY**, use the UI controls:

| Button | Action |
|---|---|
| **▶ Start** | Begin the multi-model pipeline |
| **⏸ E-Stop** | Emergency stop (disables torques immediately) |
| **🏠 Home** | Smoothly return arm to home position |
| **▶ Resume** | Continue after an E-stop |
| **🔄 Replan** | Ask the LLM to reorder remaining stages |
| **✕ Quit** | Stop inference |

The UI also shows live camera feeds (front + wrist) and hand safety detection status.

#### Stopping the System

```bash
bash ~/lerobot/launch-lerobot-demo-ui/start.sh stop
```

Or press **Ctrl+C** in the terminal where `start.sh` is running.

### Option B: Command-Line Pipeline

Run the full 4-model pipeline directly without the UI:

```bash
conda activate lerobot

python eval_pipeline.py \
  --robot-port /dev/ttyACM0 \
  --robot-id follower_hope \
  --cameras "front:/dev/video4,wrist:/dev/video8" \
  --fps 30 \
  --device cuda \
  --pipeline-config launch-lerobot-demo-ui/backend/config.py \
  --hand-detect
```

**Keyboard controls during execution:**
| Key | Action |
|-----|--------|
| `Spacebar` | Emergency stop (hold) |
| `Enter` | Resume from E-stop |
| `r` | Go to home position (smooth interpolation) |
| `→` (Right arrow) | Skip current episode |
| `Esc` | Quit |

### Utility Scripts

```bash
# Disable torques (safe to manually pose the arm)
python disable_torque.py

# View live camera feeds
python view_cameras_live.py

# Save current arm position as a waypoint
python save_position.py --name "my_position" --file waypoints.json

# Read current joint positions
python read_joints.py
```

---

## What to Expect

### Startup
1. The system loads the first ACT model (Lemon) and initializes the robot arm
2. The arm moves to its home position
3. Camera feeds start streaming

### During Operation
The pipeline runs through **four stages** in sequence:

| Stage | Object | Model | Trigger to Advance |
|-------|--------|-------|--------------------|
| 1 | 🍋 Lemon | `act_lemon_box_0226_merged_160_ckpt_040000` | shoulder_pan < -25° |
| 2 | 🧻 Tissue | `act_tissue_box_0226_merged_80_0226_221249` | shoulder_pan < -25° |
| 3 | 🥤 Cup | `act_cup_box_0301_merged_80` | shoulder_pan < -25° |
| 4 | 🧹 Cloth | `act_cloth_0301_merged_80_0301_200931` | shoulder_pan > 0° |

For each stage:
1. The ACT model runs inference, controlling the arm to pick up the object
2. When the trigger condition is met (arm reaches a certain angle → object is grasped), the model stops
3. The arm follows predefined waypoints to move the object to a drop-off bin
4. The next model loads and the cycle repeats

### Safety
- **Hand detection**: If a human hand is detected in the camera frame, the arm automatically stops and waits
- **E-stop**: Press Spacebar (or the UI Emergency Stop button) to immediately disable all servo torques
- **Go-to-home**: Press `r` (or the UI Return to Home button) to smoothly return the arm to its resting position

### Output
- Live camera frames are streamed to `/tmp/lerobot_frames/` as JPEG files
- Console output shows the current stage, model loading progress, and trigger monitoring
- The web UI displays real-time state (IDLE → RUNNING → E-STOP → DONE) with camera feeds

---

## Custom Components

This project adds the following on top of the base [HuggingFace LeRobot](https://github.com/huggingface/lerobot) library:

| Component | Description |
|-----------|-------------|
| `eval_pipeline.py` | Multi-model sequential pipeline engine |
| `eval_act_safe.py` | Single-model inference with safety controls |
| `launch-lerobot-demo-ui/` | Full-stack React + FastAPI control interface |
| Safety patches in `src/lerobot/` | E-stop, go-to-home support in `control_utils.py`, `robot.py` |
| Hand detection | MediaPipe-based auto-stop when human hand is detected |
| LLM planner | Gemini-powered vision planner for dynamic stage ordering |
| Utility scripts | `disable_torque.py`, `save_position.py`, `read_joints.py`, etc. |
| Waypoint system | JSON/CSV waypoint files for each object's pick-and-place trajectory |

---

## Trained Models

All models are hosted on HuggingFace Hub under the `FrankYuzhe` namespace:

| Model | Task | Training Data |
|-------|------|---------------|
| `FrankYuzhe/act_lemon_box_0226_merged_160_ckpt_040000` | Pick up lemon | 160 demonstrations |
| `FrankYuzhe/act_tissue_box_0226_merged_80_0226_221249` | Pick up tissue | 80 demonstrations |
| `FrankYuzhe/act_cup_box_0301_merged_80` | Pick up cup | 80 demonstrations |
| `FrankYuzhe/act_cloth_0301_merged_80_0301_200931` | Pick up cloth | 80 demonstrations |

Models were trained using the ACT architecture via LeRobot's training pipeline with data collected through teleoperation on the SO-101 arm.

---

## Troubleshooting

### Cameras not detected
- Check USB connections: `ls /dev/video*`
- Run: `lerobot-find-cameras opencv`
- Some `/dev/video*` entries are metadata-only (even numbers are usually the actual streams)

### Robot port not found
- Check USB: `ls /dev/ttyACM*`
- Run: `lerobot-find-port`
- Unplug/replug the robot USB cable

### Frontend won't load
- Ensure npm dependencies are installed: `cd ~/lerobot/launch-lerobot-demo-ui/ui && npm install`
- Check port 5173 isn't in use: `lsof -i :5173`

### Backend crash on startup
- Ensure conda env is active: `conda activate lerobot`
- Check Python deps: `pip install fastapi uvicorn opencv-python`
- Check port 8000 isn't in use: `lsof -i :8000`

### Model warmup fails
- Ensure CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
- Models are auto-downloaded from HuggingFace on first run — ensure internet access

---

## Acknowledgments

- **[HuggingFace LeRobot](https://github.com/huggingface/lerobot)** — The base robotics library this project is forked from
- **ACT (Action Chunking with Transformers)** — The imitation learning policy architecture used for all manipulation tasks

---

*This project is a fork of [huggingface/lerobot](https://github.com/huggingface/lerobot) with custom extensions for a multi-object table-clearing demonstration. For the original LeRobot documentation, see the [upstream repository](https://github.com/huggingface/lerobot).*
