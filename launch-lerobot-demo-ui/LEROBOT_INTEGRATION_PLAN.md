# LeRobot SO101 Robot Integration Plan

## Overview

This document outlines the integration of the frontend UI with a real SO101 robot using the LeRobot system for running trained ACT policies.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Same Laptop                          │
│  ┌──────────────┐      ┌──────────────────────────┐    │
│  │   Frontend   │◄────►│   FastAPI Backend        │    │
│  │   (React)    │ WS   │   (Python)               │    │
│  │   Port 5173  │      │   Port 8000              │    │
│  └──────────────┘      └────────────┬─────────────┘    │
│                                      │                  │
│                         subprocess.Popen               │
│                                      ▼                  │
│                        ┌─────────────────────────┐     │
│                        │  lerobot-record         │     │
│                        │  --policy.path=...      │     │
│                        │  (inference mode)       │     │
│                        └─────────────────────────┘     │
│                                      │                  │
│                                      ▼                  │
│                        ┌─────────────────────────┐     │
│                        │   SO101 Robot + Cameras  │     │
│                        └─────────────────────────┘     │
└─────────────────────────────────────────────────────────┘
```

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Policy Storage | Local path | Policy stored at `outputs/train/.../pretrained_model` |
| Status Display | Simplified | Parse lerobot output for user-friendly status messages |
| Pause/Resume | Removed | LeRobot CLI doesn't support pause/resume natively |
| Mock Backend | Preserved | Keep `main.py` for documentation and testing without robot |

## LeRobot Command for Inference

The backend executes this command when the user clicks "Start":

```bash
lerobot-record \
  --robot.type=so100_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=my_robot \
  --robot.cameras="{ front: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, side: {type: opencv, index_or_path: 8, width: 640, height: 480, fps: 30}}" \
  --display_data=true \
  --dataset.repo_id=local/eval_run \
  --dataset.num_episodes=1 \
  --dataset.single_task="Pick the blue cube and put on orange plate" \
  --policy.path=outputs/train/act_0122_1951/checkpoints/last/pretrained_model
```

## File Structure

```
launch_demo/
├── LEROBOT_INTEGRATION_PLAN.md  # This document
├── backend/
│   ├── main.py              # Mock backend (preserved for testing)
│   ├── main_robot.py        # Real robot backend (NEW)
│   ├── config.py            # Robot configuration (NEW)
│   └── requirements.txt
├── ui/
│   └── src/
│       ├── App.tsx          # Updated (removed Pause/Resume)
│       └── ...
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/start` | POST | Start inference (spawns lerobot process) |
| `/api/stop` | POST | Stop robot (SIGTERM to lerobot process) |
| `/api/feedback` | POST | Log user feedback |
| `/ws` | WebSocket | Real-time status updates |

## Status States

| State | Description |
|-------|-------------|
| `READY` | Robot connected, waiting for start command |
| `WORKING` | Inference running, robot executing task |
| `DONE` | Task completed successfully |
| `ERROR` | Error occurred (connection failed, etc.) |

## Configuration

Edit `backend/config.py` to match your robot setup:

```python
ROBOT_CONFIG = {
    "robot_type": "so100_follower",     # or "so101_follower"
    "robot_port": "/dev/ttyACM0",       # Check with `ls /dev/ttyACM*`
    "robot_id": "my_robot",
    "cameras": {
        "front": {
            "type": "opencv",
            "index_or_path": 2,          # Check with `v4l2-ctl --list-devices`
            "width": 640,
            "height": 480,
            "fps": 30
        },
        "side": {
            "type": "opencv",
            "index_or_path": 8,
            "width": 640,
            "height": 480,
            "fps": 30
        }
    },
    "task": "Pick the blue cube and put on orange plate",
    "policy_path": "outputs/train/act_0122_1951/checkpoints/last/pretrained_model"
}
```

## Usage

### Run with Mock Backend (Testing/Demo without robot)

```bash
# Terminal 1: Backend
cd backend
uvicorn main:app --reload --port 8000

# Terminal 2: Frontend
cd ui
npm run dev
```

### Run with Real Robot

```bash
# Terminal 1: Backend
cd backend
uvicorn main_robot:app --reload --port 8000

# Terminal 2: Frontend  
cd ui
npm run dev
```

Then open http://localhost:5173 in your browser.

## Prerequisites

1. **LeRobot installed**: `pip install lerobot`
2. **Trained ACT model**: Located at the path specified in `config.py`
3. **Robot connected**: SO101 robot connected via USB
4. **Cameras working**: Verify camera indices with `v4l2-ctl --list-devices`

## Troubleshooting

### Robot not found
```bash
# List available ports
ls /dev/ttyACM*
# Update robot_port in config.py
```

### Camera not found
```bash
# List video devices
v4l2-ctl --list-devices
# Update camera index_or_path in config.py
```

### Policy not found
Ensure the policy path exists:
```bash
ls outputs/train/act_0122_1951/checkpoints/last/pretrained_model
```
