# LeRobot SO101 Demo UI

A web-based control interface for running trained ACT (Action Chunking with Transformers) policies on the SO101 robot arm using [LeRobot](https://github.com/huggingface/lerobot).

![Architecture](https://img.shields.io/badge/Robot-SO101-blue)
![Framework](https://img.shields.io/badge/Framework-LeRobot-orange)
![Frontend](https://img.shields.io/badge/Frontend-React-61dafb)
![Backend](https://img.shields.io/badge/Backend-FastAPI-009688)

## Features

- 🤖 **Real-time robot control** via web interface
- 📊 **Live status updates** through WebSocket
- 🎯 **One-click inference** - Execute trained ACT policies
- 📝 **Feedback logging** - Rate task performance
- 🔄 **Mock mode** - Test UI without physical robot

## Architecture

```
┌──────────────┐      ┌──────────────────────────┐
│   Frontend   │◄────►│   FastAPI Backend        │
│   (React)    │ WS   │   (Python)               │
│   Port 5173  │      │   Port 8000              │
└──────────────┘      └────────────┬─────────────┘
                                   │
                      subprocess.Popen
                                   ▼
                      ┌─────────────────────────┐
                      │  lerobot-record         │
                      │  --policy.path=...      │
                      └─────────────────────────┘
                                   │
                                   ▼
                      ┌─────────────────────────┐
                      │   SO101 Robot + Cameras  │
                      └─────────────────────────┘
```

## Prerequisites

- Python 3.10+
- Node.js 18+
- [LeRobot](https://github.com/huggingface/lerobot) installed (`pip install lerobot`)
- SO101 robot arm (for real robot mode)
- Trained ACT model

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/lerobot-demo-ui.git
cd lerobot-demo-ui
```

### 2. Install dependencies

```bash
# Backend
cd backend
pip install -r requirements.txt

# Frontend
cd ../ui
npm install
```

### 3. Configure your robot

Edit `backend/config.py` to match your setup:

```python
ROBOT_CONFIG = {
    "robot_type": "so100_follower",     # or "so101_follower"
    "robot_port": "/dev/ttyACM0",       # Check with: ls /dev/ttyACM*
    "robot_id": "my_robot",
    "cameras": {
        "front": {
            "type": "opencv",
            "index_or_path": 2,          # Check with: v4l2-ctl --list-devices
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

### 4. Run the application

**With Real Robot:**

```bash
# Terminal 1 - Backend
cd backend
uvicorn main_robot:app --reload --port 8000

# Terminal 2 - Frontend
cd ui
npm run dev
```

**Mock Mode (Testing without robot):**

```bash
# Terminal 1 - Backend
cd backend
uvicorn main:app --reload --port 8000

# Terminal 2 - Frontend
cd ui
npm run dev
```

Then open http://localhost:5173 in your browser.

## Usage

1. **Start**: Click the green "Start" button to execute the trained policy
2. **Stop**: Click the red "Stop" button to immediately halt execution
3. **Feedback**: After task completion, rate the performance with 👍 or 👎

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/start` | POST | Start inference |
| `/api/stop` | POST | Stop robot |
| `/api/feedback` | POST | Submit feedback |
| `/api/config` | GET | Get configuration |
| `/ws` | WebSocket | Real-time status |

## LeRobot Workflow

This UI is designed for the **inference/evaluation** phase of the LeRobot workflow:

1. **Record Dataset** (via CLI):
   ```bash
   lerobot-record \
     --robot.type=so100_follower \
     --robot.port=/dev/ttyACM0 \
     --robot.cameras="{ front: {...}, side: {...}}" \
     --dataset.repo_id=${HF_USER}/my_dataset \
     --dataset.num_episodes=10 \
     --dataset.single_task="Pick the blue cube and put on orange plate"
   ```

2. **Train Model** (via CLI):
   ```bash
   python lerobot/scripts/train.py \
     --dataset.repo_id=${HF_USER}/my_dataset \
     --policy.type=act \
     --output_dir=outputs/train/my_model
   ```

3. **Run Inference** (via this UI):
   - Configure `backend/config.py` with your trained model path
   - Start the backend and frontend
   - Click "Start" to execute the policy

## Project Structure

```
.
├── backend/
│   ├── main.py           # Mock backend (for testing)
│   ├── main_robot.py     # Real robot backend
│   ├── config.py         # Robot configuration
│   └── requirements.txt
├── ui/
│   ├── src/
│   │   └── App.tsx       # Main React component
│   └── package.json
├── LEROBOT_INTEGRATION_PLAN.md
└── README.md
```

## Troubleshooting

### Robot not found
```bash
ls /dev/ttyACM*  # Find available ports
# Update robot_port in config.py
```

### Camera not found
```bash
v4l2-ctl --list-devices  # List video devices
# Update camera index_or_path in config.py
```

### LeRobot not installed
```bash
pip install lerobot
```

## License

MIT

## Acknowledgments

- [LeRobot](https://github.com/huggingface/lerobot) by Hugging Face
- [SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) robot design
