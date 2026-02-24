# 🤖 Custom Changes — ReTable Bot (UW MSTI Launch Project)

> **Team:** ReTable Bot — University of Washington MSTI Program  
> **Base:** [HuggingFace LeRobot](https://github.com/huggingface/lerobot) @ commit `b2ff21962478eea53c2db50333302fda3cb19b7f`  
> **Purpose:** Added **Emergency Stop** and **Go-to-Home** safety features for autonomous ACT policy deployment on SO-100/101 arms during pick-and-place tasks.

---

## What Was Changed

We added two operator-safety features that activate during **autonomous policy inference** (i.e., when a trained ACT policy controls the robot on its own). These do **not** affect the manual dataset-collection pipeline (`lerobot-record`), which remains untouched.

| File | Change |
|------|--------|
| `src/lerobot/utils/control_utils.py` | Added `emergency_stop`, `go_to_rest` keyboard events and `go_to_rest_position()` function |
| `src/lerobot/robots/robot.py` | Added `emergency_stop()`, `resume()`, `rest_position` to abstract `Robot` base class |
| `src/lerobot/robots/so_follower/so_follower.py` | SO-100/101 hardware implementation: torque disable/enable + default folded rest pose |
| `examples/tutorial/act/act_using_example.py` | ACT inference tutorial rewritten with both safety features integrated |

---

## How to Use

### Keyboard Shortcuts

During autonomous policy execution (e.g., running `act_using_example.py`), the following keys are active **when the terminal has focus**:

| Key | Action |
|-----|--------|
| **Spacebar** | 🛑 **Emergency Stop** — immediately disables torque on all motors. The arm goes limp and stops moving. |
| **Enter** | ▶️ **Resume** — re-enables torque and continues execution from where it paused. |
| **r** | 🏠 **Go to Home** — smoothly moves the arm to its rest/home position over ~2 seconds, then ends the current episode. |
| **Esc** | 🚪 **Quit** — stops the program entirely. |

> **Note:** The existing keys (→ for skip, ← for re-record) still work as before.

### Running the Example

```bash
# 1. Edit act_using_example.py to set your:
#    - device (mps / cuda / cpu)
#    - model_id (your trained ACT policy on HuggingFace Hub)
#    - dataset_id
#    - follower_port and follower_id
#    - camera config

# 2. Run it
python examples/tutorial/act/act_using_example.py
```

Once running, the robot will execute the policy autonomously. Use the keyboard shortcuts above to intervene as needed.

---

## Customizing the Rest Position

The default rest position for SO-100/101 is all body joints at midpoint (0.0) with the gripper half-open (50.0):

```python
{
    "shoulder_pan.pos": 0.0,
    "shoulder_lift.pos": 0.0,
    "elbow_flex.pos": 0.0,
    "wrist_flex.pos": 0.0,
    "wrist_roll.pos": 0.0,
    "gripper.pos": 50.0,
}
```

To change this, override the `rest_position` property in `so_follower.py`, or pass a custom dict directly:

```python
from lerobot.utils.control_utils import go_to_rest_position

my_home = {
    "shoulder_pan.pos": -10.0,
    "shoulder_lift.pos": 20.0,
    "elbow_flex.pos": -30.0,
    "wrist_flex.pos": 15.0,
    "wrist_roll.pos": 0.0,
    "gripper.pos": 100.0,  # fully open
}

go_to_rest_position(robot, rest_position=my_home, fps=30, duration_s=2.0, events=events)
```

---

## Architecture Overview

```
Keyboard Listener (pynput, runs in background thread)
    │
    ├── Spacebar pressed  ──►  events["emergency_stop"] = True
    │                              │
    │                              ▼
    │                     robot.emergency_stop()
    │                     (disables torque via Feetech bus)
    │                              │
    │                     waits for Enter key...
    │                              │
    │                     robot.resume()
    │                     (re-enables torque)
    │
    ├── 'r' pressed  ──►  events["go_to_rest"] = True
    │                              │
    │                              ▼
    │                     go_to_rest_position()
    │                     (reads current pos → interpolates → rest pose over 2s)
    │
    └── Esc pressed  ──►  events["stop_recording"] = True → exit
```

---

## Special Notes

1. **Clone this repo** (not the original HuggingFace one)
2. Install as usual: `pip install -e ".[dev]"`
3. The safety features are automatically available whenever you use `init_keyboard_listener()` from `control_utils`
4. **Always test e-stop on your specific arm before running any new policy** — verify it goes limp correctly

---

*Last updated: February 2026*
