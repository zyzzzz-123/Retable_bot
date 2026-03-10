"""
Backend for LeRobot SO101 — wraps eval_act_safe.py in pre-warm mode.

Lifecycle:
    Backend startup  → spawns eval_act_safe.py --wait-for-start
                       (loads model + connects robot, ~30s)
    WARMUP_COMPLETE  → state = READY, model & robot are hot
    User clicks Start → writes START to control file → instant inference
    Stop / Reset / Resume → same control-file IPC as before
    Quit → writes QUIT, respawns process to keep warm

Usage:
    uvicorn main_robot:app --port 8000
"""

import asyncio
import csv
import json
import os
import re
import signal
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

from config import (build_inference_command, ROBOT_CONFIG, CONTROL_FILE, LEROBOT_DIR, FRAME_DIR,
                     HAND_DETECT_ENABLED, POINTS_CSV, PIPELINE_STAGES,
                     LLM_PLANNER_ENABLED)
from llm_planner import plan_from_camera, plan_to_dict, LLMPlannerError

app = FastAPI(title="LeRobot SO101 Control Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== Points CSV Loading ====================

def _load_points_csv(csv_path: str):
    """Load joint positions from point.csv (header + N data rows)."""
    points = []
    joint_names = []
    try:
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i == 0:
                    joint_names = [c.strip() for c in row]
                    continue
                if not row or all(not c.strip() for c in row):
                    continue
                points.append([float(x) for x in row])
    except Exception as e:
        print(f"[WARN] Could not load points CSV ({csv_path}): {e}")
    return points, joint_names


POINTS, JOINT_NAMES = _load_points_csv(POINTS_CSV)
print(f"[POINTS] Loaded {len(POINTS)} positions from {POINTS_CSV}")

# ==================== Grid Frame Processing ====================

_grid_executor = ThreadPoolExecutor(max_workers=1)


def _process_grid_frame(path: str) -> bytes:
    """Read a camera frame, crop it, overlay 4×4 grid with numbers 1-16.
    Runs in a thread-pool so it doesn't block the async event loop.
    Matches the crop/grid logic in capture_video4.sh."""
    import cv2  # lazy import — CPU-only ops, no CUDA needed

    frame = cv2.imread(path)
    if frame is None:
        raise ValueError("Cannot read frame")

    height, width = frame.shape[:2]
    # Crop: horizontal 5%-55%, vertical 30%-85%
    x_start, x_end = int(width * 0.10), int(width * 0.60)
    y_start, y_end = int(height * 0.25), int(height * 0.8)
    cropped = frame[y_start:y_end, x_start:x_end]

    h, w = cropped.shape[:2]
    # Draw 3 vertical + 3 horizontal green lines → 4×4 grid
    for i in range(1, 4):
        cv2.line(cropped, (int(w * i / 4), 0), (int(w * i / 4), h), (0, 255, 0), 1)
        cv2.line(cropped, (0, int(h * i / 4)), (w, int(h * i / 4)), (0, 255, 0), 1)

    # Number each cell 1-16
    cell_w, cell_h = w / 4, h / 4
    font = cv2.FONT_HERSHEY_SIMPLEX
    for row in range(4):
        for col in range(4):
            num = row * 4 + col + 1
            text = str(num)
            (tw, th), _ = cv2.getTextSize(text, font, 0.8, 2)
            xc = int((col + 0.5) * cell_w)
            yc = int((row + 0.5) * cell_h)
            cv2.putText(cropped, text, (xc - tw // 2, yc + th // 2),
                        font, 0.8, (0, 255, 0), 2)

    _, buf = cv2.imencode(".jpg", cropped, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes()


# Arm-move state (tracks whether we're waiting for a GOTO to complete)
_arm_moving = False

# ==================== State Management ====================

class RobotState:
    def __init__(self):
        self.state = "WARMUP"      # WARMUP | READY | WORKING | PAUSED | HOMED | DONE | ERROR
        self.step = ""
        self.progress = 0
        self.message = "Starting up..."
        self.process: Optional[asyncio.subprocess.Process] = None
        self.session_log = []
        self._warmup_ready = False  # True once WARMUP_COMPLETE seen
        # Hand safety detection state
        self.hand_detect_enabled = HAND_DETECT_ENABLED
        self.hand_detected = False
        self.auto_stopped = False
        # Pipeline state
        self.pipeline_stage = ""       # Current stage name (e.g. "Lemon", "Tissue")
        self.pipeline_stage_idx = 0    # 0-based index
        self.pipeline_total = len(PIPELINE_STAGES)
        self.pipeline_status = ""      # "inference" | "waypoints" | "loading" | ""
        # LLM Planner state
        self.llm_plan = None           # dict: {"Lemon": {"status":"done","reason":"..."}, ...}
        self.llm_planning = False      # True while LLM API call is in progress
        self.llm_plan_error = ""       # Error message if LLM call failed
        self.llm_stages_to_run = []    # ["Tissue", "Cup"] — filtered stage names
        # Per-stage status list for frontend pipeline display
        # Each entry: {"name": "Lemon", "llm_status": "done"|"todo"|"", "exec_status": "pending"|"active"|"done"|"skipped"}
        self.pipeline_stages_info = self._build_stages_info()

    def _build_stages_info(self):
        """Build initial empty pipeline_stages_info (before LLM plan)."""
        # Before LLM plan, show nothing — stages will appear after plan completes
        return []

    def update_stages_from_plan(self):
        """After LLM plan, build stages_info — only include objects that exist (todo or done).
        not_found objects are excluded entirely."""
        self.pipeline_stages_info = []
        for s in PIPELINE_STAGES:
            name = s["name"]
            llm_status = ""
            if self.llm_plan and name in self.llm_plan:
                llm_status = self.llm_plan[name]["status"]
            # Skip not_found objects — they don't appear in the pipeline at all
            if llm_status == "not_found":
                continue
            # done → show as done (skipped), todo → show as pending (will execute)
            if llm_status == "done":
                exec_status = "done"
            elif name in self.llm_stages_to_run:
                exec_status = "pending"
            else:
                exec_status = "pending"
            self.pipeline_stages_info.append({
                "name": name,
                "llm_status": llm_status,
                "exec_status": exec_status,
            })
        self.pipeline_total = len([s for s in self.pipeline_stages_info if s["exec_status"] == "pending"])

    def mark_stage_active(self, stage_name: str):
        """Mark a stage as active. Any previously active stage becomes done."""
        for s in self.pipeline_stages_info:
            if s["exec_status"] == "active":
                s["exec_status"] = "done"
            if s["name"] == stage_name:
                s["exec_status"] = "active"

    def mark_stage_done(self, stage_name: str):
        """Mark a stage as done (completed execution)."""
        for s in self.pipeline_stages_info:
            if s["name"] == stage_name:
                s["exec_status"] = "done"
                break

    def to_dict(self):
        return {
            "state": self.state,
            "step": self.step,
            "progress": self.progress,
            "message": self.message,
            "hand_detect": self.hand_detect_enabled,
            "hand_detected": self.hand_detected,
            "auto_stopped": self.auto_stopped,
            "pipeline_stage": self.pipeline_stage,
            "pipeline_stage_idx": self.pipeline_stage_idx,
            "pipeline_total": self.pipeline_total,
            "pipeline_status": self.pipeline_status,
            "pipeline_stages_info": self.pipeline_stages_info,
            "llm_planning": self.llm_planning,
            "llm_plan_error": self.llm_plan_error,
        }
    
    def log_event(self, event: str, details: dict = None):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event": event,
            "state": self.state,
            **(details or {}),
        }
        self.session_log.append(entry)
        print(f"[LOG] {entry}")

    def reset_to_ready(self):
        self.state = "READY"
        self.step = ""
        self.progress = 0
        self.message = "Model loaded. Ready to start."
        self._warmup_ready = True


robot = RobotState()
connected_clients: list[WebSocket] = []

# ==================== Helpers ====================

async def broadcast_state():
    """Send current state to all connected WebSocket clients."""
    msg = json.dumps(robot.to_dict())
    dead = []
    for ws in connected_clients:
        try:
            await ws.send_text(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        connected_clients.remove(ws)


def send_control_command(cmd: str):
    """Write a command to the control file."""
    try:
        with open(CONTROL_FILE, "w") as f:
            f.write(cmd)
        print(f"[CTRL] Wrote '{cmd}' → {CONTROL_FILE}")
    except Exception as e:
        print(f"[CTRL] Error: {e}")

# ==================== Output Parser ====================

def parse_output(line: str) -> dict:
    """Parse eval_pipeline.py stdout and return state updates."""
    lo = line.lower()
    raw = line.strip()
    
    # ── Warmup phases ──
    if "warmup_phase: loading_model" in lo:
        return {"state": "WARMUP", "step": "Loading model", "message": "Loading ACT model to GPU...", "progress": 10}
    if "warmup_phase: model_loaded" in lo:
        return {"state": "WARMUP", "step": "Model loaded", "message": "Model loaded. Connecting robot...", "progress": 40}
    if "warmup_phase: connecting_robot" in lo:
        return {"state": "WARMUP", "step": "Connecting robot", "message": "Connecting to robot & cameras...", "progress": 50}
    if "warmup_phase: robot_connected" in lo:
        return {"state": "WARMUP", "step": "Robot connected", "message": "Robot connected. Finishing setup...", "progress": 80}

    # ── Lifecycle signals ──
    if "warmup_complete" in lo:
        return {"_signal": "WARMUP_COMPLETE"}
    if "ready_for_start" in lo:
        return {"_signal": "READY_FOR_START"}
    if "inference_started" in lo:
        return {"state": "WORKING", "step": "Running", "message": "Inference started!", "progress": 0,
                "_pipeline_status": "inference"}
    if "inference_done" in lo:
        return {"_signal": "INFERENCE_DONE"}
    if "pipeline_restart" in lo:
        return {"_signal": "PIPELINE_RESTART"}
    # PIPELINE_RETRY:StageName
    m = re.match(r"PIPELINE_RETRY:(.+)", raw) if raw.startswith("PIPELINE_RETRY:") else None
    if m:
        name = m.group(1)
        return {"_signal": "PIPELINE_RETRY", "_pipeline_stage": name}
    if "pipeline_complete" in lo:
        return {"_signal": "PIPELINE_COMPLETE"}

    # ── Pipeline stage signals ──

    # STAGE_LOADING:StageName
    m = re.match(r"STAGE_LOADING:(.+)", raw)
    if m:
        name = m.group(1)
        return {"state": "WORKING", "step": f"Loading {name}", 
                "message": f"Loading model for [{name}]...", "progress": 0,
                "_pipeline_stage": name, "_pipeline_status": "loading"}

    # STAGE_LOADED:StageName
    m = re.match(r"STAGE_LOADED:(.+)", raw)
    if m:
        name = m.group(1)
        return {"state": "WORKING", "step": f"{name} ready",
                "message": f"Model [{name}] loaded, starting inference...",
                "_pipeline_stage": name, "_pipeline_status": "inference"}

    # STAGE_STARTED:StageName
    m = re.match(r"STAGE_STARTED:(.+)", raw)
    if m:
        name = m.group(1)
        return {"state": "WORKING", "step": f"[{name}] Running",
                "message": f"Running [{name}] inference...", "progress": 0,
                "_pipeline_stage": name, "_pipeline_status": "inference"}

    # STAGE_TRIGGERED:StageName
    m = re.match(r"STAGE_TRIGGERED:(.+)", raw)
    if m:
        name = m.group(1)
        return {"state": "WORKING", "step": f"[{name}] Triggered",
                "message": f"[{name}] trigger condition met → moving to waypoints",
                "_pipeline_stage": name, "_pipeline_status": "waypoints"}

    # STAGE_COMPLETE:StageName:reason
    m = re.match(r"STAGE_COMPLETE:(.+):(.+)", raw)
    if m:
        name, reason = m.group(1), m.group(2)
        return {"step": f"[{name}] Complete",
                "message": f"Stage [{name}] complete ({reason})",
                "_pipeline_stage": name, "_stage_complete": name}

    # WAYPOINTS_STARTED:count
    m = re.match(r"WAYPOINTS_STARTED:(\d+)", raw)
    if m:
        count = m.group(1)
        return {"state": "WORKING", "step": "Waypoints",
                "message": f"Moving through {count} waypoints...", "progress": 0,
                "_pipeline_status": "waypoints"}

    # WAYPOINT_STARTED:idx:name
    m = re.match(r"WAYPOINT_STARTED:(\d+):(.+)", raw)
    if m:
        idx, name = m.group(1), m.group(2)
        return {"step": f"Waypoint {idx}", "message": f"Moving to waypoint {name}..."}

    # WAYPOINT_DONE:idx:name
    m = re.match(r"WAYPOINT_DONE:(\d+):(.+)", raw)
    if m:
        idx, name = m.group(1), m.group(2)
        return {"step": f"Waypoint {idx}", "message": f"Reached waypoint {name}"}

    # WAYPOINTS_DONE:count
    m = re.match(r"WAYPOINTS_DONE:(\d+)", raw)
    if m:
        return {"step": "Waypoints done", "message": "All waypoints reached",
                "_pipeline_status": "inference"}

    # PIPELINE_LOADED:count
    m = re.match(r"PIPELINE_LOADED:(\d+)", raw)
    if m:
        return {}  # informational only

    # ── GOTO point signals ──
    if "goto_started:" in lo:
        m = re.search(r"goto_started:(\S+)", lo)
        idx = m.group(1) if m else "?"
        return {"state": "HOMED", "step": f"Moving to {idx}", "message": f"Moving to position {idx}..."}
    if "goto_done:" in lo:
        m = re.search(r"goto_done:(\S+)", lo)
        idx = m.group(1) if m else "?"
        return {"state": "HOMED", "step": "Position reached", "message": f"At position {idx}, paused"}

    # ── Hand safety detection ──
    if "hand_detect_on" in lo:
        return {"_signal": "HAND_DETECT_ON"}
    if "hand_detect_off" in lo:
        return {"_signal": "HAND_DETECT_OFF"}
    if "hand detected" in lo and "auto" in lo:
        return {"state": "PAUSED", "step": "Hand E-Stop", "message": "🖐️ Hand detected — auto emergency stop",
                "_hand": True, "_auto": True}
    if "hand cleared" in lo and "auto" in lo:
        return {"state": "WORKING", "step": "Running", "message": "✅ Hand cleared — auto resumed",
                "_hand": False, "_auto": False}
    
    # ── Runtime status ──
    if "emergency stop" in lo and "auto" not in lo:
        return {"state": "PAUSED", "step": "E-Stop", "message": "Emergency stop — holding position",
                "_auto": False}
    if "home reached" in lo:
        return {"state": "HOMED", "step": "Home", "message": "At home position, inference paused"}
    if "resumed" in lo or "continuing inference" in lo:
        return {"state": "WORKING", "step": "Running", "message": "Inference resumed",
                "_hand": False, "_auto": False, "_pipeline_status": "inference"}
    
    if "episode" in lo and "/" in lo:
        return {"step": "Episode", "message": raw[:80]}

    if "step " in lo:
        m = re.search(r"step\s+(\d+)/(\d+)", lo)
        if m:
            cur, total = int(m.group(1)), int(m.group(2))
            pct = min(int(cur / total * 100), 99)
            return {"step": "Inference", "message": f"Step {cur}/{total}", "progress": pct}
    
    if "episode" in lo and "finished" in lo:
        return {"step": "Episode done", "message": raw[:80], "progress": 100}
    
    if "done" in lo and "✅" in line:
        return {"_signal": "PROCESS_DONE"}
    
    if "error" in lo or "traceback" in lo:
        return {"state": "ERROR", "message": raw[:120]}
    
    return {}

# ==================== Process Management ====================

async def spawn_warmup_process():
    """Spawn eval_act_safe.py in wait-for-start mode."""
    global robot
    
    # Clean stale control file
    if os.path.exists(CONTROL_FILE):
        os.remove(CONTROL_FILE)
    
    cmd = build_inference_command()
    print(f"[PROC] Spawning warmup: {' '.join(cmd)}")
    
    robot.state = "WARMUP"
    robot.step = "Starting"
    robot.progress = 0
    robot.message = "Loading model & connecting robot..."
    robot._warmup_ready = False
    robot.log_event("WARMUP_START", {"command": " ".join(cmd)})
    await broadcast_state()
    
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=LEROBOT_DIR,
            start_new_session=True,
        )
        robot.process = proc

        # Auto-accept calibration prompts (press Enter)
        try:
            proc.stdin.write(b"\n" * 10)
            await proc.stdin.drain()
            proc.stdin.close()
        except Exception:
            pass

        while True:
            line_bytes = await proc.stdout.readline()
            if not line_bytes:
                break
            line = line_bytes.decode("utf-8", errors="replace")
            print(f"[OUT] {line.rstrip()}")
                    
            updates = parse_output(line)
            if not updates:
                continue

            # Handle special signals
            sig = updates.pop("_signal", None)
            if sig == "WARMUP_COMPLETE":
                robot.reset_to_ready()
                robot.log_event("WARMUP_COMPLETE")
                await broadcast_state()
                continue
            if sig == "READY_FOR_START":
                robot.reset_to_ready()
                robot.log_event("READY_FOR_START")
                await broadcast_state()
                continue
            if sig == "INFERENCE_DONE":
                robot.state = "DONE"
                robot.step = "Complete"
                robot.progress = 100
                robot.message = "Pipeline complete"
                robot.hand_detected = False
                robot.auto_stopped = False
                robot.pipeline_status = ""
                robot.log_event("INFERENCE_DONE")
                await broadcast_state()
                continue
            if sig == "PIPELINE_RESTART":
                robot.state = "WORKING"
                robot.step = "Restarting"
                robot.progress = 0
                robot.message = "Restarting pipeline from stage 1..."
                robot.pipeline_stage_idx = 0
                robot.pipeline_status = "loading"
                robot.log_event("PIPELINE_RESTART")
                await broadcast_state()
                continue
            if sig == "PIPELINE_RETRY":
                stage_name = updates.pop("_pipeline_stage", robot.pipeline_stage)
                robot.state = "WORKING"
                robot.step = f"Retrying {stage_name}"
                robot.progress = 0
                robot.message = f"Retrying stage [{stage_name}]..."
                robot.pipeline_stage = stage_name
                robot.pipeline_status = "loading"
                robot.log_event("PIPELINE_RETRY", {"stage": stage_name})
                await broadcast_state()
                continue
            if sig == "PIPELINE_COMPLETE":
                robot.state = "DONE"
                robot.step = "Pipeline Complete"
                robot.progress = 100
                robot.message = "All stages complete!"
                robot.pipeline_status = ""
                robot.log_event("PIPELINE_COMPLETE")
                await broadcast_state()
                continue
            if sig == "PROCESS_DONE":
                continue
            if sig == "HAND_DETECT_ON":
                robot.hand_detect_enabled = True
                robot.log_event("HAND_DETECT_ON")
                await broadcast_state()
                continue
            if sig == "HAND_DETECT_OFF":
                robot.hand_detect_enabled = False
                robot.hand_detected = False
                robot.auto_stopped = False
                robot.log_event("HAND_DETECT_OFF")
                await broadcast_state()
                continue

            # Update hand detection sub-state
            if "_hand" in updates:
                robot.hand_detected = updates.pop("_hand")
            if "_auto" in updates:
                robot.auto_stopped = updates.pop("_auto")

            # Update pipeline sub-state
            if "_pipeline_stage" in updates:
                robot.pipeline_stage = updates.pop("_pipeline_stage")
                # Derive stage index from name using pipeline_stages_info (filtered list)
                for idx, s in enumerate(robot.pipeline_stages_info):
                    if s["name"] == robot.pipeline_stage:
                        robot.pipeline_stage_idx = idx
                        break
            if "_pipeline_status" in updates:
                new_status = updates.pop("_pipeline_status")
                robot.pipeline_status = new_status
                # Mark stage as active when loading/inference starts
                if new_status in ("loading", "inference") and robot.pipeline_stage:
                    robot.mark_stage_active(robot.pipeline_stage)
            # Check for stage completion signal
            if "_stage_complete" in updates:
                completed_name = updates.pop("_stage_complete")
                robot.mark_stage_done(completed_name)

            # Apply normal state updates
            for k in ("state", "step", "progress", "message"):
                if k in updates:
                    setattr(robot, k, updates[k])
            await broadcast_state()
            
        rc = await proc.wait()
        print(f"[PROC] Exited with code {rc}")

        if rc != 0 and rc != -15 and rc != -9:  # not killed by signal
            robot.state = "ERROR"
            robot.message = f"Process exited with code {rc}"
            robot.log_event("PROC_ERROR", {"return_code": rc})
            await broadcast_state()
        
    except Exception as e:
        robot.state = "ERROR"
        robot.message = str(e)[:120]
        robot.log_event("PROC_ERROR", {"error": str(e)})
        await broadcast_state()
    finally:
        robot.process = None


async def kill_process():
    """Kill the running subprocess tree."""
    proc = robot.process
    if not proc:
        return
    print("[PROC] Killing process group...")
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGTERM)
        try:
            await asyncio.wait_for(proc.wait(), timeout=3)
        except asyncio.TimeoutError:
            os.killpg(pgid, signal.SIGKILL)
            await proc.wait()
    except ProcessLookupError:
        pass
    except Exception as e:
        print(f"[PROC] Kill error: {e}")
        try:
            proc.kill()
            await proc.wait()
        except Exception:
            pass
    finally:
        robot.process = None

# ==================== LLM Planner Helper ====================

async def run_llm_planning() -> dict:
    """Run LLM vision planning and update robot state.

    Returns the plan dict on success.
    Raises LLMPlannerError on failure (robot state is updated with error).
    """
    robot.llm_planning = True
    robot.llm_plan_error = ""
    robot.llm_plan = None
    robot.llm_stages_to_run = []
    robot.message = "🧠 Vision planner analyzing scene..."
    robot.log_event("LLM_PLAN_START")
    await broadcast_state()

    try:
        result = await plan_from_camera()
        robot.llm_plan = plan_to_dict(result)
        robot.llm_stages_to_run = result.stages_to_run
        robot.llm_planning = False
        robot.update_stages_from_plan()

        if not result.stages_to_run:
            robot.llm_plan_error = ""
            robot.message = "All objects already done — nothing to do!"
            robot.log_event("LLM_PLAN_DONE", {
                "plan": robot.llm_plan, "stages": [], "result": "all_done"
            })
        else:
            robot.message = f"Plan: {', '.join(result.stages_to_run)}"
            robot.log_event("LLM_PLAN_DONE", {
                "plan": robot.llm_plan, "stages": result.stages_to_run
            })
        await broadcast_state()
        return robot.llm_plan

    except LLMPlannerError as e:
        robot.llm_planning = False
        robot.llm_plan_error = str(e)
        robot.message = f"LLM planner error: {str(e)[:80]}"
        robot.log_event("LLM_PLAN_ERROR", {"error": str(e)})
        await broadcast_state()
        raise


async def _auto_plan_and_start():
    """Auto-triggered after warmup: run LLM plan → send PLAN + START."""
    try:
        await run_llm_planning()
    except LLMPlannerError:
        print(f"[AUTO] LLM planning failed: {robot.llm_plan_error}")
        # Fall back to READY so user can still use QUICK RUN
        robot.state = "READY"
        robot.step = ""
        robot.message = f"LLM failed — use QUICK RUN to test stages manually"
        await broadcast_state()
        return

    if not robot.llm_stages_to_run:
        robot.state = "DONE"
        robot.step = "All Done"
        robot.progress = 100
        robot.message = "All objects already done — nothing to do!"
        await broadcast_state()
        return

    # Single atomic command — avoids race condition
    send_control_command("PLAN_START:" + ",".join(robot.llm_stages_to_run))

    robot.state = "WORKING"
    robot.step = "Starting"
    robot.progress = 0
    robot.message = f"Starting: {', '.join(robot.llm_stages_to_run)}"
    robot.pipeline_stage_idx = 0
    robot.log_event("AUTO_START", {"planned_stages": robot.llm_stages_to_run})
    await broadcast_state()


# ==================== API Endpoints ====================

class FeedbackRequest(BaseModel):
    score: str
    tags: Optional[list[str]] = None


class MoveToPointRequest(BaseModel):
    point: int


class RunStageRequest(BaseModel):
    stage: str  # stage name, e.g. "Lemon", "Cup"


@app.post("/api/start")
async def api_start():
    """Trigger LLM plan + start. Auto-triggered on warmup, can also be called manually."""
    if robot.process is None:
        return {"status": "error", "message": "Process not running. Warming up..."}
    if robot.state == "WARMUP":
        return {"status": "error", "message": "Still warming up. Please wait."}
    if robot.state == "WORKING":
        return {"status": "error", "message": "Already running"}

    # Use the shared auto-plan-and-start logic
    asyncio.create_task(_auto_plan_and_start())
    return {"status": "ok", "message": "Planning and starting..."}


@app.post("/api/stop")
async def api_stop():
    """Emergency stop — hold current position, pause inference."""
    if robot.process is None:
        return {"status": "error", "message": "Not running"}
    send_control_command("ESTOP")
    robot.state = "PAUSED"
    robot.step = "E-Stop"
    robot.message = "Emergency stop — holding position"
    robot.log_event("CMD_ESTOP")
    await broadcast_state()
    return {"status": "ok", "message": "Emergency stop sent"}


@app.post("/api/reset")
async def api_reset():
    """Go to home position."""
    if robot.process is None:
        return {"status": "error", "message": "Not running"}
    send_control_command("HOME")
    robot.state = "HOMED"
    robot.step = "Homing"
    robot.message = "Moving to home position..."
    robot.log_event("CMD_HOME")
    await broadcast_state()
    return {"status": "ok", "message": "Go-to-home sent"}


@app.post("/api/resume")
async def api_resume():
    """Resume inference after e-stop or home."""
    if robot.process is None:
        return {"status": "error", "message": "Not running"}
    send_control_command("RESUME")
    robot.state = "WORKING"
    robot.step = "Resuming"
    robot.message = "Resuming inference..."
    robot.log_event("CMD_RESUME")
    await broadcast_state()
    return {"status": "ok", "message": "Resume sent"}


@app.post("/api/restart")
async def api_restart():
    """Restart pipeline: go home → LLM plan → auto-start."""
    if robot.process is None:
        return {"status": "error", "message": "Not running"}

    # Step 1: Go home first
    send_control_command("HOME")
    robot.state = "HOMED"
    robot.step = "Homing"
    robot.message = "Going home before replanning..."
    robot.log_event("CMD_RESTART_HOME")
    await broadcast_state()

    # Run the rest in background so API returns immediately
    asyncio.create_task(_restart_plan_and_go())
    return {"status": "ok", "message": "Restarting — going home + replanning..."}


async def _restart_plan_and_go():
    """Background task for restart: wait a moment → LLM plan → PLAN + RESTART."""
    # Give robot time to start homing
    await asyncio.sleep(2.0)

    # Step 2: LLM Planning
    if LLM_PLANNER_ENABLED:
        try:
            await run_llm_planning()
        except LLMPlannerError:
            print(f"[RESTART] LLM planning failed: {robot.llm_plan_error}")
            return

        if not robot.llm_stages_to_run:
            robot.state = "DONE"
            robot.step = "All Done"
            robot.progress = 100
            robot.message = "All objects already done — nothing to do!"
            await broadcast_state()
            return

        # Single atomic command — avoids race condition
        send_control_command("PLAN_RESTART:" + ",".join(robot.llm_stages_to_run))
    else:
        send_control_command("RESTART")
    robot.state = "WORKING"
    robot.step = "Restarting"
    robot.progress = 0
    robot.message = f"Restarting: {', '.join(robot.llm_stages_to_run) if robot.llm_stages_to_run else 'all stages'}..."
    robot.pipeline_stage_idx = 0
    robot.pipeline_status = "loading"
    robot.log_event("CMD_RESTART", {"planned_stages": robot.llm_stages_to_run})
    await broadcast_state()


@app.post("/api/retry")
async def api_retry():
    """Retry current stage: go home → reload current model → start."""
    if robot.process is None:
        return {"status": "error", "message": "Not running"}
    send_control_command("RETRY")
    robot.state = "WORKING"
    robot.step = "Retrying"
    robot.progress = 0
    robot.message = f"Retrying stage [{robot.pipeline_stage}]..."
    robot.pipeline_status = "loading"
    robot.log_event("CMD_RETRY", {"stage": robot.pipeline_stage})
    await broadcast_state()
    return {"status": "ok", "message": f"Retry sent — going home and retrying {robot.pipeline_stage}"}


@app.post("/api/run-stage")
async def api_run_stage(req: RunStageRequest):
    """Test mode: directly run a single stage — no LLM, no homing.

    Sends PLAN:<stage> + RESTART immediately so the robot loads and
    executes only the requested stage.
    """
    if robot.process is None:
        return {"status": "error", "message": "Process not running. Warming up..."}
    if robot.state == "WARMUP":
        return {"status": "error", "message": "Still warming up. Please wait."}

    # Validate stage name
    valid_names = [s["name"] for s in PIPELINE_STAGES]
    if req.stage not in valid_names:
        return {"status": "error", "message": f"Unknown stage '{req.stage}'. Valid: {valid_names}"}

    # Update pipeline display: only show the selected stage
    robot.llm_plan = None
    robot.llm_plan_error = ""
    robot.llm_planning = False
    robot.llm_stages_to_run = [req.stage]
    robot.pipeline_stages_info = [{
        "name": req.stage, "llm_status": "todo", "exec_status": "pending",
    }]
    robot.pipeline_total = 1

    # Single atomic command — avoids race condition where PLAN gets overwritten
    send_control_command(f"PLAN_RESTART:{req.stage}")

    robot.state = "WORKING"
    robot.step = f"Loading {req.stage}"
    robot.progress = 0
    robot.message = f"[TEST] Running [{req.stage}] directly..."
    robot.pipeline_stage_idx = 0
    robot.pipeline_status = "loading"
    robot.log_event("RUN_STAGE_DIRECT", {"stage": req.stage})
    await broadcast_state()
    return {"status": "ok", "message": f"[TEST] Running {req.stage} directly"}


@app.post("/api/quit")
async def api_quit():
    """Quit current session. Process respawns to stay warm."""
    if robot.process:
        send_control_command("QUIT")
        await asyncio.sleep(2)
        await kill_process()

    robot.state = "WARMUP"
    robot.step = "Restarting"
    robot.progress = 0
    robot.message = "Restarting and re-warming..."
    robot.log_event("CMD_QUIT")
    await broadcast_state()

    # Respawn to keep warm
    asyncio.create_task(spawn_warmup_process())
    return {"status": "ok", "message": "Quit. Re-warming..."}


@app.post("/api/hand-detect")
async def api_hand_detect():
    """Toggle hand safety detection on/off."""
    if robot.process is None:
        return {"status": "error", "message": "Process not running"}
    new_state = not robot.hand_detect_enabled
    send_control_command("HAND_ON" if new_state else "HAND_OFF")
    robot.hand_detect_enabled = new_state
    if not new_state:
        robot.hand_detected = False
        robot.auto_stopped = False
    robot.log_event("HAND_DETECT_TOGGLE", {"enabled": new_state})
    await broadcast_state()
    return {"status": "ok", "enabled": new_state,
            "message": f"Hand detection {'enabled' if new_state else 'disabled'}"}


@app.post("/api/feedback")
async def api_feedback(feedback: FeedbackRequest):
    robot.log_event("FEEDBACK", {"score": feedback.score, "tags": feedback.tags or []})
    return {"status": "ok"}


@app.get("/api/log")
async def api_log():
    return {"log": robot.session_log}


@app.get("/api/config")
async def api_config():
    return ROBOT_CONFIG


@app.get("/api/pipeline")
async def api_pipeline():
    """Return pipeline stage configuration."""
    stages = [{"name": s["name"], "model": s["model"],
               "trigger": f"{s.get('trigger_joint','')} {s.get('trigger_op','')} {s.get('trigger_value','')}"}
              for s in PIPELINE_STAGES]
    return {
        "stages": stages,
        "total": len(stages),
        "current_stage": robot.pipeline_stage,
        "current_idx": robot.pipeline_stage_idx,
        "status": robot.pipeline_status,
    }


# ==================== LLM Planner Endpoints ====================

@app.get("/api/plan")
async def api_plan():
    """Return the latest LLM planning result."""
    return {
        "plan": robot.llm_plan,
        "planning": robot.llm_planning,
        "error": robot.llm_plan_error,
        "stages_to_run": robot.llm_stages_to_run,
        "enabled": LLM_PLANNER_ENABLED,
    }


@app.post("/api/replan")
async def api_replan():
    """Manually trigger LLM replanning (does NOT start inference)."""
    if not LLM_PLANNER_ENABLED:
        return {"status": "error", "message": "LLM planner is disabled"}
    try:
        await run_llm_planning()
        return {
            "status": "ok",
            "plan": robot.llm_plan,
            "stages_to_run": robot.llm_stages_to_run,
        }
    except LLMPlannerError:
        return {"status": "error", "message": robot.llm_plan_error}


# ==================== Camera Frame Streaming ====================

@app.get("/api/cameras")
async def api_cameras():
    """Return list of configured camera names."""
    cameras_str = ROBOT_CONFIG.get("cameras", "")
    names = [item.strip().split(":")[0].strip() for item in cameras_str.split(",") if ":" in item]
    return {"cameras": names}


@app.get("/api/frame/{cam_name}")
async def api_frame(cam_name: str):
    """Return the latest JPEG frame from a camera.

    Frames are written by eval_act_safe.py to FRAME_DIR/<cam_name>.jpg.
    """
    path = os.path.join(FRAME_DIR, f"{cam_name}.jpg")
    if not os.path.exists(path):
        return Response(status_code=404, content=b"No frame", media_type="text/plain")
    try:
        with open(path, "rb") as f:
            data = f.read()
        return Response(
            content=data,
            media_type="image/jpeg",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )
    except Exception:
        return Response(status_code=503, content=b"Frame read error", media_type="text/plain")


# ==================== Grid Frame (16-grid overlay) ====================

@app.get("/api/frame-grid/{cam_name}")
async def api_frame_grid(cam_name: str):
    """Return the camera frame cropped and overlaid with a 4×4 numbered grid."""
    path = os.path.join(FRAME_DIR, f"{cam_name}.jpg")
    if not os.path.exists(path):
        return Response(status_code=404, content=b"No frame", media_type="text/plain")
    loop = asyncio.get_event_loop()
    try:
        data = await loop.run_in_executor(_grid_executor, _process_grid_frame, path)
        return Response(
            content=data,
            media_type="image/jpeg",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )
    except Exception:
        return Response(status_code=503, content=b"Grid processing error", media_type="text/plain")


# ==================== Points / Arm Move ====================

@app.get("/api/points")
async def api_points():
    """Return the loaded joint positions from point.csv."""
    return {"count": len(POINTS), "joint_names": JOINT_NAMES, "points": POINTS}


@app.post("/api/move-to-point")
async def api_move_to_point(req: MoveToPointRequest):
    """Move the robot arm to a pre-defined joint position from point.csv.

    Sends a GOTO:N command via the control file (same mechanism as HOME).
    eval_act_safe.py converts the ROS2 radian positions to lerobot space
    and smoothly interpolates to the target.
    """
    if robot.process is None:
        return {"status": "error", "message": "Process not running"}

    if not POINTS:
        return {"status": "error", "message": "No points loaded from CSV"}

    if req.point < 1 or req.point > len(POINTS):
        return {"status": "error", "message": f"Invalid point {req.point}. Range: 1-{len(POINTS)}"}

    # Block movement during active inference or warmup
    if robot.state == "WARMUP":
        return {"status": "error", "message": "Still warming up. Please wait."}

    send_control_command(f"GOTO:{req.point}")
    robot.log_event("CMD_GOTO_POINT", {"point": req.point})
    return {"status": "ok", "message": f"Moving to position {req.point}..."}


# ==================== WebSocket ====================

@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)
    print(f"[WS] +1 client (total {len(connected_clients)})")
    await websocket.send_text(json.dumps(robot.to_dict()))
    try:
        while True:
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=30)
            except asyncio.TimeoutError:
                await websocket.send_text(json.dumps(robot.to_dict()))
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in connected_clients:
            connected_clients.remove(websocket)
        print(f"[WS] -1 client (total {len(connected_clients)})")

# ==================== Lifecycle ====================

@app.on_event("startup")
async def startup():
    print("\n" + "=" * 55)
    print("🤖 LeRobot SO101 Control Backend (Pipeline Mode)")
    print("=" * 55)
    print(f"  Pipeline   : {len(PIPELINE_STAGES)} stages")
    for i, s in enumerate(PIPELINE_STAGES):
        print(f"    {i+1}. [{s['name']}] {s['model'][:50]}...")
    print(f"  Robot      : {ROBOT_CONFIG['robot_id']} @ {ROBOT_CONFIG['robot_port']}")
    print(f"  Cameras    : {ROBOT_CONFIG['cameras']}")
    print(f"  Control    : {CONTROL_FILE}")
    print(f"  Hand       : {'ON' if HAND_DETECT_ENABLED else 'OFF'}")
    print(f"  LLM Plan : {'ON — ' + str(LLM_PLANNER_ENABLED) if LLM_PLANNER_ENABLED else 'OFF'}")
    print("-" * 55)
    print("  POST /api/start          → LLM plan + start inference")
    print("  POST /api/stop           → Emergency stop")
    print("  POST /api/reset          → Go to home")
    print("  POST /api/resume         → Resume inference")
    print("  POST /api/restart        → Home + LLM plan + restart")
    print("  POST /api/retry          → Retry current stage")
    print("  POST /api/run-stage      → Run a single stage by name")
    print("  POST /api/quit           → Quit & re-warm")
    print("  POST /api/hand-detect    → Toggle hand safety")
    print("  GET  /api/plan           → Get LLM plan result")
    print("  POST /api/replan         → Trigger LLM replan")
    print("  GET  /api/frame-grid/:c  → 16-grid camera frame")
    print("  GET  /api/points         → List preposition points")
    print(f"  POST /api/move-to-point  → Move arm (1-{len(POINTS)} points)")
    print("=" * 55)
    print("  Auto-warming up model + robot...\n")

    # Auto-spawn warmup process
    asyncio.create_task(spawn_warmup_process())


@app.on_event("shutdown")
async def shutdown():
    await kill_process()
    if os.path.exists(CONTROL_FILE):
        os.remove(CONTROL_FILE)
    print("[SHUTDOWN] Done")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
