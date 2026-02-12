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
import json
import os
import signal
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import build_inference_command, ROBOT_CONFIG, CONTROL_FILE, LEROBOT_DIR

app = FastAPI(title="LeRobot SO101 Control Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

    def to_dict(self):
        return {
            "state": self.state,
            "step": self.step,
            "progress": self.progress,
            "message": self.message,
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
    """Parse eval_act_safe.py stdout and return state updates."""
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
        return {"state": "WORKING", "step": "Running", "message": "Inference started!", "progress": 0}
    if "inference_done" in lo:
        return {"_signal": "INFERENCE_DONE"}

    # ── Runtime status ──
    if "emergency stop" in lo:
        return {"state": "PAUSED", "step": "E-Stop", "message": "Emergency stop — holding position"}
    if "home reached" in lo:
        return {"state": "HOMED", "step": "Home", "message": "At home position, inference paused"}
    if "resumed" in lo or "continuing inference" in lo:
        return {"state": "WORKING", "step": "Running", "message": "Inference resumed"}

    if "episode" in lo and "/" in lo:
        return {"step": "Episode", "message": raw[:60]}

    if "step " in lo:
        import re
        m = re.search(r"step\s+(\d+)/(\d+)", lo)
        if m:
            cur, total = int(m.group(1)), int(m.group(2))
            pct = min(int(cur / total * 100), 99)
            return {"step": "Inference", "message": f"Step {cur}/{total}", "progress": pct}

    if "episode" in lo and "finished" in lo:
        return {"step": "Episode done", "message": raw[:60], "progress": 100}

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
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=LEROBOT_DIR,
            start_new_session=True,
        )
        robot.process = proc

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
                robot.message = "Inference session complete"
                robot.log_event("INFERENCE_DONE")
                await broadcast_state()
                continue
            if sig == "PROCESS_DONE":
                continue

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

# ==================== API Endpoints ====================

class FeedbackRequest(BaseModel):
    score: str
    tags: Optional[list[str]] = None


@app.post("/api/start")
async def api_start():
    """Send START command — inference begins immediately (model already loaded)."""
    if robot.process is None:
        return {"status": "error", "message": "Process not running. Warming up..."}
    if robot.state == "WARMUP":
        return {"status": "error", "message": "Still warming up. Please wait."}
    if robot.state == "WORKING":
        return {"status": "error", "message": "Already running"}

    send_control_command("START")
    robot.state = "WORKING"
    robot.step = "Starting"
    robot.progress = 0
    robot.message = "Inference starting..."
    robot.log_event("CMD_START")
    await broadcast_state()
    return {"status": "ok", "message": "Start sent — inference beginning immediately"}


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
    print("🤖 LeRobot SO101 Control Backend (Pre-warm Mode)")
    print("=" * 55)
    print(f"  Model      : {ROBOT_CONFIG['model']}")
    print(f"  Robot      : {ROBOT_CONFIG['robot_id']} @ {ROBOT_CONFIG['robot_port']}")
    print(f"  Cameras    : {ROBOT_CONFIG['cameras']}")
    print(f"  Control    : {CONTROL_FILE}")
    print("-" * 55)
    print("  POST /api/start   → Start inference (instant)")
    print("  POST /api/stop    → Emergency stop")
    print("  POST /api/reset   → Go to home")
    print("  POST /api/resume  → Resume inference")
    print("  POST /api/quit    → Quit & re-warm")
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
