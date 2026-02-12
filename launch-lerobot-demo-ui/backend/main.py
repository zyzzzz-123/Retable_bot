"""
Mock Backend for Table Reset Control UI Demo
Simulates robot state transitions for the Gate 7 MWS demo video.
"""

import asyncio
import json
import time
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Table Reset Robot Mock Backend")

# CORS for local development
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
        self.state = "READY"
        self.step = ""
        self.progress = 0
        self.message = "Robot ready for table reset"
        self.task_running = False
        self.paused = False
        self.session_log = []
        
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
            **(details or {})
        }
        self.session_log.append(entry)
        print(f"[LOG] {entry}")

robot = RobotState()
connected_clients: list[WebSocket] = []

# ==================== WebSocket Broadcasting ====================

async def broadcast_state():
    """Send current state to all connected clients"""
    message = json.dumps(robot.to_dict())
    disconnected = []
    for ws in connected_clients:
        try:
            await ws.send_text(message)
        except:
            disconnected.append(ws)
    for ws in disconnected:
        connected_clients.remove(ws)

# ==================== Simulated Robot Task ====================

# Define the table reset workflow steps
WORKFLOW_STEPS = [
    {"step": "Scanning table", "duration": 2, "progress_end": 15},
    {"step": "Identifying items", "duration": 2, "progress_end": 25},
    {"step": "Clearing dishes", "duration": 3, "progress_end": 50},
    {"step": "Removing utensils", "duration": 2, "progress_end": 65},
    {"step": "Wiping table surface", "duration": 3, "progress_end": 85},
    {"step": "Final inspection", "duration": 2, "progress_end": 100},
]

async def run_robot_task():
    """Simulate the table reset workflow"""
    global robot
    
    robot.state = "WORKING"
    robot.progress = 0
    robot.message = "Starting table reset routine..."
    robot.log_event("TASK_STARTED")
    await broadcast_state()
    
    for step_info in WORKFLOW_STEPS:
        if not robot.task_running:
            return  # Task was stopped
        
        # Handle pause
        while robot.paused and robot.task_running:
            await asyncio.sleep(0.5)
        
        if not robot.task_running:
            return
        
        robot.step = step_info["step"]
        robot.message = f"Currently: {step_info['step']}"
        robot.log_event("STEP_STARTED", {"step": step_info["step"]})
        await broadcast_state()
        
        # Simulate step duration with progress updates
        start_progress = robot.progress
        end_progress = step_info["progress_end"]
        duration = step_info["duration"]
        steps = duration * 4  # Update 4 times per second
        
        for i in range(steps):
            if not robot.task_running:
                return
            while robot.paused and robot.task_running:
                await asyncio.sleep(0.5)
            if not robot.task_running:
                return
                
            robot.progress = int(start_progress + (end_progress - start_progress) * (i + 1) / steps)
            await broadcast_state()
            await asyncio.sleep(duration / steps)
    
    # Task completed
    if robot.task_running:
        robot.state = "DONE"
        robot.step = "Complete"
        robot.progress = 100
        robot.message = "Table reset completed successfully!"
        robot.task_running = False
        robot.log_event("TASK_COMPLETED")
        await broadcast_state()

# ==================== API Endpoints ====================

class FeedbackRequest(BaseModel):
    score: str  # "up" or "down"
    tags: Optional[list[str]] = None

@app.post("/api/start")
async def start_robot():
    """Start the table reset routine"""
    global robot
    
    if robot.task_running:
        return {"status": "error", "message": "Task already running"}
    
    robot.task_running = True
    robot.paused = False
    robot.log_event("COMMAND_START", {"ack_time": datetime.now().isoformat()})
    
    # Run task in background
    asyncio.create_task(run_robot_task())
    
    return {"status": "ok", "message": "Robot task started"}

@app.post("/api/stop")
async def stop_robot():
    """Stop the robot immediately"""
    global robot
    
    robot.task_running = False
    robot.paused = False
    robot.state = "READY"
    robot.step = ""
    robot.progress = 0
    robot.message = "Task stopped by user"
    robot.log_event("COMMAND_STOP", {"ack_time": datetime.now().isoformat()})
    
    await broadcast_state()
    return {"status": "ok", "message": "Robot stopped"}

@app.post("/api/pause")
async def pause_robot():
    """Pause the current task"""
    global robot
    
    if not robot.task_running:
        return {"status": "error", "message": "No task running"}
    
    robot.paused = True
    robot.state = "PAUSED"
    robot.message = "Task paused by user"
    robot.log_event("COMMAND_PAUSE", {"ack_time": datetime.now().isoformat()})
    
    await broadcast_state()
    return {"status": "ok", "message": "Robot paused"}

@app.post("/api/resume")
async def resume_robot():
    """Resume a paused task"""
    global robot
    
    if not robot.paused:
        return {"status": "error", "message": "Robot not paused"}
    
    robot.paused = False
    robot.state = "WORKING"
    robot.message = f"Resuming: {robot.step}"
    robot.log_event("COMMAND_RESUME", {"ack_time": datetime.now().isoformat()})
    
    await broadcast_state()
    return {"status": "ok", "message": "Robot resumed"}

@app.post("/api/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """Submit user feedback after task completion"""
    global robot
    
    robot.log_event("FEEDBACK_SUBMITTED", {
        "score": feedback.score,
        "tags": feedback.tags or [],
        "ack_time": datetime.now().isoformat()
    })
    
    return {"status": "ok", "message": "Feedback received"}

@app.get("/api/log")
async def get_session_log():
    """Get the session log for measurability evidence"""
    return {"log": robot.session_log}

# ==================== WebSocket Endpoint ====================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time status updates"""
    await websocket.accept()
    connected_clients.append(websocket)
    print(f"[WS] Client connected. Total: {len(connected_clients)}")
    
    # Send initial state
    await websocket.send_text(json.dumps(robot.to_dict()))
    
    try:
        while True:
            # Keep connection alive, handle any incoming messages
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30)
            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_text(json.dumps(robot.to_dict()))
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in connected_clients:
            connected_clients.remove(websocket)
        print(f"[WS] Client disconnected. Total: {len(connected_clients)}")

# ==================== Startup ====================

@app.on_event("startup")
async def startup():
    print("\n" + "="*50)
    print("🤖 Table Reset Robot Mock Backend")
    print("="*50)
    print("API Endpoints:")
    print("  POST /api/start   - Start table reset")
    print("  POST /api/stop    - Stop robot")
    print("  POST /api/pause   - Pause task")
    print("  POST /api/resume  - Resume task")
    print("  POST /api/feedback - Submit feedback")
    print("  GET  /api/log     - Get session log")
    print("  WS   /ws          - Real-time status")
    print("="*50 + "\n")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
