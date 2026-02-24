"""
Preflight Camera Calibration Server

A lightweight FastAPI server that helps users:
1. Discover available video devices
2. Preview live snapshots from each camera
3. Assign camera roles (front / wrist)
4. Detect available robot serial ports
5. Save the mapping to config.py

IMPORTANT: This server does NOT import OpenCV or any GPU libraries.
All camera operations are delegated to _camera_worker.py which runs
in a separate subprocess. This prevents CUDA driver corruption that
would break the main inference process (eval_act_safe.py).

Usage:
    uvicorn preflight_server:app --host 0.0.0.0 --port 8000
"""

import asyncio
import glob
import json
import os
import re
import signal
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

# ── Constants ──────────────────────────────────────────────────────────────
CONFIG_PATH = Path(__file__).parent / "config.py"
WORKER_SCRIPT = Path(__file__).parent / "_camera_worker.py"
CACHE_TTL = 3.0          # seconds to cache a snapshot per device (subprocess = slower)
DETECT_TIMEOUT = 60      # seconds to wait for camera detection

# ── Snapshot cache ─────────────────────────────────────────────────────────
_snapshot_cache: dict[str, tuple[float, bytes]] = {}


# ── Lifespan ───────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    _snapshot_cache.clear()


app = FastAPI(title="LeRobot Preflight Check", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Subprocess helpers ─────────────────────────────────────────────────────
# All camera work happens in _camera_worker.py subprocess.
# This ensures the FastAPI process NEVER loads OpenCV / CUDA,
# preventing GPU driver corruption that would break inference.

def _worker_detect_cameras() -> list[dict[str, Any]]:
    """Run _camera_worker.py detect in a subprocess, return camera list."""
    result = subprocess.run(
        [sys.executable, str(WORKER_SCRIPT), "detect"],
        capture_output=True,
        timeout=DETECT_TIMEOUT,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"Camera detection failed: {stderr}")

    stdout = result.stdout.decode(errors="replace").strip()
    if not stdout:
        return []
    return json.loads(stdout)


def _worker_capture_snapshot(device: str) -> bytes:
    """Run _camera_worker.py snapshot <device> in a subprocess, return JPEG bytes."""
    # Check cache first
    now = time.time()
    cached = _snapshot_cache.get(device)
    if cached and (now - cached[0]) < CACHE_TTL:
        return cached[1]

    result = subprocess.run(
        [sys.executable, str(WORKER_SCRIPT), "snapshot", device],
        capture_output=True,
        timeout=15,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"Snapshot failed for {device}: {stderr}")

    jpeg_bytes = result.stdout
    if not jpeg_bytes:
        raise RuntimeError(f"Empty snapshot from {device}")

    _snapshot_cache[device] = (time.time(), jpeg_bytes)
    return jpeg_bytes


# ── Non-camera helpers ─────────────────────────────────────────────────────

def _detect_serial_ports() -> list[dict[str, str]]:
    """Detect available serial ports (robot motor bus connections)."""
    ports: list[dict[str, str]] = []
    try:
        from serial.tools import list_ports
        for p in list_ports.comports():
            ports.append({
                "device": p.device,
                "description": p.description or "",
                "manufacturer": p.manufacturer or "",
                "hwid": p.hwid or "",
            })
    except ImportError:
        for pattern in ["/dev/ttyACM*", "/dev/ttyUSB*"]:
            for dev in sorted(glob.glob(pattern)):
                ports.append({"device": dev, "description": "", "manufacturer": "", "hwid": ""})
    return ports


def _read_current_config() -> dict[str, Any]:
    """Parse the current config.py and return ROBOT_CONFIG values."""
    if not CONFIG_PATH.exists():
        return {}
    content = CONFIG_PATH.read_text()
    m = re.search(r'"cameras"\s*:\s*"([^"]*)"', content)
    cameras_str = m.group(1) if m else ""
    m2 = re.search(r'"robot_port"\s*:\s*"([^"]*)"', content)
    robot_port = m2.group(1) if m2 else ""
    camera_map = {}
    if cameras_str:
        for pair in cameras_str.split(","):
            pair = pair.strip()
            if ":" in pair:
                role, dev = pair.split(":", 1)
                camera_map[role.strip()] = dev.strip()
    return {
        "cameras": camera_map,
        "cameras_raw": cameras_str,
        "robot_port": robot_port,
    }


def _save_config(cameras_str: str, robot_port: str | None = None) -> None:
    """Update ROBOT_CONFIG in config.py with new camera mapping and optionally robot port."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")

    content = CONFIG_PATH.read_text()

    content = re.sub(
        r'("cameras"\s*:\s*)"([^"]*)"',
        f'\\1"{cameras_str}"',
        content,
    )

    if robot_port:
        content = re.sub(
            r'("robot_port"\s*:\s*)"([^"]*)"',
            f'\\1"{robot_port}"',
            content,
        )

    CONFIG_PATH.write_text(content)


# ── API Models ─────────────────────────────────────────────────────────────

class CameraAssignment(BaseModel):
    role: str    # "front" or "wrist"
    device: str  # e.g. "/dev/video4"


class SaveConfigRequest(BaseModel):
    cameras: list[CameraAssignment]
    robot_port: str | None = None


# ── API Routes ─────────────────────────────────────────────────────────────

@app.get("/api/preflight/detect-cameras")
async def detect_cameras():
    """
    Detect all usable video capture devices.
    Camera work runs in a separate subprocess (_camera_worker.py).
    """
    loop = asyncio.get_event_loop()
    try:
        cameras = await loop.run_in_executor(None, _worker_detect_cameras)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"cameras": cameras, "count": len(cameras)}


@app.get("/api/preflight/snapshot/{device_path:path}")
async def get_snapshot(device_path: str):
    """
    Capture a snapshot from a specific video device.
    Camera work runs in a separate subprocess (_camera_worker.py).
    """
    device = f"/{device_path}"
    if not os.path.exists(device):
        raise HTTPException(status_code=404, detail=f"Device {device} not found")

    loop = asyncio.get_event_loop()
    try:
        jpeg_bytes = await loop.run_in_executor(None, _worker_capture_snapshot, device)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return Response(content=jpeg_bytes, media_type="image/jpeg")


@app.get("/api/preflight/detect-ports")
async def detect_ports():
    """Detect available serial ports for robot connection."""
    loop = asyncio.get_event_loop()
    ports = await loop.run_in_executor(None, _detect_serial_ports)
    return {"ports": ports}


@app.get("/api/preflight/current-config")
async def get_current_config():
    """Return the current camera and port configuration from config.py."""
    config = _read_current_config()
    return {"config": config}


@app.post("/api/preflight/save-config")
async def save_config(req: SaveConfigRequest):
    """Save camera assignments and robot port to config.py."""
    parts = [f"{c.role}:{c.device}" for c in req.cameras]
    cameras_str = ",".join(parts)

    try:
        _save_config(cameras_str, req.robot_port)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save config: {e}")

    return {
        "status": "ok",
        "message": "Configuration saved successfully",
        "cameras": cameras_str,
        "robot_port": req.robot_port,
    }


@app.post("/api/preflight/launch-control")
async def launch_control():
    """
    Switch from preflight mode to main robot control mode.
    1. Spawns main_robot:app uvicorn process in background (detached)
    2. Schedules self-termination of this preflight server after 1.5s
    """
    backend_dir = Path(__file__).parent
    conda_sh = os.path.expanduser("~/miniconda3/etc/profile.d/conda.sh")

    # Explicitly unset CUDA_VISIBLE_DEVICES so main_robot sees the GPU
    cmd = (
        f"unset CUDA_VISIBLE_DEVICES && "
        f"source {conda_sh} && conda activate lerobot && "
        f"sleep 2 && "
        f"cd {backend_dir} && "
        f"exec uvicorn main_robot:app --host 0.0.0.0 --port 8000"
    )

    # Clean environment — remove CUDA_VISIBLE_DEVICES if set
    clean_env = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}

    subprocess.Popen(
        ["bash", "-c", cmd],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        env=clean_env,
        start_new_session=True,
    )

    async def self_terminate():
        await asyncio.sleep(1.5)
        os.kill(os.getpid(), signal.SIGTERM)

    asyncio.create_task(self_terminate())

    return {"status": "ok", "message": "Switching to control mode — please wait ~30s for warmup"}


@app.get("/api/preflight/health")
async def health():
    return {"status": "ok", "service": "preflight"}


@app.get("/api/health")
async def health_generic():
    """Generic health check — used by frontend to detect backend type."""
    return {"status": "ok", "service": "preflight"}
