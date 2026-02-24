"""
Preflight Camera Calibration Server

A lightweight FastAPI server that helps users:
1. Discover available video devices
2. Preview live snapshots from each camera
3. Assign camera roles (front / wrist)
4. Detect available robot serial ports
5. Save the mapping to config.py

Usage:
    uvicorn preflight_server:app --host 0.0.0.0 --port 8000
"""

import asyncio
import glob
import os
import platform
import re
import signal
import subprocess
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import cv2
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

# ── Constants ──────────────────────────────────────────────────────────────
CONFIG_PATH = Path(__file__).parent / "config.py"
CACHE_TTL = 1.5          # seconds to cache a snapshot per device
WARMUP_FRAMES = 5        # frames to discard for auto-exposure settling
SOLID_COLOR_THRESHOLD = 0.97   # fraction of pixels within ±10 of median = solid color

# ── Global camera lock — only one camera open at a time ───────────────────
# This prevents resource contention when multiple browser polls arrive
_camera_lock = threading.Lock()

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


# ── Helpers ────────────────────────────────────────────────────────────────

def _is_capture_device(path: str) -> bool:
    """
    On Linux, USB cameras expose pairs of /dev/videoN nodes:
      - Even index  (0, 2, 4, …)  → actual capture stream
      - Odd index   (1, 3, 5, …)  → metadata / control node (returns garbage/green)

    Filter to even-numbered devices only to avoid metadata nodes.
    Exception: if only odd-numbered devices exist for a given path, keep them.
    """
    try:
        num = int(re.search(r"\d+$", path).group())
        return num % 2 == 0
    except (AttributeError, ValueError):
        return True  # non-numeric path → keep it


def _is_solid_color_frame(frame) -> bool:
    """
    Return True if the frame is dominated by a single solid color
    (i.e. blank green, all-black, etc.) and is therefore not useful.
    Uses the fact that a usable camera frame has variance across its pixels.
    """
    import numpy as np
    if frame is None:
        return True
    # Convert to grayscale for a simple check
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    median_val = float(np.median(gray))
    # Fraction of pixels within ±12 of the median
    within = np.sum(np.abs(gray.astype(float) - median_val) < 12)
    fraction = within / gray.size
    return fraction > SOLID_COLOR_THRESHOLD


def _detect_video_devices() -> list[dict[str, Any]]:
    """
    Scan /dev/video* and return metadata for each usable capture camera.
    - Filters to even-numbered devices (skips metadata/control nodes)
    - Validates that each device returns a real (non-solid-color) frame
    - Sequential access (protected by _camera_lock)
    """
    cameras: list[dict[str, Any]] = []

    if platform.system() == "Linux":
        all_paths = sorted(glob.glob("/dev/video*"))
        # Filter to capture nodes (even-numbered)
        paths = [p for p in all_paths if _is_capture_device(p)]
    else:
        paths = [str(i) for i in range(20)]

    for path in paths:
        target = path if platform.system() == "Linux" else int(path)
        with _camera_lock:
            cap = cv2.VideoCapture(target)
            if not cap.isOpened():
                cap.release()
                continue

            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
            fourcc = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])

            # Validate: try to grab a real frame (discard warmup frames)
            valid = False
            last_frame = None
            for _ in range(WARMUP_FRAMES + 1):
                ret, frame = cap.read()
                if ret and frame is not None:
                    last_frame = frame

            cap.release()

        if last_frame is not None and not _is_solid_color_frame(last_frame):
            valid = True

        if valid:
            cameras.append({
                "device": path if platform.system() == "Linux" else int(path),
                "width": w,
                "height": h,
                "fps": round(fps, 1),
                "fourcc": fourcc,
            })

    return cameras


def _capture_snapshot_jpeg(device: str, quality: int = 80) -> bytes:
    """
    Open a camera, grab one stable frame, return JPEG bytes, then release.
    Protected by _camera_lock to prevent concurrent camera opens.
    """
    # Check cache first
    now = time.time()
    cached = _snapshot_cache.get(device)
    if cached and (now - cached[0]) < CACHE_TTL:
        return cached[1]

    with _camera_lock:
        cap = cv2.VideoCapture(device)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Cannot open {device}")

        try:
            # Discard warmup frames for auto-exposure
            ret, frame = False, None
            for _ in range(WARMUP_FRAMES + 1):
                ret, frame = cap.read()

            if not ret or frame is None:
                raise RuntimeError(f"Failed to read frame from {device}")

            if _is_solid_color_frame(frame):
                raise RuntimeError(f"Device {device} returned a solid-color (unusable) frame")

            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            jpeg_bytes = buf.tobytes()
        finally:
            cap.release()

    _snapshot_cache[device] = (time.time(), jpeg_bytes)
    return jpeg_bytes


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
        # Fallback: scan /dev/ttyACM* and /dev/ttyUSB*
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
    - Filters to even-numbered /dev/video* (skips metadata/control nodes)
    - Validates each device returns a real non-solid-color frame
    """
    loop = asyncio.get_event_loop()
    cameras = await loop.run_in_executor(None, _detect_video_devices)
    return {"cameras": cameras, "count": len(cameras)}


@app.get("/api/preflight/snapshot/{device_path:path}")
async def get_snapshot(device_path: str):
    """
    Capture a snapshot from a specific video device.
    device_path is the path without leading /  e.g. "dev/video4"
    """
    device = f"/{device_path}"
    if not os.path.exists(device):
        raise HTTPException(status_code=404, detail=f"Device {device} not found")

    loop = asyncio.get_event_loop()
    try:
        jpeg_bytes = await loop.run_in_executor(None, _capture_snapshot_jpeg, device)
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
    """
    Save camera assignments and robot port to config.py.
    """
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
       (enough time to send the response back to the browser)
    The frontend should poll /api/health until main_robot responds.
    """
    backend_dir = Path(__file__).parent
    conda_sh = os.path.expanduser("~/miniconda3/etc/profile.d/conda.sh")

    # Build the shell command to restart as main_robot
    cmd = (
        f"source {conda_sh} && conda activate lerobot && "
        f"sleep 2 && "
        f"cd {backend_dir} && "
        f"exec uvicorn main_robot:app --host 0.0.0.0 --port 8000"
    )

    # Launch as a completely detached process so it survives after preflight dies
    subprocess.Popen(
        ["bash", "-c", cmd],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        start_new_session=True,   # detach from current process group
    )

    # Schedule self-termination after 1.5s (response will have been sent by then)
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
    """Generic health check — used by frontend to detect when main_robot is up."""
    return {"status": "ok", "service": "preflight"}
