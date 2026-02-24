#!/usr/bin/env python3
"""
Camera Worker — runs in a SEPARATE subprocess from the FastAPI server.

This script is invoked by preflight_server.py via subprocess.run().
It handles all OpenCV camera operations in isolation so that:
  1. The FastAPI process never loads OpenCV / CUDA
  2. No CUDA driver corruption can occur
  3. Short-lived subprocess cleans up all GPU/camera resources on exit

Usage:
  python _camera_worker.py detect            → JSON list of cameras to stdout
  python _camera_worker.py snapshot <device>  → JPEG bytes to stdout
"""

import glob
import json
import os
import platform
import re
import sys

# ── Force CPU-only — prevent any CUDA context in this worker ──────────────
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2
import numpy as np

# NOTE: We use CAP_ANY (not CAP_V4L2) — V4L2 backend can cause green tint
# on some cameras due to incorrect YUYV→BGR conversion.
# CUDA safety is already ensured by CUDA_VISIBLE_DEVICES="" above.
WARMUP_FRAMES = 2    # fewer warmup frames = faster snapshots
SOLID_COLOR_THRESHOLD = 0.95  # reject frames that are >95% one color (green/black)


def _is_capture_device(path: str) -> bool:
    """Even-numbered /dev/video* = capture stream; odd = metadata (skip)."""
    try:
        num = int(re.search(r"\d+$", path).group())
        return num % 2 == 0
    except (AttributeError, ValueError):
        return True


def _set_mjpeg(cap) -> None:
    """Try to switch camera to MJPEG format — faster USB transfer, no color issues."""
    mjpg = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    cap.set(cv2.CAP_PROP_FOURCC, mjpg)


def detect_cameras() -> list[dict]:
    """
    Scan /dev/video* and return metadata for each openable camera.
    Fast: only checks isOpened() + reads metadata. No frame capture.
    """
    cameras = []
    if platform.system() == "Linux":
        all_paths = sorted(glob.glob("/dev/video*"))
        paths = [p for p in all_paths if _is_capture_device(p)]
    else:
        paths = [str(i) for i in range(20)]

    for path in paths:
        target = path if platform.system() == "Linux" else int(path)
        cap = cv2.VideoCapture(target)
        if not cap.isOpened():
            cap.release()
            continue

        _set_mjpeg(cap)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])
        cap.release()

        cameras.append({
            "device": path if platform.system() == "Linux" else int(path),
            "width": w,
            "height": h,
            "fps": round(fps, 1),
            "fourcc": fourcc,
        })

    return cameras


def _is_solid_color(frame) -> bool:
    """Return True if >95% of pixels are within ±12 of the median (solid green/black)."""
    if frame is None:
        return True
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    median_val = float(np.median(gray))
    within = np.sum(np.abs(gray.astype(float) - median_val) < 12)
    return (within / gray.size) > SOLID_COLOR_THRESHOLD


def capture_snapshot(device: str, quality: int = 80) -> bytes:
    """Open camera, grab one stable frame, return JPEG bytes.
    Rejects solid-color frames (green/black from metadata or depth nodes)."""
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open {device}")

    try:
        _set_mjpeg(cap)

        ret, frame = False, None
        for _ in range(WARMUP_FRAMES + 1):
            ret, frame = cap.read()

        if not ret or frame is None:
            raise RuntimeError(f"Failed to read frame from {device}")

        if _is_solid_color(frame):
            raise RuntimeError(f"Device {device} returned a solid-color frame (unusable)")

        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes()
    finally:
        cap.release()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: _camera_worker.py detect | snapshot <device>", file=sys.stderr)
        sys.exit(1)

    command = sys.argv[1]

    if command == "detect":
        cameras = detect_cameras()
        json.dump(cameras, sys.stdout)
        sys.stdout.flush()

    elif command == "snapshot":
        if len(sys.argv) < 3:
            print("Usage: _camera_worker.py snapshot <device>", file=sys.stderr)
            sys.exit(1)
        device = sys.argv[2]
        try:
            jpeg_bytes = capture_snapshot(device)
            sys.stdout.buffer.write(jpeg_bytes)
            sys.stdout.buffer.flush()
        except RuntimeError as e:
            print(str(e), file=sys.stderr)
            sys.exit(2)

    else:
        print(f"Unknown command: {command}", file=sys.stderr)
        sys.exit(1)
