#!/bin/bash
# Capture images from /dev/video4 at 1Hz (1 image per second)

source ~/miniconda3/etc/profile.d/conda.sh
conda activate lerobot

OUTPUT_DIR="/home/robotlab/lerobot/outputs/captured_images"
CAMERA_DEVICE="/dev/video4"
FPS=1  # 1 image per second

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "  Video4 Image Capture Script"
echo "=========================================="
echo "Camera: $CAMERA_DEVICE"
echo "Rate: ${FPS} Hz (1 image per second)"
echo "Output: $OUTPUT_DIR"
echo "=========================================="
echo "Press Ctrl+C to stop"
echo ""

# Fixed filename - always overwrite the latest image
FILENAME="${OUTPUT_DIR}/opencv__${CAMERA_DEVICE//\//_}_latest.png"

# Capture loop
while true; do
    
    # Use opencv-python to capture image
    python3 << EOF
import cv2
import sys
import time

cap = cv2.VideoCapture("$CAMERA_DEVICE")
if not cap.isOpened():
    print(f"Error: Cannot open camera $CAMERA_DEVICE", file=sys.stderr)
    sys.exit(1)

# Set camera properties for stability
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

# Read a few frames to let camera stabilize (skip first 3 frames)
for _ in range(3):
    cap.read()

# Now capture the actual frame
ret, frame = cap.read()
cap.release()

if ret:
    # Crop image: 
    # Horizontal: left to right 5%-55%
    # Vertical: from bottom 15% to bottom 70% (i.e., from top 30% to top 85%)
    height, width = frame.shape[:2]
    x_start = int(width * 0.05)
    x_end = int(width * 0.55)
    # From bottom 70% = from top 30%, from bottom 15% = from top 85%
    y_start = int(height * 0.30)  # From top 30% (from bottom 70%)
    y_end = int(height * 0.85)    # From top 85% (from bottom 15%)
    
    # Crop: frame[y_start:y_end, x_start:x_end]
    cropped = frame[y_start:y_end, x_start:x_end]
    
    # Draw grid lines: 4x4 = 16 cells
    h, w = cropped.shape[:2]
    # Draw 3 vertical lines (dividing width into 4 parts)
    for i in range(1, 4):
        x = int(w * i / 4)
        cv2.line(cropped, (x, 0), (x, h), (0, 255, 0), 1)  # Green lines
    # Draw 3 horizontal lines (dividing height into 4 parts)
    for i in range(1, 4):
        y = int(h * i / 4)
        cv2.line(cropped, (0, y), (w, y), (0, 255, 0), 1)  # Green lines
    
    # Add numbers 1-16 in each cell (from top-left to bottom-right)
    cell_w = w / 4
    cell_h = h / 4
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    color = (0, 255, 0)  # Green color
    
    for row in range(4):
        for col in range(4):
            cell_num = row * 4 + col + 1
            # Center position of each cell
            x_center = int((col + 0.5) * cell_w)
            y_center = int((row + 0.5) * cell_h)
            # Get text size for centering
            text = str(cell_num)
            (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            x_text = x_center - text_w // 2
            y_text = y_center + text_h // 2
            # Draw number
            cv2.putText(cropped, text, (x_text, y_text), font, font_scale, color, thickness)
    
    cv2.imwrite("$FILENAME", cropped)
    print(f"Captured and cropped: $FILENAME (Original: {width}x{height}, Cropped: {x_end-x_start}x{y_end-y_start})")
else:
    print(f"Error: Failed to capture frame", file=sys.stderr)
    sys.exit(1)
EOF
    
    if [ $? -eq 0 ]; then
        echo "[$(date +'%H:%M:%S')] Updated: $(basename $FILENAME)"
    else
        echo "[$(date +'%H:%M:%S')] Error capturing frame"
    fi
    
    # Wait 1 second (1Hz = 1 image per second)
    sleep 1
done
