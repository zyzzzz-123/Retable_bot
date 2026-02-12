#!/bin/bash
# Start LeRobot Demo UI - Frontend and Backend

echo "=========================================="
echo "  LeRobot Demo UI Launcher"
echo "=========================================="

# Kill any existing processes on ports 8000 and 5173
echo "Cleaning up existing processes..."
fuser -k 8000/tcp 2>/dev/null
fuser -k 5173/tcp 2>/dev/null
sleep 1

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Start Backend
echo ""
echo "Starting Backend on port 8000..."
cd "$SCRIPT_DIR/backend"
source /home/robotlab/miniconda3/etc/profile.d/conda.sh
conda activate lerobot
uvicorn main_robot:app --port 8000 &
BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID"

# Wait for backend to start
sleep 2

# Start Frontend
echo ""
echo "Starting Frontend on port 5173..."
cd "$SCRIPT_DIR/ui"
npm run dev &
FRONTEND_PID=$!
echo "Frontend PID: $FRONTEND_PID"

echo ""
echo "=========================================="
echo "  Services Started!"
echo "=========================================="
echo "  Backend:  http://localhost:8000"
echo "  Frontend: http://localhost:5173"
echo "=========================================="
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for user interrupt
trap "echo 'Stopping services...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" SIGINT SIGTERM
wait
