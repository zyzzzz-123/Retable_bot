#!/bin/bash
# Start SO-ARM ROS2 Only (No MoveIt)
# 只启动 ROS2 基础组件，用于 TF2 检测 EE 位置，不启动 MoveIt
#
# 启动的组件:
#   1. robot_state_publisher  → 发布 TF (正运动学)
#   2. ros2_control_node      → 硬件接口 (读取舵机)
#   3. joint_state_broadcaster → 发布 /joint_states
#   4. arm_controller (可选)   → 发送关节指令
#
# 之后可以用 TF2 查询 EE 位置:
#   ros2 run tf2_ros tf2_echo base_link jaw

echo "=========================================="
echo "  SO-ARM ROS2 Only (No MoveIt)"
echo "  仅启动 ROS2 基础组件"
echo "=========================================="

# Check if serial port exists
SERIAL_PORT="/dev/ttyACM1"
if [ ! -e "$SERIAL_PORT" ]; then
    echo "⚠️  Warning: Serial port $SERIAL_PORT not found!"
    echo "   请检查机械臂是否已连接"
    echo ""
    echo "Available serial ports:"
    ls /dev/ttyACM* /dev/ttyUSB* 2>/dev/null || echo "   No serial ports found"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Source ROS2 environment
echo ""
echo "Setting up ROS2 environment..."
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash

# Paths
URDF_XACRO=$(ros2 pkg prefix so_arm_description --share)/urdf/so101.urdf.xacro
CONTROLLERS_YAML=$(find ~/ros2_ws/src -path "*/so_arm_bringup/config/ros2_controllers.yaml" | head -1)
RVIZ_CONFIG=$(find ~/ros2_ws/src -path "*/so_arm_bringup/config/hardware_view.rviz" | head -1)

# Generate URDF
echo "Generating URDF with hardware interface..."
ROBOT_DESCRIPTION=$(xacro "$URDF_XACRO" use_hardware:=true)

if [ -z "$ROBOT_DESCRIPTION" ]; then
    echo "❌ Error: Failed to generate URDF"
    exit 1
fi

# Replace hardcoded port in URDF with actual serial port
ROBOT_DESCRIPTION=$(echo "$ROBOT_DESCRIPTION" | sed "s|/dev/ttyACM0|$SERIAL_PORT|g")

echo ""
echo "=========================================="
echo "  Launching ROS2 Hardware Interface"
echo "=========================================="
echo ""
echo "  Serial Port: $SERIAL_PORT"
echo "  Controllers: $CONTROLLERS_YAML"
echo "  Components:"
echo "    ✅ robot_state_publisher (TF2)"
echo "    ✅ ros2_control_node (Hardware)"
echo "    ✅ joint_state_broadcaster"
echo "    ✅ arm_controller"
echo "    ❌ MoveIt (not started)"
echo ""
echo "  检测 EE 位置:"
echo "    ros2 run tf2_ros tf2_echo base_link jaw"
echo ""
echo "=========================================="
echo ""

# Cleanup function
cleanup() {
    echo ""
    echo "Stopping all ROS2 nodes..."
    kill $RSP_PID $CTRL_PID $RVIZ_PID 2>/dev/null
    wait 2>/dev/null
    echo "All nodes stopped."
}
trap cleanup SIGINT SIGTERM

# 1. Robot State Publisher
echo "[1/4] Starting robot_state_publisher..."
ros2 run robot_state_publisher robot_state_publisher \
    --ros-args -p robot_description:="$ROBOT_DESCRIPTION" &
RSP_PID=$!
sleep 1

# 2. ROS2 Control Node
echo "[2/4] Starting ros2_control_node..."
ros2 run controller_manager ros2_control_node \
    --ros-args \
    -p robot_description:="$ROBOT_DESCRIPTION" \
    --params-file "$CONTROLLERS_YAML" \
    --remap /controller_manager/robot_description:=/robot_description &
CTRL_PID=$!
sleep 2

# 3. Joint State Broadcaster
echo "[3/4] Spawning joint_state_broadcaster..."
ros2 run controller_manager spawner joint_state_broadcaster \
    --controller-manager /controller_manager
sleep 1

# 4. Arm Controller
echo "[4/4] Spawning arm_controller..."
ros2 run controller_manager spawner arm_controller \
    --controller-manager /controller_manager

echo ""
echo "=========================================="
echo "  ✅ ROS2 Hardware Interface Ready!"
echo "=========================================="
echo ""
echo "  可用命令:"
echo "    # 查看 EE 位置 (实时)"
echo "    ros2 run tf2_ros tf2_echo base_link jaw"
echo ""
echo "    # 查看关节状态"
echo "    ros2 topic echo /joint_states"
echo ""
echo "    # 查看所有 TF"
echo "    ros2 run tf2_tools view_frames"
echo ""
echo "  Press Ctrl+C to stop"
echo ""

# Optional: Start RViz for visualization
if [ -n "$RVIZ_CONFIG" ] && [ -f "$RVIZ_CONFIG" ]; then
    read -p "启动 RViz 可视化? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Starting RViz..."
        ros2 run rviz2 rviz2 -d "$RVIZ_CONFIG" &
        RVIZ_PID=$!
    fi
fi

# Wait for processes
wait $RSP_PID $CTRL_PID 2>/dev/null

echo ""
echo "ROS2 has stopped."
