#!/usr/bin/env python3
"""
Move SO-ARM robot through joint positions from point.csv
按照 point.csv 中的关节角度顺序移动机械臂
"""
import csv
import sys
import time
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration


class ArmController(Node):
    def __init__(self):
        super().__init__("arm_controller_node")
        self.action_client = ActionClient(
            self, FollowJointTrajectory, "/arm_controller/follow_joint_trajectory"
        )
        self.get_logger().info("Arm controller node initialized")
        
    def get_current_joint_positions(self):
        """Get current joint positions from /joint_states"""
        from sensor_msgs.msg import JointState
        from rclpy.qos import qos_profile_sensor_data
        
        sub = self.create_subscription(
            JointState, 
            "/joint_states", 
            lambda msg: setattr(self, '_current_joint_state', msg),
            qos_profile_sensor_data
        )
        
        # Wait for one message
        self._current_joint_state = None
        timeout = time.time() + 2.0
        while self._current_joint_state is None and time.time() < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
        
        self.destroy_subscription(sub)
        
        if self._current_joint_state is None:
            return None
        
        # Create a dict for easy lookup
        current_positions = {}
        for i, name in enumerate(self._current_joint_state.name):
            current_positions[name] = self._current_joint_state.position[i]
        
        return current_positions
    
    def move_to_joint_positions(self, joint_positions, joint_names, move_time=0.7, hold_time=0.7):
        """
        Move robot to specified joint positions
        
        Args:
            joint_positions: List of joint angles [Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw]
            joint_names: List of joint names
            move_time: Time to reach the position (seconds)
            hold_time: Time to hold at the position (seconds)
        """
        # Wait for action server
        if not self.action_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("arm_controller action server not available")
            return False
        
        # Get current positions
        current_positions = self.get_current_joint_positions()
        if current_positions is None:
            self.get_logger().warn("Could not get current joint positions, using target positions as start")
            start_positions = joint_positions
        else:
            start_positions = [current_positions.get(name, 0.0) for name in joint_names]
        
        # Create trajectory message with start and end points
        traj = JointTrajectory()
        traj.joint_names = joint_names
        
        # Start point (current position) at time 0
        start_point = JointTrajectoryPoint()
        start_point.positions = start_positions
        start_point.velocities = [0.0] * len(joint_names)
        start_point.accelerations = [0.0] * len(joint_names)
        start_point.time_from_start = Duration(sec=0, nanosec=0)
        
        # End point (target position)
        end_point = JointTrajectoryPoint()
        end_point.positions = joint_positions
        end_point.velocities = [0.0] * len(joint_names)
        end_point.accelerations = [0.0] * len(joint_names)
        end_point.time_from_start = Duration()
        end_point.time_from_start.sec = int(move_time)
        end_point.time_from_start.nanosec = int((move_time - int(move_time)) * 1e9)
        
        traj.points = [start_point, end_point]
        
        # Create and send goal
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj
        
        self.get_logger().info(f"Moving from {[f'{p:.4f}' for p in start_positions]} to {[f'{p:.4f}' for p in joint_positions]}")
        
        future = self.action_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)
        goal_handle = future.result()
        
        if not goal_handle.accepted:
            self.get_logger().error("Trajectory rejected")
            return False
        
        self.get_logger().info("Trajectory accepted, executing...")
        
        # Wait for execution to complete
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=move_time + 1.0)
        
        # Wait additional hold time
        time.sleep(hold_time)
        
        return True


def read_csv_points(csv_file):
    """Read joint positions from CSV file"""
    points = []
    joint_names = None
    
    try:
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if i == 0:  # Header row
                    joint_names = row
                    continue
                if not row or all(not cell.strip() for cell in row):  # Skip empty rows
                    continue
                # Convert to float
                positions = [float(x) for x in row]
                points.append(positions)
    except FileNotFoundError:
        print(f"Error: File {csv_file} not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)
    
    return points, joint_names


def main():
    csv_file = "point.csv"
    move_time = 0.7  # Time to reach position (seconds)
    hold_time = 0.7  # Time to hold at position (seconds)
    
    # Read points from CSV
    print(f"Reading joint positions from {csv_file}...")
    points, joint_names = read_csv_points(csv_file)
    
    if not points:
        print("Error: No points found in CSV file")
        sys.exit(1)
    
    if not joint_names:
        # Default joint names if header is missing
        joint_names = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
    
    print(f"Found {len(points)} points (regions 1-{len(points)})")
    print(f"Joint names: {joint_names}")
    print(f"Move time: {move_time}s, Hold time: {hold_time}s")
    print()
    
    # Initialize ROS2
    rclpy.init()
    controller = ArmController()
    
    try:
        # Wait a bit for action server to be ready
        print("Waiting for action server...")
        time.sleep(1.0)
        print("✅ Ready!")
        print()
        
        # Interactive loop: ask user for region number
        while True:
            try:
                print(f"Available regions: 1-{len(points)}")
                user_input = input("Enter region number to move to (or 'q' to quit): ").strip()
                
                if user_input.lower() == 'q':
                    print("Exiting...")
                    break
                
                region_num = int(user_input)
                
                if region_num < 1 or region_num > len(points):
                    print(f"❌ Invalid region number. Please enter a number between 1 and {len(points)}")
                    print()
                    continue
                
                # Get the point (region_num is 1-indexed, points list is 0-indexed)
                point = points[region_num - 1]
                
                print(f"Moving to region {region_num}: {[f'{p:.4f}' for p in point]}")
                success = controller.move_to_joint_positions(point, joint_names, move_time, hold_time)
                
                if success:
                    print(f"  ✅ Reached region {region_num} and held for {hold_time}s")
                else:
                    print(f"  ❌ Failed to reach region {region_num}")
                print()
                
            except ValueError:
                print("❌ Invalid input. Please enter a number or 'q' to quit.")
                print()
            except KeyboardInterrupt:
                print("\n⚠️  Interrupted by user")
                break
        
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
