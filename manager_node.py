#!/usr/bin/env python3
"""Entry point that mirrors a typical ROS-style manager executable."""

import argparse
import sys

import ros_task_manager
import vision_task_manager

from typing import Optional
def main() -> int:
    parser = argparse.ArgumentParser(description="Manager node for edge_detection")
    parser.add_argument("--mode", choices=["ros", "vision"], default="ros",
                        help="Select which automation workflow to launch")
    parser.add_argument("--ros-setup", default="/opt/ros/noetic/setup.bash",
                        help="Path to your ROS distribution setup script")
    parser.add_argument("--workspace-setup", required=True,
                        help="Path to the catkin workspace devel/setup.bash")
    parser.add_argument("--bag-file", required=True,
                        help="ROS bag file to replay")
    parser.add_argument("--gripper-name", default="robotiq2f_140",
                        help="Gripper name passed to mira_picker display.launch")
    parser.add_argument("--publish-joint-state", action="store_true",
                        help="Publish joint_state from the robot description launch")
    parser.add_argument("--publish-robot-state", action="store_true",
                        help="Publish robot_state from the robot description launch")
    parser.add_argument("--rosbag-extra-args", default="--clock -l",
                        help="Extra arguments appended to rosbag play")
    parser.add_argument("--start-relay", action="store_true",
                        help="When in ROS mode, launch the topic_tools relay for edge points")
    parser.add_argument("--relay-input", default="/camera/depth_registered/points",
                        help="Input topic for topic_tools relay (ROS mode only)")
    parser.add_argument("--relay-output", default="/edge_points",
                        help="Output topic for topic_tools relay (ROS mode only)")
    parser.add_argument("--launch-delay", type=float, default=3.0,
                        help="Delay between launching long-running processes")

    args = parser.parse_args()

    if args.mode == "ros":
        return ros_task_manager.main(
            ros_setup=args.ros_setup,
            workspace_setup=args.workspace_setup,
            bag_file=args.bag_file,
            gripper_name=args.gripper_name,
            publish_joint_state=args.publish_joint_state,
            publish_robot_state=args.publish_robot_state,
            rosbag_extra_args=args.rosbag_extra_args,
            start_relay=args.start_relay,
            relay_input=args.relay_input,
            relay_output=args.relay_output,
            launch_delay=args.launch_delay,
        )

    return vision_task_manager.main(
        ros_setup=args.ros_setup,
        workspace_setup=args.workspace_setup,
        bag_file=args.bag_file,
        gripper_name=args.gripper_name,
        publish_joint_state=args.publish_joint_state,
        publish_robot_state=args.publish_robot_state,
        rosbag_extra_args=args.rosbag_extra_args,
        launch_delay=args.launch_delay,
    )


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
