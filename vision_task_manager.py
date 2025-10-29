#!/usr/bin/env python3
"""Automate the vision-only edge detection ROS pipeline."""

import argparse
import shlex
import sys
import time
from pathlib import Path
from typing import Optional
from process_manager import ProcessGroup, run_once, wait_for_ros_master


def _resolve_path(path: str, description: str) -> str:
    if not path:
        raise ValueError(f"{description} path must be provided")
    candidate = Path(path).expanduser().resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"Expected {description} at {candidate}")
    return str(candidate)


def _boolean_flag(value: bool) -> str:
    return "true" if value else "false"


def run_workflow(
    *,
    ros_setup: Optional[str],
    workspace_setup: str,
    bag_file: str,
    gripper_name: str,
    publish_joint_state: bool,
    publish_robot_state: bool,
    rosbag_extra_args: str,
    launch_delay: float,
) -> int:
    try:
        ros_setup_path = _resolve_path(ros_setup, "ROS setup") if ros_setup else None
        workspace_setup_path = _resolve_path(workspace_setup, "workspace setup")
        bag_path = _resolve_path(bag_file, "bag file")
    except (ValueError, FileNotFoundError) as exc:
        print(exc, file=sys.stderr)
        return 1

    env_sources = [src for src in (ros_setup_path, workspace_setup_path) if src]
    group = ProcessGroup(env_sources)

    print("[manager] starting roscore…")
    group.start("roscore", "roscore", delay=launch_delay)
    try:
        wait_for_ros_master(env_sources)
    except RuntimeError as exc:
        print(f"[manager] {exc}", file=sys.stderr)
        return 1

    print("[manager] setting /use_sim_time = true")
    run_once(env_sources, "rosparam set /use_sim_time true")

    display_cmd = (
        f"roslaunch mira_picker display.launch "
        f"gripper_name:={shlex.quote(gripper_name)} "
        f"publish_joint_state:={_boolean_flag(publish_joint_state)} "
        f"publish_robot_state:={_boolean_flag(publish_robot_state)}"
    )
    print("[manager] launching robot display")
    group.start("robot_display", display_cmd, delay=launch_delay)

    rosbag_cmd = f"rosbag play {rosbag_extra_args} {shlex.quote(bag_path)}"
    print("[manager] launching rosbag playback")
    group.start("rosbag_play", rosbag_cmd, delay=launch_delay)

    print("[manager] launching vision edge pipeline")
    group.start("vision_edges", "roslaunch edge_detection vision_edges.launch", delay=launch_delay)

    print("[manager] all processes launched – press Ctrl+C to stop")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[manager] received shutdown request")
    finally:
        group.shutdown()

    return 0


def main(
    *,
    ros_setup: Optional[str] = None,
    workspace_setup: Optional[str] = None,
    bag_file: Optional[str] = None,
    gripper_name: Optional[str] = None,
    publish_joint_state: bool = False,
    publish_robot_state: bool = False,
    rosbag_extra_args: Optional[str] = None,
    launch_delay: Optional[float] = None,
) -> int:
    if any(v is None for v in (workspace_setup, bag_file)):
        parser = argparse.ArgumentParser(description="Launch the vision edge detection workflow")
        parser.add_argument("--ros-setup", default="/opt/ros/noetic/setup.bash",
                            help="Path to the ROS distribution setup script")
        parser.add_argument("--workspace-setup", required=True,
                            help="Path to the catkin workspace devel/setup.bash")
        parser.add_argument("--bag-file", required=True,
                            help="Bag file containing RGB imagery without point clouds")
        parser.add_argument("--gripper-name", default="robotiq2f_140",
                            help="gripper_name passed to the robot display launch")
        parser.add_argument("--publish-joint-state", action="store_true",
                            help="Publish joint_state from the robot description launch")
        parser.add_argument("--publish-robot-state", action="store_true",
                            help="Publish robot_state from the robot description launch")
        parser.add_argument("--rosbag-extra-args", default="--clock -l",
                            help="Extra arguments appended to rosbag play")
        parser.add_argument("--launch-delay", type=float, default=3.0,
                            help="Delay between launching each long-running process")

        args = parser.parse_args()
        return run_workflow(
            ros_setup=args.ros_setup,
            workspace_setup=args.workspace_setup,
            bag_file=args.bag_file,
            gripper_name=args.gripper_name,
            publish_joint_state=args.publish_joint_state,
            publish_robot_state=args.publish_robot_state,
            rosbag_extra_args=args.rosbag_extra_args,
            launch_delay=args.launch_delay,
        )

    return run_workflow(
        ros_setup=ros_setup or "/opt/ros/noetic/setup.bash",
        workspace_setup=workspace_setup,
        bag_file=bag_file,
        gripper_name=gripper_name or "robotiq2f_140",
        publish_joint_state=publish_joint_state,
        publish_robot_state=publish_robot_state,
        rosbag_extra_args=rosbag_extra_args or "--clock -l",
        launch_delay=launch_delay if launch_delay is not None else 3.0,
    )


if __name__ == "__main__":
    sys.exit(main())
