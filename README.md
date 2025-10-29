# Vision Programming Challenge – Edge Detection Toolkit

This repository provides a reference implementation for edge detection, featuring a modular image-processing pipeline, ROS integrations for RGB-D data, and automation helpers—primarily the manager_node—so evaluation tasks can be reproduced quickly.

- **Basic task** – run the standalone edge detector against still images, optionally saving overlays.
- **Vision_ROS task** – consume RGB or RGB-D frames from bag files, publish edge overlays, and produce 3D edge point clouds.
- **Robot_ROS task** – visualise edge points alongside the robot model in RViz.
- **Advanced** – run every element above end-to-end.  See the `result/` folder for demo recordings generated with these scripts.


## 1. Prerequisites

| Component | Notes |
|-----------|-------|
| ROS Noetic | Install and configure for your platform. Ensure `roscore`, `roslaunch`, and `rosbag` are on your `PATH`. |
| Catkin workspace | Clone this package into `<workspace>/src`, then build with `catkin_make` or `catkin build`. |
| Python 3.8+ | The scripts target Python ≥3.8 (tested on 3.8.10). |
| Python dependencies | `numpy`, `opencv-python` (`pip install numpy opencv-python`). |
| Bag files | Obtain the `withpointcloud.bag` and `withoutpointcloud.bag` archives supplied with the challenge. |

After building, source the workspace before running any ROS commands:

```bash
source <workspace>/devel/setup.bash
```


## 2. Project layout

```
.
├── README.md
├── edge_detection/
│   ├── CMakeLists.txt
│   ├── data/
│   │   ├── results               # edge_detection result
│   ├── launch/                   # ROS launch files
│   ├── msg/                      # Message definitions
│   ├── scripts/
│   │   ├── edge_detector.py      # Standalone CLI & library entry point
│   │   ├── edge_from_rgbd_node.py# RGB-D processing node (Vision_ROS)
│   │   └── edge_markers_node.py  # RViz marker publisher (Robot_ROS)
│   └── srv/                      # Service definitions
├── manager_node.py               # Unified CLI wrapper (select ROS workflow)
├── process_manager.py            # Shared process orchestration helpers
├── ros_task_manager.py           # Robot_ROS automation script
├── vision_task_manager.py        # Vision_ROS automation script
└── result/                       # Sample run recordings (.webm)
```



## 3. Basic edge detection workflow

The `edge_detection/scripts/edge_detector.py` CLI accepts either a single image or an entire directory.

```bash
# Run on the sample dataset and save binary edge masks
rosrun edge_detection edge_detector.py \
  --dir "$(rospack find edge_detection)/data" \
  --method canny --auto-threshold --save

# Save green overlays in addition to the binary outputs
rosrun edge_detection edge_detector.py \
  --dir "$(rospack find edge_detection)/data" \
  --method canny --auto-threshold --save --save-overlay
```

Key switches:

- `--method {canny,sobel,laplacian,scharr}` – pick the detector backend.
- `--auto-threshold` – derive Canny thresholds from local gradient statistics.
- `--min-blob-area`, `--min-blob-area-px` – scrub speckle-sized components.
- `--nlm-h` – add non-local means denoising before Canny (set to 0 to disable).
- `--save-overlay` – write an RGB copy with the detected edges painted bright green.


The `EdgeDetector` class follows a multi-stage pipeline tuned for checkerboard imagery:

1. **Multiple detction method** – canny,sobel,lapacian,schaar and method included
2. **Contrast & denoise** – optional CLAHE, configurable Gaussian/median/bilateral filtering, plus fast non-local means for stubborn sensor noise.
3. **Adaptive thresholds** – gradient statistics yield robust high/low bounds even across lighting changes.
4. **Multi-scale Canny** – fine and coarse responses are fused, ensuring thin lines are retained without losing global contours.
5. **Morphology & cleanup** – close gaps, remove isolated blobs using relative/absolute area thresholds, and thin the result for crisp overlays.
6. **Optional thinning/overlay** – skeletonise edges and paint them bright green on the original RGB frame when requested.

Parameters are exposed through CLI switches and ROS dynamic reconfigure hooks so you can trade off sensitivity vs. noise suppression per dataset


## 4. ROS automation helpers for ros_task and vision_task both

Running the ROS pipelines normally requires four separate terminals.  The manager scripts at the repository root launch the same stack programmatically and keep every process in sync.  Both helpers expect that the workspace and ROS distribution setup scripts are available.

### 4.0 Unified manager node

Prefer a single entry point that mirrors the sample code snippet you shared? Use the root-level `manager_node.py` to select either workflow:

```bash
python3 manager_node.py \
  --mode ros \
  --workspace-setup ~/catkin_workspace/devel/setup.bash \
  --bag-file /path/withpointcloud.bag \
  --start-relay

python3 manager_node.py \
  --mode vision \
  --workspace-setup ~/catkin_workspace/devel/setup.bash \
  --bag-file /path/withoutpointcloud.bag
```

The CLI matches the example structure—arguments are parsed upfront and forwarded to the appropriate manager.

### 4.1 Robot_ROS pipeline (RGB-D → RViz markers)

```bash
python3 ros_task_manager.py \
  --workspace-setup ~/catkin_workspace/devel/setup.bash \
  --bag-file /path/withpointcloud.bag \
  --start-relay
```

What it does:

1. Starts `roscore`.
2. Sets `/use_sim_time` so nodes follow bag timestamps.
3. Launches the robot model with `mira_picker display.launch` (override the gripper with `--gripper-name`, toggle state publishers via `--publish-joint-state`, `--publish-robot-state`).
4. Plays the point-cloud bag (`--rosbag-extra-args` lets you add `--pause`, custom rate, etc.).
5. Launches `edge_markers_from_cloud.launch` to extract 3D edge points and publish RViz markers.
6. Optionally starts a topic relay from `/camera/depth_registered/points` to `/edge_points` when `--start-relay` is supplied.

Press `Ctrl+C` to stop; the script tears every subprocess down in reverse order.

### 4.2 Vision_ROS pipeline (RGB frames → edge overlays)

```bash
python3 vision_task_manager.py \
  --workspace-setup ~/catkin_workspace/devel/setup.bash \
  --bag-file /path/withoutpointcloud.bag
```

This launches the same robot model, replays the RGB-only bag, and runs `vision_edges.launch` to visualise the detected edges in RViz.  The same flags for `--gripper-name`, `--publish-joint-state`, `--publish-robot-state`, and `--rosbag-extra-args` are available.


## 5. Manual ROS launch sequence (for reference)

If you prefer to drive each terminal yourself, replicate the original instructions:

```
# Terminal 1
roscore

# Terminal 2
rosparam set /use_sim_time true
source <workspace>/devel/setup.bash
roslaunch mira_picker display.launch gripper_name:=robotiq2f_140 \
  publish_joint_state:=false publish_robot_state:=false

# Terminal 3 (RGB-D bag)
rosbag play --clock -l /path/to/withpointcloud.bag

# Terminal 4 (Robot_ROS)
roslaunch edge_detection edge_markers_from_cloud.launch

# Terminal 3 (Vision_ROS)
rosbag play --clock -l /path/to/withoutpointcloud.bag

# Terminal 4 (Vision_ROS)
roslaunch edge_detection vision_edges.launch

# Terminal 5 (Robot_ROS)
rosrun topic_tools relay /camera/depth_registered/points /edge_points
```




## 6. Results
The `result/` directory contains screen captures from the automated workflows:
- `ros_task1.webm` – Robot_ROS run showing the robot model and edge markers in RViz.
- `ros_task2.webm` – Robot_ROS run with the synchronized camera view.
- `vision_task1.webm` – Vision_ROS run highlighting the edge overlays in RViz.
- `vision_task2.webm` – Vision_ROS run focused on the RGB playback stream.
- `camera_edge_view.webm` – Combined camera feed with edge overlays for quick review.
