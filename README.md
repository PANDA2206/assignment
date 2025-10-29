# Vision Programming Challenge – Edge Detection Toolkit

This repository contains a reference implementation for the edge-detection coding challenge.  It provides a robust, modular image processing pipeline, ROS integrations for RGB-D data, and automation helpers so the different evaluation tasks can be reproduced quickly.

- **Basic task** – run the standalone edge detector against still images, optionally saving overlays.
- **Vision_ROS task** – consume RGB-D frames from bag files, publish edge overlays, and produce 3D edge point clouds.
- **Robot_ROS task** – visualise edge points alongside the robot model in RViz.
- **Advanced** – run every element above end-to-end.


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
edge_detection/
├── data/                     # Sample images & results folder
├── launch/                   # ROS launch files
├── msg/, srv/                # Message/service definitions for ROS tasks
├── scripts/                  # Core edge detector + utility nodes
│   ├── edge_detector.py      # Standalone CLI & library entry point
│   ├── edge_from_rgbd_node.py# RGB-D processing node (Vision_ROS)
│   ├── edge_markers_node.py  # RViz marker publisher (Robot_ROS)
│   ├── edge_service_node.py  # Edge detection service entry point
│   ├── process_manager.py    # Shared helper for automation scripts
│   ├── ros_task_manager.py   # Automates Robot_ROS workflow
│   └── vision_task_manager.py# Automates Vision_ROS workflow
└── include/, src/            # C++ stubs (if you port the pipeline)
```


## 3. Basic edge detection workflow

The `scripts/edge_detector.py` CLI accepts either a single image or an entire directory.

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

The Python module can also be imported and used programmatically:

```python
from edge_detector import EdgeDetector
det = EdgeDetector(method="canny", auto_threshold=True)
edges = det.detect_edges(image)
overlay = det.draw_edges_green(image, edges)
```


## 4. ROS automation helpers

Running the ROS pipelines normally requires four separate terminals.  The manager scripts in `scripts/` launch the same stack programmatically and keep every process in sync.  Both helpers expect that the workspace and ROS distribution setup scripts are available.

### 4.0 Unified manager node

Prefer a single entry point that mirrors the sample code snippet you shared? Use `scripts/manager_node.py` to select either workflow:

```bash
python3 scripts/manager_node.py \
  --mode ros \
  --workspace-setup ~/catkin_workspace/devel/setup.bash \
  --bag-file /home/pankaj/Desktop/withpointcloud.bag \
  --start-relay

python3 scripts/manager_node.py \
  --mode vision \
  --workspace-setup ~/catkin_workspace/devel/setup.bash \
  --bag-file /home/pankaj/Desktop/withoutpointcloud.bag
```

The CLI matches the example structure—arguments are parsed upfront and forwarded to the appropriate manager.

### 4.1 Robot_ROS pipeline (RGB-D → RViz markers)

```bash
python3 scripts/ros_task_manager.py \
  --workspace-setup ~/catkin_workspace/devel/setup.bash \
  --bag-file /home/pankaj/Desktop/withpointcloud.bag \
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
python3 scripts/vision_task_manager.py \
  --workspace-setup ~/catkin_workspace/devel/setup.bash \
  --bag-file /home/pankaj/Desktop/withoutpointcloud.bag
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
```


## 6. Algorithm overview

The `EdgeDetector` class follows a multi-stage pipeline tuned for checkerboard imagery:

1. **Contrast & denoise** – optional CLAHE, configurable Gaussian/median/bilateral filtering, plus fast non-local means for stubborn sensor noise.
2. **Adaptive thresholds** – gradient statistics yield robust high/low bounds even across lighting changes.
3. **Multi-scale Canny** – fine and coarse responses are fused, ensuring thin lines are retained without losing global contours.
4. **Morphology & cleanup** – close gaps, remove isolated blobs using relative/absolute area thresholds, and thin the result for crisp overlays.
5. **Optional thinning/overlay** – skeletonise edges and paint them bright green on the original RGB frame when requested.

Parameters are exposed through CLI switches and ROS dynamic reconfigure hooks so you can trade off sensitivity vs. noise suppression per dataset.


## 7. Extending or customising

- **Additional filters:** Drop new gradient operators into `EdgeDetector.VALID_METHODS` and extend the dispatch table.
- **Service/API integration:** The ROS service defined in `srv/` demonstrates how to wrap the detector; client examples live in `scripts/edge_service_node.py`.
- **Robot visualisation:** `scripts/edge_markers_node.py` shows how 3D edge points are published as `visualization_msgs/MarkerArray` objects.
- **Testing:** Add your own unit tests under `tests/` (not provided here) and run them with `pytest`.


## 8. Troubleshooting

- **`ModuleNotFoundError: cv2`** – install OpenCV for the active interpreter (`python3 -m pip install opencv-python`).
- **`roslaunch` cannot find files** – ensure you sourced both `/opt/ros/noetic/setup.bash` and `<workspace>/devel/setup.bash` before launching.
- **Bag playback too fast** – change `--rosbag-extra-args` to `--clock --rate 0.5` (or pass `--pause`).
- **RViz shows no data** – verify `/edge_points` or image topics exist using `rostopic list`; if empty, confirm the bag file path and that `/use_sim_time` is set.


## 9. Roadmap / possible improvements

- Add automated tests for the multi-scale Canny thresholds and blob cleanup heuristics.
- Port the Python pipeline to C++ to match the challenge’s preferred language.
- Provide Docker images pre-configured with ROS Noetic, the workspace, and sample bag files.
- Integrate dynamic reconfigure for live parameter tuning during ROS playback.

Happy hacking!  If you run into issues, capture the CLI output and the commands used so we can reproduce and assist.
