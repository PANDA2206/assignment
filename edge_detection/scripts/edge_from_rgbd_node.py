#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RGB-D -> robust 2D edges (via EdgeDetector) -> back-project -> /edge_points

Topics in:
  ~rgb_topic   (Image)       default /camera/color/image_raw
  ~depth_topic (Image)       default /camera/depth/image_rect_raw
  ~camera_info (CameraInfo)  default /camera/depth/camera_info

Topic out:
  ~cloud_topic (PointCloud2) default /edge_points

Service:
  ~set_edge_params (SetEdgeParams) live-tune canny/stride/depth limits etc.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import cv2
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
import sensor_msgs.point_cloud2 as pc2
from message_filters import ApproximateTimeSynchronizer, Subscriber

# Local detector
from edge_detector import EdgeDetector

# ROS srv/msg (assumed to exist in your package)
from edge_detection.srv import SetEdgeParams, SetEdgeParamsResponse
try:
    from edge_detection.msg import EdgeDetectionResult
except Exception:
    # Fallback tiny stub so the code still runs even if msg isn't compiled
    from collections import namedtuple
    EdgeDetectionResult = namedtuple("EdgeDetectionResult", ["input_path","output_path","num_edge_pixels"])

class EdgeFromRGBD:
    def __init__(self):
        # ---- Params (topics)
        self.rgb_topic    = rospy.get_param("~rgb_topic",   "/camera/color/image_raw")
        self.depth_topic  = rospy.get_param("~depth_topic", "/camera/depth/image_rect_raw")
        self.info_topic   = rospy.get_param("~camera_info", "/camera/depth/camera_info")
        self.cloud_topic  = rospy.get_param("~cloud_topic", "/edge_points")

        # ---- Behavior/config
        self.edge_on_rgb      = True  # Always use RGB for edge detection
        self.strict_assert    = bool(rospy.get_param("~strict_assert", True))
        self.frame_override   = rospy.get_param("~frame_override", "")
        self.stamp_latest     = bool(rospy.get_param("~stamp_latest", True))

        # Edge detection parameters
        self.canny_low   = float(rospy.get_param("~canny_low", 200.0))
        self.canny_high  = float(rospy.get_param("~canny_high", 250.0))
        self.stride      = int(rospy.get_param("~stride", 2))
        self.max_points  = int(rospy.get_param("~max_points", 60000))
        self.min_depth   = float(rospy.get_param("~min_depth", 0.15))
        self.max_depth   = float(rospy.get_param("~max_depth", 6.0))
        self.depth_scale = float(rospy.get_param("~depth_scale", 1000.0))
        self.sync_slop   = float(rospy.get_param("~sync_slop", 0.2))
        self.sync_queue  = int(rospy.get_param("~sync_queue", 10))

        # Initialize objects
        self.bridge = CvBridge()
        self.cam_info = None
        self.det = EdgeDetector(
            method="canny",
            canny_low=self.canny_low,
            canny_high=self.canny_high,
            denoise=None,
            use_clahe=False,
            pad=2
        )

        # ---- ROS I/O
        # Publishers for point cloud and edge visualization
        self.pub_cloud = rospy.Publisher(self.cloud_topic, PointCloud2, queue_size=1)
        self.pub_edges = rospy.Publisher("~edge_image", Image, queue_size=1)

        # Synchronized subscribers for RGB and depth
        self.sub_rgb   = Subscriber(self.rgb_topic, Image)
        self.sub_depth = Subscriber(self.depth_topic, Image)
        self.sync = ApproximateTimeSynchronizer([self.sub_rgb, self.sub_depth],
                                               queue_size=self.sync_queue, 
                                               slop=self.sync_slop)
        self.sync.registerCallback(self.cb_rgbd)

        # Camera info subscriber
        self.sub_info = rospy.Subscriber(self.info_topic, CameraInfo, 
                                       self.cb_info, queue_size=1)

        # Service for live parameter tuning
        self.srv = rospy.Service("~set_edge_params", SetEdgeParams, 
                                self.handle_set_edge_params)

        rospy.loginfo("edge_from_rgbd_node: RGB=%s  Depth=%s  Info=%s -> %s | edge_on_rgb=%s strict=%s",
                      self.rgb_topic, self.depth_topic, self.info_topic, self.cloud_topic,
                      self.edge_on_rgb, self.strict_assert)

    # ---------- helpers ----------
    def _warn_or_fail(self, cond: bool, msg: str):
        if cond:
            return True
        if self.strict_assert:
            rospy.logerr(msg)
            raise RuntimeError(msg)
        else:
            rospy.logwarn_throttle(2.0, msg)
            return False

    @staticmethod
    def _to_mono(img):
        """Convert image to mono8 format."""
        if img.ndim == 2:
            return (img > 0).astype(np.uint8) * 255 if img.dtype != np.uint8 else img
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def _depth_at_pixels(self, depth, xs, ys):
        """Get depth values at given pixel coordinates."""
        if depth.dtype == np.uint16:
            return depth[ys, xs].astype(np.float32) / self.depth_scale
        elif depth.dtype in (np.float32, np.float64):
            return depth[ys, xs].astype(np.float32)
        else:
            raise RuntimeError(f"Unexpected depth dtype {depth.dtype}")

    @staticmethod
    def _remove_small_components(edges_mask: np.ndarray, min_pixels: int) -> np.ndarray:
        """Remove tiny connected components from a binary mono8 mask."""
        if min_pixels <= 1:
            return edges_mask
        _, labels = cv2.connectedComponents((edges_mask > 0).astype(np.uint8))
        keep = np.zeros_like(edges_mask, dtype=np.uint8)
        max_label = labels.max()
        for lab in range(1, max_label + 1):
            m = (labels == lab)
            if int(m.sum()) >= min_pixels:
                keep[m] = 255
        return keep

    # ---------- service ----------
    def handle_set_edge_params(self, req):
        # Use getattr to be robust to srv variations
        self.canny_low   = float(getattr(req, "canny_low", self.canny_low))
        self.canny_high  = float(getattr(req, "canny_high", self.canny_high))
        self.stride      = int(getattr(req, "stride", self.stride))
        self.min_depth   = float(getattr(req, "min_depth", self.min_depth))
        self.max_depth   = float(getattr(req, "max_depth", self.max_depth))

        # Update detector thresholds
        self.det.canny_low  = self.canny_low
        self.det.canny_high = self.canny_high

        rospy.loginfo("Updated params: canny=(%.1f, %.1f) stride=%d depth=[%.2f, %.2f]",
                      self.canny_low, self.canny_high, self.stride, self.min_depth, self.max_depth)
        return SetEdgeParamsResponse(ok=True, results=[])

    # ---------- callbacks ----------
    def cb_info(self, msg: CameraInfo):
        ok = self._warn_or_fail(msg.width > 0 and msg.height > 0, "CameraInfo has zero size.")
        ok = ok and self._warn_or_fail(msg.K[0] > 0 and msg.K[4] > 0, "Invalid intrinsics (fx/fy).")
        if ok:
            self.cam_info = msg

    def cb_rgbd(self, rgb_msg: Image, depth_msg: Image):
        if self.cam_info is None:
            rospy.logwarn_throttle(5.0, "Waiting for CameraInfo on %s", self.info_topic)
            return

        # Decode RGB and depth images
        try:
            rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")  # 16UC1/32FC1
        except Exception as e:
            rospy.logwarn(f"Error decoding images: {e}")
            return

        # Verify depth encoding
        depth_enc = depth_msg.encoding
        if not self._warn_or_fail(depth_enc in ("16UC1", "32FC1"),
                                  f"Unsupported depth encoding '{depth_enc}'"):
            return

        # Verify image sizes match
        w, h = self.cam_info.width, self.cam_info.height
        rh, rw = rgb.shape[:2]
        if not self._warn_or_fail((rw == w and rh == h),
                            f"CameraInfo {w}x{h} != RGB {rw}x{rh}"):
            return

        # Detect edges using our existing edge detector
        edges = self._to_mono(self.det.detect_edges(rgb))

        # Collect edge pixels with subsampling
        ys, xs = np.nonzero(edges)
        if ys.size == 0:
            rospy.logwarn_throttle(2.0, "No edge pixels in this frame.")
            return
        if self.stride > 1:
            xs = xs[::self.stride]
            ys = ys[::self.stride]

        # Get depth values at edge pixels
        Z = self._depth_at_pixels(depth, xs, ys)
        valid = np.isfinite(Z) & (Z > self.min_depth) & (Z < self.max_depth)
        xs, ys, Z = xs[valid], ys[valid], Z[valid]

        if xs.size == 0:
            rospy.logwarn_throttle(2.0, "No valid depth at edge pixels.")
            return

        # Limit number of points if needed
        if xs.size > self.max_points:
            sel = np.linspace(0, xs.size - 1, self.max_points).astype(np.int32)
            xs, ys, Z = xs[sel], ys[sel], Z[sel]

        # Back-project to 3D using camera parameters
        fx, fy = float(self.cam_info.K[0]), float(self.cam_info.K[4])
        cx, cy = float(self.cam_info.K[2]), float(self.cam_info.K[5])
        X = (xs - cx) / fx * Z
        Y = (ys - cy) / fy * Z
        pts = np.stack([X, Y, Z], axis=1).astype(np.float32)

        # Prepare and send messages
        header = depth_msg.header
        if self.frame_override:
            header.frame_id = self.frame_override
        if self.stamp_latest:
            header.stamp = rospy.Time(0)

        # Publish point cloud and edge image
        cloud = pc2.create_cloud_xyz32(header, pts.tolist())
        self.pub_cloud.publish(cloud)

        edge_msg = self.bridge.cv2_to_imgmsg(edges, encoding="mono8")
        edge_msg.header = header
        self.pub_edges.publish(edge_msg)

        rospy.loginfo_throttle(1.0, "edge_from_rgbd: edges=%d -> 3D points=%d (frame=%s)",
                              int(np.count_nonzero(edges)), int(pts.shape[0]), header.frame_id)

    @staticmethod
    def _to_mono(img):
        if img.ndim == 2:
            return (img > 0).astype(np.uint8) * 255 if img.dtype != np.uint8 else img
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def _depth_at_pixels(self, depth, xs, ys):
        if depth.dtype == np.uint16:
            return depth[ys, xs].astype(np.float32) / self.depth_scale
        elif depth.dtype in (np.float32, np.float64):
            return depth[ys, xs].astype(np.float32)
        else:
            raise RuntimeError(f"Unexpected depth dtype {depth.dtype}")

    def _rescue_with_neighbors(self, depth, xs, ys):
        H, W = depth.shape[:2]
        xs2, ys2, Z2 = [], [], []
        for u, v in zip(xs, ys):
            zbest = None; uu_best = u; vv_best = v
            for dv in (-1, 0, 1):
                vv = v + dv
                if vv < 0 or vv >= H: continue
                for du in (-1, 0, 1):
                    uu = u + du
                    if uu < 0 or uu >= W: continue
                    zraw = depth[vv, uu]
                    if depth.dtype == np.uint16:
                        z = float(zraw) / self.depth_scale
                    else:
                        z = float(zraw)
                    if np.isfinite(z) and (self.min_depth < z < self.max_depth):
                        zbest = z; uu_best = uu; vv_best = vv
                        break
                if zbest is not None:
                    break
            if zbest is not None:
                xs2.append(uu_best); ys2.append(vv_best); Z2.append(zbest)
        return (np.asarray(xs2, dtype=np.int32),
                np.asarray(ys2, dtype=np.int32),
                np.asarray(Z2,  dtype=np.float32))


def main():
    rospy.init_node("edge_from_rgbd_node")
    try:
        EdgeFromRGBD()
        rospy.spin()
    except RuntimeError as e:
        rospy.logfatal(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()

