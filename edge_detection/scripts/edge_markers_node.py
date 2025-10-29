#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
edge_markers_node.py
PointCloud2 -> RViz Markers (green points) with optional TF transform.

ROS params (private, ~):
  ~cloud_topic   (str)  : input point cloud topic (default: "/edge_points")
  ~target_frame  (str)  : frame to visualize in (default: "base_link")
  ~stride        (int)  : subsample every Nth point (default: 2)
  ~max_points    (int)  : cap number of points per message (default: 30000)
  ~scale         (float): marker point size in meters (default: 0.004)
  ~lifetime      (float): marker lifetime seconds (default: 0.05)
  ~z_colorize    (bool) : color points by Z (else solid green) (default: false)
"""

import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
import tf2_ros
import tf.transformations as tft


def transform_points_xyz(points_xyz: np.ndarray, tf_mat_4x4: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to Nx3 points."""
    homo = np.concatenate([points_xyz, np.ones((points_xyz.shape[0], 1), dtype=points_xyz.dtype)], axis=1)
    out = homo @ tf_mat_4x4.T
    return out[:, :3]


class EdgeMarkersNode:
    def __init__(self):
        # --- params
        self.cloud_topic  = rospy.get_param("~input_cloud", "/edge_points")
        self.target_frame = rospy.get_param("~target_frame", "camera_depth_optical_frame")
        self.stride      = int(rospy.get_param("~stride", 2))
        self.max_points  = int(rospy.get_param("~max_points", 60000))
        self.scale      = float(rospy.get_param("~point_size", 0.03))
        self.lifetime   = float(rospy.get_param("~lifetime", 0.1))
        self.alpha      = float(rospy.get_param("~alpha", 1.0))
        self.marker_ns = rospy.get_param("~marker_ns", "edge_markers")
        # Use latest TF regardless of timestamps
        self.use_latest_tf = True
        
        # --- pubs/subs
        self.pub = rospy.Publisher("~markers", Marker, queue_size=2)
        
        # --- TF setup
        self.tf_buf = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_lst = tf2_ros.TransformListener(self.tf_buf)
        
        # Wait for TF and subscribe
        rospy.sleep(1.0)
        self.sub = rospy.Subscriber(self.cloud_topic, PointCloud2, self._cloud_cb, queue_size=2)

        rospy.loginfo("Edge markers node initialized. Input: %s, Target frame: %s",
                      self.cloud_topic, self.target_frame)

    def _cloud_cb(self, msg: PointCloud2):
        # Get source frame and timestamp
        src_frame = msg.header.frame_id or self.target_frame
        stamp = rospy.Time(0)  # Always use latest TF
        T = None
        
        try:
            # Get transform
            tr = self.tf_buf.lookup_transform(self.target_frame, src_frame, stamp, 
                                            rospy.Duration(1.0))
            t = tr.transform.translation
            q = tr.transform.rotation
            T = tft.quaternion_matrix([q.x, q.y, q.z, q.w])
            T[0:3, 3] = [t.x, t.y, t.z]
            out_frame = self.target_frame
        except Exception as e:
            rospy.logwarn_throttle(5.0, "Using source frame: %s", str(e))
            out_frame = src_frame

        # Collect points (subsampled)
        pts = []
        for i, p in enumerate(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)):
            if (i % self.stride) != 0:
                continue
            pts.append((p[0], p[1], p[2]))
            if len(pts) >= self.max_points:
                break

        if not pts:
            return

        P = np.asarray(pts, dtype=np.float32)
        if T is not None and out_frame == self.target_frame:
            P = transform_points_xyz(P, T)

        # Build marker
        mk = Marker()
        mk.header.stamp = rospy.Time.now() 
        mk.header.frame_id = out_frame
        mk.ns = self.marker_ns
        mk.id = 0
        mk.type = Marker.POINTS
        mk.action = Marker.ADD
        mk.scale.x = self.scale
        mk.scale.y = self.scale
        mk.lifetime = rospy.Duration(self.lifetime)
        

        # Points
        mk.points = [Point(x=float(x), y=float(y), z=float(z)) for x, y, z in P]

        # Set marker color (green with configured alpha)
        mk.color.r = 0.0
        mk.color.g = 1.0
        mk.color.b = 0.0
        mk.color.a = self.alpha
            
        # Log marker info (throttled)
        rospy.loginfo_throttle(5.0, "Published %d edge points in frame %s", 
                             len(mk.points), mk.header.frame_id)

        self.pub.publish(mk)


def main():
    rospy.init_node("edge_markers_node")
    EdgeMarkersNode()
    rospy.spin()


if __name__ == "__main__":
    main()

