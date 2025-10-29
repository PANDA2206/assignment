# EdgeDetectionResult.msg
# Message structure for the result of edge detection on a single image.

# Header for timestamp and frame information
Header header

# Absolute or relative path of the input image
string input_path

# Path of the saved edge image (optional, empty if not saved)
string output_path

# Number of edge pixels detected in the result
uint32 num_edge_pixels

# 3D point cloud of edge points
sensor_msgs/PointCloud2 edge_points

# Camera info used for 3D projection
sensor_msgs/CameraInfo camera_info

# Edge detection parameters used
float32 canny_low
float32 canny_high
