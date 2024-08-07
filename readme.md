## 🧭YOLOv5-ROS Navigation🧭

Two main parts:

1. **YOLOv5-ROS for General Object Detection**
2. **YOLOv5-ROS for Navigation (calculating distance and heading angle to the target)**

### 💥YOLOv5-ROS for Object Detection

This part performs standard YOLOv5 object detection using images from a ROS topic.

###### Script `yolo_v5_nonvisual.py`

Subscribes to a ROS image topic, receives image data, performs object detection, and publishes the results to a specified ROS topic.

###### Launch File `yolo_v5_nonvisual.launch`

Configures and starts the YOLOv5 and ROS integration node.

```
- Parameter Description
yolov5_path: Path to the YOLOv5 model.
use_cpu: Whether to use CPU for inference (default is false, using GPU).
weight_path: Path to the YOLOv5 model weight file.
image_topic: Name of the image topic to subscribe to.
pub_topic: Name of the topic to publish detection results to.
camera_frame: Name of the camera coordinate frame.
conf: Detection confidence threshold.
```

### 💥YOLOv5-ROS for Navigation

This part extends YOLOv5 object detection to include distance and heading angle calculations for navigation.

###### Script `yolo_v5_visual.py`

Builds on object detection by subscribing to lidar data topics, combining camera and lidar calibration parameters, calculating the target object's distance and heading angle, and publishing the results to a specified ROS topic.

###### Launch File `yolo_v5_visual.launch`

Configures and starts the YOLOv5 and ROS integration node with navigation capabilities.

```
- Parameter Description
yolov5_path: Path to the YOLOv5 model.
use_cpu: Whether to use CPU for inference (default is false, using GPU).
camera_matrix: Camera intrinsic matrix for image rectification.
dist_coeffs: Camera distortion coefficients for image rectification.
calibration_result: Calibration results between the camera and lidar, used for distance, heading angle calculation, and reprojection.
weight_path: Path to the YOLOv5 model weight file.
image_topic: Name of the image topic to subscribe to.
pub_topic: Name of the topic to publish detection results to.
camera_frame: Name of the camera coordinate frame.
conf: Detection confidence threshold.
```

### ⭐Usage

To run the general object detection node:

```bash
roslaunch yolov5_nonvisual yolo_v5_nonvisual.launch
```

To run the navigation-enhanced object detection node, first start the lidar node to receive data, and then launch the navigation node:

```bash
roslaunch yolov5_ros_visual yolo_v5_visual.launch
```

The result of running the navigation-enhanced object detection node is shown in `result.gif`.


## License
This project is licensed under the MIT License. See the LICENSE file for details.

### ⭐How to Perform Camera-Lidar Calibration to Obtain `calibration_result`

Refer to [TurtleZhong/camera_lidar_calibration_v2: ROS VERSION: A tool used for calibrating a 2D laser range finder (LRF) and camera](https://github.com/TurtleZhong/camera_lidar_calibration_v2) for detailed instructions.
