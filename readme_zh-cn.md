
## 🧭YOLOv5-ROS 导航🧭

两个主要部分：

1. **YOLOv5-ROS 用于一般目标检测**
2. **YOLOv5-ROS 用于导航（计算到目标的距离和航向角）**

### 💥YOLOv5-ROS 用于目标检测

这一部分使用从 ROS 主题获取的图像进行标准的 YOLOv5 目标检测

###### 脚本 `yolo_v5_nonvisual.py`

订阅 ROS 图像主题，接收图像数据并进行目标检测，然后将结果发布到指定的 ROS 主题

###### 启动文件 `yolo_v5_nonvisual.launch`

配置并启动 YOLOv5 和 ROS 集成节点

```
- 参数说明
yolov5_path: YOLOv5 模型所在的路径
use_cpu: 是否使用 CPU 进行推理（默认为 false，使用 GPU）
weight_path: YOLOv5 模型权重文件的路径
image_topic: 订阅的图像主题名称
pub_topic: 发布检测结果的主题名称
camera_frame: 相机坐标系的名称
conf: 检测置信度阈值
```

### 💥YOLOv5-ROS 用于导航

这一部分扩展了 YOLOv5 目标检测，增加了计算目标距离和航向角的功能用于导航

###### 脚本 `yolo_v5_visual.py`

在目标检测的基础上，订阅激光雷达数据主题，结合相机和激光雷达的标定参数，计算出目标物体的距离和航向角，并将结果发布到指定的 ROS 主题

###### 启动文件 `yolo_v5_visual.launch`

配置并启动具有导航功能的 YOLOv5 和 ROS 集成节点

```
- 参数说明
yolov5_path: YOLOv5 模型所在的路径
use_cpu: 是否使用 CPU 进行推理（默认为 false，使用 GPU）
camera_matrix: 相机内参矩阵，用于图像矫正
dist_coeffs: 相机畸变系数，用于图像矫正
calibration_result: 相机与激光雷达的标定结果，用于计算距离，航向角和重投影
weight_path: YOLOv5 模型权重文件的路径
image_topic: 订阅的图像主题名称
pub_topic: 发布检测结果的主题名称
camera_frame: 相机坐标系的名称
conf: 检测置信度阈值
```

### ⭐使用方法

运行一般目标检测节点：

```bash
roslaunch yolov5_nonvisual yolo_v5_nonvisual.launch
```

运行具有导航功能的目标检测节点需要先启动激光雷达节点来接收数据，然后启动导航节点：

```bash
roslaunch ydlidar ydlidar.launch 
roslaunch yolov5_ros_visual yolo_v5_visual.launch
```

运行具有导航功能的目标检测节点结果如 `result.gif` 所示

### ⭐如何进行相机-雷达联合标定获得 `calibration_result`

参考 [TurtleZhong/camera_lidar_calibration_v2: ROS VERSION: A tool used for calibrating a 2D laser range finder (LRF) and camera](https://github.com/TurtleZhong/camera_lidar_calibration_v2) 以获取详细说明
