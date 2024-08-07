#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

import rospy
import cv2
import torch
import numpy as np
import math
from time import time
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, LaserScan
from yolov5_ros.msg import BoundingBox, BoundingBoxes
from laser_geometry import LaserProjection
from utils.utils import scale_coords, non_max_suppression
import torch.backends.cudnn as cudnn
from sensor_msgs import point_cloud2

class Yolo_Dect:
    def __init__(self):
        # Load parameters from the ROS parameter server
        yolov5_path = rospy.get_param('/yolov5_path', '')
        weight_path = rospy.get_param('~weight_path', '')
        image_topic = rospy.get_param('~image_topic', '/usb_cam/image_raw')
        pub_topic = rospy.get_param('~pub_topic', '/yolov5/BoundingBoxes')
        self.camera_frame = rospy.get_param('~camera_frame', '')
        conf = rospy.get_param('~conf', '0.5')

        # Camera parameters
        self.camera_matrix = np.array(rospy.get_param('~camera_matrix'))
        self.dist_coeffs = np.array(rospy.get_param('~dist_coeffs'))

        # Load the YOLO model
        self.model = torch.load(weight_path, map_location='cuda')['model'].float()

        # Check if the model should use CPU or GPU
        if rospy.get_param('/use_cpu', 'false'):
            self.model.cpu()
        else:
            self.model.cuda()

        self.model.conf = conf
        self.color_image = None
        self.getImageStatus = False

        # Load class colors for bounding boxes
        self.classes_colors = {}

        # Subscribe to image and laser scan topics
        self.color_sub = rospy.Subscriber(image_topic, Image, self.image_callback, queue_size=1, buff_size=52428800)
        self.laser_sub = rospy.Subscriber('/scan', LaserScan, self.laser_callback, queue_size=1)

        self.boundingBoxes = BoundingBoxes()
        self.bridge = CvBridge()
        self.projector = LaserProjection()

        self.K = self.camera_matrix
        self.D = self.dist_coeffs
        self.Rcl, self.tcl = self.load_calibration_parameters()

        # Set up publishers for bounding boxes and detection images
        self.position_pub = rospy.Publisher(pub_topic, BoundingBoxes, queue_size=5)
        self.image_pub = rospy.Publisher('/yolov5/detection_image', Image, queue_size=1)

        self.yolo_boxes = []

        # Wait for the first image
        while not self.getImageStatus:
            rospy.loginfo("waiting for image.")
            rospy.sleep(2)

    def load_calibration_parameters(self):
        # Load calibration parameters from the ROS parameter server
        calibration_result = rospy.get_param('~calibration_result')
        q_x, q_y, q_z, q_w, t_x, t_y, t_z = calibration_result
        Rcl = np.array([
            [1 - 2*(q_y**2 + q_z**2), 2*(q_x*q_y - q_z*q_w), 2*(q_x*q_z + q_y*q_w)],
            [2*(q_x*q_y + q_z*q_w), 1 - 2*(q_x**2 + q_z**2), 2*(q_y*q_z - q_x*q_w)],
            [2*(q_x*q_z - q_y*q_w), 2*(q_y*q_z + q_x*q_w), 1 - 2*(q_x**2 + q_y**2)]
        ])
        tcl = np.array([[t_x], [t_y], [t_z]])
        return Rcl, tcl

    def undistort_image(self, image):
        """
        Undistort the input image using the camera matrix and distortion coefficients.
        """
        h, w = image.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h))
        undistorted_image = cv2.undistort(image, self.camera_matrix, self.dist_coeffs, None, new_camera_matrix)
        x, y, w, h = roi
        undistorted_image = undistorted_image[y:y+h, x:x+w]
        return undistorted_image

    def image_callback(self, image):
        """
        Callback function for the image topic. Performs YOLO detection on the image.
        """
        self.boundingBoxes = BoundingBoxes()
        self.boundingBoxes.header = image.header
        self.boundingBoxes.image_header = image.header
        self.getImageStatus = True

        # Convert the image to a numpy array and undistort it
        self.color_image = np.frombuffer(image.data, dtype=np.uint8).reshape(image.height, image.width, -1)
        self.color_image = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB)
        self.color_image = self.undistort_image(self.color_image)

        # Prepare the image for YOLO detection
        self.color_image_tensor = torch.from_numpy(self.color_image.astype(np.float32)).permute(2, 0, 1) / 255.0
        self.color_image_tensor = self.color_image_tensor.unsqueeze(0).to('cuda')

        # Perform YOLO detection
        start_time = time()
        results = self.model(self.color_image_tensor)[0]
        inference_time = time() - start_time

        results = non_max_suppression(results, self.model.conf, 0.5)

        # Process detections
        if results[0] is not None:
            for det in results:
                det[:, :4] = scale_coords(self.color_image_tensor[0].shape[1:], det[:, :4], self.color_image.shape).round()
                boxs = det.cpu().numpy()
                self.process_detections(boxs, image.height, image.width, inference_time)
                self.yolo_boxes = boxs

        cv2.waitKey(3)

    def laser_callback(self, laser_msg):
        """
        Callback function for the laser scan topic. Projects laser points onto the image and associates them with bounding boxes.
        """
        if not self.getImageStatus or len(self.yolo_boxes) == 0:
            return

        cloud = self.projector.projectLaser(laser_msg)
        pts_uv = []

        # Read points from the point cloud and project them onto the image
        for point in point_cloud2.read_points(cloud, skip_nans=True):
            x, y, z = point[:3]
            point_l = np.array([[x], [y], [z]])
            point_c = np.dot(self.Rcl, point_l) + self.tcl

            if point_c[2, 0] <= 0:
                continue

            point_c[0, 0] /= point_c[2, 0]
            point_c[1, 0] /= point_c[2, 0]
            point_c[2, 0] = 1.0

            uv = np.dot(self.K, point_c)
            pt_uv = (uv[0, 0], uv[1, 0])
            pts_uv.append((pt_uv, (x, y, z)))

        self.laser_pts_uv = pts_uv

        # Associate laser points with bounding boxes
        for bbox in self.boundingBoxes.bounding_boxes:
            center_x = (bbox.xmin + bbox.xmax) / 2
            center_y = (bbox.ymin + bbox.ymax) / 2

            closest_distance = float('inf')
            closest_point = None

            for pt_uv, (x, y, z) in pts_uv:
                if bbox.xmin <= pt_uv[0] <= bbox.xmax and bbox.ymin <= pt_uv[1] <= bbox.ymax:
                    distance = np.linalg.norm([x, y, z])
                    if distance < closest_distance:
                        closest_distance = distance
                        closest_point = (x, y, z)

            if closest_point:
                x, y, z = closest_point
                mean_distance = np.linalg.norm([x, y, z]) - 0.25
                heading_angle = np.arctan2(y, x)

                heading_angle *= (-1)
                if heading_angle > 0:
                    heading_angle -= math.pi/2
                if heading_angle < 0:
                    heading_angle += math.pi/2

                bbox.mean_distance = float(mean_distance)
                bbox.heading_angle = float(heading_angle)

        # Publish bounding boxes with associated distances and headings
        if self.boundingBoxes.bounding_boxes:
            self.position_pub.publish(self.boundingBoxes)
            print(self.boundingBoxes)
            print("----------------------------")
            self.yolo_boxes = []

    def process_detections(self, boxs, height, width, inference_time):
        """
        Process YOLO detections, draw bounding boxes on the image, and publish the results.
        """
        img = self.color_image.copy()
        self.boundingBoxes.bounding_boxes = []

        for box in boxs:
            boundingBox = BoundingBox()
            boundingBox.probability = np.float64(box[4])
            boundingBox.xmin = np.int64(box[0])
            boundingBox.ymin = np.int64(box[1])
            boundingBox.xmax = np.int64(box[2])
            boundingBox.ymax = np.int64(box[3])
            boundingBox.num = np.int16(len(boxs))
            boundingBox.Class = str(box[-1])

            self.boundingBoxes.bounding_boxes.append(boundingBox)

            if box[-1] in self.classes_colors.keys():
                color = self.classes_colors[box[-1]]
            else:
                color = np.random.randint(0, 183, 3)
                self.classes_colors[box[-1]] = color

            # Draw bounding box on the image
            cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (int(color[0]), int(color[1]), int(color[2])), 2)

        # Draw laser points on the image
        if hasattr(self, 'laser_pts_uv'):
            for pt_uv, (x, y, z) in self.laser_pts_uv:
                for bbox in self.boundingBoxes.bounding_boxes:
                    if bbox.xmin <= pt_uv[0] <= bbox.xmax and bbox.ymin <= pt_uv[1] <= bbox.ymax:
                        cv2.circle(img, (int(pt_uv[0]), int(pt_uv[1])), 1, (0, 0, 255), 1)
                        distance = np.linalg.norm([x, y, z])
                        cv2.putText(img, f"{distance:.2f}m", (int(pt_uv[0]) + 5, int(pt_uv[1]) + 5), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # Display FPS on the image
        fps_text = f'FPS: {1.0 / inference_time:.2f}'
        cv2.putText(img, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Publish bounding boxes and display the image
        self.position_pub.publish(self.boundingBoxes)
        cv2.imshow('YOLOv5', img)
        cv2.waitKey(3)

def main():
    rospy.init_node('yolov5_ros', anonymous=True)
    yolo_dect = Yolo_Dect()
    rospy.spin()

if __name__ == "__main__":
    main()
