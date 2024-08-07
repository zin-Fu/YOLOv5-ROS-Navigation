#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

import cv2
import torch
import rospy
import numpy as np
from time import time

from std_msgs.msg import Header
from sensor_msgs.msg import Image
from yolov5_ros.msg import BoundingBox, BoundingBoxes

import torch.backends.cudnn as cudnn

from utils.utils import scale_coords, non_max_suppression

class Yolo_Dect:
    def __init__(self):
        # Load parameters from ROS parameter server
        yolov5_path = rospy.get_param('/yolov5_path', '')
        weight_path = rospy.get_param('~weight_path', '')
        image_topic = rospy.get_param('~image_topic', '/usb_cam/image_raw')
        pub_topic = rospy.get_param('~pub_topic', '/yolov5/BoundingBoxes')
        self.camera_frame = rospy.get_param('~camera_frame', '')
        conf = rospy.get_param('~conf', '0.5')

        # Camera parameters
        self.camera_matrix = np.array([[417.4378384222683, 0, 316.1704948005648],
                                       [0, 416.629339408546, 237.6943393173432],
                                       [0, 0, 1]])
        self.dist_coeffs = np.array([-0.2994228197129783, 0.0762218269094439, -0.001955744778229992, -0.0005395207018594055, 0])

        # Load YOLO model
        self.model = torch.load(weight_path, map_location='cuda')['model'].float()

        # Use CPU or GPU for model inference
        if rospy.get_param('/use_cpu', 'false'):
            self.model.cpu()
        else:
            self.model.cuda()

        self.model.conf = conf
        self.color_image = Image()
        self.depth_image = Image()
        self.getImageStatus = False

        # Load class colors for bounding boxes
        self.classes_colors = {}

        # Subscribe to image topic
        self.color_sub = rospy.Subscriber(image_topic, Image, self.image_callback, queue_size=1, buff_size=52428800)

        # Output publishers
        self.position_pub = rospy.Publisher(pub_topic, BoundingBoxes, queue_size=5)
        self.image_pub = rospy.Publisher('/yolov5/detection_image', Image, queue_size=1)

        # Wait for first image message
        while not self.getImageStatus:
            rospy.loginfo("waiting for image.")
            rospy.sleep(2)

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

        cv2.waitKey(3)

    def process_detections(self, boxs, height, width, inference_time):
        """
        Process YOLO detections, draw bounding boxes on the image, and publish the results.
        """
        img = self.color_image.copy()
        current_detections = set()
        count = 0
        for i in boxs:
            count += 1

        for box in boxs:
            boundingBox = BoundingBox()
            boundingBox.probability = np.float64(box[4])
            boundingBox.xmin = np.int64(box[0])
            boundingBox.ymin = np.int64(box[1])
            boundingBox.xmax = np.int64(box[2])
            boundingBox.ymax = np.int64(box[3])
            boundingBox.num = np.int16(count)
            boundingBox.Class = str(box[-1])

            current_detections.add(boundingBox.Class)
            self.boundingBoxes.bounding_boxes.append(boundingBox)

            if box[-1] in self.classes_colors.keys():
                color = self.classes_colors[box[-1]]
            else:
                color = np.random.randint(0, 183, 3)
                self.classes_colors[box[-1]] = color

            # Draw bounding box on the image
            cv2.rectangle(img, (int(box[0]), int(box[1])),
                          (int(box[2]), int(box[3])), (int(color[0]), int(color[1]), int(color[2])), 2)

        # Publish bounding boxes and display FPS on the image
        self.position_pub.publish(self.boundingBoxes)
        print(self.boundingBoxes)
        print("--------------------")

        fps_text = f'FPS: {1.0 / inference_time:.2f}'
        cv2.putText(img, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('YOLOv5', img)

def main():
    rospy.init_node('yolov5_ros', anonymous=True)
    yolo_dect = Yolo_Dect()
    rospy.spin()

if __name__ == "__main__":
    main()
