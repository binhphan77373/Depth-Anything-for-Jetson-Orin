#!/usr/bin/env python3

import cv2
import numpy as np
import torch
import threading
import queue
import time
import sys

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from ultralytics import YOLO
from depth import DepthEngine

class OptimizedImageProcessor(Node):
    def __init__(self):
        super().__init__('optimized_image_subscriber')
        
        # Threading and Queue Management
        self.frame_queue = queue.Queue(maxsize=100)
        self.processed_queue = queue.Queue(maxsize=100)
        self.stop_event = threading.Event()
        
        # ROS2 Subscription
        self.subscription = self.create_subscription(
            Image,
            'aria/rgb_image',
            self.listener_callback,
            10
        )

        self.publisher = self.create_publisher(
            Image,
            'image_results',
            10
        )
        self.bridge = CvBridge()
        
        # Device and Model Management
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.depth_engine = None

    def _initialize_models(self):
        """Lazy initialization of models"""
        if self.depth_engine is None:
            with torch.cuda.device(self.device):
                self.depth_engine = DepthEngine()
        if self.model is None:
            self.model = YOLO("/home/orin/Test/Depth-Anything-for-Jetson-Orin/weights/yolo11n.onnx")

    def listener_callback(self, msg):
        """ROS2 image callback to add frames to processing queue"""
        try:
            rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            
            try:
                self.frame_queue.put(rgb_image, timeout=1/30)
            except queue.Full:
                pass
        except Exception as e:
            self.get_logger().error(f'Error in listener callback: {e}')


    def run_processing(self):
        """Main processing method with multithreading and performance tracking"""

        while rclpy.ok():
            rclpy.spin_once(self)

            try:
                # Non-blocking frame retrieval
                frame = self.frame_queue.get(timeout=1/30)
                
                # Process frame
                processed_frame = self.process_frame(frame)                                                                                                  
                self.processed_frames += 1

                img_msg = self.bridge.cv2_to_imgmsg(processed_frame, encoding='rgb8')
                self.publisher.publish(img_msg)

            except queue.Empty:
                time.sleep(0.01)
                continue

def main(args=None):
    rclpy.init(args=args)
    node = OptimizedImageProcessor()
    
    try:
        torch.cuda.init()
        node.run_processing()
    except Exception as e:
        print(f"Error in main processing: {e}")
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

