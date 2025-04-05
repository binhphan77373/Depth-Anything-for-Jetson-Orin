import cv2
import numpy as np
import torch
import threading
import queue
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from ultralytics import YOLO
from depth_anythingv1 import DepthEngine

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
            1 #10
        )
        self.bridge = CvBridge()
        
        # Device and Model Management
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.depth_engine = None
        
        # Performance Tracking
        self.processed_frames = 0
        self.start_time = time.time()
        
        # FPS Tracking
        self.fps = 0
        self.frame_times = []

    def _initialize_models(self):
        """Lazy initialization of models"""
        if self.depth_engine is None:
            with torch.cuda.device(self.device):
                self.depth_engine = DepthEngine()
        if self.model is None:
            self.model = YOLO("yolo11n.onnx")

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

    def process_frame(self, frame):
        """Process single frame with object detection and depth"""
        try:
            if self.model is None or self.depth_engine is None:
                self._initialize_models()

            with torch.cuda.device(self.device):
                # Depth Estimation
                depth = self._process_depth(frame)
                # Object Detection
                object_frame = frame.copy()
                results = self.model(object_frame)

                # Process Detection Results
                for result in results:
                    if hasattr(result, 'boxes'):
                        # Draw Bounding Boxes
                        for box in result.boxes:
                            cv2.rectangle(frame, 
                                         (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                                         (int(box.xyxy[0][2]), int(box.xyxy[0][3])), 
                                         (255, 0, 0), 2)
                            
                            # Object Label
                            cv2.putText(frame, 
                                        f"{result.names[int(box.cls[0])]}", 
                                        (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)

                        # Depth Calculations
                        depths = self._extract_depth_from_boxes(
                            [box.xyxy[0] for box in result.boxes], 
                            depth
                        )
                        
                        objects = self._extract_bounding_boxes_and_depth(
                            [box.xyxy[0] for box in result.boxes], 
                            [result.names[int(box.cls[0])] for box in result.boxes], 
                            depths
                        )

                        # Display Depth Information
                        for obj in objects:
                            cv2.putText(frame, 
                                        f"{obj['depth']:.2f} m", 
                                        (obj['x1'], obj['y1'] - 10), 
                                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
                            
                            # Check if object depth is less than 1.5 meters
                            # if obj['depth'] < 1.5:
                            #     warning_text = f"Warning: {obj['class']} too close! {obj['depth']:.2f} m"
                            #     cv2.putText(frame, 
                            #                 warning_text,
                            #                 (obj['x1'], obj['y1'] - 30), 
                            #                 cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

                return frame
        except Exception as e:
            self.get_logger().error(f'Frame processing error: {e}')
            return frame

    def _calculate_fps(self):
        """Calculate FPS using a rolling window of frame times"""
        current_time = time.time()
        self.frame_times.append(current_time)
        
        # Keep only the last 30 frame times
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)
        
        # Calculate FPS if we have enough frames
        if len(self.frame_times) > 1:
            self.fps = len(self.frame_times) / (current_time - self.frame_times[0])
        
        return self.fps

    def _process_depth(self, depth_frame):
        """Depth estimation with error handling"""
        try:
            depth = self.depth_engine.infer(depth_frame)
            depth_resized = cv2.resize(depth, (depth_frame.shape[1], depth_frame.shape[0]))
            depth_resized = cv2.cvtColor(depth_resized, cv2.COLOR_BGR2GRAY)
            return 255 - depth_resized
        except Exception as e:
            self.get_logger().error(f'Depth estimation error: {e}')
            return np.zeros_like(depth_frame, dtype=np.uint8)

    def _extract_depth_from_boxes(self, boxes, depth_map):
        """Extract depth for detected objects"""
        object_depths = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            object_depth_map = depth_map[y1:y2, x1:x2]
            median_depth = np.mean(object_depth_map)
            object_depths.append(median_depth)
        return object_depths

    def _extract_bounding_boxes_and_depth(self, detected_boxes, detected_labels, depths):
        """Create detailed object information"""
        objects = []
        for i, box in enumerate(detected_boxes):
            x1, y1, x2, y2 = map(int, box)
            obj = {
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'class': detected_labels[i],
                'depth': depths[i]
            }
            objects.append(obj)
        return objects

    def run_processing(self):
        """Main processing method with multithreading and performance tracking"""
        cv2.namedWindow("Processed Image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Processed Image", 640, 640)

        while rclpy.ok() and not self._quit_keypress():
            rclpy.spin_once(self)

            try:
                # Non-blocking frame retrieval
                frame = self.frame_queue.get(timeout=1/30)
                
                # Process frame
                processed_frame = self.process_frame(frame)                                                                                                  
                self.processed_frames += 1

                # Calculate FPS
                fps = self._calculate_fps()

                # Display FPS on frame
                cv2.putText(processed_frame, 
                            f"FPS: {fps:.2f}", 
                            (10, 30),  # Top-left corner
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            1,  # Font scale
                            (0, 255, 0),  # Green color
                            2)  # Thickness

                # Display
                cv2.imshow("Processed Image", processed_frame)

            except queue.Empty:
                time.sleep(0.01)
                continue

        total_time = time.time() - self.start_time
        avg_fps = self.processed_frames / total_time
        self.get_logger().info(f"Processed Frames: {self.processed_frames}")
        self.get_logger().info(f"Total Time: {total_time:.2f} seconds")
        self.get_logger().info(f"Average FPS: {avg_fps:.2f}")

    def _quit_keypress(self):
        """Check for quit key"""
        key = cv2.waitKey(1)
        return key == 27 or key == ord('q')

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