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
from depth import DepthEngine

def calculate_distance(depth_map, box, depth_scale=1.0, inverse=True):
    """
    Calculate the distance from the object to the camera based on depth map
    
    depth_map: numpy array containing depth information
    box: bounding box of the object [x1, y1, x2, y2]
    depth_scale: scale factor to convert depth values to real-world distance (meters)
    inverse: invert the distance (True if small values are far, False if large values are far)
    
    Returns: average distance of the object (in meters)
    """
    # Extract object region from depth map
    x1, y1, x2, y2 = map(int, box)
    
    # Get the center region of the object (30% of the area)
    center_width = int((x2 - x1) * 0.3)
    center_height = int((y2 - y1) * 0.3)
    center_x1 = x1 + (x2 - x1)//2 - center_width//2
    center_y1 = y1 + (y2 - y1)//2 - center_height//2
    center_x2 = center_x1 + center_width
    center_y2 = center_y1 + center_height
    
    # Ensure center region is within image bounds
    center_x1 = max(0, center_x1)
    center_y1 = max(0, center_y1)
    center_x2 = min(depth_map.shape[1], center_x2)
    center_y2 = min(depth_map.shape[0], center_y2)
    
    # Get the depth map portion corresponding to the object's center region
    object_depth = depth_map[center_y1:center_y2, center_x1:center_x2]
    
    # Calculate average distance (ignore zero values if any)
    if object_depth.size > 0:
        # Remove values that are too small or too large (outliers)
        valid_depths = object_depth[object_depth > 0.01]
        if valid_depths.size > 0:
            # Use median instead of mean to reduce noise impact
            avg_depth = np.median(valid_depths)
            
            # Apply conversion scale
            if inverse:
                # Invert distance: large value = near, small value = far
                # Use 1.0 as standard value for inversion
                # This coefficient needs to be adjusted based on depth map value range
                norm_factor = 1.0
                distance = norm_factor / (avg_depth + 0.001) * depth_scale
            else:
                # Keep original: large value = far, small value = near
                distance = avg_depth * depth_scale
                
            return distance
    
    # Return -1 if distance cannot be calculated
    return -1

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
        
        # Depth configuration
        # Source camera calibration (Fisheye624)
        source_focal_length = 608.032  # mm
        source_principal_point = [717.302, 708.448]  # pixels
        source_image_size = [2880, 2880]  # pixels
        
        # Destination camera calibration (Linear)
        dest_focal_length = 150.0  # mm
        dest_principal_point = [255.5, 255.5]  # pixels
        dest_image_size = [512, 512]  # pixels
        
        # Calculate scale factors
        # 1. Focal length scale
        focal_length_scale = source_focal_length / dest_focal_length
        
        # 2. Image size scale (using average of width and height ratios)
        width_scale = source_image_size[0] / dest_image_size[0]
        height_scale = source_image_size[1] / dest_image_size[1]
        image_size_scale = (width_scale + height_scale) / 2.0
        
        # 3. Principal point offset compensation
        # Calculate the relative position of principal points
        source_center_ratio_x = source_principal_point[0] / source_image_size[0]
        source_center_ratio_y = source_principal_point[1] / source_image_size[1]
        dest_center_ratio_x = dest_principal_point[0] / dest_image_size[0]
        dest_center_ratio_y = dest_principal_point[1] / dest_image_size[1]
        
        # Calculate center offset compensation factor
        center_offset_scale = np.sqrt(
            (source_center_ratio_x - dest_center_ratio_x)**2 + 
            (source_center_ratio_y - dest_center_ratio_y)**2
        ) + 1.0  # Add 1.0 to ensure scale is at least 1.0
        
        # Combined scale factor
        self.depth_scale = focal_length_scale * image_size_scale * center_offset_scale
        self.inverse_depth = True

    def _initialize_models(self):
        """Lazy initialization of models"""
        if self.depth_engine is None:
            with torch.cuda.device(self.device):
                self.depth_engine = DepthEngine(raw=True)
        if self.model is None:
            self.model = YOLO("/home/orin/test_ws/Depth-Anything-for-Jetson-Orin/weights/yolo11n.onnx")

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
    
    def processing_frame(self, frame):
        """Process single frame with object detection and depth"""
        try:
            if self.model is None or self.depth_engine is None:
                self._initialize_models()

            with torch.cuda.device(self.device):
                # Depth Estimation - Get raw depth map from depth_engine.process_frame
                try:
                    depth_raw = self.depth_engine.process_frame(frame.copy())
                    
                    # Check if depth_raw is valid
                    if depth_raw is None or depth_raw.size == 0 or np.isnan(depth_raw).any():
                        self.get_logger().warn("Invalid depth map, performing object detection only")
                        depth_available = False
                    else:
                        depth_available = True
                except Exception as e:
                    self.get_logger().error(f'Depth engine error: {e}')
                    depth_available = False
                    depth_raw = None
                
                # Object Detection
                results = self.model(frame)

                # Process Detection Results
                for result in results:
                    if hasattr(result, 'boxes'):
                        # Bounding boxes for each detected object
                        boxes = [box.xyxy[0] for box in result.boxes]
                        labels = [result.names[int(box.cls[0])] for box in result.boxes]
                        
                        # Calculate distances if depth map is valid
                        distances = []
                        if depth_available:
                            for box in boxes:
                                try:
                                    dist = calculate_distance(
                                        depth_raw, box, 
                                        depth_scale=self.depth_scale,
                                        inverse=self.inverse_depth
                                    )
                                    distances.append(dist)
                                except Exception as e:
                                    self.get_logger().error(f"Error calculating distance: {e}")
                                    distances.append(-1)
                        else:
                            # If no depth, set all distances to -1
                            distances = [-1] * len(boxes)
                        
                        # Draw bounding box and display distance
                        for i, (box, label, distance) in enumerate(zip(boxes, labels, distances)):
                            x1, y1, x2, y2 = map(int, box)
                            
                            # Draw bounding box
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                            
                            # Display object label
                            cv2.putText(frame, 
                                        f"{label}", 
                                        (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
                            
                            # Display distance
                            if distance > 0:
                                cv2.putText(frame, 
                                            f"{distance:.2f}m", 
                                            (x1, y1 - 30), 
                                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
                                
                                # Warning when object is too close (optional)
                                if distance < 1.5:
                                    cv2.putText(frame, 
                                                f"Warning: {label} too close!", 
                                                (x1, y1 - 50), 
                                                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

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
                processed_frame = self.processing_frame(frame)                                                                                                  
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