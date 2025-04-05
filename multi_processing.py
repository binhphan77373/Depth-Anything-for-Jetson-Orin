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
    Tính khoảng cách từ đối tượng đến camera dựa trên depth map
    
    depth_map: mảng numpy chứa thông tin độ sâu
    box: bounding box của đối tượng [x1, y1, x2, y2]
    depth_scale: tỷ lệ để chuyển đổi từ giá trị độ sâu sang khoảng cách thực tế (mét)
    inverse: đảo ngược khoảng cách (True nếu giá trị nhỏ là xa, False nếu giá trị lớn là xa)
    
    Trả về: khoảng cách trung bình của đối tượng (đơn vị: mét)
    """
    # Trích xuất vùng đối tượng từ depth map
    x1, y1, x2, y2 = map(int, box)
    
    # Lấy vùng trung tâm của đối tượng (30% diện tích giữa)
    center_width = int((x2 - x1) * 0.3)
    center_height = int((y2 - y1) * 0.3)
    center_x1 = x1 + (x2 - x1)//2 - center_width//2
    center_y1 = y1 + (y2 - y1)//2 - center_height//2
    center_x2 = center_x1 + center_width
    center_y2 = center_y1 + center_height
    
    # Đảm bảo vùng trung tâm nằm trong ảnh
    center_x1 = max(0, center_x1)
    center_y1 = max(0, center_y1)
    center_x2 = min(depth_map.shape[1], center_x2)
    center_y2 = min(depth_map.shape[0], center_y2)
    
    # Lấy phần depth map tương ứng với vùng trung tâm của đối tượng
    object_depth = depth_map[center_y1:center_y2, center_x1:center_x2]
    
    # Tính khoảng cách trung bình (bỏ qua giá trị 0 nếu có)
    if object_depth.size > 0:
        # Loại bỏ các giá trị quá nhỏ hoặc quá lớn (outliers)
        valid_depths = object_depth[object_depth > 0.01]
        if valid_depths.size > 0:
            # Sử dụng trung vị thay vì trung bình để giảm ảnh hưởng của nhiễu
            avg_depth = np.median(valid_depths)
            
            # Áp dụng tỷ lệ chuyển đổi
            if inverse:
                # Đảo ngược khoảng cách: giá trị lớn = gần, giá trị nhỏ = xa
                # Sử dụng 1.0 làm giá trị chuẩn để đảo ngược
                # Cần điều chỉnh hệ số này tùy theo dải giá trị của depth map
                norm_factor = 1.0
                distance = norm_factor / (avg_depth + 0.001) * depth_scale
            else:
                # Giữ nguyên: giá trị lớn = xa, giá trị nhỏ = gần
                distance = avg_depth * depth_scale
                
            return distance
    
    # Trả về -1 nếu không thể tính khoảng cách
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
        self.depth_scale = 10.0
        self.inverse_depth = True

    def _initialize_models(self):
        """Lazy initialization of models"""
        if self.depth_engine is None:
            with torch.cuda.device(self.device):
                self.depth_engine = DepthEngine(raw=True)  # Quan trọng: cần raw depth map
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
                # Depth Estimation - Lấy raw depth map
                depth_raw = self._process_depth(frame)
                
                # Object Detection
                object_frame = frame.copy()
                results = self.model(object_frame)

                # Process Detection Results
                for result in results:
                    if hasattr(result, 'boxes'):
                        # Bounding boxes mỗi đối tượng được nhận diện
                        boxes = [box.xyxy[0] for box in result.boxes]
                        labels = [result.names[int(box.cls[0])] for box in result.boxes]
                        
                        # Tính khoảng cách của các đối tượng
                        distances = []
                        for box in boxes:
                            dist = calculate_distance(
                                depth_raw, box, 
                                depth_scale=self.depth_scale,
                                inverse=self.inverse_depth
                            )
                            distances.append(dist)
                        
                        # Vẽ bounding box và hiển thị khoảng cách
                        for i, (box, label, distance) in enumerate(zip(boxes, labels, distances)):
                            x1, y1, x2, y2 = map(int, box)
                            
                            # Vẽ bounding box
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                            
                            # Hiển thị nhãn đối tượng
                            cv2.putText(frame, 
                                        f"{label}", 
                                        (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
                            
                            # Hiển thị khoảng cách
                            if distance > 0:
                                cv2.putText(frame, 
                                            f"{distance:.2f}m", 
                                            (x1, y1 - 30), 
                                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
                                
                                # Cảnh báo khi đối tượng quá gần (tuỳ chọn)
                                if distance < 1.5:
                                    cv2.putText(frame, 
                                                f"Cảnh báo: {label} quá gần!", 
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

    def _process_depth(self, depth_frame):
        """
        Xử lý depth estimation và trả về raw depth map để tính khoảng cách chính xác
        """
        try:
            # Lấy raw depth map thay vì depth map đã xử lý màu
            depth_raw = self.depth_engine.infer(depth_frame)
            return depth_raw
        except Exception as e:
            self.get_logger().error(f'Depth estimation error: {e}')
            return np.zeros((depth_frame.shape[0], depth_frame.shape[1]), dtype=np.float32)

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