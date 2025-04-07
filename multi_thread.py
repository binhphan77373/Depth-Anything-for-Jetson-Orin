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
        
        # Chuyển depth map sang ảnh xám
        self.convert_depth_to_gray = True

    def _initialize_models(self):
        """Lazy initialization of models"""
        if self.depth_engine is None:
            with torch.cuda.device(self.device):
                self.depth_engine = DepthEngine(raw=True)  # Quan trọng: cần raw depth map
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
                # Depth Estimation - Lấy raw depth map từ depth_engine.process_frame
                try:
                    depth_raw = self.depth_engine.process_frame(frame.copy())
                    
                    # Kiểm tra depth_raw có hợp lệ hay không
                    if depth_raw is None or depth_raw.size == 0 or np.isnan(depth_raw).any():
                        self.get_logger().warn("Depth map không hợp lệ, chỉ thực hiện phát hiện đối tượng")
                        depth_available = False
                    else:
                        depth_available = True
                        
                        # Chuyển depth map sang ảnh xám trước khi tính khoảng cách
                        if self.convert_depth_to_gray:
                            # Chuẩn hóa depth map về khoảng 0-255
                            normalized_depth = cv2.normalize(depth_raw, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                            # Chuyển sang ảnh xám
                            depth_gray = cv2.cvtColor(normalized_depth, cv2.COLOR_GRAY2BGR)
                            cv2.cvtColor(depth_gray, cv2.COLOR_BGR2GRAY)
                            # Đảo ngược depth map xám (giá trị pixel lớn = xa, nhỏ = gần)
                            depth_raw = 255 - normalized_depth
                            
                            # Hiển thị depth map xám
                            cv2.imshow("Depth Map (Grayscale)", depth_raw)
                except Exception as e:
                    self.get_logger().error(f'Depth engine error: {e}')
                    depth_available = False
                    depth_raw = None
                
                # Object Detection
                results = self.model(frame)

                # Process Detection Results
                for result in results:
                    if hasattr(result, 'boxes'):
                        # Bounding boxes mỗi đối tượng được nhận diện
                        boxes = [box.xyxy[0] for box in result.boxes]
                        labels = [result.names[int(box.cls[0])] for box in result.boxes]
                        
                        # Tính khoảng cách nếu depth map hợp lệ
                        if depth_available:
                            for box in boxes:
                                x1, y1, x2, y2 = map(int, box)
                                
                                # Tính tọa độ trung tâm của bounding box
                                center_x = int((x1 + x2) / 2)
                                center_y = int((y1 + y2) / 2)
                                
                                # Lấy giá trị độ sâu tại tâm bounding box thay vì dùng giá trị trung bình
                                depth_value = float(depth_raw[center_y, center_x])
                                
                                # Tính khoảng cách - đảo ngược depth map nếu cần
                                if self.convert_depth_to_gray:
                                    # Depth map đã được đảo ngược, giá trị pixel lớn = gần
                                    # Chuẩn hóa về khoảng cách thực tế (giả sử 0-10 mét)
                                    distance = (1.0 - depth_value / 255.0) * 10.0
                                else:
                                    # Cách tính khoảng cách gốc
                                    distance = depth_value
                                
                                self.get_logger().info(f"Khoảng cách đến {labels[boxes.index(box)]}: {distance:.2f} m")
                                
                                # Vẽ bounding box lên hình ảnh
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                
                                # Vẽ điểm tâm của đối tượng
                                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
                                
                                # Hiển thị thông tin nhãn và khoảng cách
                                label_text = f"{labels[boxes.index(box)]}: {distance:.2f}m"
                                cv2.putText(frame, label_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        else:
                            # Nếu không có thông tin độ sâu, chỉ vẽ bounding box và nhãn
                            for box in boxes:
                                x1, y1, x2, y2 = map(int, box)
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                label_text = f"{labels[boxes.index(box)]}"
                                cv2.putText(frame, label_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                # Hiển thị trạng thái chuyển đổi depth map
                cv2.putText(frame, 
                           f"Depth Gray: {'Bật' if self.convert_depth_to_gray else 'Tắt'}", 
                           (10, 60),  
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           1, 
                           (0, 255, 0), 
                           2)

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
        # Thêm phím 'd' để bật/tắt chế độ depth map grayscale
        if key == ord('d'):
            self.convert_depth_to_gray = not self.convert_depth_to_gray
            self.get_logger().info(f"Chuyển đổi depth map sang ảnh xám: {'Bật' if self.convert_depth_to_gray else 'Tắt'}")
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