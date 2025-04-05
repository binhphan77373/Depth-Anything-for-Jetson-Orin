#!/usr/bin/env python3

import cv2
import numpy as np
import torch
import threading
import queue
import time
import sys
import argparse

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from ultralytics import YOLO
from depth import DepthEngine

# Thêm khóa đồng bộ hóa cho CUDA
cuda_lock = threading.Lock()

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
        
        # Xử lý tham số
        parser = argparse.ArgumentParser(description='YOLO và Depth Anything trên ROS2')
        parser.add_argument('--engine', type=str, default="./weights/depth_anything_vits14_518.trt", 
                           help='Đường dẫn đến TensorRT engine')
        parser.add_argument('--yolo_model', type=str, default="./weights/yolo11n.onnx", 
                           help='Đường dẫn đến mô hình YOLO')
        parser.add_argument('--depth_scale', type=float, default=10.0, 
                           help='Tỷ lệ chuyển đổi độ sâu sang khoảng cách thực tế (mét)')
        parser.add_argument('--inverse', action='store_true', default=True,
                           help='Đảo ngược tính toán khoảng cách (mặc định: True)')
        parser.add_argument('--show', action='store_true', help='Hiển thị kết quả trong cửa sổ')
        
        # Lấy tham số từ ROS2
        # Lưu ý: Trong ROS2, chúng ta không thể sử dụng argparse trực tiếp,
        # nên cần thêm các tham số vào ROS parameter hoặc sử dụng giá trị mặc định
        self.get_logger().info('Khởi tạo các tham số mặc định')
        self.declare_parameter('engine_path', './weights/depth_anything_vits14_518.trt')
        self.declare_parameter('yolo_model_path', './weights/yolo11n.onnx')
        self.declare_parameter('depth_scale', 10.0)
        self.declare_parameter('inverse', True)
        self.declare_parameter('show', False)
        
        self.engine_path = self.get_parameter('engine_path').get_parameter_value().string_value
        self.yolo_model_path = self.get_parameter('yolo_model_path').get_parameter_value().string_value
        self.depth_scale = self.get_parameter('depth_scale').get_parameter_value().double_value
        self.inverse = self.get_parameter('inverse').get_parameter_value().bool_value
        self.show = self.get_parameter('show').get_parameter_value().bool_value
        
        # Threading and Queue Management
        self.frame_queue = queue.Queue(maxsize=100)
        self.processed_queue = queue.Queue(maxsize=100)
        self.stop_event = threading.Event()
        
        # ROS2 Subscription
        self.subscription = self.create_subscription(
            Image,
            'aria/rgb_image',
            self.listener_callback,
            1
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
        self.processed_frames = 0
        
        # Kiểm tra và hiển thị thông tin CUDA
        if torch.cuda.is_available():
            self.get_logger().info(f'CUDA khả dụng: {torch.cuda.is_available()}')
            self.get_logger().info(f'Số lượng GPU: {torch.cuda.device_count()}')
            self.get_logger().info(f'GPU hiện tại: {torch.cuda.current_device()}')
            self.get_logger().info(f'Tên GPU: {torch.cuda.get_device_name(0)}')
        else:
            self.get_logger().warning('CUDA không khả dụng! Sẽ sử dụng CPU.')
        
        # Hiển thị thông tin cấu hình
        self.get_logger().info(f'Engine path: {self.engine_path}')
        self.get_logger().info(f'YOLO model path: {self.yolo_model_path}')
        self.get_logger().info(f'Depth scale: {self.depth_scale}')
        self.get_logger().info(f'Inverse distance: {self.inverse}')
        self.get_logger().info(f'Show results: {self.show}')
        
        # Khởi tạo CUDA trong luồng chính
        if torch.cuda.is_available():
            try:
                with cuda_lock:
                    torch.cuda.init()
                    torch.cuda.set_device(0)  # Đặt thiết bị mặc định
                self.get_logger().info('Đã khởi tạo CUDA trong luồng chính')
            except Exception as e:
                self.get_logger().error(f'Lỗi khởi tạo CUDA: {e}')
        
        # Khởi tạo luồng xử lý
        self.processing_thread = threading.Thread(target=self.processing_worker)
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def _initialize_models(self):
        """Lazy initialization of models"""
        # Sử dụng khóa để ngăn truy cập đồng thời
        with cuda_lock:
            if self.depth_engine is None:
                try:
                    # Đảm bảo thiết bị hiện tại là thiết bị đúng
                    if torch.cuda.is_available():
                        torch.cuda.set_device(0)
                        current_device = torch.cuda.current_device()
                        self.get_logger().info(f'Khởi tạo DepthEngine trên thiết bị {current_device}')
                    
                    self.depth_engine = DepthEngine(
                        trt_engine_path=self.engine_path,
                        stream=False,
                        record=False,
                        save=False,
                        grayscale=False,
                        raw=True  # Cần giá trị độ sâu thô để tính khoảng cách chính xác
                    )
                    self.get_logger().info('Đã khởi tạo Depth Engine')
                except Exception as e:
                    self.get_logger().error(f'Lỗi khởi tạo DepthEngine: {str(e)}')
                    raise
                
            if self.model is None:
                try:
                    self.model = YOLO(self.yolo_model_path)
                    self.get_logger().info('Đã khởi tạo model YOLO')
                except Exception as e:
                    self.get_logger().error(f'Lỗi khởi tạo YOLO: {str(e)}')
                    raise

    def listener_callback(self, msg):
        """ROS2 image callback to add frames to processing queue"""
        try:
            rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')  # Chuyển sang BGR cho OpenCV
            
            try:
                self.frame_queue.put(rgb_image, timeout=1/30)
            except queue.Full:
                self.get_logger().warning('Frame queue đầy, bỏ qua frame này')
                pass
        except Exception as e:
            self.get_logger().error(f'Lỗi trong callback: {e}')

    def process_frame(self, frame):
        """Xử lý frame với YOLO và Depth Anything"""
        try:
            # Đảm bảo các model đã được khởi tạo
            self._initialize_models()
            
            # Sử dụng khóa để đảm bảo chỉ một luồng truy cập CUDA tại một thời điểm
            with cuda_lock:
                # Lấy bản đồ độ sâu
                depth_raw = self.depth_engine.process_frame(frame.copy())
                
                # Phát hiện đối tượng với YOLO
                yolo_results = self.model(frame)
            
            # Phần còn lại của xử lý có thể diễn ra bên ngoài khóa
            # Tạo bản sao của frame để vẽ
            annotated_frame = frame.copy()
            
            # Xử lý từng đối tượng được phát hiện
            for result in yolo_results:
                boxes = result.boxes.cpu().numpy()
                
                for i, box in enumerate(boxes):
                    # Lấy tọa độ bounding box
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Lấy class id và confidence
                    class_id = int(box.cls[0])
                    conf = box.conf[0]
                    
                    # Lấy tên lớp từ model
                    class_name = result.names[class_id]
                    
                    # Tính khoảng cách của đối tượng
                    distance = calculate_distance(depth_raw, [x1, y1, x2, y2], 
                                                self.depth_scale, self.inverse)
                    
                    # Vẽ bounding box
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Hiển thị thông tin đối tượng và khoảng cách
                    if distance > 0:
                        label = f"{class_name}: {conf:.2f}, {distance:.2f}m"
                    else:
                        label = f"{class_name}: {conf:.2f}"
                    
                    # Vẽ background cho text
                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(annotated_frame, (x1, y1-20), (x1+w, y1), (0, 255, 0), -1)
                    
                    # Vẽ text
                    cv2.putText(annotated_frame, label, (x1, y1-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Hiển thị kết quả nếu được yêu cầu
            if self.show:
                cv2.imshow('YOLO + Depth', annotated_frame)
                cv2.waitKey(1)
                
            return annotated_frame
            
        except Exception as e:
            self.get_logger().error(f'Lỗi xử lý frame: {str(e)}')
            import traceback
            self.get_logger().error(traceback.format_exc())
            return frame  # Trả về frame gốc nếu có lỗi

    def processing_worker(self):
        """Worker thread for processing frames"""
        try:
            # Khởi tạo CUDA cho luồng worker
            if torch.cuda.is_available():
                with cuda_lock:
                    torch.cuda.init()
                    torch.cuda.set_device(0)  # Đặt thiết bị mặc định
                self.get_logger().info('Đã khởi tạo CUDA trong luồng worker')
        except Exception as e:
            self.get_logger().error(f'Lỗi khởi tạo CUDA trong worker: {e}')
        
        while not self.stop_event.is_set():
            try:
                # Lấy frame từ queue
                frame = self.frame_queue.get(timeout=0.1)
                
                # Xử lý frame
                t0 = time.time()
                processed_frame = self.process_frame(frame)                                                                                                 
                processing_time = time.time() - t0
                
                self.processed_frames += 1
                if self.processed_frames % 10 == 0:
                    self.get_logger().info(f'Đã xử lý {self.processed_frames} frames, thời gian xử lý: {processing_time:.4f}s')
                
                # Chuyển đổi sang định dạng ROS message và gửi đi
                img_msg = self.bridge.cv2_to_imgmsg(processed_frame, encoding='bgr8')
                self.publisher.publish(img_msg)
                
                # Đánh dấu công việc đã hoàn thành
                self.frame_queue.task_done()
                
            except queue.Empty:
                # Không có frame mới để xử lý
                continue
            except Exception as e:
                self.get_logger().error(f'Lỗi trong worker: {str(e)}')
                import traceback
                self.get_logger().error(traceback.format_exc())

    def run_processing(self):
        """Main processing method with ROS2 spinning"""
        try:
            self.get_logger().info('Bắt đầu xử lý...')
            rclpy.spin(self)
        except KeyboardInterrupt:
            self.get_logger().info('Đã nhận tín hiệu dừng...')
        finally:
            self.stop_event.set()
            if self.processing_thread.is_alive():
                self.processing_thread.join(timeout=1/30)
            if self.show:
                cv2.destroyAllWindows()
            if hasattr(self, 'depth_engine') and self.depth_engine is not None:
                self.depth_engine.close()
            self.get_logger().info('Đã dừng xử lý.')

def main(args=None):
    rclpy.init(args=args)
    node = OptimizedImageProcessor()
    
    try:
        # Đã chuyển việc khởi tạo CUDA vào luồng worker
        node.run_processing()
    except Exception as e:
        node.get_logger().error(f"Lỗi trong xử lý chính: {e}")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

