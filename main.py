import cv2
import numpy as np
from ultralytics import YOLO
from depth import DepthEngine
import argparse

def calculate_distance(depth_map, box, depth_scale=1.0):
    """
    Tính khoảng cách từ đối tượng đến camera dựa trên depth map
    
    depth_map: mảng numpy chứa thông tin độ sâu
    box: bounding box của đối tượng [x1, y1, x2, y2]
    depth_scale: tỷ lệ để chuyển đổi từ giá trị độ sâu sang khoảng cách thực tế (mét)
    
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
            distance = avg_depth * depth_scale
            return distance
    
    # Trả về -1 nếu không thể tính khoảng cách
    return -1

def main():
    # Xử lý tham số dòng lệnh
    parser = argparse.ArgumentParser(description='YOLO và Depth Anything trên video')
    parser.add_argument('--video', type=str, default="/home/orin/Test/Depth-Anything-for-Jetson-Orin/AriaEverydayActivities_1.0.0_loc5_script4_seq6_rec1_preview_rgb.mp4", 
                        help='Đường dẫn đến file video đầu vào')
    parser.add_argument('--output', type=str, default="./output_video.mp4", 
                        help='Đường dẫn file video đầu ra')
    parser.add_argument('--engine', type=str, default="./weights/depth_anything_vits14_518.trt", 
                        help='Đường dẫn đến TensorRT engine')
    parser.add_argument('--yolo_model', type=str, default="./weights/yolo11n.onnx", 
                        help='Đường dẫn đến mô hình YOLO')
    parser.add_argument('--show', action='store_true', help='Hiển thị kết quả trong quá trình xử lý')
    parser.add_argument('--save_frames', action='store_true', help='Lưu các frame kết quả')
    parser.add_argument('--depth_scale', type=float, default=10.0, 
                       help='Tỷ lệ chuyển đổi độ sâu sang khoảng cách thực tế (mét)')
    args = parser.parse_args()
    
    # Khởi tạo mô hình YOLO
    model = YOLO(args.yolo_model)
    
    # Khởi tạo DepthEngine
    depth_engine = DepthEngine(
        trt_engine_path=args.engine,
        stream=args.show,
        record=False,  # Chúng ta sẽ tự xử lý việc ghi video
        save=args.save_frames,
        grayscale=False,
        raw=True  # Cần giá trị độ sâu thô để tính khoảng cách chính xác
    )
    
    # Mở video đầu vào
    cap = cv2.VideoCapture(args.video)
    
    # Lấy thông tin video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Thiết lập video writer
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(args.output, fourcc, fps, (frame_width * 2, frame_height))
    
    frame_count = 0
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                print("Đã xử lý hết video.")
                break
            
            frame_count += 1
            print(f"Đang xử lý frame {frame_count}")
            
            # Xử lý depth - lấy raw depth map để tính khoảng cách
            depth_raw = depth_engine.process_frame(frame.copy())
            
            # Tạo bản depth map có màu cho hiển thị
            # depth_colored = cv2.applyColorMap(
            #     cv2.normalize(depth_raw, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U),
            #     cv2.COLORMAP_INFERNO
            # )
            
            # Thực hiện phát hiện đối tượng với YOLO
            yolo_results = model(frame)
            
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
                    distance = calculate_distance(depth_raw, [x1, y1, x2, y2], args.depth_scale)
                    
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
            
            # Kết hợp frame đã annotate và depth map
            #combined_result = np.concatenate((annotated_frame, depth_colored), axis=1)
            
            # Ghi frame kết quả
            out.write(annotated_frame)
            
            # Hiển thị kết quả nếu được yêu cầu
            if args.show:
                cv2.imshow('YOLO + Depth', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
    except Exception as e:
        print(f"Lỗi: {e}")
    
    finally:
        # Giải phóng tài nguyên
        cap.release()
        out.release()
        depth_engine.close()
        if args.show:
            cv2.destroyAllWindows()
        print("Đã hoàn thành xử lý.")

if __name__ == "__main__":
    main()