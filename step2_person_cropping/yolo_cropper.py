"""
Step 2: Person Cropping with YOLO v8
Nhiệm vụ: Detect người và crop frame để tránh nhiễu
"""

import cv2
import numpy as np
from typing import Optional, Tuple, NamedTuple
from ultralytics import YOLO


class BoundingBox(NamedTuple):
    """Bounding box của người được detect."""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float


class YoloCropper:
    """Detect và crop người sử dụng YOLO v8."""
    
    def __init__(self, model_path: str = "yolov8n.pt", confidence: float = 0.5):
        """
        Khởi tạo YOLO detector.
        
        Args:
            model_path: Đường dẫn tới model YOLO (sẽ tự download nếu chưa có)
            confidence: Ngưỡng confidence tối thiểu
        """
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.person_class_id = 0  # COCO class ID for "person"
    
    def detect_person(self, frame: np.ndarray) -> Optional[BoundingBox]:
        """
        Detect người trong frame.
        
        Args:
            frame: Ảnh BGR từ webcam
            
        Returns:
            BoundingBox nếu detect được người, None nếu không
        """
        results = self.model(frame, verbose=False, conf=self.confidence)
        
        # Tìm person với confidence cao nhất
        best_bbox = None
        best_conf = 0
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                if cls_id == self.person_class_id and conf > best_conf:
                    best_conf = conf
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    best_bbox = BoundingBox(x1, y1, x2, y2, conf)
        
        return best_bbox
    
    def crop_person(self, frame: np.ndarray, padding: int = 20) -> Tuple[Optional[np.ndarray], Optional[BoundingBox]]:
        """
        Detect và crop người từ frame.
        
        Args:
            frame: Ảnh BGR từ webcam
            padding: Số pixel padding thêm xung quanh bounding box
            
        Returns:
            Tuple[cropped_frame, bounding_box] hoặc (None, None) nếu không detect được
        """
        bbox = self.detect_person(frame)
        
        if bbox is None:
            return None, None
        
        # Thêm padding và clip theo kích thước frame
        h, w = frame.shape[:2]
        x1 = max(0, bbox.x1 - padding)
        y1 = max(0, bbox.y1 - padding)
        x2 = min(w, bbox.x2 + padding)
        y2 = min(h, bbox.y2 + padding)
        
        cropped = frame[y1:y2, x1:x2]
        
        return cropped, BoundingBox(x1, y1, x2, y2, bbox.confidence)
    
    def draw_bbox(self, frame: np.ndarray, bbox: BoundingBox, color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """Vẽ bounding box lên frame."""
        frame_copy = frame.copy()
        cv2.rectangle(frame_copy, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), color, 2)
        cv2.putText(frame_copy, f"Person: {bbox.confidence:.2f}", 
                    (bbox.x1, bbox.y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return frame_copy


# Test module
if __name__ == "__main__":
    print("Testing YoloCropper...")
    print("Downloading YOLO model if needed...")
    cropper = YoloCropper()
    
    # Test với webcam
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        cropped, bbox = cropper.crop_person(frame)
        if bbox:
            print(f"Detected person: {bbox}")
            frame_with_box = cropper.draw_bbox(frame, bbox)
            cv2.imshow("Detection", frame_with_box)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("No person detected")
    print("Test completed!")
