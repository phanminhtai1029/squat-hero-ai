"""
Step 1: Webcam Frame Capture
Nhiệm vụ: Mở webcam và capture frame real-time
"""

import cv2
import numpy as np
from typing import Optional


class WebcamCapture:
    """Capture frames từ webcam real-time."""
    
    def __init__(self, camera_id: int = 0, width: int = 640, height: int = 480):
        """
        Khởi tạo webcam capture.
        
        Args:
            camera_id: ID của camera (0 = camera mặc định)
            width: Chiều rộng frame
            height: Chiều cao frame
        """
        self.camera_id = camera_id
        self.cap = cv2.VideoCapture(camera_id)
        
        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Không thể mở camera {camera_id}")
    
    def get_frame(self) -> Optional[np.ndarray]:
        """
        Lấy 1 frame từ webcam.
        
        Returns:
            np.ndarray: Frame ảnh BGR, hoặc None nếu lỗi
        """
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame
    
    def get_fps(self) -> float:
        """Lấy FPS thực tế của camera."""
        return self.cap.get(cv2.CAP_PROP_FPS)
    
    def release(self) -> None:
        """Giải phóng camera."""
        if self.cap.isOpened():
            self.cap.release()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


# Test module
if __name__ == "__main__":
    print("Testing WebcamCapture...")
    with WebcamCapture() as cam:
        print(f"Camera FPS: {cam.get_fps()}")
        frame = cam.get_frame()
        if frame is not None:
            print(f"Frame shape: {frame.shape}")
            cv2.imshow("Test", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    print("Test passed!")
