"""
Step 3: Pose Detection with MediaPipe
Nhiệm vụ: Xác định các điểm (keypoints) trên cơ thể người
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, Optional, NamedTuple, List


class Landmark(NamedTuple):
    """Một điểm landmark trên cơ thể."""
    x: float  # Normalized [0, 1]
    y: float  # Normalized [0, 1]
    z: float  # Depth
    visibility: float  # Confidence


class PoseDetector:
    """Detect pose và keypoints sử dụng MediaPipe."""
    
    # Mapping tên landmark quan trọng cho squat
    SQUAT_LANDMARKS = {
        'left_hip': 23,
        'right_hip': 24,
        'left_knee': 25,
        'right_knee': 26,
        'left_ankle': 27,
        'right_ankle': 28,
        'left_shoulder': 11,
        'right_shoulder': 12,
    }
    
    def __init__(self, 
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Khởi tạo MediaPipe Pose.
        
        Args:
            min_detection_confidence: Ngưỡng confidence cho detection
            min_tracking_confidence: Ngưỡng confidence cho tracking
        """
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,  # Video mode for better tracking
            model_complexity=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
    
    def detect_pose(self, frame: np.ndarray) -> Optional[Dict[str, Landmark]]:
        """
        Detect pose và trả về các keypoints quan trọng cho squat.
        
        Args:
            frame: Ảnh BGR
            
        Returns:
            Dict với key là tên landmark, value là Landmark object
            None nếu không detect được
        """
        # MediaPipe cần RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Tăng performance
        rgb_frame.flags.writeable = False
        results = self.pose.process(rgb_frame)
        rgb_frame.flags.writeable = True
        
        if not results.pose_landmarks:
            return None
        
        # Extract các landmark quan trọng
        landmarks = {}
        for name, idx in self.SQUAT_LANDMARKS.items():
            lm = results.pose_landmarks.landmark[idx]
            landmarks[name] = Landmark(lm.x, lm.y, lm.z, lm.visibility)
        
        return landmarks
    
    def detect_all_landmarks(self, frame: np.ndarray) -> Optional[List[Landmark]]:
        """Detect tất cả 33 landmarks."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self.pose.process(rgb_frame)
        
        if not results.pose_landmarks:
            return None
        
        return [
            Landmark(lm.x, lm.y, lm.z, lm.visibility)
            for lm in results.pose_landmarks.landmark
        ]
    
    def draw_pose(self, frame: np.ndarray) -> np.ndarray:
        """Vẽ skeleton lên frame."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self.pose.process(rgb_frame)
        rgb_frame.flags.writeable = True
        
        frame_copy = frame.copy()
        
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame_copy,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
        
        return frame_copy
    
    def get_pixel_coords(self, landmark: Landmark, frame_width: int, frame_height: int) -> tuple:
        """Chuyển đổi normalized coords sang pixel coords."""
        return (int(landmark.x * frame_width), int(landmark.y * frame_height))
    
    def close(self):
        """Giải phóng resources."""
        self.pose.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Test module
if __name__ == "__main__":
    print("Testing PoseDetector...")
    
    detector = PoseDetector()
    
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        landmarks = detector.detect_pose(frame)
        if landmarks:
            print("Detected landmarks:")
            for name, lm in landmarks.items():
                print(f"  {name}: x={lm.x:.3f}, y={lm.y:.3f}, vis={lm.visibility:.2f}")
            
            # Vẽ pose
            frame_with_pose = detector.draw_pose(frame)
            cv2.imshow("Pose Detection", frame_with_pose)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("No pose detected")
    
    detector.close()
    print("Test completed!")
