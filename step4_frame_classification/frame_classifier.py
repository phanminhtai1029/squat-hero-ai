"""
Step 4: Frame Classification
=============================
Nhiệm vụ: Phân loại frame thành tư thế (pose) hoặc chuyển động (transition)

Logic: Dựa vào góc gối và angular velocity để xác định:
- POSE_STANDING: Đứng ổn định (góc > 160°, velocity thấp)
- POSE_SQUAT: Squat ổn định (góc < 90°, velocity thấp)
- TRANSITION_DOWN: Đang đi xuống
- TRANSITION_UP: Đang đi lên
"""

import numpy as np
from typing import Dict, Optional, NamedTuple, Deque
from enum import Enum
from collections import deque


class FrameType(Enum):
    """Loại frame trong động tác squat."""
    POSE_STANDING = "pose_standing"       # Tư thế đứng ổn định
    POSE_SQUAT = "pose_squat"             # Tư thế squat ổn định
    TRANSITION_DOWN = "transition_down"   # Đang chuyển động xuống
    TRANSITION_UP = "transition_up"       # Đang chuyển động lên
    UNKNOWN = "unknown"                   # Không xác định


class ClassificationResult(NamedTuple):
    """Kết quả phân loại frame."""
    frame_type: FrameType
    knee_angle: float
    back_angle: float
    angular_velocity: float  # Độ/frame
    is_stable: bool          # True nếu là pose frame
    confidence: float        # Độ tin cậy phân loại (0-1)


class FrameClassifier:
    """
    Phân loại frame dựa trên góc và tốc độ thay đổi góc.
    
    Sử dụng buffer để tính angular velocity và xác định
    frame đang ở tư thế ổn định hay đang chuyển động.
    """
    
    # Ngưỡng góc (degrees)
    STANDING_THRESHOLD = 160   # Góc gối > 160 = đứng
    SQUAT_THRESHOLD = 90       # Góc gối < 90 = squat đủ sâu
    
    # Ngưỡng stability (degrees/frame)
    STABILITY_THRESHOLD = 3.0  # Angular velocity < 3°/frame = ổn định
    
    # Buffer size cho tính velocity
    BUFFER_SIZE = 5
    
    def __init__(self, 
                 standing_threshold: float = 160,
                 squat_threshold: float = 90,
                 stability_threshold: float = 3.0):
        """
        Khởi tạo FrameClassifier.
        
        Args:
            standing_threshold: Ngưỡng góc gối cho tư thế đứng
            squat_threshold: Ngưỡng góc gối cho tư thế squat
            stability_threshold: Ngưỡng angular velocity để coi là ổn định
        """
        self.STANDING_THRESHOLD = standing_threshold
        self.SQUAT_THRESHOLD = squat_threshold
        self.STABILITY_THRESHOLD = stability_threshold
        
        # Buffer lưu history góc gối
        self.angle_buffer: Deque[float] = deque(maxlen=self.BUFFER_SIZE)
        
        # State tracking
        self.prev_frame_type = FrameType.UNKNOWN
        self.frames_in_current_state = 0
    
    def calculate_angle(self, a: tuple, b: tuple, c: tuple) -> float:
        """
        Tính góc tại điểm b tạo bởi 3 điểm a-b-c.
        
        Args:
            a: Điểm đầu (x, y)
            b: Điểm giữa - vertex (x, y)
            c: Điểm cuối (x, y)
            
        Returns:
            Góc tính bằng degrees (0-180)
        """
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        
        angle = np.degrees(np.arccos(cosine_angle))
        return angle
    
    def get_knee_angle(self, landmarks: Dict) -> float:
        """Tính góc đầu gối (hip-knee-ankle)."""
        # Chọn chân có visibility cao hơn
        left_vis = getattr(landmarks.get('left_knee'), 'visibility', 0) or 0
        right_vis = getattr(landmarks.get('right_knee'), 'visibility', 0) or 0
        
        if left_vis >= right_vis:
            hip = (landmarks['left_hip'].x, landmarks['left_hip'].y)
            knee = (landmarks['left_knee'].x, landmarks['left_knee'].y)
            ankle = (landmarks['left_ankle'].x, landmarks['left_ankle'].y)
        else:
            hip = (landmarks['right_hip'].x, landmarks['right_hip'].y)
            knee = (landmarks['right_knee'].x, landmarks['right_knee'].y)
            ankle = (landmarks['right_ankle'].x, landmarks['right_ankle'].y)
        
        return self.calculate_angle(hip, knee, ankle)
    
    def get_back_angle(self, landmarks: Dict) -> float:
        """Tính góc lưng (shoulder-hip-knee)."""
        left_vis = getattr(landmarks.get('left_hip'), 'visibility', 0) or 0
        right_vis = getattr(landmarks.get('right_hip'), 'visibility', 0) or 0
        
        if left_vis >= right_vis:
            shoulder = (landmarks['left_shoulder'].x, landmarks['left_shoulder'].y)
            hip = (landmarks['left_hip'].x, landmarks['left_hip'].y)
            knee = (landmarks['left_knee'].x, landmarks['left_knee'].y)
        else:
            shoulder = (landmarks['right_shoulder'].x, landmarks['right_shoulder'].y)
            hip = (landmarks['right_hip'].x, landmarks['right_hip'].y)
            knee = (landmarks['right_knee'].x, landmarks['right_knee'].y)
        
        return self.calculate_angle(shoulder, hip, knee)
    
    def calculate_angular_velocity(self) -> float:
        """
        Tính angular velocity dựa trên buffer góc.
        
        Returns:
            Angular velocity (degrees/frame), hoặc 0 nếu buffer chưa đủ
        """
        if len(self.angle_buffer) < 2:
            return 0.0
        
        # Tính trung bình thay đổi góc giữa các frame liên tiếp
        angles = list(self.angle_buffer)
        velocities = [abs(angles[i] - angles[i-1]) for i in range(1, len(angles))]
        
        return sum(velocities) / len(velocities)
    
    def classify(self, landmarks: Dict) -> ClassificationResult:
        """
        Phân loại frame hiện tại.
        
        Args:
            landmarks: Dict các landmark từ PoseDetector
            
        Returns:
            ClassificationResult với loại frame và thông tin liên quan
        """
        # Tính các góc
        knee_angle = self.get_knee_angle(landmarks)
        back_angle = self.get_back_angle(landmarks)
        
        # Thêm vào buffer
        self.angle_buffer.append(knee_angle)
        
        # Tính angular velocity
        angular_velocity = self.calculate_angular_velocity()
        
        # Xác định frame type
        is_stable = angular_velocity < self.STABILITY_THRESHOLD
        
        if knee_angle > self.STANDING_THRESHOLD:
            if is_stable:
                frame_type = FrameType.POSE_STANDING
            else:
                # Đang từ squat lên standing
                frame_type = FrameType.TRANSITION_UP
        elif knee_angle < self.SQUAT_THRESHOLD:
            if is_stable:
                frame_type = FrameType.POSE_SQUAT
            else:
                # Vừa xuống đến squat nhưng chưa ổn định
                frame_type = FrameType.TRANSITION_DOWN
        else:
            # Góc ở giữa (90-160)
            if self.prev_frame_type in [FrameType.POSE_STANDING, FrameType.TRANSITION_DOWN, FrameType.UNKNOWN]:
                frame_type = FrameType.TRANSITION_DOWN
            else:
                frame_type = FrameType.TRANSITION_UP
        
        # Tính confidence
        confidence = self._calculate_confidence(knee_angle, angular_velocity, is_stable)
        
        # Update state
        if frame_type == self.prev_frame_type:
            self.frames_in_current_state += 1
        else:
            self.frames_in_current_state = 1
            self.prev_frame_type = frame_type
        
        return ClassificationResult(
            frame_type=frame_type,
            knee_angle=knee_angle,
            back_angle=back_angle,
            angular_velocity=angular_velocity,
            is_stable=is_stable,
            confidence=confidence
        )
    
    def _calculate_confidence(self, knee_angle: float, angular_velocity: float, is_stable: bool) -> float:
        """
        Tính độ tin cậy của phân loại.
        
        Returns:
            Confidence score từ 0 đến 1
        """
        confidence = 1.0
        
        # Giảm confidence nếu góc ở biên
        if 155 < knee_angle < 165:  # Gần ngưỡng standing
            confidence *= 0.8
        if 85 < knee_angle < 95:    # Gần ngưỡng squat
            confidence *= 0.8
        
        # Giảm confidence nếu velocity cao nhưng góc ổn định
        if not is_stable and angular_velocity > 10:
            confidence *= 0.7
        
        # Tăng confidence nếu ở pose rõ ràng
        if is_stable and (knee_angle > 170 or knee_angle < 80):
            confidence = min(confidence * 1.1, 1.0)
        
        return round(confidence, 2)
    
    def is_pose_frame(self, classification: ClassificationResult) -> bool:
        """Check xem frame có phải là pose frame không."""
        return classification.frame_type in [FrameType.POSE_STANDING, FrameType.POSE_SQUAT]
    
    def reset(self):
        """Reset state và buffer."""
        self.angle_buffer.clear()
        self.prev_frame_type = FrameType.UNKNOWN
        self.frames_in_current_state = 0


# Test module
if __name__ == "__main__":
    print("Testing FrameClassifier...")
    
    classifier = FrameClassifier()
    
    # Mock landmarks cho test
    from collections import namedtuple
    Landmark = namedtuple('Landmark', ['x', 'y', 'z', 'visibility'])
    
    # Simulate sequence: Standing -> Going Down -> Squat -> Going Up -> Standing
    test_sequences = [
        # Standing
        {'left_hip': Landmark(0.5, 0.4, 0, 0.9), 'left_knee': Landmark(0.5, 0.6, 0, 0.9), 
         'left_ankle': Landmark(0.5, 0.8, 0, 0.9), 'left_shoulder': Landmark(0.5, 0.2, 0, 0.9),
         'right_hip': Landmark(0.5, 0.4, 0, 0.8), 'right_knee': Landmark(0.5, 0.6, 0, 0.8),
         'right_ankle': Landmark(0.5, 0.8, 0, 0.8), 'right_shoulder': Landmark(0.5, 0.2, 0, 0.8)},
    ]
    
    for landmarks in test_sequences:
        result = classifier.classify(landmarks)
        print(f"Frame Type: {result.frame_type.value}")
        print(f"Knee Angle: {result.knee_angle:.1f}°")
        print(f"Angular Velocity: {result.angular_velocity:.2f}°/frame")
        print(f"Is Stable: {result.is_stable}")
        print(f"Confidence: {result.confidence}")
        print("-" * 40)
    
    print("Test completed!")
