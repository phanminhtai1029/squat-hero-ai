"""
Step 4: Pose Comparison
Nhiệm vụ: So sánh pose hiện tại với pose chuẩn và đưa ra đánh giá
"""

import numpy as np
from typing import Dict, List, NamedTuple, Optional
from enum import Enum


class SquatPhase(Enum):
    """Trạng thái của động tác squat."""
    STANDING = "standing"     # Đứng thẳng
    GOING_DOWN = "going_down" # Đang ngồi xuống
    SQUAT = "squat"           # Đang ở tư thế squat
    GOING_UP = "going_up"     # Đang đứng lên


class FormError(Enum):
    """Các lỗi form khi squat."""
    NONE = "none"
    KNEE_CAVE_IN = "knee_cave_in"     # Đầu gối khép vào trong
    BACK_ROUNDING = "back_rounding"   # Cong lưng
    NOT_DEEP_ENOUGH = "not_deep_enough"  # Squat không đủ sâu
    KNEES_OVER_TOES = "knees_over_toes"  # Đầu gối quá ngón chân


class ComparisonResult(NamedTuple):
    """Kết quả so sánh pose."""
    phase: SquatPhase
    knee_angle: float
    back_angle: float
    errors: List[FormError]
    rep_count: int
    is_good_form: bool
    feedback: str


class PoseComparator:
    """So sánh pose và đếm reps."""
    
    # Ngưỡng góc (degrees)
    STANDING_THRESHOLD = 160  # Góc gối > 160 = đứng
    SQUAT_THRESHOLD = 90      # Góc gối < 90 = squat đủ sâu
    BACK_ANGLE_MIN = 100      # Góc lưng tối thiểu khi squat
    
    def __init__(self):
        self.rep_count = 0
        self.current_phase = SquatPhase.STANDING
        self.was_in_squat = False  # Để đếm reps
    
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
        # Sử dụng chân trái hoặc phải dựa vào visibility
        left_vis = landmarks.get('left_knee', (0, 0, 0, 0))[3] if 'left_knee' in landmarks else 0
        right_vis = landmarks.get('right_knee', (0, 0, 0, 0))[3] if 'right_knee' in landmarks else 0
        
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
        left_vis = landmarks.get('left_hip', (0, 0, 0, 0))[3] if 'left_hip' in landmarks else 0
        right_vis = landmarks.get('right_hip', (0, 0, 0, 0))[3] if 'right_hip' in landmarks else 0
        
        if left_vis >= right_vis:
            shoulder = (landmarks['left_shoulder'].x, landmarks['left_shoulder'].y)
            hip = (landmarks['left_hip'].x, landmarks['left_hip'].y)
            knee = (landmarks['left_knee'].x, landmarks['left_knee'].y)
        else:
            shoulder = (landmarks['right_shoulder'].x, landmarks['right_shoulder'].y)
            hip = (landmarks['right_hip'].x, landmarks['right_hip'].y)
            knee = (landmarks['right_knee'].x, landmarks['right_knee'].y)
        
        return self.calculate_angle(shoulder, hip, knee)
    
    def determine_phase(self, knee_angle: float) -> SquatPhase:
        """Xác định phase hiện tại dựa vào góc gối."""
        if knee_angle > self.STANDING_THRESHOLD:
            return SquatPhase.STANDING
        elif knee_angle < self.SQUAT_THRESHOLD:
            return SquatPhase.SQUAT
        elif self.current_phase in [SquatPhase.STANDING, SquatPhase.GOING_DOWN]:
            return SquatPhase.GOING_DOWN
        else:
            return SquatPhase.GOING_UP
    
    def check_form_errors(self, knee_angle: float, back_angle: float) -> List[FormError]:
        """Kiểm tra các lỗi form."""
        errors = []
        
        # Kiểm tra squat không đủ sâu khi ở phase going_up
        if self.current_phase == SquatPhase.GOING_UP and knee_angle > self.SQUAT_THRESHOLD + 20:
            errors.append(FormError.NOT_DEEP_ENOUGH)
        
        # Kiểm tra cong lưng (góc lưng quá nhỏ)
        if back_angle < self.BACK_ANGLE_MIN and self.current_phase in [SquatPhase.SQUAT, SquatPhase.GOING_DOWN]:
            errors.append(FormError.BACK_ROUNDING)
        
        return errors if errors else [FormError.NONE]
    
    def compare(self, landmarks: Dict) -> ComparisonResult:
        """
        So sánh pose hiện tại và trả về kết quả đánh giá.
        
        Args:
            landmarks: Dict các landmark từ PoseDetector
            
        Returns:
            ComparisonResult với đầy đủ thông tin đánh giá
        """
        # Tính các góc
        knee_angle = self.get_knee_angle(landmarks)
        back_angle = self.get_back_angle(landmarks)
        
        # Xác định phase
        new_phase = self.determine_phase(knee_angle)
        
        # Đếm reps: khi chuyển từ SQUAT/GOING_UP sang STANDING
        if self.was_in_squat and new_phase == SquatPhase.STANDING:
            self.rep_count += 1
            self.was_in_squat = False
        
        if new_phase == SquatPhase.SQUAT:
            self.was_in_squat = True
        
        self.current_phase = new_phase
        
        # Kiểm tra lỗi form
        errors = self.check_form_errors(knee_angle, back_angle)
        is_good = FormError.NONE in errors
        
        # Tạo feedback
        feedback = self._generate_feedback(new_phase, knee_angle, errors)
        
        return ComparisonResult(
            phase=new_phase,
            knee_angle=knee_angle,
            back_angle=back_angle,
            errors=errors,
            rep_count=self.rep_count,
            is_good_form=is_good,
            feedback=feedback
        )
    
    def _generate_feedback(self, phase: SquatPhase, knee_angle: float, errors: List[FormError]) -> str:
        """Generate feedback text for user."""
        if phase == SquatPhase.STANDING:
            return "Ready! Start your squat"
        elif phase == SquatPhase.GOING_DOWN:
            return f"Going down... Angle: {knee_angle:.0f}"
        elif phase == SquatPhase.SQUAT:
            if FormError.BACK_ROUNDING in errors:
                return "Keep your back straight!"
            return f"Good! Hold position. Angle: {knee_angle:.0f}"
        else:  # GOING_UP
            if FormError.NOT_DEEP_ENOUGH in errors:
                return "Go deeper!"
            return f"Standing up! Angle: {knee_angle:.0f}"
    
    def reset(self):
        """Reset counter và state."""
        self.rep_count = 0
        self.current_phase = SquatPhase.STANDING
        self.was_in_squat = False


# Test module
if __name__ == "__main__":
    print("Testing PoseComparator...")
    
    comparator = PoseComparator()
    
    # Mock landmarks cho test
    from collections import namedtuple
    Landmark = namedtuple('Landmark', ['x', 'y', 'z', 'visibility'])
    
    # Test case: Standing pose
    standing_landmarks = {
        'left_hip': Landmark(0.5, 0.5, 0, 0.9),
        'left_knee': Landmark(0.5, 0.7, 0, 0.9),
        'left_ankle': Landmark(0.5, 0.9, 0, 0.9),
        'left_shoulder': Landmark(0.5, 0.3, 0, 0.9),
        'right_hip': Landmark(0.5, 0.5, 0, 0.8),
        'right_knee': Landmark(0.5, 0.7, 0, 0.8),
        'right_ankle': Landmark(0.5, 0.9, 0, 0.8),
        'right_shoulder': Landmark(0.5, 0.3, 0, 0.8),
    }
    
    result = comparator.compare(standing_landmarks)
    print(f"Phase: {result.phase.value}")
    print(f"Knee angle: {result.knee_angle:.1f}°")
    print(f"Feedback: {result.feedback}")
    print(f"Reps: {result.rep_count}")
    
    print("\nTest completed!")
