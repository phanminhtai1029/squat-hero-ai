"""
Step 5: Standard Pose Comparison
=================================
Nhiệm vụ: So sánh frame tư thế với tư thế chuẩn

Chỉ so sánh khi nhận được POSE frame (không phải TRANSITION frame).
Load tư thế chuẩn từ JSON và đưa ra đánh giá chi tiết.
"""

import json
import os
from typing import Dict, List, NamedTuple, Optional
from enum import Enum
from pathlib import Path


class FormError(Enum):
    """Các lỗi form khi squat."""
    NONE = "none"
    KNEE_TOO_SHALLOW = "knee_too_shallow"     # Squat không đủ sâu
    BACK_ROUNDING = "back_rounding"           # Cong lưng
    BACK_LEANING = "back_leaning"             # Nghiêng người quá nhiều
    KNEE_CAVE_IN = "knee_cave_in"             # Đầu gối khép vào trong


class Severity(Enum):
    """Mức độ nghiêm trọng của lỗi."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class FormFeedback(NamedTuple):
    """Feedback cho một lỗi form cụ thể."""
    error: FormError
    message: str
    severity: Severity


class ComparisonResult(NamedTuple):
    """Kết quả so sánh với tư thế chuẩn."""
    pose_name: str              # Tên tư thế đang so sánh
    knee_angle: float
    back_angle: float
    similarity_score: float     # Điểm tương đồng (0-100%)
    is_good_form: bool
    errors: List[FormFeedback]  # Danh sách lỗi và feedback
    tips: List[str]             # Gợi ý cải thiện
    rep_count: int


class StandardPoseComparator:
    """
    So sánh pose frame với tư thế chuẩn.
    
    Load tư thế chuẩn từ JSON file và đánh giá mức độ
    tương đồng cũng như phát hiện các lỗi form.
    """
    
    def __init__(self, poses_file: str = None):
        """
        Khởi tạo StandardPoseComparator.
        
        Args:
            poses_file: Path đến file JSON chứa tư thế chuẩn.
                       Nếu None, sẽ load từ file mặc định.
        """
        if poses_file is None:
            # Load từ file mặc định trong cùng thư mục
            current_dir = Path(__file__).parent
            poses_file = current_dir / "standard_poses.json"
        
        self.standard_poses = self._load_poses(poses_file)
        
        # State tracking cho rep counting
        self.rep_count = 0
        self.was_in_squat = False
        self.prev_pose_name = None
    
    def _load_poses(self, poses_file) -> Dict:
        """Load tư thế chuẩn từ JSON file."""
        try:
            with open(poses_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: Poses file not found at {poses_file}. Using defaults.")
            return self._get_default_poses()
        except json.JSONDecodeError as e:
            print(f"Warning: Invalid JSON in {poses_file}: {e}. Using defaults.")
            return self._get_default_poses()
    
    def _get_default_poses(self) -> Dict:
        """Trả về tư thế chuẩn mặc định nếu không load được từ file."""
        return {
            "squat": {
                "name": "Squat chuẩn",
                "angles": {
                    "knee": {"ideal": 85, "min": 70, "max": 95, "tolerance": 10},
                    "back": {"ideal": 110, "min": 100, "max": 130, "tolerance": 15}
                },
                "errors": {},
                "tips": ["Giữ lưng thẳng", "Squat sâu"]
            },
            "standing": {
                "name": "Tư thế đứng",
                "angles": {
                    "knee": {"ideal": 175, "min": 160, "max": 180, "tolerance": 10},
                    "back": {"ideal": 170, "min": 160, "max": 180, "tolerance": 10}
                },
                "errors": {},
                "tips": []
            }
        }
    
    def calculate_similarity(self, actual_angle: float, ideal_config: Dict) -> float:
        """
        Tính điểm tương đồng cho một góc so với chuẩn.
        
        Args:
            actual_angle: Góc thực tế
            ideal_config: Dict chứa {ideal, min, max, tolerance}
            
        Returns:
            Điểm từ 0 đến 100
        """
        ideal = ideal_config["ideal"]
        tolerance = ideal_config.get("tolerance", 10)
        
        # Tính khoảng cách so với ideal
        diff = abs(actual_angle - ideal)
        
        if diff <= tolerance / 2:
            # Rất gần ideal: 90-100 điểm
            return 100 - (diff / tolerance * 20)
        elif diff <= tolerance:
            # Trong khoảng tolerance: 70-90 điểm
            return 90 - (diff / tolerance * 20)
        elif diff <= tolerance * 2:
            # Gấp đôi tolerance: 50-70 điểm
            return 70 - ((diff - tolerance) / tolerance * 20)
        else:
            # Quá xa: 0-50 điểm
            return max(0, 50 - (diff - tolerance * 2) * 2)
    
    def detect_errors(self, pose_name: str, knee_angle: float, back_angle: float) -> List[FormFeedback]:
        """
        Phát hiện các lỗi form dựa trên tư thế chuẩn.
        
        Args:
            pose_name: Tên tư thế (squat/standing)
            knee_angle: Góc đầu gối thực tế
            back_angle: Góc lưng thực tế
            
        Returns:
            Danh sách FormFeedback cho các lỗi phát hiện
        """
        errors = []
        
        if pose_name not in self.standard_poses:
            return errors
        
        pose_config = self.standard_poses[pose_name]
        
        # Kiểm tra các lỗi từ config
        if pose_name == "squat":
            knee_config = pose_config["angles"]["knee"]
            back_config = pose_config["angles"]["back"]
            
            # Kiểm tra squat không đủ sâu
            if knee_angle > knee_config["max"]:
                errors.append(FormFeedback(
                    error=FormError.KNEE_TOO_SHALLOW,
                    message="⚠️ Squat chưa đủ sâu! Hạ thấp hơn.",
                    severity=Severity.WARNING
                ))
            
            # Kiểm tra cong lưng
            if back_angle < back_config["min"]:
                errors.append(FormFeedback(
                    error=FormError.BACK_ROUNDING,
                    message="⚠️ Lưng bị cong! Giữ thẳng lưng.",
                    severity=Severity.ERROR
                ))
            
            # Kiểm tra nghiêng người
            if back_angle > back_config["max"]:
                errors.append(FormFeedback(
                    error=FormError.BACK_LEANING,
                    message="⚠️ Người nghiêng quá nhiều! Ngực hướng lên.",
                    severity=Severity.WARNING
                ))
        
        if not errors:
            errors.append(FormFeedback(
                error=FormError.NONE,
                message="✓ Form tốt! Tiếp tục giữ.",
                severity=Severity.INFO
            ))
        
        return errors
    
    def compare(self, 
                knee_angle: float, 
                back_angle: float,
                is_squat_pose: bool) -> ComparisonResult:
        """
        So sánh pose hiện tại với tư thế chuẩn.
        
        Args:
            knee_angle: Góc đầu gối
            back_angle: Góc lưng
            is_squat_pose: True nếu đang ở tư thế squat, False nếu đứng
            
        Returns:
            ComparisonResult với đánh giá chi tiết
        """
        pose_name = "squat" if is_squat_pose else "standing"
        pose_config = self.standard_poses.get(pose_name, {})
        
        # Tính similarity score
        knee_config = pose_config.get("angles", {}).get("knee", {"ideal": 85 if is_squat_pose else 175})
        back_config = pose_config.get("angles", {}).get("back", {"ideal": 110 if is_squat_pose else 170})
        
        knee_score = self.calculate_similarity(knee_angle, knee_config)
        back_score = self.calculate_similarity(back_angle, back_config)
        
        # Tổng hợp score (weighted average)
        similarity_score = (knee_score * 0.6 + back_score * 0.4)
        
        # Phát hiện lỗi
        errors = self.detect_errors(pose_name, knee_angle, back_angle)
        is_good = all(e.error == FormError.NONE for e in errors)
        
        # Đếm reps: Khi chuyển từ squat sang standing
        if is_squat_pose:
            self.was_in_squat = True
        elif self.was_in_squat and not is_squat_pose:
            self.rep_count += 1
            self.was_in_squat = False
        
        # Lấy tips
        tips = pose_config.get("tips", [])
        
        self.prev_pose_name = pose_name
        
        return ComparisonResult(
            pose_name=pose_config.get("name", pose_name),
            knee_angle=knee_angle,
            back_angle=back_angle,
            similarity_score=round(similarity_score, 1),
            is_good_form=is_good,
            errors=errors,
            tips=tips,
            rep_count=self.rep_count
        )
    
    def get_feedback_text(self, result: ComparisonResult) -> str:
        """Tạo feedback text ngắn gọn từ ComparisonResult."""
        if result.is_good_form:
            return f"✓ {result.pose_name} - Điểm: {result.similarity_score}%"
        else:
            # Lấy lỗi đầu tiên
            first_error = result.errors[0] if result.errors else None
            if first_error:
                return first_error.message
            return f"{result.pose_name} - Điểm: {result.similarity_score}%"
    
    def reset(self):
        """Reset state."""
        self.rep_count = 0
        self.was_in_squat = False
        self.prev_pose_name = None


# Test module
if __name__ == "__main__":
    print("Testing StandardPoseComparator...")
    
    comparator = StandardPoseComparator()
    
    # Test squat pose
    print("\n--- Test Squat Pose (good form) ---")
    result = comparator.compare(knee_angle=85, back_angle=110, is_squat_pose=True)
    print(f"Pose: {result.pose_name}")
    print(f"Similarity: {result.similarity_score}%")
    print(f"Good Form: {result.is_good_form}")
    print(f"Feedback: {comparator.get_feedback_text(result)}")
    
    print("\n--- Test Squat Pose (bad form - shallow) ---")
    result = comparator.compare(knee_angle=120, back_angle=110, is_squat_pose=True)
    print(f"Pose: {result.pose_name}")
    print(f"Similarity: {result.similarity_score}%")
    print(f"Good Form: {result.is_good_form}")
    print(f"Errors: {[e.message for e in result.errors]}")
    
    print("\n--- Test Standing Pose ---")
    result = comparator.compare(knee_angle=170, back_angle=175, is_squat_pose=False)
    print(f"Pose: {result.pose_name}")
    print(f"Similarity: {result.similarity_score}%")
    print(f"Rep Count: {result.rep_count}")
    
    print("\n✓ Test completed!")
