"""
Form Scorer - Cosine Similarity Based
======================================
Đánh giá quality của động tác bằng cách so sánh với reference poses.

Input:  Keypoints (132,) + Pose label từ Step 4
Output: Score (0-100%) + Feedback
"""

import numpy as np
from typing import Dict, List, NamedTuple, Optional
from pathlib import Path
import json


class ScoringResult(NamedTuple):
    """Kết quả đánh giá form."""
    pose_label: str          # Tên động tác
    score: float             # Điểm (0-100)
    is_good_form: bool       # True nếu score > threshold
    feedback: str            # Feedback chính
    details: Dict            # Chi tiết (góc, so sánh...)
    rep_count: int           # Số reps đã đếm


class FormScorer:
    """
    Đánh giá form động tác bằng Cosine Similarity.
    
    So sánh keypoints hiện tại với reference poses đã lưu sẵn.
    """
    
    # Ngưỡng để coi là good form
    GOOD_FORM_THRESHOLD = 70.0
    
    def __init__(self, reference_dir: Optional[str] = None):
        """
        Khởi tạo FormScorer.
        
        Args:
            reference_dir: Thư mục chứa reference poses (.npy files)
        """
        if reference_dir is None:
            current_dir = Path(__file__).parent
            reference_dir = current_dir / "reference_poses"
        
        self.reference_dir = Path(reference_dir)
        self.reference_poses: Dict[str, np.ndarray] = {}
        
        # State cho rep counting
        self.rep_count = 0
        self.prev_pose = None
        self.was_in_pose = {}
        
        # Load reference poses
        self._load_references()
    
    def _load_references(self):
        """Load tất cả reference poses từ thư mục."""
        if not self.reference_dir.exists():
            print(f"Reference dir not found: {self.reference_dir}")
            self._create_default_references()
            return
        
        for npy_file in self.reference_dir.glob("*.npy"):
            pose_name = npy_file.stem.replace("_reference", "")
            try:
                self.reference_poses[pose_name] = np.load(npy_file)
                print(f"Loaded reference: {pose_name}")
            except Exception as e:
                print(f"Failed to load {npy_file}: {e}")
        
        if not self.reference_poses:
            print("No reference poses found. Creating defaults.")
            self._create_default_references()
    
    def _create_default_references(self):
        """
        Tạo reference poses mặc định (dummy).
        Trong thực tế, cần thu thập từ người tập đúng form.
        """
        # Đảm bảo thư mục tồn tại
        self.reference_dir.mkdir(parents=True, exist_ok=True)
        
        # Tạo dummy reference cho mỗi pose
        poses_defaults = {
            'squat': self._create_squat_reference(),
            'lunge': self._create_lunge_reference(),
            'plank': self._create_plank_reference(),
            'warrior_i': self._create_warrior_reference(),
            'tree_pose': self._create_tree_reference(),
        }
        
        for pose_name, reference in poses_defaults.items():
            self.reference_poses[pose_name] = reference
            # Save to file
            save_path = self.reference_dir / f"{pose_name}_reference.npy"
            np.save(save_path, reference)
            print(f"Created default reference: {pose_name}")
    
    def _create_squat_reference(self) -> np.ndarray:
        """Reference cho squat chuẩn."""
        # Format: 33 keypoints × 4 (x, y, z, visibility)
        # Đây là dummy data, cần thay bằng data thực
        ref = np.zeros(132, dtype=np.float32)
        
        # Đặt các keypoints quan trọng (normalized coords)
        # left_shoulder (index 11): x, y, z, vis
        ref[11*4:11*4+4] = [0.45, 0.25, 0, 0.95]
        # right_shoulder (index 12)
        ref[12*4:12*4+4] = [0.55, 0.25, 0, 0.95]
        # left_hip (index 23)
        ref[23*4:23*4+4] = [0.45, 0.50, 0, 0.95]
        # right_hip (index 24)
        ref[24*4:24*4+4] = [0.55, 0.50, 0, 0.95]
        # left_knee (index 25)
        ref[25*4:25*4+4] = [0.42, 0.70, 0, 0.95]
        # right_knee (index 26)
        ref[26*4:26*4+4] = [0.58, 0.70, 0, 0.95]
        # left_ankle (index 27)
        ref[27*4:27*4+4] = [0.40, 0.90, 0, 0.95]
        # right_ankle (index 28)
        ref[28*4:28*4+4] = [0.60, 0.90, 0, 0.95]
        
        return ref
    
    def _create_lunge_reference(self) -> np.ndarray:
        """Reference cho lunge chuẩn."""
        ref = np.zeros(132, dtype=np.float32)
        # Asymmetric pose: một chân trước, một chân sau
        ref[23*4:23*4+4] = [0.50, 0.50, 0, 0.95]  # left_hip
        ref[24*4:24*4+4] = [0.50, 0.50, 0, 0.95]  # right_hip
        ref[25*4:25*4+4] = [0.35, 0.70, 0, 0.95]  # left_knee (forward)
        ref[26*4:26*4+4] = [0.65, 0.80, 0, 0.95]  # right_knee (back)
        ref[27*4:27*4+4] = [0.30, 0.90, 0, 0.95]  # left_ankle
        ref[28*4:28*4+4] = [0.70, 0.95, 0, 0.95]  # right_ankle
        return ref
    
    def _create_plank_reference(self) -> np.ndarray:
        """Reference cho plank chuẩn."""
        ref = np.zeros(132, dtype=np.float32)
        # Horizontal pose
        ref[11*4:11*4+4] = [0.30, 0.50, 0, 0.95]  # left_shoulder
        ref[12*4:12*4+4] = [0.30, 0.55, 0, 0.95]  # right_shoulder
        ref[23*4:23*4+4] = [0.50, 0.50, 0, 0.95]  # left_hip
        ref[24*4:24*4+4] = [0.50, 0.55, 0, 0.95]  # right_hip
        ref[27*4:27*4+4] = [0.70, 0.50, 0, 0.95]  # left_ankle
        ref[28*4:28*4+4] = [0.70, 0.55, 0, 0.95]  # right_ankle
        return ref
    
    def _create_warrior_reference(self) -> np.ndarray:
        """Reference cho Warrior I pose."""
        ref = np.zeros(132, dtype=np.float32)
        # Arms up, front knee bent
        ref[11*4:11*4+4] = [0.45, 0.20, 0, 0.95]  # left_shoulder
        ref[12*4:12*4+4] = [0.55, 0.20, 0, 0.95]  # right_shoulder
        ref[15*4:15*4+4] = [0.45, 0.05, 0, 0.95]  # left_wrist (up)
        ref[16*4:16*4+4] = [0.55, 0.05, 0, 0.95]  # right_wrist (up)
        ref[25*4:25*4+4] = [0.40, 0.65, 0, 0.95]  # left_knee (bent)
        ref[26*4:26*4+4] = [0.60, 0.50, 0, 0.95]  # right_knee (straight)
        return ref
    
    def _create_tree_reference(self) -> np.ndarray:
        """Reference cho Tree pose."""
        ref = np.zeros(132, dtype=np.float32)
        # Standing on one leg
        ref[23*4:23*4+4] = [0.50, 0.45, 0, 0.95]  # left_hip
        ref[24*4:24*4+4] = [0.50, 0.45, 0, 0.95]  # right_hip
        ref[25*4:25*4+4] = [0.50, 0.70, 0, 0.95]  # left_knee (standing)
        ref[26*4:26*4+4] = [0.45, 0.55, 0, 0.95]  # right_knee (lifted, to side)
        ref[27*4:27*4+4] = [0.50, 0.95, 0, 0.95]  # left_ankle (on ground)
        ref[28*4:28*4+4] = [0.45, 0.65, 0, 0.95]  # right_ankle (on left leg)
        return ref
    
    def preprocess_keypoints(self, landmarks: Dict) -> np.ndarray:
        """Chuyển landmarks dict thành vector 132 chiều."""
        # Mapping keypoint names to indices
        KEYPOINT_INDICES = {
            'left_shoulder': 11, 'right_shoulder': 12,
            'left_hip': 23, 'right_hip': 24,
            'left_knee': 25, 'right_knee': 26,
            'left_ankle': 27, 'right_ankle': 28,
        }
        
        features = np.zeros(132, dtype=np.float32)
        
        for name, idx in KEYPOINT_INDICES.items():
            if name in landmarks:
                lm = landmarks[name]
                features[idx*4] = lm.x
                features[idx*4 + 1] = lm.y
                features[idx*4 + 2] = lm.z
                features[idx*4 + 3] = lm.visibility
        
        return features
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Tính cosine similarity giữa 2 vectors."""
        # Chỉ so sánh các điểm có visibility > 0
        mask = (vec1[3::4] > 0.5) & (vec2[3::4] > 0.5)  # visibility at every 4th position
        
        if not np.any(mask):
            return 0.0
        
        # Expand mask to cover x, y, z (not just visibility)
        full_mask = np.zeros(132, dtype=bool)
        for i, m in enumerate(mask):
            if m:
                full_mask[i*4:i*4+3] = True  # x, y, z (not visibility)
        
        v1 = vec1[full_mask]
        v2 = vec2[full_mask]
        
        if len(v1) == 0:
            return 0.0
        
        dot = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.0
        
        similarity = dot / (norm1 * norm2)
        return float(similarity)
    
    def calculate_angles(self, landmarks: Dict) -> Dict[str, float]:
        """Tính các góc quan trọng."""
        def angle(a, b, c):
            a, b, c = np.array(a), np.array(b), np.array(c)
            ba, bc = a - b, c - b
            cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
            return np.degrees(np.arccos(np.clip(cos, -1, 1)))
        
        angles = {}
        
        if all(k in landmarks for k in ['left_hip', 'left_knee', 'left_ankle']):
            angles['left_knee'] = angle(
                (landmarks['left_hip'].x, landmarks['left_hip'].y),
                (landmarks['left_knee'].x, landmarks['left_knee'].y),
                (landmarks['left_ankle'].x, landmarks['left_ankle'].y)
            )
        
        if all(k in landmarks for k in ['left_shoulder', 'left_hip', 'left_knee']):
            angles['left_back'] = angle(
                (landmarks['left_shoulder'].x, landmarks['left_shoulder'].y),
                (landmarks['left_hip'].x, landmarks['left_hip'].y),
                (landmarks['left_knee'].x, landmarks['left_knee'].y)
            )
        
        return angles
    
    def score(self, landmarks: Dict, pose_label: str) -> ScoringResult:
        """
        Score the pose form.
        
        Args:
            landmarks: Dict keypoints from PoseDetector
            pose_label: Label from classifier (e.g., "natarajasana")
            
        Returns:
            ScoringResult with score, feedback, and details
        """
        # Keep original label for display
        original_label = pose_label
        
        # Preprocess
        current_keypoints = self.preprocess_keypoints(landmarks)
        
        # Map unknown poses to closest known reference for scoring
        # But keep original label for display
        scoring_label = pose_label
        if pose_label not in self.reference_poses:
            # Map yoga poses to closest reference
            yoga_mapping = {
                'natarajasana': 'tree_pose',  # Dancer -> Tree (single leg)
                'vrksasana': 'tree_pose',
                'virabhadrasana': 'warrior_i',
                'trikonasana': 'warrior_i',  # Triangle -> Warrior
                'utkatasana': 'squat',  # Chair pose -> Squat
                'chaturanga': 'plank',
                'adho_mukha': 'plank',  # Downward dog -> Plank
            }
            scoring_label = yoga_mapping.get(pose_label, 'tree_pose')  # Default to tree for yoga
        
        reference = self.reference_poses.get(scoring_label)
        
        if reference is None:
            return ScoringResult(
                pose_label=original_label,  # Use original label
                score=75.0,  # Give neutral score for unknown poses
                is_good_form=True,
                feedback=f"Detected: {original_label.replace('_', ' ').title()}",
                details={},
                rep_count=self.rep_count
            )
        
        # Calculate cosine similarity
        similarity = self.cosine_similarity(current_keypoints, reference)
        
        # Convert to 0-100 score
        score = (similarity + 1) / 2 * 100
        score = max(0, min(100, score))
        
        # Calculate angles
        angles = self.calculate_angles(landmarks)
        
        # Evaluate form
        is_good = score >= self.GOOD_FORM_THRESHOLD
        
        # Generate feedback with ORIGINAL label (not fallback)
        feedback = self._generate_feedback(original_label, score, angles)
        
        # Rep counting
        self._update_rep_count(original_label, score)
        
        return ScoringResult(
            pose_label=original_label,  # Use original label
            score=round(score, 1),
            is_good_form=is_good,
            feedback=feedback,
            details={
                'angles': angles,
                'similarity': round(similarity, 3)
            },
            rep_count=self.rep_count
        )
    
    def _generate_feedback(self, pose_label: str, score: float, angles: Dict) -> str:
        """Generate pose-specific feedback."""
        pose_name = pose_label.replace('_', ' ').title()
        
        if score >= 90:
            return f"Excellent {pose_name}!"
        elif score >= 80:
            return f"Good {pose_name}! Hold steady"
        elif score >= 70:
            return f"Nice {pose_name}. Score: {score:.0f}%"
        elif score >= 50:
            # Pose-specific corrections
            if pose_label == 'tree_pose':
                return "Focus on balance, steady your gaze"
            elif pose_label == 'warrior_i':
                return "Bend front knee more, arms up"
            elif pose_label == 'squat':
                if 'left_knee' in angles and angles['left_knee'] > 110:
                    return "Bend knees deeper!"
                if 'left_back' in angles and angles['left_back'] < 100:
                    return "Keep back straight!"
            elif pose_label == 'lunge':
                return "Lower your hips, back knee down"
            elif pose_label == 'plank':
                return "Keep body straight, core tight"
            return f"Keep practicing. Score: {score:.0f}%"
        else:
            return f"Adjust your {pose_name} form"
    
    def _update_rep_count(self, pose_label: str, score: float):
        """Cập nhật số reps."""
        # Đếm rep khi chuyển từ pose sang non-pose
        is_in_pose = score >= 60  # Threshold để coi là đang trong pose
        
        if pose_label not in self.was_in_pose:
            self.was_in_pose[pose_label] = False
        
        if self.was_in_pose[pose_label] and not is_in_pose:
            self.rep_count += 1
        
        self.was_in_pose[pose_label] = is_in_pose
        self.prev_pose = pose_label
    
    def reset(self):
        """Reset state."""
        self.rep_count = 0
        self.prev_pose = None
        self.was_in_pose = {}


# Test module
if __name__ == "__main__":
    print("Testing FormScorer...")
    
    scorer = FormScorer()
    
    # Mock landmarks (squat pose)
    from collections import namedtuple
    Landmark = namedtuple('Landmark', ['x', 'y', 'z', 'visibility'])
    
    test_landmarks = {
        'left_hip': Landmark(0.45, 0.50, 0, 0.9),
        'left_knee': Landmark(0.42, 0.70, 0, 0.9),
        'left_ankle': Landmark(0.40, 0.90, 0, 0.9),
        'left_shoulder': Landmark(0.45, 0.25, 0, 0.9),
        'right_hip': Landmark(0.55, 0.50, 0, 0.9),
        'right_knee': Landmark(0.58, 0.70, 0, 0.9),
        'right_ankle': Landmark(0.60, 0.90, 0, 0.9),
        'right_shoulder': Landmark(0.55, 0.25, 0, 0.9),
    }
    
    result = scorer.score(test_landmarks, "squat")
    print(f"Pose: {result.pose_label}")
    print(f"Score: {result.score}%")
    print(f"Good form: {result.is_good_form}")
    print(f"Feedback: {result.feedback}")
    print(f"Details: {result.details}")
    
    print("\n✓ Test completed!")
