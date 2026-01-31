"""
Pose Classifier - MLP Neural Network
=====================================
Phân loại động tác từ 33 keypoints (132 features).

Input:  Vector (132,) - 33 keypoints × 4 attributes (x, y, z, visibility)
Output: Pose label + Confidence score
"""

import numpy as np
from typing import Dict, NamedTuple, Optional
from pathlib import Path
import os

# Attempt to import PyTorch, fallback to numpy-based if not available
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not found. Using rule-based fallback classifier.")


# Danh sách 5 động tác hỗ trợ
POSE_CLASSES = [
    "squat",        # 0
    "lunge",        # 1
    "plank",        # 2
    "warrior_i",    # 3 - Virabhadrasana I
    "tree_pose"     # 4 - Vrksasana
]


class ClassificationResult(NamedTuple):
    """Kết quả phân loại động tác."""
    label: str           # Tên động tác
    class_id: int        # ID của class (0-4)
    confidence: float    # Độ tin cậy (0-1)
    all_probs: Dict[str, float]  # Xác suất của tất cả classes


if TORCH_AVAILABLE:
    class MLPModel(nn.Module):
        """MLP model cho pose classification."""
        
        def __init__(self, input_size: int = 132, num_classes: int = 5):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(input_size, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, num_classes)
            )
        
        def forward(self, x):
            return self.model(x)


class PoseClassifier:
    """
    Classifier để phân loại động tác từ keypoints.
    
    Sử dụng MLP neural network nếu có model trained,
    hoặc fallback về rule-based classification.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Khởi tạo PoseClassifier.
        
        Args:
            model_path: Đường dẫn đến file weights (.pth).
                       Nếu None, sẽ dùng rule-based fallback.
        """
        self.model = None
        self.use_neural_network = False
        
        if model_path is None:
            # Tìm trong thư mục mặc định
            current_dir = Path(__file__).parent
            default_path = current_dir / "models" / "pose_classifier.pth"
            if default_path.exists():
                model_path = str(default_path)
        
        if model_path and TORCH_AVAILABLE and os.path.exists(model_path):
            self._load_model(model_path)
        else:
            print("Using rule-based fallback classifier (no trained model found)")
    
    def _load_model(self, model_path: str):
        """Load trained model weights."""
        try:
            # Load label encoder first to get correct classes
            encoder_path = Path(model_path).parent / "label_encoder.pkl"
            if encoder_path.exists():
                import pickle
                with open(encoder_path, 'rb') as f:
                    self.label_encoder = pickle.load(f)
                    self.classes = list(self.label_encoder.classes_)
                    print(f"Loaded classes: {self.classes}")
            else:
                self.classes = POSE_CLASSES
                self.label_encoder = None
            
            num_classes = len(self.classes)
            self.model = MLPModel(num_classes=num_classes)
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
            self.model.eval()
            self.use_neural_network = True
            print(f"Loaded model from {model_path}")
        except Exception as e:
            print(f"Failed to load model: {e}")
            self.model = None
            self.use_neural_network = False
            self.classes = POSE_CLASSES
    
    def preprocess_keypoints(self, landmarks: Dict) -> np.ndarray:
        """
        Chuyển landmarks dict thành vector 132 chiều.
        
        Args:
            landmarks: Dict từ PoseDetector với keys như 'left_hip', 'left_knee'...
            
        Returns:
            numpy array shape (132,)
        """
        # Thứ tự 33 keypoints theo MediaPipe
        KEYPOINT_ORDER = [
            'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
            'right_eye_inner', 'right_eye', 'right_eye_outer',
            'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
            'left_index', 'right_index', 'left_thumb', 'right_thumb',
            'left_hip', 'right_hip', 'left_knee', 'right_knee',
            'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
            'left_foot_index', 'right_foot_index'
        ]
        
        features = []
        
        # Nếu landmarks là dict đơn giản (chỉ có 8 keypoints quan trọng)
        if 'left_hip' in landmarks and len(landmarks) < 33:
            # Dùng các keypoints có sẵn, padding phần còn lại
            important_keys = ['left_shoulder', 'right_shoulder', 
                            'left_hip', 'right_hip',
                            'left_knee', 'right_knee',
                            'left_ankle', 'right_ankle']
            
            for key in important_keys:
                if key in landmarks:
                    lm = landmarks[key]
                    features.extend([lm.x, lm.y, lm.z, lm.visibility])
                else:
                    features.extend([0.0, 0.0, 0.0, 0.0])
            
            # Padding to 132
            while len(features) < 132:
                features.extend([0.0, 0.0, 0.0, 0.0])
        else:
            # Full 33 keypoints
            for key in KEYPOINT_ORDER:
                if key in landmarks:
                    lm = landmarks[key]
                    features.extend([lm.x, lm.y, lm.z, lm.visibility])
                else:
                    features.extend([0.0, 0.0, 0.0, 0.0])
        
        return np.array(features[:132], dtype=np.float32)
    
    def classify(self, landmarks: Dict) -> ClassificationResult:
        """
        Phân loại động tác từ landmarks.
        
        Args:
            landmarks: Dict các landmark từ PoseDetector
            
        Returns:
            ClassificationResult với label, class_id, confidence và all_probs
        """
        keypoints = self.preprocess_keypoints(landmarks)
        
        if self.use_neural_network and self.model is not None:
            return self._classify_with_nn(keypoints)
        else:
            return self._classify_rule_based(landmarks)
    
    def _classify_with_nn(self, keypoints: np.ndarray) -> ClassificationResult:
        """Phân loại sử dụng neural network."""
        with torch.no_grad():
            x = torch.tensor(keypoints).unsqueeze(0)  # (1, 132)
            logits = self.model(x)  # (1, num_classes)
            probs = torch.softmax(logits, dim=1).squeeze().numpy()
        
        class_id = int(np.argmax(probs))
        label = self.classes[class_id]
        confidence = float(probs[class_id])
        all_probs = {self.classes[i]: float(probs[i]) for i in range(len(self.classes))}
        
        return ClassificationResult(
            label=label,
            class_id=class_id,
            confidence=confidence,
            all_probs=all_probs
        )
    
    def _classify_rule_based(self, landmarks: Dict) -> ClassificationResult:
        """
        Fallback: Rule-based classification for yoga poses.
        Used when no trained model is available.
        """
        # Calculate key angles
        left_knee_angle = self._calculate_angle(
            (landmarks['left_hip'].x, landmarks['left_hip'].y),
            (landmarks['left_knee'].x, landmarks['left_knee'].y),
            (landmarks['left_ankle'].x, landmarks['left_ankle'].y)
        )
        
        right_knee_angle = self._calculate_angle(
            (landmarks['right_hip'].x, landmarks['right_hip'].y),
            (landmarks['right_knee'].x, landmarks['right_knee'].y),
            (landmarks['right_ankle'].x, landmarks['right_ankle'].y)
        )
        
        back_angle = self._calculate_angle(
            (landmarks['left_shoulder'].x, landmarks['left_shoulder'].y),
            (landmarks['left_hip'].x, landmarks['left_hip'].y),
            (landmarks['left_knee'].x, landmarks['left_knee'].y)
        )
        
        # Get positions
        left_ankle_x = landmarks['left_ankle'].x
        right_ankle_x = landmarks['right_ankle'].x
        left_knee_x = landmarks['left_knee'].x
        right_knee_x = landmarks['right_knee'].x
        left_knee_y = landmarks['left_knee'].y
        right_knee_y = landmarks['right_knee'].y
        left_ankle_y = landmarks['left_ankle'].y
        right_ankle_y = landmarks['right_ankle'].y
        hip_y = landmarks['left_hip'].y
        
        # Initialize probabilities
        probs = {pose: 0.1 for pose in POSE_CLASSES}
        
        # ========== Tree Pose Detection ==========
        # One leg bent sideways, foot near other knee
        # Check if one ankle is significantly higher and closer to center
        ankle_height_diff = abs(left_ankle_y - right_ankle_y)
        ankle_close_to_knee = (
            (abs(left_ankle_x - right_knee_x) < 0.1 and left_ankle_y < right_knee_y + 0.1) or
            (abs(right_ankle_x - left_knee_x) < 0.1 and right_ankle_y < left_knee_y + 0.1)
        )
        
        if ankle_height_diff > 0.15 or ankle_close_to_knee:
            # One foot is lifted - likely tree pose
            probs['tree_pose'] = 0.75
        
        # ========== Warrior I Detection ==========
        # Front knee bent (< 130), back leg straight (> 150)
        # Legs spread apart
        leg_spread = abs(left_ankle_x - right_ankle_x)
        
        if leg_spread > 0.25:  # Feet spread apart
            if (left_knee_angle < 130 and right_knee_angle > 150) or \
               (right_knee_angle < 130 and left_knee_angle > 150):
                probs['warrior_i'] = 0.75
        
        # ========== Lunge Detection ==========
        # One knee forward, one back, significant Y difference
        knee_y_diff = abs(left_knee_y - right_knee_y)
        if knee_y_diff > 0.12 and leg_spread > 0.15:
            probs['lunge'] = 0.6
        
        # ========== Squat Detection ==========
        # Both knees bent, feet close together
        if left_knee_angle < 120 and right_knee_angle < 120 and leg_spread < 0.3:
            probs['squat'] = 0.8 if min(left_knee_angle, right_knee_angle) < 100 else 0.6
        
        # ========== Plank Detection ==========
        # Body nearly horizontal (back_angle > 150)
        if back_angle > 150 and hip_y > 0.6:  # Hip low, body horizontal
            probs['plank'] = 0.7
        
        # ========== Standing (no special pose) ==========
        # Both legs straight, standing upright
        if left_knee_angle > 160 and right_knee_angle > 160 and leg_spread < 0.2:
            # Standing straight - could be preparing for tree pose
            if ankle_height_diff < 0.05:
                # Both feet on ground, just standing
                probs['tree_pose'] = 0.5  # Might be preparing for tree
        
        # Normalize probabilities
        total = sum(probs.values())
        probs = {k: v/total for k, v in probs.items()}
        
        class_id = max(range(len(POSE_CLASSES)), key=lambda i: probs[POSE_CLASSES[i]])
        label = POSE_CLASSES[class_id]
        confidence = probs[label]
        
        return ClassificationResult(
            label=label,
            class_id=class_id,
            confidence=confidence,
            all_probs=probs
        )
    
    def _calculate_angle(self, a: tuple, b: tuple, c: tuple) -> float:
        """Tính góc tại điểm b."""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        ba = a - b
        bc = c - b
        
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        cosine = np.clip(cosine, -1.0, 1.0)
        
        return np.degrees(np.arccos(cosine))


# Test module
if __name__ == "__main__":
    print("Testing PoseClassifier...")
    
    classifier = PoseClassifier()
    
    # Mock landmarks
    from collections import namedtuple
    Landmark = namedtuple('Landmark', ['x', 'y', 'z', 'visibility'])
    
    test_landmarks = {
        'left_hip': Landmark(0.5, 0.5, 0, 0.9),
        'left_knee': Landmark(0.5, 0.7, 0, 0.9),
        'left_ankle': Landmark(0.5, 0.9, 0, 0.9),
        'left_shoulder': Landmark(0.5, 0.3, 0, 0.9),
        'right_hip': Landmark(0.55, 0.5, 0, 0.9),
        'right_knee': Landmark(0.55, 0.7, 0, 0.9),
        'right_ankle': Landmark(0.55, 0.9, 0, 0.9),
        'right_shoulder': Landmark(0.55, 0.3, 0, 0.9),
    }
    
    result = classifier.classify(test_landmarks)
    print(f"Label: {result.label}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"All probs: {result.all_probs}")
    
    print("\n✓ Test completed!")
