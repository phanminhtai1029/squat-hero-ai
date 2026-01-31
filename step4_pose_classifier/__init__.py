"""
Step 4: Pose Classifier
========================
Phân loại động tác từ keypoints sử dụng MLP Neural Network.

Hỗ trợ 5 động tác:
- Squat
- Lunge
- Plank
- Warrior I (Virabhadrasana I)
- Tree Pose (Vrksasana)
"""

from .classifier import PoseClassifier, POSE_CLASSES

__all__ = ['PoseClassifier', 'POSE_CLASSES']
