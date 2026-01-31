"""
Yoga Pose Recognition Pipeline

5-Step Pipeline:
1. Frame Capture - Capture frames from webcam/video
2. Person Detection - Detect person using YOLO
3. Pose Estimation - Estimate pose using MediaPipe
4. Frame Classification - Classify KEY_POSE vs TRANSITION
5. Pose Matching - Match pose against database
"""

from .step1_frame_capture import FrameCapture, VideoCapture, WebcamCapture
from .step2_person_detection import PersonDetector
from .step3_pose_estimation import PoseEstimator
from .step4_frame_classifier import FrameClassifier
from .step5_pose_matcher import PoseMatcher

__all__ = [
    'FrameCapture',
    'VideoCapture', 
    'WebcamCapture',
    'PersonDetector',
    'PoseEstimator',
    'FrameClassifier',
    'PoseMatcher'
]
