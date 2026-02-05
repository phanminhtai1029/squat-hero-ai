"""
Yoga Pose AI Configuration
==========================

Central configuration file for all pipeline parameters.
"""

# =============================================================================
# Camera Settings
# =============================================================================
CAMERA_ID = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
TARGET_FPS = 30

# =============================================================================
# YOLOv8-Pose Settings (replaces YOLO person detection + MediaPipe)
# =============================================================================
YOLOV8_POSE_MODEL = "yolov8s-pose.pt"  # Options: yolov8n-pose.pt, yolov8s-pose.pt, yolov8m-pose.pt
YOLOV8_CONFIDENCE = 0.5  # Detection confidence threshold
YOLOV8_DEVICE = None  # None=auto-detect, "cuda" or "cpu"

# Legacy YOLO settings (deprecated, kept for backward compatibility)
YOLO_MODEL_PATH = "yolov8n.pt"
YOLO_CONFIDENCE = 0.5
PERSON_CLASS_ID = 0

# Legacy MediaPipe settings (deprecated)
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
MODEL_COMPLEXITY = 1

# =============================================================================
# Frame Classifier Settings
# =============================================================================
VELOCITY_THRESHOLD = 0.015  # Movement threshold for KEY_POSE detection
WINDOW_SIZE = 5             # Frames to consider for velocity
STABILITY_FRAMES = 3        # Consecutive stable frames required

# =============================================================================
# Pose Matcher Settings
# =============================================================================
DATABASE_PATH = "data/pose_database.yaml"
MATCH_METHOD = "euclidean"  # Options: "cosine", "euclidean", "weighted", "combined"

# Similarity thresholds
THRESHOLD_HIGH = 0.85
THRESHOLD_MEDIUM = 0.70
THRESHOLD_LOW = 0.55

# =============================================================================
# Display Settings
# =============================================================================
WINDOW_NAME = "Yoga Pose AI - Real-time Analysis"
FONT_SCALE = 0.7

# Colors (BGR format)
COLOR_GREEN = (0, 255, 0)
COLOR_ORANGE = (0, 165, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
