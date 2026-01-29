# Squat AI Configuration

# Camera settings
CAMERA_ID = 0
TARGET_FPS = 30

# YOLO settings
YOLO_MODEL = "yolov8n.pt"
YOLO_CONFIDENCE = 0.5
PERSON_CLASS_ID = 0  # COCO class for person

# Pose detection settings
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# Squat thresholds (degrees)
STANDING_ANGLE_THRESHOLD = 160  # > 160 = standing
SQUAT_ANGLE_THRESHOLD = 90      # < 90 = full squat

# Display settings
WINDOW_NAME = "Squat AI - Real-time Analysis"
FONT_SCALE = 1.0
