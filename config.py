# Pose AI Configuration
# =====================

# Camera settings
CAMERA_ID = 0
TARGET_FPS = 30

# YOLO settings (Step 2)
YOLO_MODEL = "yolov8n.pt"
YOLO_CONFIDENCE = 0.5
PERSON_CLASS_ID = 0  # COCO class for person

# Pose detection settings (Step 3 - MediaPipe)
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# Pose Classifier settings (Step 4)
POSE_CLASSIFIER_MODEL = "step4_pose_classifier/models/pose_classifier.pth"
POSE_CLASSES = ["squat", "lunge", "plank", "warrior_i", "tree_pose"]

# Form Scorer settings (Step 5)
GOOD_FORM_THRESHOLD = 70.0  # Score >= 70 = good form
REFERENCE_POSES_DIR = "step5_form_scorer/reference_poses"

# Legacy settings (backward compatibility)
STANDING_ANGLE_THRESHOLD = 160  # > 160 = standing
SQUAT_ANGLE_THRESHOLD = 90      # < 90 = full squat

# Display settings
WINDOW_NAME = "Pose AI - Real-time Analysis"
FONT_SCALE = 1.0

# Pipeline mode
USE_AI_PIPELINE = True  # True = new 5-step AI pipeline, False = legacy 4-step
