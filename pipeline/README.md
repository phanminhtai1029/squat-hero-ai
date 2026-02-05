# Pipeline Structure

## 4-Step Real-time Yoga Pose Recognition Pipeline

### Active Pipeline (YOLOv8-Pose):

```
┌──────────────────────────────────────────────────────────────┐
│  Step 1: Frame Capture                                       │
│  File: step1_frame_capture.py                                │
│  Input: Webcam/Video/Image                                   │
│  Output: Frame (BGR image)                                   │
└──────────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────────┐
│  Step 2: Pose Estimation (YOLOv8-Pose)                       │
│  File: step3_pose_estimation.py                              │
│  Input: Frame                                                │
│  Output: PoseResult (17 COCO keypoints + bbox)               │
│  Note: Combines person detection + pose estimation           │
│  Performance: ~11.5ms on RTX 3060 (87 FPS)                   │
└──────────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────────┐
│  Step 3: Frame Classification                                │
│  File: step4_frame_classifier.py                             │
│  Input: PoseResult (17 keypoints)                            │
│  Output: KEY_POSE or TRANSITION                              │
│  Algorithm: Velocity-based threshold (rule-based)            │
│  Performance: ~0.01ms                                        │
└──────────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────────┐
│  Step 4: Pose Matching                                       │
│  File: step5_pose_matcher.py                                 │
│  Input: PoseResult (only for KEY_POSE frames)                │
│  Output: Matched pose name + similarity score                │
│  Algorithm: Cosine similarity on 8 joint angles              │
│  Performance: ~0.5ms                                         │
└──────────────────────────────────────────────────────────────┘

Total Pipeline: ~12ms → 83 FPS
```

---

## File Naming vs Pipeline Steps:

**Note:** File names don't match step numbers due to migration from 5-step to 4-step pipeline:

| Pipeline Step | File Name | Status |
|--------------|-----------|--------|
| Step 1: Frame Capture | `step1_frame_capture.py` | ✅ Active |
| Step 2: Pose Estimation | `step3_pose_estimation.py` | ✅ Active (YOLOv8-Pose) |
| Step 3: Frame Classifier | `step4_frame_classifier.py` | ✅ Active |
| Step 4: Pose Matcher | `step5_pose_matcher.py` | ✅ Active |
| ~~Step 2: Person Detection~~ | `step2_person_detection.py` | ⚠️ DEPRECATED |

---

## Deprecated Files:

### `step2_person_detection.py` - DEPRECATED
- **Reason:** YOLOv8-Pose handles person detection + pose estimation together
- **Replaced by:** `step3_pose_estimation.py` (PoseEstimator class)
- **Status:** Kept for backward compatibility with evaluation scripts
- **Do NOT use** for new code

---

## Key Components:

### 1. Frame Capture (`step1_frame_capture.py`)
- `WebcamCapture` - Real-time webcam input
- `VideoCapture` - Video file input
- `ImageCapture` - Single image input

### 2. Pose Estimation (`step3_pose_estimation.py`)
- **Model:** YOLOv8-Small-Pose (yolov8s-pose.pt)
- **Keypoints:** 17 COCO format keypoints
- **Features:**
  - GPU acceleration (CUDA)
  - Person detection + pose in single forward pass
  - Returns normalized coordinates [0, 1]
  - Includes confidence scores per keypoint

### 3. Frame Classifier (`step4_frame_classifier.py`)
- **Algorithm:** Velocity-based threshold
- **Uses:** 13 essential keypoints (excludes eyes/ears)
- **Detects:** Stable pose (KEY_POSE) vs Movement (TRANSITION)
- **Parameters:**
  - `velocity_threshold`: 0.015 (configurable)
  - `window_size`: 5 frames
  - `stability_frames`: 3 consecutive stable frames required

### 4. Pose Matcher (`step5_pose_matcher.py`)
- **Database:** YAML file with reference poses
- **Features:**
  - 8 joint angles extraction
  - Multiple similarity metrics (cosine, euclidean, weighted)
  - Confidence scoring
- **Current Accuracy:** 43% (baseline with rule-based angles)

---

## Migration from MediaPipe to YOLOv8-Pose:

### Before (5-step pipeline):
```
Step 1: Frame Capture → Step 2: Person Detection (YOLO) → 
Step 3: Pose Estimation (MediaPipe) → Step 4: Frame Classifier → 
Step 5: Pose Matcher

Performance: 42.69ms → 23.4 FPS
```

### After (4-step pipeline):
```
Step 1: Frame Capture → Step 2: Pose Estimation (YOLOv8-Pose) → 
Step 3: Frame Classifier → Step 4: Pose Matcher

Performance: 11.92ms → 83.9 FPS (3.6x faster!)
```

---

## Usage:

```python
from pipeline.step3_pose_estimation import PoseEstimator
from pipeline.step4_frame_classifier import FrameClassifier
from pipeline.step5_pose_matcher import PoseMatcher

# Initialize pipeline
pose_estimator = PoseEstimator(model_path="yolov8s-pose.pt")
frame_classifier = FrameClassifier(velocity_threshold=0.015)
pose_matcher = PoseMatcher(database_path="data/pose_database.yaml")

# Process frame
pose_result = pose_estimator.estimate(frame)
classification = frame_classifier.classify_from_pose_result(pose_result)

if classification.frame_type == FrameType.KEY_POSE:
    match_result = pose_matcher.match_from_pose_result(pose_result)
    print(f"Detected: {match_result.display_name}")
```

---

## Future Improvements:

1. **Accuracy Enhancement:**
   - Train ML-based encoder for pose embeddings (replace angle-based matching)
   - Use Yoga-82 dataset (28k images, 82 poses)
   - Target: 85-95% accuracy

2. **Performance:**
   - Already optimized with YOLOv8-Pose GPU acceleration
   - Consider YOLOv8-Nano for edge devices (100+ FPS)

3. **Robustness:**
   - Add data augmentation
   - Handle occlusions better
   - Multi-person support
