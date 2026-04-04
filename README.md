# 🏋️ Squat Hero AI

<p align="center">
  <strong>Real-time AI-Powered Squat Form Analysis & Rep Counter</strong>
</p>


<p align="center">
  <a href="#features">Features</a> •
  <a href="#demo">Demo</a> •
  <a href="#installation">Installation</a> •
  <a href="#usage">Usage</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#configuration">Configuration</a>
</p>

---

## 📖 Overview

**Squat Hero AI** is a real-time computer vision application that uses AI to analyze your squat form, count repetitions, and provide instant feedback. Built with Python, it leverages:

- **YOLOv8** for person detection
- **MediaPipe Pose** for body keypoint estimation
- **OpenCV** for real-time video processing

Perfect for fitness enthusiasts, personal trainers, or anyone looking to improve their squat technique!

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎯 **Real-time Analysis** | Instant feedback on your squat form via webcam |
| 📊 **Rep Counter** | Automatic counting of completed squats |
| 📐 **Angle Detection** | Measures knee and back angles for form assessment |
| ⚠️ **Form Correction** | Detects common form errors (back rounding, insufficient depth, etc.) |
| 🎬 **Benchmark Mode** | Analyze recorded videos for detailed performance metrics |
| 📈 **JSON Reports** | Export benchmark results for tracking progress |

---

## 🔄 Pipeline Architecture

The system uses a **4-step modular pipeline**:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         SQUAT HERO AI PIPELINE                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌───────────────┐   ┌───────────────┐   ┌───────────────┐   ┌─────────────────┐
│  │   STEP 1      │   │   STEP 2      │   │   STEP 3      │   │    STEP 4       │
│  │ Frame Capture │──▶│Person Cropping│──▶│Pose Detection │──▶│Pose Comparison  │
│  │   (Webcam)    │   │   (YOLOv8)    │   │  (MediaPipe)  │   │   (Analysis)    │
│  └───────────────┘   └───────────────┘   └───────────────┘   └─────────────────┘
│         │                   │                   │                     │
│         ▼                   ▼                   ▼                     ▼
│      Raw Frame         Bounding Box       33 Keypoints         Rep Count +
│                        + Cropped Area     (Body Landmarks)     Form Feedback
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

### Module Breakdown

| Step | Module | Technology | Purpose |
|------|--------|------------|---------|
| 1 | `step1_frame_capture/` | OpenCV | Capture frames from webcam or video file |
| 2 | `step2_person_cropping/` | YOLOv8 | Detect and isolate person in frame |
| 3 | `step3_pose_detection/` | MediaPipe | Extract 33 body keypoints |
| 4 | `step4_pose_comparison/` | NumPy | Analyze form, count reps, provide feedback |

---

## 🚀 Installation

### Prerequisites

- Python 3.8+
- Webcam (for real-time mode)
- pip (Python package manager)

### Step 1: Clone the Repository

```bash
git clone https://github.com/phanminhtai1029/squat-hero-ai.git
cd squat-hero-ai
```

### Step 2: Create Virtual Environment (Recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- `opencv-python>=4.8.0` - Computer vision library
- `mediapipe>=0.10.0` - Pose estimation
- `numpy>=1.24.0` - Numerical computing
- `ultralytics>=8.0.0` - YOLOv8 implementation

> **Note:** The YOLOv8 model (`yolov8n.pt`) is included in the repository for convenience.

---

## 🎮 Usage

### Real-time Mode (Webcam)

Launch the application with your webcam:

```bash
python main.py
```

**Controls:**
| Key | Action |
|-----|--------|
| `Q` | Quit application |
| `R` | Reset rep counter |

### Benchmark Mode (Video Analysis)

Analyze a recorded video for detailed metrics:

```bash
# Basic usage
python benchmark.py --video path/to/your/video.mp4

# Save output video with overlays
python benchmark.py --video video.mp4 --save-output

# Export results to JSON
python benchmark.py --video video.mp4 --save-json results.json

# Run without preview window
python benchmark.py --video video.mp4 --no-preview
```

**Benchmark Options:**
| Option | Description |
|--------|-------------|
| `--video`, `-v` | Path to input video (required) |
| `--save-output`, `-s` | Save processed video with overlays |
| `--output`, `-o` | Custom output video path |
| `--save-json`, `-j` | Export metrics to JSON file |
| `--no-preview` | Disable preview window |

---

## 🔧 Configuration

All configurable parameters are in `config.py`:

```python
# Camera settings
CAMERA_ID = 0              # Camera device ID
TARGET_FPS = 30            # Target frame rate

# YOLO settings
YOLO_MODEL = "yolov8n.pt"  # Model path
YOLO_CONFIDENCE = 0.5      # Detection threshold

# Pose detection settings
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# Squat thresholds (degrees)
STANDING_ANGLE_THRESHOLD = 160  # > 160° = standing position
SQUAT_ANGLE_THRESHOLD = 90      # < 90° = full squat depth

# Display settings
WINDOW_NAME = "Squat AI - Real-time Analysis"
```

---

## 📊 Squat Phase Detection

The system identifies 4 phases of a squat:

```
STANDING ──────▶ GOING_DOWN ──────▶ SQUAT ──────▶ GOING_UP ──────▶ STANDING
  (>160°)          (90°-160°)        (<90°)        (90°-160°)        (>160°)
                                                                      [+1 rep]
```

| Phase | Knee Angle | Description |
|-------|------------|-------------|
| `STANDING` | > 160° | Upright position, ready to squat |
| `GOING_DOWN` | 90° - 160° | Descending into squat |
| `SQUAT` | < 90° | Full squat depth achieved |
| `GOING_UP` | 90° - 160° | Ascending back to standing |

---

## ⚠️ Form Error Detection

The system checks for common squat form errors:

| Error | Detection Method | Feedback |
|-------|-----------------|----------|
| **Not Deep Enough** | Knee angle > threshold when standing up | "⚠️ Squat sâu hơn!" |
| **Back Rounding** | Back angle < 100° during squat | "⚠️ Thẳng lưng lên!" |

---

## 📁 Project Structure

```
squat-hero-ai/
├── main.py                      # 🚀 Main application entry point
├── benchmark.py                 # 📊 Video benchmark runner
├── config.py                    # ⚙️  Configuration settings
├── requirements.txt             # 📦 Python dependencies
├── yolov8n.pt                   # 🤖 Pre-trained YOLO model
│
├── step1_frame_capture/         # 📸 Frame capture module
│   ├── __init__.py
│   ├── webcam_capture.py        #    Real-time webcam capture
│   └── video_extractor.py       #    Video file frame extraction
│
├── step2_person_cropping/       # 👤 Person detection module
│   ├── __init__.py
│   └── yolo_cropper.py          #    YOLOv8-based person detection
│
├── step3_pose_detection/        # 🦴 Pose estimation module
│   ├── __init__.py
│   └── pose_detector.py         #    MediaPipe pose detection
│
├── step4_pose_comparison/       # 📐 Pose analysis module
│   ├── __init__.py
│   └── pose_comparator.py       #    Form analysis & rep counting
│
└── utils/                       # 🛠️  Utility functions
    ├── __init__.py
    └── visualization.py         #    UI overlay drawing
```

---

## 🔍 Key Classes

### `SquatAIPipeline` (main.py)
The main orchestrator that connects all 4 pipeline steps.

### `WebcamCapture` (step1_frame_capture/webcam_capture.py)
Handles real-time webcam frame capture with configurable resolution.

### `YoloCropper` (step2_person_cropping/yolo_cropper.py)
Uses YOLOv8 to detect and crop the person from each frame.

### `PoseDetector` (step3_pose_detection/pose_detector.py)
Extracts 33 body keypoints using MediaPipe Pose, focusing on joints relevant for squat analysis:
- Hip (left/right)
- Knee (left/right)
- Ankle (left/right)
- Shoulder (left/right)

### `PoseComparator` (step4_pose_comparison/pose_comparator.py)
Analyzes pose data to:
- Calculate knee and back angles
- Determine squat phase
- Count repetitions
- Detect form errors
- Generate feedback messages

---

## 📈 Benchmark Output Example

```
==================================================
BENCHMARK RESULT
==================================================
Video: workout_session.mp4
Timestamp: 2024-01-15T10:30:45
--------------------------------------------------
METRICS:
  Total Frames: 1800
  Detection Rate: 98.5%
  Total Reps Detected: 15
  Avg FPS: 28.3
  Processing Time: 62.45s
--------------------------------------------------
ANGLE STATS:
  Min Knee Angle: 72.3°
  Max Knee Angle: 178.2°
  Avg Knee Angle: 125.8°
--------------------------------------------------
Form Errors Detected: 3
==================================================
```

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| Camera not detected | Check `CAMERA_ID` in config.py (try 0, 1, 2...) |
| Low FPS | Reduce camera resolution or close other applications |
| Pose not detected | Ensure full body is visible and well-lit |
| YOLO model download error | Manually download `yolov8n.pt` from Ultralytics |

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 👨‍💻 Author

**Phan Minh Tài**
**Trịnh Khải Nguyên**
**Lê Hoàng Hữu**

---

<p align="center">
  Made with ❤️ and 🏋️ for fitness enthusiasts
</p>
