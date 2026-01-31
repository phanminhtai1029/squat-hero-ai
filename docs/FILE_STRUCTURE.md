# 📁 Cấu Trúc File - Squat Hero AI MVP

Generated: 2026-01-31

## Project Structure

```
squat-hero-ai/
├── pipeline/                           # Core 5-step pipeline
│   ├── __init__.py                     # Package exports
│   ├── step1_frame_capture.py          # Video/Webcam/Image capture
│   ├── step2_person_detection.py       # YOLO person detection  
│   ├── step3_pose_estimation.py        # MediaPipe pose landmarks
│   ├── step4_frame_classifier.py       # KEY_POSE vs TRANSITION
│   └── step5_pose_matcher.py           # Cosine similarity matching
│
├── utils/
│   ├── __init__.py
│   ├── angle_calculator.py             # 8 joint angles calculator
│   └── visualization.py                # Drawing utilities
│
├── evaluation/
│   ├── evaluate_pipeline.py            # Full evaluation script
│   └── evaluation_report.json          # Results JSON
│
├── data/
│   ├── pose_database.yaml              # 8 yoga pose references
│   └── Yoga_Poses-Dataset/             # Training dataset (484 images)
│
├── docs/
│   ├── IMPLEMENTATION_PLAN_MVP.md      # MVP plan
│   ├── YOGA_POSE_AI_PIPELINE.md        # Architecture docs
│   ├── FILE_STRUCTURE.md               # This file
│   └── EVALUATION_REPORT.md            # Evaluation analysis
│
├── main.py                             # Entry point (NEW)
├── config.py                           # Configuration
├── requirements.txt                    # Dependencies
├── README.md                           # Project readme
├── yolov8n.pt                          # YOLO weights
└── pose_landmarker.task                # MediaPipe model
```

## Usage

```bash
# Activate virtual environment
source .venv/bin/activate

# Run with webcam
python main.py

# Run on image
python main.py --image path/to/image.jpg

# Run evaluation
python evaluation/evaluate_pipeline.py
```
