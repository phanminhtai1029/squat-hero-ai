"""
Pipeline Review & Training Report
===================================
Script tạo báo cáo tổng hợp sau khi train xong.

Usage:
    python generate_report.py
    python generate_report.py --output reports/report_v1.md
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import numpy as np


def get_model_info() -> Dict:
    """Lấy thông tin model đã train."""
    model_path = Path("step4_pose_classifier/models/pose_classifier.pth")
    encoder_path = Path("step4_pose_classifier/models/label_encoder.pkl")
    
    info = {
        "model_exists": model_path.exists(),
        "encoder_exists": encoder_path.exists(),
        "model_size": None,
        "classes": None
    }
    
    if model_path.exists():
        info["model_size"] = f"{model_path.stat().st_size / 1024:.1f} KB"
    
    if encoder_path.exists():
        try:
            import pickle
            with open(encoder_path, 'rb') as f:
                encoder = pickle.load(f)
                info["classes"] = list(encoder.classes_)
        except:
            pass
    
    return info


def get_reference_poses_info() -> Dict:
    """Lấy thông tin reference poses."""
    ref_dir = Path("step5_form_scorer/reference_poses")
    
    info = {
        "directory_exists": ref_dir.exists(),
        "poses": []
    }
    
    if ref_dir.exists():
        for npy_file in ref_dir.glob("*.npy"):
            pose_name = npy_file.stem.replace("_reference", "")
            info["poses"].append(pose_name)
    
    return info


def get_dataset_info() -> Dict:
    """Lấy thông tin dataset."""
    csv_path = Path("data/processed/pose_dataset.csv")
    
    info = {
        "exists": csv_path.exists(),
        "samples": 0,
        "classes": {}
    }
    
    if csv_path.exists():
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            info["samples"] = len(df)
            info["classes"] = df['label'].value_counts().to_dict()
        except:
            pass
    
    return info


def generate_report(output_path: Optional[str] = None) -> str:
    """
    Tạo báo cáo markdown.
    
    Returns:
        Nội dung báo cáo dạng markdown
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    model_info = get_model_info()
    ref_info = get_reference_poses_info()
    dataset_info = get_dataset_info()
    
    report = f"""# 📊 Pose AI Pipeline Report

**Generated:** {timestamp}

---

## 1. Pipeline Overview

| Step | Component | Technology | Status |
|------|-----------|------------|--------|
| 1 | Frame Capture | OpenCV | ✅ Ready |
| 2 | Person Cropping | YOLOv8 | ✅ Ready |
| 3 | Pose Detection | MediaPipe | ✅ Ready |
| 4 | Pose Classifier | MLP (PyTorch) | {'✅ Trained' if model_info['model_exists'] else '⚠️ Not trained'} |
| 5 | Form Scorer | Cosine Similarity | ✅ Ready |

---

## 2. Supported Poses

| # | Pose | Reference |
|---|------|-----------|
"""
    
    poses = model_info.get('classes') or ['squat', 'lunge', 'plank', 'warrior_i', 'tree_pose']
    for i, pose in enumerate(poses, 1):
        ref_status = "✅" if pose in ref_info['poses'] else "❌"
        report += f"| {i} | {pose.replace('_', ' ').title()} | {ref_status} |\n"
    
    report += f"""
---

## 3. Model Information

| Property | Value |
|----------|-------|
| Model file | `step4_pose_classifier/models/pose_classifier.pth` |
| Status | {'✅ Exists' if model_info['model_exists'] else '❌ Not found'} |
| Size | {model_info['model_size'] or 'N/A'} |
| Architecture | MLP (132 → 256 → 128 → 64 → 5) |
| Input | 33 keypoints × 4 = 132 features |
| Output | 5 classes (softmax) |

---

## 4. Dataset Information

| Property | Value |
|----------|-------|
| Dataset file | `data/processed/pose_dataset.csv` |
| Status | {'✅ Exists' if dataset_info['exists'] else '❌ Not found'} |
| Total samples | {dataset_info['samples']} |

"""
    
    if dataset_info['classes']:
        report += "### Class Distribution\n\n"
        report += "| Class | Count |\n|-------|-------|\n"
        for cls, count in dataset_info['classes'].items():
            report += f"| {cls} | {count} |\n"
    
    report += f"""
---

## 5. Reference Poses

| Property | Value |
|----------|-------|
| Directory | `step5_form_scorer/reference_poses/` |
| Total poses | {len(ref_info['poses'])} |

"""
    
    if ref_info['poses']:
        report += "### Available References\n\n"
        for pose in ref_info['poses']:
            report += f"- ✅ `{pose}_reference.npy`\n"
    
    report += """
---

## 6. Training Metrics

> Fill in after training:

| Metric | Value |
|--------|-------|
| Epochs | _ |
| Best Accuracy | _% |
| Train Loss | _ |
| Val Loss | _ |

---

## 7. Evaluation Results

> Fill in after evaluation:

### Confusion Matrix

```
              Predicted
             squat lunge plank warrior tree
Actual squat   _     _     _     _      _
       lunge   _     _     _     _      _
       plank   _     _     _     _      _
     warrior   _     _     _     _      _
        tree   _     _     _     _      _
```

### Per-class Metrics

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| squat | _ | _ | _ |
| lunge | _ | _ | _ |
| plank | _ | _ | _ |
| warrior_i | _ | _ | _ |
| tree_pose | _ | _ | _ |

---

## 8. Usage

```bash
# Run pipeline
python main.py

# Run with legacy mode
python main.py --legacy

# Benchmark on video
python benchmark.py --video path/to/video.mp4
```

---

## 9. Next Steps

- [ ] Collect more training data
- [ ] Fine-tune hyperparameters
- [ ] Add more poses
- [ ] Improve form detection accuracy
- [ ] Deploy to production

---

*Report generated by `generate_report.py`*
"""
    
    # Save if output path provided
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Report saved to: {output_path}")
    
    return report


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate pipeline report")
    parser.add_argument("--output", "-o", default="reports/pipeline_report.md",
                       help="Output file path")
    
    args = parser.parse_args()
    
    print("Generating report...")
    report = generate_report(args.output)
    
    # Also print summary
    print("\n" + "=" * 50)
    print("REPORT SUMMARY")
    print("=" * 50)
    
    model_info = get_model_info()
    print(f"Model trained: {'Yes' if model_info['model_exists'] else 'No'}")
    
    dataset_info = get_dataset_info()
    print(f"Dataset samples: {dataset_info['samples']}")
    
    ref_info = get_reference_poses_info()
    print(f"Reference poses: {len(ref_info['poses'])}")
    
    print("=" * 50)


if __name__ == "__main__":
    main()
