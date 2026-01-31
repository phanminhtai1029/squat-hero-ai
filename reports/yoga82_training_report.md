# 📊 Pose AI Pipeline Report

**Generated:** 2026-01-30 20:21:58

---

## 1. Pipeline Overview

| Step | Component | Technology | Status |
|------|-----------|------------|--------|
| 1 | Frame Capture | OpenCV | ✅ Ready |
| 2 | Person Cropping | YOLOv8 | ✅ Ready |
| 3 | Pose Detection | MediaPipe | ✅ Ready |
| 4 | Pose Classifier | MLP (PyTorch) | ✅ Trained |
| 5 | Form Scorer | Cosine Similarity | ✅ Ready |

---

## 2. Supported Poses

| # | Pose | Reference |
|---|------|-----------|
| 1 | Adho Mukha Svanasana | ❌ |
| 2 | Adho Mukha Vriksasana | ❌ |
| 3 | Agnistambhasana | ❌ |
| 4 | Ananda Balasana | ❌ |
| 5 | Anantasana | ❌ |
| 6 | Anjaneyasana | ❌ |
| 7 | Ardha Bhekasana | ❌ |
| 8 | Ardha Chandrasana | ❌ |
| 9 | Ardha Matsyendrasana | ❌ |
| 10 | Ardha Pincha Mayurasana | ❌ |
| 11 | Ardha Uttanasana | ❌ |
| 12 | Ashtanga Namaskara | ❌ |
| 13 | Astavakrasana | ❌ |
| 14 | Baddha Konasana | ❌ |
| 15 | Bakasana | ❌ |
| 16 | Balasana | ❌ |
| 17 | Bhairavasana | ❌ |
| 18 | Bharadvajasana I | ❌ |
| 19 | Bhekasana | ❌ |
| 20 | Bhujangasana | ❌ |
| 21 | Bhujapidasana | ❌ |
| 22 | Bitilasana | ❌ |
| 23 | Camatkarasana | ❌ |
| 24 | Chakravakasana | ❌ |
| 25 | Chaturanga Dandasana | ❌ |
| 26 | Dandasana | ❌ |
| 27 | Dhanurasana | ❌ |
| 28 | Durvasasana | ❌ |
| 29 | Dwi Pada Viparita Dandasana | ❌ |
| 30 | Eka Pada Koundinyanasana I | ❌ |
| 31 | Eka Pada Koundinyanasana Ii | ❌ |
| 32 | Eka Pada Rajakapotasana | ❌ |
| 33 | Eka Pada Rajakapotasana Ii | ❌ |
| 34 | Ganda Bherundasana | ❌ |
| 35 | Garbha Pindasana | ❌ |
| 36 | Garudasana | ❌ |
| 37 | Gomukhasana | ❌ |
| 38 | Halasana | ❌ |
| 39 | Hanumanasana | ❌ |
| 40 | Janu Sirsasana | ❌ |
| 41 | Kapotasana | ❌ |
| 42 | Krounchasana | ❌ |
| 43 | Kurmasana | ❌ |
| 44 | Lolasana | ❌ |
| 45 | Makara Adho Mukha Svanasana | ❌ |
| 46 | Makarasana | ❌ |
| 47 | Malasana | ❌ |
| 48 | Marichyasana I | ❌ |
| 49 | Marichyasana Iii | ❌ |
| 50 | Marjaryasana | ❌ |
| 51 | Matsyasana | ❌ |
| 52 | Mayurasana | ❌ |
| 53 | Natarajasana | ❌ |
| 54 | Padangusthasana | ❌ |
| 55 | Padmasana | ❌ |
| 56 | Parighasana | ❌ |
| 57 | Paripurna Navasana | ❌ |
| 58 | Parivrtta Janu Sirsasana | ❌ |
| 59 | Parivrtta Parsvakonasana | ❌ |
| 60 | Parivrtta Trikonasana | ❌ |
| 61 | Parsva Bakasana | ❌ |
| 62 | Parsvottanasana | ❌ |
| 63 | Pasasana | ❌ |
| 64 | Paschimottanasana | ❌ |
| 65 | Phalakasana | ❌ |
| 66 | Pincha Mayurasana | ❌ |
| 67 | Prasarita Padottanasana | ❌ |
| 68 | Purvottanasana | ❌ |
| 69 | Salabhasana | ❌ |
| 70 | Salamba Bhujangasana | ❌ |
| 71 | Salamba Sarvangasana | ❌ |
| 72 | Salamba Sirsasana | ❌ |
| 73 | Savasana | ❌ |
| 74 | Setu Bandha Sarvangasana | ❌ |
| 75 | Simhasana | ❌ |
| 76 | Sukhasana | ❌ |
| 77 | Supta Baddha Konasana | ❌ |
| 78 | Supta Matsyendrasana | ❌ |
| 79 | Supta Padangusthasana | ❌ |
| 80 | Supta Virasana | ❌ |
| 81 | Tadasana | ❌ |
| 82 | Tittibhasana | ❌ |
| 83 | Tolasana | ❌ |
| 84 | Tulasana | ❌ |
| 85 | Upavistha Konasana | ❌ |
| 86 | Urdhva Dhanurasana | ❌ |
| 87 | Urdhva Hastasana | ❌ |
| 88 | Urdhva Mukha Svanasana | ❌ |
| 89 | Urdhva Prasarita Eka Padasana | ❌ |
| 90 | Ustrasana | ❌ |
| 91 | Utkatasana | ❌ |
| 92 | Uttana Shishosana | ❌ |
| 93 | Uttanasana | ❌ |
| 94 | Utthita Ashwa Sanchalanasana | ❌ |
| 95 | Utthita Hasta Padangustasana | ❌ |
| 96 | Utthita Parsvakonasana | ❌ |
| 97 | Utthita Trikonasana | ❌ |
| 98 | Vajrasana | ❌ |
| 99 | Vasisthasana | ❌ |
| 100 | Viparita Karani | ❌ |
| 101 | Virabhadrasana I | ❌ |
| 102 | Virabhadrasana Ii | ❌ |
| 103 | Virabhadrasana Iii | ❌ |
| 104 | Virasana | ❌ |
| 105 | Vriksasana | ❌ |
| 106 | Vrischikasana | ❌ |
| 107 | Yoganidrasana | ❌ |

---

## 3. Model Information

| Property | Value |
|----------|-------|
| Model file | `step4_pose_classifier/models/pose_classifier.pth` |
| Status | ✅ Exists |
| Size | 333.3 KB |
| Architecture | MLP (132 → 256 → 128 → 64 → 5) |
| Input | 33 keypoints × 4 = 132 features |
| Output | 5 classes (softmax) |

---

## 4. Dataset Information

| Property | Value |
|----------|-------|
| Dataset file | `data/processed/pose_dataset.csv` |
| Status | ✅ Exists |
| Total samples | 1035 |

### Class Distribution

| Class | Count |
|-------|-------|
| plank | 261 |
| warrior2 | 249 |
| downdog | 198 |
| goddess | 172 |
| tree | 155 |

---

## 5. Reference Poses

| Property | Value |
|----------|-------|
| Directory | `step5_form_scorer/reference_poses/` |
| Total poses | 5 |

### Available References

- ✅ `lunge_reference.npy`
- ✅ `plank_reference.npy`
- ✅ `squat_reference.npy`
- ✅ `tree_pose_reference.npy`
- ✅ `warrior_i_reference.npy`

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
