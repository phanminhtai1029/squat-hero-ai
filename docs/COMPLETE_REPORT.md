# 📋 Báo Cáo Hoàn Chỉnh: Rule-Based Yoga Pose Recognition

**Ngày thực hiện**: 2026-01-31  
**Repository**: https://github.com/phanminhtai1029/squat-hero-ai  
**Branch**: `feature/rule-based-mvp`

---

## 1. Tổng Quan Dự Án

### 1.1 Mục Tiêu
Xây dựng hệ thống nhận diện tư thế yoga sử dụng **rule-based approach** (không training ML) để đánh giá khả năng và giới hạn của phương pháp này.

### 1.2 Pipeline 5 Bước

```
Input (Webcam/Video/Image)
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ Step 1: Frame Capture (OpenCV)                                   │
│ Step 2: Person Detection (YOLOv8 - pretrained)                   │
│ Step 3: Pose Estimation (MediaPipe - pretrained)                 │
│ Step 4: Frame Classification (Rule-based: velocity threshold)    │
│ Step 5: Pose Matching (Rule-based: similarity metrics)           │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
Output: Pose Name + Similarity Score
```

---

## 2. Công Việc Đã Thực Hiện

### 2.1 Dataset
- Tải về **Yoga_Poses-Dataset** từ GitHub (484 ảnh, 8 poses)
- Poses: Downward Dog, Triangle, Warrior, Tree, Dancer, Half Moon, Goddess, Bound Angle

### 2.2 Refactor Codebase
**Trước refactor:**
```
step1_frame_capture/
step2_person_cropping/
step3_pose_detection/
step4_pose_comparison/
venv/ (7.3GB duplicate)
```

**Sau refactor:**
```
pipeline/
├── step1_frame_capture.py
├── step2_person_detection.py
├── step3_pose_estimation.py
├── step4_frame_classifier.py
└── step5_pose_matcher.py
```
→ Tiết kiệm ~7.3GB, code sạch hơn

### 2.3 Implement Rule-Based Pipeline

| Step | Algorithm/Model | Mô tả |
|------|----------------|-------|
| Step 1 | OpenCV VideoCapture | Đọc frames từ camera/video |
| Step 2 | YOLOv8n (pretrained) | Detect person bounding box |
| Step 3 | MediaPipe Pose (pretrained) | Extract 33 body landmarks |
| Step 4 | Velocity Threshold | Classify KEY_POSE vs TRANSITION |
| Step 5 | Similarity Metrics | Match angles với database |

### 2.4 Pose Database
Tạo `pose_database.yaml` với 8 poses:
- 8 joint angles per pose
- Weights (importance) per angle
- Tolerance ranges (min-max acceptable)

---

## 3. Thử Nghiệm & Kết Quả

### 3.1 Các Phương Pháp Matching Đã Test

| Method | Công thức | Top-1 Accuracy |
|--------|-----------|----------------|
| **Cosine Similarity** | `cos(A,B) = A·B / (||A||×||B||)` | **43.2%** ⭐ |
| Euclidean Similarity | `exp(-||A-B|| × 2)` | 33.9% |
| Weighted Euclidean | `exp(-||w×(A-B)|| × 3)` | 34.7% |
| Combined | `0.4×euc + 0.3×weight + 0.3×tol` | 31.4% |

### 3.2 Kết Quả Chi Tiết Per-Pose

| Pose | Cosine | Euclidean | Weighted | Combined |
|------|--------|-----------|----------|----------|
| baddha_konasana | 100% | 100% | 100% | 100% |
| downward_dog | 93.3% | 100% | 100% | 100% |
| triangle | 93.3% | 6.7% | 6.7% | 0% |
| veerabhadrasana | 33.3% | 26.7% | 6.7% | 0% |
| utkata_konasana | 0% | 26.7% | 26.7% | 6.7% |
| ardha_chandrasana | 7.7% | 0% | 15.4% | 38.5% |
| vrukshasana | 13.3% | 6.7% | 6.7% | 6.7% |
| natarajasana | 0% | 0% | 13.3% | 0% |

### 3.3 Performance Metrics (từ evaluation trước đó)

| Metric | Giá trị |
|--------|---------|
| Detection Rate | 99.4% |
| Avg Latency | 37ms |
| FPS | 27 |

---

## 4. Phân Tích Giới Hạn Rule-Based

### 4.1 Tại Sao Accuracy Thấp?

1. **8 góc không đủ phân biệt**: Nhiều poses có góc tương tự
2. **Thiếu thông tin vị trí**: Góc giống nhau nhưng vị trí khác
3. **Biến thể trong dataset**: Ảnh thực tế khác "pose chuẩn"
4. **Mirror confusion**: Left/right không được xử lý

### 4.2 Các Cải Thiện Đã Thử

| Strategy | Kết quả |
|----------|---------|
| Euclidean thay Cosine | ❌ Giảm accuracy |
| Weighted angles | ❌ Không cải thiện đáng kể |
| Angle tolerances | ❌ Không cải thiện đáng kể |
| Combined method | ❌ Giảm accuracy |

### 4.3 Các Cải Thiện Không Khả Thi

| Approach | Tại sao không scale |
|----------|---------------------|
| Position features | Phải tune cho từng pose → O(N) effort |
| Mirror handling | 2× tính toán cho mọi frame |
| Category-first | Category có thể rất lớn, boundary không rõ |
| Multi-reference | N poses × M refs = O(N×M) complexity |

---

## 5. Kết Luận

### 5.1 Rule-Based Đạt Giới Hạn

| Metric | Giá trị |
|--------|---------|
| Best accuracy | **43.2%** (Cosine) |
| Theoretical max (với full tuning) | ~70-80% |
| Scalability | ❌ Không scale được |

### 5.2 Vấn Đề Cốt Lõi

> **Rule-based không có khả năng "học"** để phân biệt các poses tương tự. Mọi cải thiện đều yêu cầu manual tuning, không scalable với nhiều poses.

### 5.3 Khuyến Nghị

Để đạt accuracy >80% và scale với hàng trăm poses, cần chuyển sang **ML-based approach**:
- Learned pose embeddings
- Neural network classifier
- Metric learning

---

## 6. Files Đã Tạo/Sửa

| File | Mô tả |
|------|-------|
| `pipeline/__init__.py` | Package exports |
| `pipeline/step1_frame_capture.py` | Frame capture module |
| `pipeline/step2_person_detection.py` | YOLO person detection |
| `pipeline/step3_pose_estimation.py` | MediaPipe pose estimation |
| `pipeline/step4_frame_classifier.py` | Velocity-based classifier |
| `pipeline/step5_pose_matcher.py` | Multi-method pose matcher |
| `utils/angle_calculator.py` | Joint angle computation |
| `data/pose_database.yaml` | 8 poses với weights & tolerances |
| `evaluation/evaluate_pipeline.py` | Full evaluation script |
| `evaluation/evaluation_report.json` | Kết quả đánh giá |
| `evaluation/comparison_results.json` | So sánh các methods |
| `docs/RULE_BASED_REVIEW.md` | Review chi tiết |
| `docs/FILE_STRUCTURE.md` | Cấu trúc project |
| `docs/EVALUATION_REPORT.md` | Báo cáo đánh giá |
| `main.py` | Entry point (refactored) |

---

## 7. Hướng Đi Tiếp Theo

Pending user direction - likely ML-based approach for improved accuracy and scalability.

---

**End of Report**
