# 📊 Báo Cáo Đánh Giá - Yoga Pose Recognition Pipeline

**Ngày chạy**: 2026-01-31  
**Dataset**: Yoga_Poses-Dataset (GitHub)  
**Số ảnh**: 484 ảnh  

---

## 📈 Tổng Quan Kết Quả

| Metric | Giá trị | Target MVP | Status |
|--------|---------|------------|--------|
| **Top-1 Accuracy** | 27.2% | >80% | ❌ Cần cải thiện |
| **Top-3 Accuracy** | 46.6% | >95% | ❌ Cần cải thiện |
| **Detection Rate** | 99.4% | >95% | ✅ Đạt |
| **Avg Latency** | 37.0ms | <80ms | ✅ Đạt |
| **Avg FPS** | 27.0 | >12 | ✅ Đạt |

---

## 🎯 Độ Chính Xác Theo Pose

| Pose | Accuracy | Correct/Total | Nhận xét |
|------|----------|---------------|----------|
| 🟢 **Downward Dog** | 98.3% | 59/60 | ✅ Xuất sắc |
| 🟢 **Triangle** | 76.7% | 46/60 | ✅ Tốt |
| 🟡 **Tree Pose (Vrukshasana)** | 42.1% | 24/57 | ⚠️ Cần tune |
| 🔴 **Dancer (Natarajasana)** | 1.7% | 1/60 | ❌ Nhầm với Tree |
| 🔴 **Warrior (Veerabhadrasana)** | 1.7% | 1/60 | ❌ Nhầm với Tree |
| 🔴 **Half Moon** | 0.0% | 0/58 | ❌ Nhầm với Tree/Triangle |
| 🔴 **Bound Angle** | 0.0% | 0/60 | ❌ Nhầm với nhiều pose |
| 🔴 **Goddess Pose** | 0.0% | 0/66 | ❌ Nhầm với nhiều pose |

---

## 🔍 Phân Tích Confusion Matrix

### Các nhầm lẫn phổ biến:

1. **Natarajasana → Vrukshasana (52/60)**: Cả hai đều là pose 1 chân
2. **Veerabhadrasana → Vrukshasana (57/60)**: Góc tương tự
3. **UtkataKonasana → Vrukshasana (29/66)**: Góc database cần điều chỉnh
4. **ArdhaChandrasana → Vrukshasana (29/58) + Triangle (24/58)**: Pose nghiêng

### Nhận xét:
- Many poses được nhầm thành **Vrukshasana** (Tree Pose)
- Database angles chưa phản ánh đúng sự khác biệt giữa các pose
- Cần thêm features khác ngoài 8 góc (relative positions, symmetry)

---

## 💡 Khuyến Nghị Cải Thiện

### Immediate (1-2 ngày):
1. **Tune reference angles** - Re-compute từ subset ảnh "chuẩn" nhất
2. **Lower similarity threshold** - Giảm từ 0.92 xuống 0.85
3. **Add more angles** - Thêm góc back spine, neck

### Short-term (1 tuần):
1. **Add relative position features** - Không chỉ góc mà còn vị trí tương đối
2. **Per-pose thresholds** - Mỗi pose có ngưỡng riêng
3. **Train simple classifier** - MLP trên angle + position features

---

## 📁 Files Đã Tạo

| File | Mô tả |
|------|-------|
| `pipeline/step1_frame_capture.py` | Video/Webcam/Image capture |
| `pipeline/step2_person_detection.py` | YOLO person detection |
| `pipeline/step3_pose_estimation.py` | MediaPipe Tasks API |
| `pipeline/step4_frame_classifier.py` | Velocity-based KEY_POSE detection |
| `pipeline/step5_pose_matcher.py` | Cosine similarity matching |
| `utils/angle_calculator.py` | 8 joint angles |
| `data/pose_database.yaml` | 8 yoga pose references |
| `evaluation/evaluate_pipeline.py` | Full evaluation script |
| `evaluation/evaluation_report.json` | JSON results |

---

## 🏆 Kết Luận

**Pipeline HOẠT ĐỘNG** nhưng accuracy cần cải thiện:
- ✅ Performance tốt (27 FPS, <40ms latency)
- ✅ Detection rate cao (99.4%)
- ✅ 2 poses hoạt động xuất sắc (Downward Dog, Triangle)
- ❌ Cần tune database angles cho các pose khác
- ❌ Cần thêm features để phân biệt poses tương tự
