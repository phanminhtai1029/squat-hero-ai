# 📊 BÁO CÁO ĐÁNH GIÁ MÔ HÌNH PHÂN LOẠI TƯ THẾ YOGA

---

## 1. TỔNG QUAN DỰ ÁN

### 1.1 Mục tiêu

Xây dựng hệ thống AI phân loại tư thế yoga real-time sử dụng camera, hỗ trợ người dùng nhận dạng và đánh giá form khi tập yoga.

### 1.2 Kiến trúc Pipeline

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frame Capture │───▶│ Person Cropping │───▶│ Pose Detection  │
│    (OpenCV)     │    │    (YOLOv8)     │    │  (MediaPipe)    │
└─────────────────┘    └─────────────────┘    └────────┬────────┘
                                                       │
                                                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Form Scorer   │◀───│ Pose Classifier │◀───│Frame Classifier │
│(Cosine Similar.)│    │   (MLP - AI)    │    │(Angular Velocity)│
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

| Bước | Component | Công nghệ | Vai trò |
|------|-----------|-----------|---------|
| 1 | Frame Capture | OpenCV | Lấy frame từ webcam/video |
| 2 | Person Cropping | YOLOv8 | Crop vùng chứa người |
| 3 | Pose Detection | MediaPipe BlazePose | Trích xuất 33 keypoints |
| 4 | Frame Classification | Angular Velocity | Phân biệt pose vs chuyển động |
| 5 | Pose Classifier | MLP Neural Network | Nhận dạng loại tư thế |
| 6 | Form Scorer | Cosine Similarity | Đánh giá form đúng/sai |

---

## 2. DỮ LIỆU HUẤN LUYỆN

### 2.1 Dataset: Yoga-82

| Thông số | Giá trị |
|----------|---------|
| **Tên dataset** | Yoga-82 |
| **Nguồn** | Kaggle |
| **Tổng số mẫu** | 5,593 |
| **Số lượng classes** | 107 tư thế yoga |
| **Phân chia Train/Test** | 80% / 20% (4,474 / 1,119) |
| **Định dạng input** | Vector 132 chiều (33 keypoints × 4 thuộc tính) |

### 2.2 Top 10 Classes theo số lượng mẫu

| # | Tên tư thế | Số mẫu |
|---|-----------|--------|
| 1 | chaturanga dandasana | 88 |
| 2 | ardha matsyendrasana | 86 |
| 3 | bitilasana | 84 |
| 4 | ustrasana | 75 |
| 5 | garudasana | 74 |
| 6 | bakasana | 74 |
| 7 | vasisthasana | 73 |
| 8 | bhujangasana | 70 |
| 9 | supta baddha konasana | 70 |
| 10 | gomukhasana | 70 |

---

## 3. MÔ HÌNH MLP

### 3.1 Kiến trúc Neural Network

```
Input Layer (132 neurons)
    ↓
Dense Layer (256 neurons) + ReLU + BatchNorm + Dropout(0.3)
    ↓
Dense Layer (128 neurons) + ReLU + BatchNorm + Dropout(0.3)
    ↓
Dense Layer (64 neurons) + ReLU
    ↓
Output Layer (107 neurons) + Softmax
```

### 3.2 Thông số huấn luyện

| Thông số | Giá trị |
|----------|---------|
| **Optimizer** | Adam |
| **Learning Rate** | 0.001 |
| **Loss Function** | CrossEntropyLoss |
| **Batch Size** | 32 |
| **Epochs** | 100 |
| **Dropout Rate** | 0.3 |

### 3.3 Framework & Thư viện

| Thư viện | Phiên bản | Vai trò |
|----------|-----------|---------|
| PyTorch | 2.x | Framework deep learning |
| MediaPipe | 0.10.x | Trích xuất keypoints |
| OpenCV | 4.x | Xử lý hình ảnh |
| scikit-learn | 1.x | Tiền xử lý & đánh giá |
| NumPy | 1.x | Tính toán số học |
| Pandas | 2.x | Xử lý dữ liệu |

---

## 4. KẾT QUẢ ĐÁNH GIÁ

### 4.1 Tổng quan các chỉ số

| Chỉ số | Macro Average | Weighted Average |
|--------|---------------|------------------|
| **Accuracy** | **73.37%** | **73.37%** |
| **Precision** | 72.35% | 73.48% |
| **Recall** | 71.58% | 73.37% |
| **F1-Score** | 70.85% | 72.34% |

> **Giải thích:**
>
> - **Accuracy**: Tỷ lệ dự đoán đúng trên tổng số mẫu
> - **Precision**: Trong số các mẫu dự đoán là class X, bao nhiêu % thực sự là class X
> - **Recall**: Trong số các mẫu thực sự là class X, bao nhiêu % được dự đoán đúng
> - **F1-Score**: Trung bình điều hòa của Precision và Recall

### 4.2 Tiến trình huấn luyện

| Epoch | Train Loss | Val Loss | Val Accuracy |
|-------|------------|----------|--------------|
| 1 | 4.1709 | 3.3847 | 19.84% |
| 10 | 1.4071 | 1.2856 | 64.52% |
| 20 | 1.0969 | 1.1475 | 68.90% |
| 30 | 0.9466 | 1.0681 | 71.94% |
| 40 | 0.8688 | 1.0724 | 72.21% |
| 50 | 0.7944 | 1.0353 | 72.21% |
| 60 | 0.7609 | 1.0195 | 73.64% |
| 70 | 0.7215 | 1.0191 | **74.08%** |
| 80 | 0.6917 | 1.0375 | 73.91% |
| 90 | 0.7078 | 1.0215 | 74.08% |
| 100 | 0.6678 | 1.0209 | 73.37% |

**Best Validation Accuracy: 74.53%** (lưu model)

### 4.3 Top 10 Classes có độ chính xác cao nhất

| # | Tư thế | Precision | Recall | F1-Score | Samples |
|---|--------|-----------|--------|----------|---------|
| 1 | paschimottanasana | 100.0% | 100.0% | **100.0%** | 10 |
| 2 | vriksasana (Tree Pose) | 100.0% | 100.0% | **100.0%** | 12 |
| 3 | virabhadrasana iii (Warrior 3) | 100.0% | 91.7% | 95.7% | 12 |
| 4 | virabhadrasana ii (Warrior 2) | 100.0% | 90.9% | 95.2% | 11 |
| 5 | ardha pincha mayurasana | 90.0% | 100.0% | 94.7% | 9 |
| 6 | hanumanasana (Splits) | 87.5% | 100.0% | 93.3% | 7 |
| 7 | ustrasana (Camel Pose) | 93.3% | 93.3% | 93.3% | 15 |
| 8 | bhujapidasana | 85.7% | 100.0% | 92.3% | 12 |
| 9 | ananda balasana | 84.6% | 100.0% | 91.7% | 11 |
| 10 | dandasana (Staff Pose) | 91.7% | 91.7% | 91.7% | 12 |

### 4.4 Top 10 cặp tư thế hay bị nhầm lẫn

| # | Tư thế thật | Bị nhầm thành | Số lần |
|---|-------------|---------------|--------|
| 1 | virasana | vajrasana | 9 |
| 2 | chakravakasana | bitilasana | 7 |
| 3 | eka pada koundinyanasana i | eka pada koundinyanasana ii | 7 |
| 4 | viparita karani | salamba sarvangasana | 6 |
| 5 | eka pada koundinyanasana ii | eka pada koundinyanasana i | 5 |
| 6 | tulasana | tolasana | 5 |
| 7 | baddha konasana | supta baddha konasana | 4 |
| 8 | bhujangasana | urdhva mukha svanasana | 4 |
| 9 | chakravakasana | marjaryasana | 4 |
| 10 | eka pada rajakapotasana ii | eka pada rajakapotasana | 4 |

> **Nhận xét:** Các tư thế hay bị nhầm lẫn thường có đặc điểm tương tự nhau (cùng họ tư thế hoặc biến thể của nhau).

---

## 5. PHÂN TÍCH & NHẬN XÉT

### 5.1 Điểm mạnh

✅ **Độ chính xác khá với 107 classes**: Đạt 73.37% accuracy với 107 loại tư thế yoga là kết quả khả quan, xét đến độ khó của bài toán multi-class classification.

✅ **Một số tư thế đạt 100% F1**: Các tư thế có đặc điểm riêng biệt như vriksasana (Tree Pose), paschimottanasana được nhận dạng hoàn hảo.

✅ **Real-time processing**: Pipeline có thể chạy real-time với webcam nhờ sử dụng MLP thay vì CNN nặng.

### 5.2 Điểm yếu & Thách thức

⚠️ **Class imbalance**: Một số class có rất ít mẫu (14-30 samples), ảnh hưởng đến chất lượng học.

⚠️ **Tư thế tương tự bị nhầm**: Các biến thể của cùng một tư thế (vd: warrior i, ii, iii) đôi khi bị nhầm lẫn.

⚠️ **Chưa có tư thế không phải yoga**: Model chỉ train với yoga poses, chưa có class "unknown" hoặc "không phải yoga".


---

## 6. HƯỚNG DẪN SỬ DỤNG

### 6.1 Chạy Real-time với Webcam

```bash
cd squat_ai
python main.py
```

**Phím điều khiển:**

- `Q` - Thoát chương trình
- `R` - Reset bộ đếm

### 6.2 Chạy Demo trên Ảnh

```bash
python inference_demo.py --num 5
python inference_demo.py --images path/to/image1.jpg path/to/image2.jpg
```

### 6.3 Huấn luyện lại Model

```bash
# 1. Chuẩn bị dataset
python training/prepare_dataset.py --input data/raw/DATASET --output data/processed/yoga82_dataset.csv

# 2. Huấn luyện
python training/train_classifier.py --data data/processed/yoga82_dataset.csv --epochs 100

# 3. Đánh giá
python training/evaluate_model.py --data data/processed/yoga82_dataset.csv
```

---

## 7. CẤU TRÚC THƯ MỤC DỰ ÁN

```
squat_ai/
├── main.py                          # Entry point chính
├── config.py                        # Cấu hình
├── requirements.txt                 # Dependencies
│
├── step1_frame_capture/             # Bước 1: Capture frame
├── step2_person_cropping/           # Bước 2: YOLO cropping
├── step3_pose_detection/            # Bước 3: MediaPipe pose
├── step4_frame_classification/      # Bước 4: Frame classifier
├── step4_pose_classifier/           # Bước 5: MLP classifier
│   └── models/
│       ├── pose_classifier.pth      # Model weights
│       └── label_encoder.pkl        # Label encoder (107 classes)
├── step5_form_scorer/               # Bước 6: Form scoring
│
├── training/
│   ├── prepare_dataset.py           # Trích xuất keypoints
│   ├── train_classifier.py          # Huấn luyện model
│   └── evaluate_model.py            # Đánh giá model
│
├── data/
│   ├── raw/                         # Ảnh gốc
│   └── processed/                   # Dataset đã xử lý
│       └── yoga82_dataset.csv
│
└── reports/
    ├── evaluation_results.json      # Kết quả đánh giá (JSON)
    └── bao_cao_danh_gia.md          # Báo cáo này
```

---

## 8. KẾT LUẬN

### 8.1 Tóm tắt kết quả

| Tiêu chí | Kết quả |
|----------|---------|
| **Accuracy** | 73.37% |
| **F1-Score (Macro)** | 70.85% |
| **Số tư thế hỗ trợ** | 107 |
| **Thời gian inference** | Real-time |
| **Đánh giá chung** | ✅ Khả quan |

### 8.2 Hướng phát triển

1. **Thu thập thêm dữ liệu** cho các class có ít mẫu
2. **Thêm data augmentation** (rotation, scaling, noise)
3. **Thử nghiệm các kiến trúc khác** (CNN, Transformer)
4. **Thêm tính năng đếm reps** chính xác hơn
5. **Phát triển mobile app** để dễ sử dụng

---
