# Báo Cáo Huấn Luyện Mô Hình DNN Trên Bộ Dữ Liệu 3DYoga90

**Nhóm thực hiện:** Squat Hero AI Team
**Tham khảo:** *"3DYoga90: A Hierarchical Video Dataset for Yoga Pose Understanding"* — arXiv:2310.10131

---

## 1. Tóm Tắt (Abstract)

Báo cáo trình bày kết quả huấn luyện mạng nơ-ron sâu (DNN) cho bài toán nhận diện tư thế yoga sử dụng bộ dữ liệu **3DYoga90**. Ba kiến trúc mô hình (DNN-Small, DNN-Base, DNN-Large) được huấn luyện trên **5.526 chuỗi khung xương 3D** với hệ thống phân loại 3 cấp bậc: 6 danh mục (L1), 20 nhóm (L2), và 90 tư thế cụ thể (L3).

Thí nghiệm được tiến hành qua **2 phiên bản**: phiên bản baseline (198-dim features) tái tạo kết quả từ bài báo gốc, và phiên bản nâng cấp (484-dim features) bổ sung góc khớp, khoảng cách tương đối và percentile. Kết quả tốt nhất đạt **95,02% (L1)**, **89,68% (L2)** và **83,04% (L3)**.

---

## 2. Giới Thiệu (Introduction)

### 2.1 Bối cảnh

Nhận diện tư thế yoga từ video là bài toán thị giác máy tính có nhiều ứng dụng thực tế: hỗ trợ tập luyện tại nhà, đánh giá tư thế tự động, và phát hiện sai lệch để phòng tránh chấn thương. Bộ dữ liệu 3DYoga90 cung cấp một benchmark chuẩn với hệ thống phân cấp 3 mức cho 90 tư thế yoga.

### 2.2 Mục tiêu

- Tái tạo thí nghiệm Table 4 trong bài báo gốc (3 mô hình × 3 cấp độ = 9 tổ hợp).
- Nâng cấp đặc trưng bằng góc khớp (joint angles) và khoảng cách tương đối.
- Xây dựng pipeline suy luận (inference) trên video và webcam thời gian thực.
- Đánh giá và so sánh hiệu suất giữa 2 phiên bản feature.

---

## 3. Bộ Dữ Liệu (Dataset)

### 3.1 Tổng quan 3DYoga90

| Thông số | Giá trị |
| --- | --- |
| Tổng chuỗi (sequences) | 5.526 |
| Tập huấn luyện (train) | 4.683 (84,7%) |
| Tập kiểm tra (test) | 843 (15,3%) |
| Tập xác thực (val) | 10% từ train (~468 mẫu) |
| Nguồn skeleton | BlazePose (33 keypoints × 3 tọa độ) |
| Định dạng | Parquet files |

### 3.2 Hệ thống phân cấp 3 mức

```text
Level 1 (6 danh mục)         Level 2 (20 nhóm)              Level 3 (90 tư thế)
├── standing (đứng)      ──→  standing-straight          ──→  mountain, tree, warrior I...
├── sitting (ngồi)       ──→  sitting-legs-front         ──→  cobbler, staff, boat...
├── balancing (thăng bằng)──→  balancing-arm             ──→  crow, firefly...
├── inverted (đảo ngược) ──→  inverted-legs-up           ──→  headstand, shoulderstand...
├── reclining (nằm)      ──→  reclining-face-up          ──→  corpse, bridge...
└── wheel (uốn cong)     ──→  wheel-up-facing            ──→  cobra, upward-dog...
```

---

## 4. Trích Xuất Đặc Trưng (Feature Extraction)

### 4.1 Phiên bản Baseline (V1) — 198 chiều

Mỗi chuỗi video được tóm tắt bằng thống kê đơn giản:

- **Mean** (trung bình) của 33 keypoints × 3 tọa độ = 99 giá trị
- **Std** (độ lệch chuẩn) = 99 giá trị
- **Tổng: 198 chiều**

### 4.2 Phiên bản Nâng Cấp (V2) — 484 chiều

Bổ sung thêm 3 loại đặc trưng mới:

#### A. Landmark Statistics Mở Rộng (396 chiều)

Thay vì chỉ mean + std, thêm **percentile 25%** và **percentile 75%**:

```text
Stats = [mean(99) + std(99) + p25(99) + p75(99)] = 396 chiều
```

→ Capture phân phối tốt hơn, đặc biệt khi tư thế có biến thiên không đối xứng.

#### B. 12 Góc Khớp — Joint Angles (48 chiều)

Tính góc tại 12 khớp quan trọng của cơ thể cho mỗi frame:

| # | Khớp | Điểm A | Điểm trụ | Điểm B |
| --- | --- | --- | --- | --- |
| 1-2 | Khuỷu tay (T/P) | Vai | Khuỷu | Cổ tay |
| 3-4 | Đầu gối (T/P) | Hông | Đầu gối | Mắt cá |
| 5-6 | Vai (T/P) | Khuỷu | Vai | Hông |
| 7-8 | Hông (T/P) | Vai | Hông | Đầu gối |
| 9-10 | Mắt cá (T/P) | Hông | Mắt cá | Đầu gối |
| 11-12 | Cột sống (T/P) | Mũi | Vai | Hông |

Mỗi góc được normalize về [0, 1] bằng cách chia cho π.

```text
Angles = [mean(12) + std(12) + p25(12) + p75(12)] = 48 chiều
```

#### C. 10 Khoảng Cách Tương Đối — Distances (40 chiều)

Khoảng cách Euclidean giữa các cặp keypoint quan trọng:

| # | Cặp | Ý nghĩa |
| --- | --- | --- |
| 1 | Tay trái ↔ Tay phải | Độ dang rộng tay |
| 2 | Chân trái ↔ Chân phải | Độ dang rộng chân |
| 3-4 | Tay ↔ Chân cùng phía | Tư thế gập cơ thể |
| 5-6 | Tay ↔ Chân chéo | Tư thế xoay/vặn |
| 7-8 | Mũi ↔ Hông (T/P) | Độ cúi/ngửa thân |
| 9-10 | Vai ↔ Hông chéo | Xoay thân trên/dưới |

```text
Distances = [mean(10) + std(10) + p25(10) + p75(10)] = 40 chiều
```

#### Tổng kết Feature Vector

```text
V2 Feature = Landmarks(396) + Angles(48) + Distances(40) = 484 chiều
```

---

## 5. Kiến Trúc Mô Hình (Model Architecture)

Ba biến thể DNN sử dụng **Fully Connected layers** với **BatchNorm + ReLU + Dropout(0.4)**:

```text
                    DNN-Small              DNN-Base               DNN-Large
                    ─────────              ─────────              ──────────
Input (484-dim) ──→ Dense(1024)        ──→ Dense(1024)        ──→ Dense(1024)
                    ↓ BN+ReLU+Drop         ↓ BN+ReLU+Drop         ↓ BN+ReLU+Drop
                    Dense(512)             Dense(512)             Dense(512)
                    ↓ BN+ReLU+Drop         ↓ BN+ReLU+Drop         ↓ BN+ReLU+Drop
                    Dense(n_classes)        Dense(256)             Dense(256)
                                           ↓ BN+ReLU+Drop         ↓ BN+ReLU+Drop
                                           Dense(n_classes)        Dense(128)
                                                                   ↓ BN+ReLU+Drop
                                                                   Dense(n_classes)
```

---

## 6. Cấu Hình Huấn Luyện (Training Configuration)

| Siêu tham số | Giá trị |
| --- | --- |
| Optimizer | Adam |
| Learning Rate | 3.33 × 10⁻⁴ |
| LR Scheduler | ReduceLROnPlateau (factor=0.8, patience=5) |
| Batch Size | 256 |
| Max Epochs | 100 |
| Early Stopping | patience=10 epochs |
| Dropout | 0.4 |
| Val Split | 10% từ tập train |
| Loss Function | CrossEntropyLoss |
| Hardware | NVIDIA GPU (CUDA) |

---

## 7. Kết Quả Thực Nghiệm (Experimental Results)

### 7.1 Bảng kết quả chính — So sánh Baseline vs Enhanced

| Model | Feature | L1 (6 cls) | L2 (20 cls) | L3 (90 cls) |
| --- | --- | :---: | :---: | :---: |
| DNN-Small | Baseline (198-dim) | 93,24% | 88,14% | 80,43% |
| DNN-Small | **Enhanced (484-dim)** | **95,02%** | 88,26% | **83,04%** |
| | | *+1,78%* | *+0,12%* | *+2,61%* |
| DNN-Base | Baseline (198-dim) | 93,71% | 88,14% | 81,02% |
| DNN-Base | **Enhanced (484-dim)** | 93,36% | 88,73% | 82,33% |
| | | *-0,35%* | *+0,59%* | *+1,31%* |
| DNN-Large | Baseline (198-dim) | 94,07% | 88,49% | 82,33% |
| DNN-Large | **Enhanced (484-dim)** | 93,12% | **89,68%** | 81,73% |
| | | *-0,95%* | *+1,19%* | *-0,60%* |

### 7.2 Kết quả tốt nhất (Best Overall)

| Cấp độ | Model tốt nhất | Feature | Accuracy |
| --- | --- | --- | :---: |
| **L1** (6 danh mục) | DNN-Small | Enhanced | **95,02%** |
| **L2** (20 nhóm) | DNN-Large | Enhanced | **89,68%** |
| **L3** (90 tư thế) | DNN-Small | Enhanced | **83,04%** |

### 7.3 Phân tích kết quả

1. **DNN-Small hưởng lợi nhiều nhất** từ enhanced features, tăng +2,61% ở L3 (80,43% → 83,04%). Điều này cho thấy feature tốt hơn quan trọng hơn mạng sâu hơn.

2. **DNN-Large cải thiện rõ nhất ở L2** (+1,19%), nhưng giảm nhẹ ở L1 và L3 — dấu hiệu overfitting khi mạng quá sâu với feature đã informative.

3. **Góc khớp đặc biệt hữu ích** cho L3 vì nhiều tư thế yoga chỉ khác nhau ở góc bẻ của tay/chân, thông tin mà mean/std của tọa độ thô không capture được.

4. **Percentile (p25, p75)** giúp mô hình phân biệt tư thế ổn định vs chuyển tiếp — p25 và p75 capture "biên" của chuyển động.

5. **Giới hạn của DNN**: Accuracy L3 ~83% là gần trần của kiến trúc DNN do nó chỉ nhìn thống kê tổng hợp, không học được trình tự thời gian. Để đột phá >85%, cần chuyển sang LSTM hoặc Transformer.

---

## 8. Ứng Dụng Demo (Demo Applications)

### 8.1 Nhận diện trên video

```bash
python demo_video.py --url "<YouTube_URL>" --model small --max-sec 60
```

### 8.2 Nhận diện real-time qua webcam

```bash
python realtime_webcam.py --model small
```

### 8.3 Demo hàng loạt

```bash
python run_demo_batch.py
```

### 8.4 Kết quả demo mẫu

| Video | L1 | L2 | L3 | Conf |
| --- | --- | --- | --- | ---: |
| 10-Min Yoga Beginners | sitting | sitting-legs-front | cobbler | 97,9% |
| Yoga Short (twist) | sitting | sitting-legs-behind | bharadvaja's-twist | 98,5% |

---

## 9. Cấu Trúc Dự Án

```text
squat_ai/
├── train.py                    # Script huấn luyện DNN (enhanced 484-dim)
├── demo_video.py               # Demo nhận diện trên video
├── realtime_webcam.py          # Demo webcam thời gian thực
├── run_demo_batch.py           # Demo hàng loạt
├── download_skeleton.py        # Tải dữ liệu skeleton
├── requirements.txt            # Thư viện cần thiết
├── checkpoints/                # 9 checkpoint + label_maps.pkl
└── 3DYoga90/
    └── data/
        ├── 3DYoga90.csv
        └── landmarks/
            ├── official_dataset/  # 5526 parquet files
            ├── feat_cache/        # Cache v1 (198-dim)
            └── feat_cache_v2/     # Cache v2 (484-dim)
```

---

## 10. Thư Viện Sử Dụng

| Thư viện | Phiên bản | Mục đích |
| --- | --- | --- |
| PyTorch | ≥2.0.0 | Framework deep learning |
| MediaPipe | ≥0.10.0 | Trích xuất pose landmarks |
| OpenCV | ≥4.8.0 | Xử lý ảnh/video |
| Pandas | ≥2.0.0 | Xử lý dữ liệu tabular |
| PyArrow | ≥12.0.0 | Đọc file Parquet |
| scikit-learn | ≥1.3.0 | Label encoding, train/test split |
| gdown | ≥4.7.0 | Tải dữ liệu từ Google Drive |
| yt-dlp | ≥2023.10 | Tải video từ YouTube |

---

## 11. Kết Luận (Conclusion)

1. **Tái tạo thành công** kết quả Table 4 từ bài báo gốc, sau đó cải thiện bằng enhanced features.
2. **Enhanced features (484-dim)** tăng accuracy tốt nhất cho DNN-Small: **+2,61%** ở L3 và **+1,78%** ở L1.
3. **Kết quả tốt nhất**: L1 = 95,02%, L2 = 89,68%, L3 = 83,04%.
4. **Giới hạn**: Kiến trúc DNN có trần accuracy ~83-85% cho L3 do mất thông tin temporal.
5. **Hướng phát triển**: Chuyển sang LSTM/Transformer để đạt >85% L3 accuracy.

---

## Tài Liệu Tham Khảo

1. S. Kim et al., *"3DYoga90: A Hierarchical Video Dataset for Yoga Pose Understanding"*, arXiv:2310.10131, 2023.
2. Google MediaPipe, *BlazePose: On-device Real-time Body Pose tracking*, 2020.
3. PyTorch Documentation, <https://pytorch.org/docs/>
