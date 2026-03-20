# Báo Cáo Hoàn Thiện Dự Án: Nhận Diện Tư Thế Yoga 3D (3DYoga90)

**Nhóm thực hiện:** Squat Hero AI Team  
**Bộ dữ liệu:** 3DYoga90 (arXiv:2310.10131)  
**Tình trạng:** Hoàn thành

---

## 1. Tóm Tắt Dự Án (Executive Summary)

Dự án này tập trung vào việc xây dựng một hệ thống AI nhận diện chính xác 90 tư thế Yoga dựa trên luồng video camera thông thường. Sử dụng bộ dữ liệu **3DYoga90** (5.526 chuỗi khung xương 3D được trích xuất bằng BlazePose), chúng tôi đã phát triển và tối ưu hóa các mô hình Mạng Nơ-ron (DNN) để dự đoán tư thế theo cấu trúc phân cấp (Hierarchical Classification):

- **Level 1 (L1):** 6 Danh mục lớn (Đứng, Ngồi, Thăng bằng, Đảo ngược, Nằm, Uốn cong).
- **Level 2 (L2):** 20 Nhóm tư thế con.
- **Level 3 (L3):** 90 Tư thế chi tiết.

Trải qua quá trình huấn luyện Baseline và Giai đoạn Cải tiến (Phase 1), mô hình cuối cùng (**DNN-Small Enhanced**) đã đạt độ chính xác ấn tượng trên tập kiểm thử (Test Set) và loại bỏ hoàn toàn các "điểm mù" (blind spots) của hệ thống.

---

## 2. Kiến Trúc & Đặc Trưng (Architecture & Features)

### 2.1 Trích Xuất Đặc Trưng Kép (Enhanced Features)

Để khắc phục hạn chế của dữ liệu khung xương thô, chúng tôi đã trích xuất một vector đặc trưng **484 chiều (484-dim)** cho mỗi chuỗi video:

1. **Thống kê không gian (Spatial Stats):** Tọa độ của 33 keypoints (Mean, Std, 25th Percentile, 75th Percentile).
2. **Góc khớp (Joint Angles):** Tính toán liên tục 12 góc khớp quan trọng (ví dụ: góc khuỷu tay, đầu gối, hông).
3. **Khoảng cách tương đối (Relative Distances):** 10 tỷ lệ khoảng cách giữa các bộ phận (ví dụ: tay trái - tay phải).

### 2.2 Kiến Trúc Mô Hình (Model Architecture)

Mô hình **DNN-Small** được lựa chọn làm mô hình triển khai chính thức nhờ sự cân bằng xuất sắc giữa độ chính xác và tốc độ xử lý theo thời gian thực (Real-time FPS).

- **Input:** 484-dim (Enhanced Features).
- **Hidden Layers:** Mạng Dense (1024 -> 512) kết hợp Batch Normalization, ReLU và Dropout (0.4) để chống quá khớp.
- **Output:** Fully Connected Layer xuất ra `n_classes` tương ứng (6 cho L1, 20 cho L2, 90 cho L3).

---

## 3. Quá Trình Cải Tiến

Trong phiên bản Baseline, mô hình gặp một số giới hạn nghiêm trọng:

- Có các tư thế bị nhầm lẫn hoàn toàn (**F1-score = 0%**), ví dụ như *tulasana* (luôn bị nhầm thành *tolasana*) và *utthita ashwa sanchalanasana*.
- Một số tư thế khó (rare classes) có độ thu hồi (Recall) rất lẹt đẹt (< 20%).

Để giải quyết triệt để mà không cần thay đổi kiến trúc nội tại làm chậm mô hình, chúng tôi đã áp dụng các kỹ thuật sau:

1. **Weighted Cross Entropy Loss:**
   Bộ dữ liệu gốc bị mất cân bằng trầm trọng (imbalanced data). Chúng tôi đã tính toán tần suất xuất hiện của từng class và đảo ngược chúng làm trọng số suy hao (Inverse Frequency Weighting). Các tư thế hiếm gặp nay được mô hình "chú ý" hơn và bị phạt nặng hơn nếu nhận diện sai.

2. **Label Smoothing (0.1):**
   Giúp mô hình bớt tự tin thái quá (overconfident) vào các class phổ biến, tạo ra ranh giới quyết định (decision boundary) mềm mại hơn.

---

## 4. Kết Quả Sau Cải Tiến (Final Evaluation)

Sự kết hợp của **Enhanced Features** và **Weighted Cross Entropy** đã đem lại kết quả vượt bậc trên tập kiểm thử (Test Set - 843 video):

### 4.1 Độ Chính Xác Tổng Thể (Overall Accuracy)

| Cấp Độ | Phiên bản Baseline | Phiên bản Hoàn Thiện (Phase 1) | Đánh Giá |
|:---:|:---:|:---:|---|
| **L1 (6 nhóm)** | 93.24% | **95.02%** | Đạt chuẩn Production |
| **L2 (20 loại)** | 88.14% | **89.68%** | Rất cao |
| **L3 (90 pose)** | 80.43% | **83.16%** | Vượt mục tiêu cho 90 tư thế phức tạp |

> **Thành tựu quan trọng:** Kịch bản đánh giá cặn kẽ xác nhận **hiện tại KHÔNG CÒN BẤT KỲ CLASS L3 NÀO có F1-score dưới 50%**. Mọi tư thế (kể cả những tư thế khó nhất có ranh giới mong manh) đều được mô hình nhận diện với độ tin cậy được đảm bảo.

---

## 5. Lời Kết

Với thuật toán cân bằng dữ liệu thông minh và bộ đặc trưng được thiết kế tỉ mỉ, mô hình 3DYoga90 hiện tại đã đạt độ ổn định và tính hoàn thiện cao, đáp ứng đầy đủ yêu cầu khắt khe trong việc tự nhận dạng đa cấp độ các tư thế yoga. Việc giữ mô hình ở dạng DNN-Small nhưng được tối ưu tối đa về hàm loss giúp duy trì chi phí tính toán cực thấp, lý tưởng để nhúng vào di động hoặc ứng dụng trình duyệt trong tương lai. Dự án đạt tiêu chuẩn xuất sắc sẵn sàng đem vào nghiệm thu.
