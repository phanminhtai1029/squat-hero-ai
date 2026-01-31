# 📋 Kế Hoạch Thực Hiện MVP - Yoga Pose AI

> **Mục tiêu**: Xây dựng hệ thống nhận diện yoga poses với rule-based approach, có thể hoạt động trong 1 tuần.

---

## 📊 Tổng Quan

### Pipeline Đơn Giản

```
Input: Webcam/Video
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 1-2-3: GIỐNG CŨ (đã có sẵn)                                           │
│  Frame Capture → Person Detection (YOLO) → Pose Estimation (MediaPipe)      │
└─────────────────────────────────────────────────────────────────────────────┘
    │
    ▼ 13 Keypoints (x, y, z, visibility)
    │
┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 4: FRAME CLASSIFICATION (Rule-based)                                   │
│  Input: Pose history (last 5 frames)                                         │
│  Output: KEY_POSE / TRANSITION                                               │
│  Method: Velocity threshold                                                  │
└─────────────────────────────────────────────────────────────────────────────┘
    │
    ▼ Chỉ xử lý nếu KEY_POSE
    │
┌─────────────────────────────────────────────────────────────────────────────┐
│  Step 5: POSE MATCHING (Rule-based)                                          │
│  Input: Current pose angles                                                  │
│  Output: Pose name + Similarity %                                            │
│  Method: Cosine similarity với angle database                               │
└─────────────────────────────────────────────────────────────────────────────┘
    │
    ▼
Output: "Warrior II - 87% match"
```

### Timeline

| Phase | Task | Thời gian |
|-------|------|-----------|
| Phase 1 | Setup & Refactor | 1 ngày |
| Phase 2 | Frame Classifier | 1 ngày |
| Phase 3 | Pose Matcher | 1-2 ngày |
| Phase 4 | Pose Database | 1 ngày |
| Phase 5 | Testing & Evaluation | 1-2 ngày |
| **Tổng** | | **5-7 ngày** |

---

## 🔧 Phase 1: Setup & Refactor (1 ngày)

### 1.1 Mục tiêu
- Refactor code hiện tại thành 5 steps rõ ràng
- Tách riêng từng module

### 1.2 Cấu trúc thư mục mới

```
squat-hero-ai/
├── main.py                          # Entry point
├── config.py                        # Configuration
├── pipeline/
│   ├── __init__.py
│   ├── step1_frame_capture.py       # Giữ nguyên
│   ├── step2_person_detection.py    # Giữ nguyên
│   ├── step3_pose_estimation.py     # Giữ nguyên
│   ├── step4_frame_classifier.py    # MỚI
│   └── step5_pose_matcher.py        # MỚI
├── data/
│   └── pose_database.yaml           # Reference poses
├── utils/
│   ├── angle_calculator.py          # Tính góc
│   ├── pose_normalizer.py           # Normalize pose
│   └── visualization.py             # Giữ nguyên
├── evaluation/
│   ├── evaluate_frame_classifier.py
│   ├── evaluate_pose_matcher.py
│   └── evaluate_pipeline.py
└── tests/
    └── test_data/                   # Test videos/images
```

### 1.3 Deliverables
- [ ] Refactored codebase
- [ ] Step 1-3 hoạt động như cũ
- [ ] Placeholder cho Step 4-5

---

## 🔧 Phase 2: Frame Classifier (1 ngày)

### 2.1 Algorithm

```python
# Pseudo-code
class RuleBasedFrameClassifier:
    def __init__(self, window_size=5, velocity_threshold=0.02):
        self.window_size = window_size
        self.threshold = velocity_threshold
        self.history = []  # Lưu N frames gần nhất
    
    def classify(self, current_pose):
        self.history.append(current_pose)
        if len(self.history) > self.window_size:
            self.history.pop(0)
        
        if len(self.history) < 2:
            return "TRANSITION"
        
        # Tính velocity trung bình
        velocity = self.compute_average_velocity()
        
        if velocity < self.threshold:
            return "KEY_POSE"
        else:
            return "TRANSITION"
    
    def compute_average_velocity(self):
        total_velocity = 0
        for i in range(1, len(self.history)):
            prev = self.history[i-1]
            curr = self.history[i]
            diff = np.abs(curr - prev)
            total_velocity += np.mean(diff)
        return total_velocity / (len(self.history) - 1)
```

### 2.2 Parameters cần tune

| Parameter | Giá trị đề xuất | Ý nghĩa |
|-----------|-----------------|---------|
| `window_size` | 5 frames | Số frames để tính velocity |
| `velocity_threshold` | 0.02 | Ngưỡng phân biệt KEY_POSE/TRANSITION |

### 2.3 Evaluation Metrics

#### Dataset cần chuẩn bị
```
test_data/frame_classification/
├── video_001.mp4
├── video_001_labels.json  # Ground truth: [0,0,1,1,1,0,0,0,...]
├── video_002.mp4
├── video_002_labels.json
└── ...
```

#### Metrics

| Metric | Công thức | Mục tiêu |
|--------|-----------|----------|
| **Accuracy** | (TP + TN) / Total | > 85% |
| **Precision (KEY_POSE)** | TP_key / (TP_key + FP_key) | > 80% |
| **Recall (KEY_POSE)** | TP_key / (TP_key + FN_key) | > 90% |
| **F1 Score** | 2 * P * R / (P + R) | > 85% |

> **Lưu ý**: Recall quan trọng hơn Precision! Bỏ sót KEY_POSE → không match → tệ hơn là match nhầm TRANSITION.

#### Evaluation Script

```python
def evaluate_frame_classifier(classifier, test_videos):
    results = {
        'total': 0,
        'correct': 0,
        'tp_key': 0, 'fp_key': 0, 'fn_key': 0,
        'tp_trans': 0, 'fp_trans': 0, 'fn_trans': 0
    }
    
    for video, labels in test_videos:
        for frame, true_label in zip(video.frames(), labels):
            pred_label = classifier.classify(frame.pose)
            
            results['total'] += 1
            if pred_label == true_label:
                results['correct'] += 1
            
            # Update confusion matrix...
    
    accuracy = results['correct'] / results['total']
    precision_key = results['tp_key'] / (results['tp_key'] + results['fp_key'])
    recall_key = results['tp_key'] / (results['tp_key'] + results['fn_key'])
    f1_key = 2 * precision_key * recall_key / (precision_key + recall_key)
    
    return {
        'accuracy': accuracy,
        'precision_key': precision_key,
        'recall_key': recall_key,
        'f1_key': f1_key
    }
```

### 2.4 Deliverables
- [ ] `step4_frame_classifier.py` hoàn chỉnh
- [ ] Test data với ground truth labels
- [ ] Evaluation script
- [ ] Báo cáo accuracy

---

## 🔧 Phase 3: Pose Matcher (1-2 ngày)

### 3.1 Algorithm

```python
# Pseudo-code
class RuleBasedPoseMatcher:
    # Định nghĩa 8 góc quan trọng
    ANGLE_DEFINITIONS = {
        'left_elbow':    ('left_shoulder', 'left_elbow', 'left_wrist'),
        'right_elbow':   ('right_shoulder', 'right_elbow', 'right_wrist'),
        'left_knee':     ('left_hip', 'left_knee', 'left_ankle'),
        'right_knee':    ('right_hip', 'right_knee', 'right_ankle'),
        'left_hip':      ('left_shoulder', 'left_hip', 'left_knee'),
        'right_hip':     ('right_shoulder', 'right_hip', 'right_knee'),
        'left_shoulder': ('left_elbow', 'left_shoulder', 'left_hip'),
        'right_shoulder':('right_elbow', 'right_shoulder', 'right_hip'),
    }
    
    def __init__(self, database_path):
        self.database = self.load_database(database_path)
    
    def match(self, landmarks):
        # 1. Tính 8 góc
        current_angles = self.compute_angles(landmarks)
        
        # 2. Normalize angles to [0, 1]
        normalized_angles = [a / 180.0 for a in current_angles]
        
        # 3. So sánh với database
        matches = []
        for pose_name, pose_data in self.database.items():
            ref_angles = pose_data['angles_normalized']
            similarity = self.cosine_similarity(normalized_angles, ref_angles)
            matches.append((pose_name, similarity))
        
        # 4. Sort và trả về best match
        matches.sort(key=lambda x: -x[1])
        
        best_pose, best_similarity = matches[0]
        
        return MatchResult(
            pose_name=best_pose,
            similarity=best_similarity,
            top_3_matches=matches[:3]
        )
    
    def cosine_similarity(self, a, b):
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)
```

### 3.2 Angle Embedding

```
Mỗi pose được biểu diễn bởi 8 góc:

Pose "Warrior II":
┌───────────────────────────────────────────────────┐
│  left_elbow:     175° (tay gần thẳng)             │
│  right_elbow:    178° (tay gần thẳng)             │
│  left_knee:      92°  (gập 90°)                   │
│  right_knee:     168° (gần thẳng)                 │
│  left_hip:       108° (nghiêng)                   │
│  right_hip:      152° (gần thẳng)                 │
│  left_shoulder:  165°                             │
│  right_shoulder: 170°                             │
└───────────────────────────────────────────────────┘

Normalized: [0.97, 0.99, 0.51, 0.93, 0.60, 0.84, 0.92, 0.94]
```

### 3.3 Similarity Threshold

| Similarity | Interpretation |
|------------|----------------|
| > 0.95 | Perfect match |
| 0.85 - 0.95 | Good match |
| 0.70 - 0.85 | Possible match, cần verify |
| < 0.70 | No match / Unknown pose |

### 3.4 Evaluation Metrics

#### Dataset cần chuẩn bị
```
test_data/pose_matching/
├── warrior_i/
│   ├── image_001.jpg
│   ├── image_001_keypoints.json
│   ├── image_002.jpg
│   └── ...
├── warrior_ii/
│   └── ...
├── tree_pose/
│   └── ...
└── ... (20-50 poses × 10-20 images mỗi pose)
```

#### Metrics

| Metric | Công thức | Mục tiêu |
|--------|-----------|----------|
| **Top-1 Accuracy** | Đúng ở vị trí #1 | > 80% |
| **Top-3 Accuracy** | Đúng trong Top 3 | > 95% |
| **Mean Similarity (True)** | Avg similarity khi match đúng | > 0.85 |
| **Mean Similarity (False)** | Avg similarity khi match sai | < 0.70 |
| **Confusion Matrix** | Pose A nhầm thành Pose B bao nhiêu lần | Để debug |

#### Evaluation Script

```python
def evaluate_pose_matcher(matcher, test_data):
    results = {
        'total': 0,
        'top1_correct': 0,
        'top3_correct': 0,
        'similarities_correct': [],
        'similarities_wrong': [],
        'confusion': defaultdict(lambda: defaultdict(int))
    }
    
    for true_pose, images in test_data.items():
        for image in images:
            keypoints = extract_keypoints(image)
            match_result = matcher.match(keypoints)
            
            results['total'] += 1
            predicted_pose = match_result.pose_name
            similarity = match_result.similarity
            
            # Top-1
            if predicted_pose == true_pose:
                results['top1_correct'] += 1
                results['similarities_correct'].append(similarity)
            else:
                results['similarities_wrong'].append(similarity)
            
            # Top-3
            top3_poses = [m[0] for m in match_result.top_3_matches]
            if true_pose in top3_poses:
                results['top3_correct'] += 1
            
            # Confusion matrix
            results['confusion'][true_pose][predicted_pose] += 1
    
    return {
        'top1_accuracy': results['top1_correct'] / results['total'],
        'top3_accuracy': results['top3_correct'] / results['total'],
        'mean_sim_correct': np.mean(results['similarities_correct']),
        'mean_sim_wrong': np.mean(results['similarities_wrong']),
        'confusion_matrix': dict(results['confusion'])
    }
```

### 3.5 Deliverables
- [ ] `step5_pose_matcher.py` hoàn chỉnh
- [ ] `angle_calculator.py` utility
- [ ] Test data với ground truth labels
- [ ] Evaluation script
- [ ] Confusion matrix analysis

---

## 🔧 Phase 4: Pose Database (1 ngày)

### 4.1 Database Format

```yaml
# pose_database.yaml
poses:
  # ═══════════════════════════════════════════════════
  # STANDING POSES
  # ═══════════════════════════════════════════════════
  
  warrior_i:
    display_name: "Warrior I (Virabhadrasana I)"
    category: "standing"
    difficulty: "beginner"
    angles_degrees:
      left_elbow: 175
      right_elbow: 175
      left_knee: 95
      right_knee: 165
      left_hip: 115
      right_hip: 145
      left_shoulder: 175
      right_shoulder: 175
    angles_normalized: [0.97, 0.97, 0.53, 0.92, 0.64, 0.81, 0.97, 0.97]
    description: "Lunge với 2 tay giơ cao trên đầu"
  
  warrior_ii:
    display_name: "Warrior II (Virabhadrasana II)"
    category: "standing"
    difficulty: "beginner"
    angles_degrees:
      left_elbow: 175
      right_elbow: 178
      left_knee: 92
      right_knee: 168
      left_hip: 108
      right_hip: 152
      left_shoulder: 165
      right_shoulder: 170
    angles_normalized: [0.97, 0.99, 0.51, 0.93, 0.60, 0.84, 0.92, 0.94]
    description: "Lunge với 2 tay dang ngang"
  
  tree_pose:
    display_name: "Tree Pose (Vrksasana)"
    category: "balancing"
    difficulty: "beginner"
    angles_degrees:
      left_elbow: 175
      right_elbow: 175
      left_knee: 170    # Chân đứng thẳng
      right_knee: 45    # Chân gập gác lên đùi
      left_hip: 170
      right_hip: 90
      left_shoulder: 170
      right_shoulder: 170
    angles_normalized: [0.97, 0.97, 0.94, 0.25, 0.94, 0.50, 0.94, 0.94]
    description: "Đứng 1 chân, tay chắp trước ngực hoặc giơ cao"
  
  # ... thêm 20-50 poses khác

# ═══════════════════════════════════════════════════
# METADATA
# ═══════════════════════════════════════════════════
metadata:
  version: "1.0"
  total_poses: 30
  last_updated: "2025-01-31"
  categories:
    - standing
    - balancing
    - seated
    - prone
    - supine
```

### 4.2 Cách Tạo Database

```
Bước 1: Chọn 20-50 yoga poses phổ biến

Bước 2: Với mỗi pose:
    a) Tìm ảnh/video reference của pose chuẩn
    b) Chạy qua MediaPipe → lấy keypoints
    c) Tính 8 góc
    d) Verify bằng mắt: góc có hợp lý không?
    e) Thêm vào database

Bước 3: Test lại với ảnh khác của cùng pose
    → Similarity có > 0.85 không?
    → Nếu không, điều chỉnh lại góc reference
```

### 4.3 Tool Hỗ Trợ Tạo Database

```python
# create_database_entry.py
def create_pose_entry(image_path, pose_name):
    """Tool để tạo entry cho database."""
    
    # 1. Load image
    image = cv2.imread(image_path)
    
    # 2. Detect pose
    landmarks = mediapipe_detect(image)
    
    # 3. Tính góc
    angles = calculate_all_angles(landmarks)
    
    # 4. In ra để copy vào database
    print(f"\n{pose_name}:")
    print(f"  angles_degrees:")
    for name, angle in angles.items():
        print(f"    {name}: {angle:.0f}")
    
    normalized = [a / 180.0 for a in angles.values()]
    print(f"  angles_normalized: {normalized}")
    
    # 5. Visualize để verify
    visualize_pose_with_angles(image, landmarks, angles)
    cv2.imshow("Verify", image)
    cv2.waitKey(0)

# Usage:
# python create_database_entry.py --image warrior2.jpg --name warrior_ii
```

### 4.4 Deliverables
- [ ] `pose_database.yaml` với 20-50 poses
- [ ] Tool tạo database entries
- [ ] Documentation cho mỗi pose

---

## 🔧 Phase 5: Testing & Evaluation (1-2 ngày)

### 5.1 End-to-End Pipeline Evaluation

#### Test Scenarios

| Scenario | Mô tả | Kỳ vọng |
|----------|-------|---------|
| **Happy Path** | User làm pose chuẩn, đứng yên | Nhận diện đúng, similarity > 0.90 |
| **Slight Variation** | Pose hơi khác chuẩn (±10°) | Nhận diện đúng, similarity 0.80-0.90 |
| **Moving** | User đang chuyển động | Frame Classifier trả về TRANSITION |
| **Unknown Pose** | Pose không có trong DB | Top similarity < 0.70, có thể reject |
| **Bad Detection** | MediaPipe detect sai | Graceful handling, không crash |
| **Multiple Poses** | User làm nhiều poses liên tiếp | Nhận diện đúng từng pose |

#### Evaluation Script

```python
def evaluate_full_pipeline(pipeline, test_videos):
    """Đánh giá toàn bộ pipeline end-to-end."""
    
    results = {
        'frame_classifier': {
            'total_frames': 0,
            'key_poses_detected': 0,
            'transitions_detected': 0
        },
        'pose_matcher': {
            'total_key_frames': 0,
            'correct_matches': 0,
            'top3_correct': 0,
            'avg_similarity_correct': [],
            'avg_similarity_wrong': []
        },
        'pipeline': {
            'latency_ms': [],
            'fps': []
        }
    }
    
    for video_path, ground_truth in test_videos:
        cap = cv2.VideoCapture(video_path)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            start_time = time.time()
            
            # Run pipeline
            output = pipeline.process_frame(frame)
            
            end_time = time.time()
            latency = (end_time - start_time) * 1000
            results['pipeline']['latency_ms'].append(latency)
            
            # Evaluate frame classifier
            results['frame_classifier']['total_frames'] += 1
            if output.frame_type == 'KEY_POSE':
                results['frame_classifier']['key_poses_detected'] += 1
            else:
                results['frame_classifier']['transitions_detected'] += 1
            
            # Evaluate pose matcher (only for key frames)
            if output.frame_type == 'KEY_POSE' and output.match_result:
                results['pose_matcher']['total_key_frames'] += 1
                
                true_pose = ground_truth.get_pose_at_frame(frame_idx)
                pred_pose = output.match_result.pose_name
                similarity = output.match_result.similarity
                
                if pred_pose == true_pose:
                    results['pose_matcher']['correct_matches'] += 1
                    results['pose_matcher']['avg_similarity_correct'].append(similarity)
                else:
                    results['pose_matcher']['avg_similarity_wrong'].append(similarity)
    
    # Compute final metrics
    return {
        'frame_classifier': {
            'key_pose_ratio': results['frame_classifier']['key_poses_detected'] / 
                              results['frame_classifier']['total_frames']
        },
        'pose_matcher': {
            'top1_accuracy': results['pose_matcher']['correct_matches'] / 
                             results['pose_matcher']['total_key_frames'],
            'mean_similarity_correct': np.mean(results['pose_matcher']['avg_similarity_correct']),
            'mean_similarity_wrong': np.mean(results['pose_matcher']['avg_similarity_wrong'])
        },
        'pipeline': {
            'avg_latency_ms': np.mean(results['pipeline']['latency_ms']),
            'avg_fps': 1000 / np.mean(results['pipeline']['latency_ms'])
        }
    }
```

### 5.2 Expected Results

#### Frame Classifier

| Metric | Target | Acceptable |
|--------|--------|------------|
| Accuracy | > 90% | > 85% |
| Precision (KEY_POSE) | > 85% | > 80% |
| Recall (KEY_POSE) | > 95% | > 90% |

#### Pose Matcher

| Metric | Target | Acceptable |
|--------|--------|------------|
| Top-1 Accuracy | > 85% | > 75% |
| Top-3 Accuracy | > 98% | > 95% |
| Mean Similarity (Correct) | > 0.88 | > 0.82 |
| Mean Similarity (Wrong) | < 0.65 | < 0.72 |

#### Pipeline Performance

| Metric | Target | Acceptable |
|--------|--------|------------|
| Latency | < 50ms | < 80ms |
| FPS | > 20 | > 12 |

### 5.3 Deliverables
- [ ] End-to-end evaluation script
- [ ] Test video dataset với ground truth
- [ ] Báo cáo kết quả chi tiết
- [ ] Confusion matrix
- [ ] Error analysis

---

## 📈 Tổng Kết & Quyết Định

### Quyết Định Dựa Trên Kết Quả

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         DECISION TREE                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Sau Phase 5, đánh giá kết quả:                                         │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ Pose Matcher Top-1 Accuracy                                      │    │
│  ├─────────────────────────────────────────────────────────────────┤    │
│  │                                                                  │    │
│  │  > 85%  ──────▶  ✅ DONE! Ship MVP                              │    │
│  │                                                                  │    │
│  │  75-85% ──────▶  🔧 Tune thresholds, thêm features              │    │
│  │                   - Thêm relative positions                      │    │
│  │                   - Thêm symmetry features                       │    │
│  │                   - Tune velocity threshold                      │    │
│  │                                                                  │    │
│  │  < 75%  ──────▶  🧠 Consider training simple encoder            │    │
│  │                   - Thu thập thêm data                           │    │
│  │                   - Train MLP encoder                            │    │
│  │                                                                  │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Checkpoints

| Checkpoint | Criteria | Action nếu FAIL |
|------------|----------|-----------------|
| Phase 2 Done | Frame Classifier F1 > 85% | Tune window_size, velocity_threshold |
| Phase 3 Done | Pose Matcher Top-1 > 75% | Thêm features, tune database |
| Phase 5 Done | Pipeline FPS > 12 | Optimize code, reduce computations |

---

## 📝 Appendix: Danh Sách 30 Yoga Poses Đề Xuất

### Beginner (15 poses)
1. Mountain Pose (Tadasana)
2. Tree Pose (Vrksasana)
3. Warrior I (Virabhadrasana I)
4. Warrior II (Virabhadrasana II)
5. Triangle Pose (Trikonasana)
6. Downward Dog (Adho Mukha Svanasana)
7. Child's Pose (Balasana)
8. Cat Pose (Marjaryasana)
9. Cow Pose (Bitilasana)
10. Cobra Pose (Bhujangasana)
11. Bridge Pose (Setu Bandhasana)
12. Seated Forward Bend (Paschimottanasana)
13. Corpse Pose (Savasana)
14. Chair Pose (Utkatasana)
15. Extended Side Angle (Utthita Parsvakonasana)

### Intermediate (15 poses)
16. Warrior III (Virabhadrasana III)
17. Half Moon (Ardha Chandrasana)
18. Eagle Pose (Garudasana)
19. Dancer Pose (Natarajasana)
20. Boat Pose (Navasana)
21. Crow Pose (Bakasana)
22. Side Plank (Vasisthasana)
23. Pigeon Pose (Eka Pada Rajakapotasana)
24. Camel Pose (Ustrasana)
25. Wheel Pose (Urdhva Dhanurasana)
26. Shoulder Stand (Sarvangasana)
27. Plow Pose (Halasana)
28. Fish Pose (Matsyasana)
29. Bow Pose (Dhanurasana)
30. Headstand Prep (Sirsasana Prep)

---

> **Ghi chú**: 
> - Kế hoạch này có thể điều chỉnh dựa trên kết quả thực tế
> - Ưu tiên hoàn thành Phase 1-3 trước, Phase 4-5 có thể mở rộng sau
> - Documentation và code comments là quan trọng để maintain sau này
