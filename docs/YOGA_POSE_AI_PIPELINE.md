# 🧘 Yoga Pose AI - Pipeline Chi Tiết

> **Mục tiêu**: Xây dựng hệ thống AI có khả năng nhận diện và đánh giá hàng trăm động tác yoga khác nhau, với khả năng mở rộng dễ dàng mà không cần retrain model.

---

## 📋 Mục Lục

1. [Tổng Quan Kiến Trúc](#1-tổng-quan-kiến-trúc)
2. [Step 1: Frame Capture](#2-step-1-frame-capture)
3. [Step 2: Person Detection](#3-step-2-person-detection)
4. [Step 3: Pose Estimation](#4-step-3-pose-estimation)
5. [Step 4: Frame Classification](#5-step-4-frame-classification)
6. [Step 5: Pose Matching](#6-step-5-pose-matching)
7. [Pose Database Design](#7-pose-database-design)
8. [Training Strategy](#8-training-strategy)
9. [Evaluation Metrics](#9-evaluation-metrics)
10. [Challenges & Solutions](#10-challenges--solutions)
11. [Implementation Roadmap](#11-implementation-roadmap)

---

## 1. Tổng Quan Kiến Trúc

### 1.1 High-Level Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              YOGA POSE AI PIPELINE                                   │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                       │
│   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────────┐   ┌─────────────────┐   │
│   │ Step 1  │   │ Step 2  │   │ Step 3  │   │   Step 4    │   │     Step 5      │   │
│   │ Frame   │──▶│ Person  │──▶│  Pose   │──▶│   Frame     │──▶│     Pose        │   │
│   │ Capture │   │ Detect  │   │ Estimate│   │   Classify  │   │    Matching     │   │
│   └─────────┘   └─────────┘   └─────────┘   └─────────────┘   └─────────────────┘   │
│        │             │             │               │                   │             │
│        ▼             ▼             ▼               ▼                   ▼             │
│     BGR Frame    BBox (x,y,w,h)  33 Keypoints   KEY_POSE or      ┌──────────────┐   │
│     (H x W x 3)  + Confidence    (x,y,z,vis)    TRANSITION       │  Pose Name   │   │
│                                                      │            │  Similarity  │   │
│                                                      │            │  Form Errors │   │
│                                                 Only if           └──────────────┘   │
│                                                 KEY_POSE ──────────────▶ │           │
│                                                                          ▼           │
│                                                                    ┌──────────┐      │
│                                                                    │ Pose DB  │      │
│                                                                    │ (Vector) │      │
│                                                                    └──────────┘      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 So Sánh Với Approach Truyền Thống

| Aspect | Classification Approach | Retrieval Approach (Đề xuất) |
|--------|------------------------|------------------------------|
| **Thêm pose mới** | Retrain toàn bộ model | Chỉ thêm vào database |
| **Output** | Class ID (discrete) | Similarity score (continuous) |
| **Pose chưa thấy** | Predict sai | Tìm pose gần nhất |
| **Giải thích** | Khó | Dễ ("giống 87% với Warrior II") |
| **Scale** | O(n) parameters | O(1) model + O(n) database |

### 1.3 Technology Stack

| Component | Technology | Lý do chọn |
|-----------|------------|------------|
| Frame Capture | OpenCV | Standard, cross-platform |
| Person Detection | YOLOv8 | Fast, accurate, well-maintained |
| Pose Estimation | MediaPipe Pose | Real-time, 33 keypoints, no GPU required |
| Frame Classification | LSTM / GRU / Transformer | Temporal modeling |
| Pose Encoder | MLP / GNN | Learns pose embeddings |
| Vector Database | FAISS / Milvus / Pinecone | Efficient similarity search |
| Training | PyTorch | Flexible, large community |

---

## 2. Step 1: Frame Capture

### 2.1 Mô Tả
Capture frames từ nguồn video (webcam real-time hoặc video file).

### 2.2 Input / Output

| | Type | Description |
|---|------|-------------|
| **Input** | Camera ID hoặc Video Path | Nguồn video |
| **Output** | `np.ndarray` (H, W, 3) | BGR frame |

### 2.3 Implementation Hiện Tại

```python
class FrameCapture:
    """Capture frames từ webcam hoặc video file."""
    
    def __init__(self, source: Union[int, str]):
        self.cap = cv2.VideoCapture(source)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
    
    def get_frame(self) -> Optional[np.ndarray]:
        ret, frame = self.cap.read()
        return frame if ret else None
```

### 2.4 Cải Tiến Có Thể

- [ ] Frame buffering để giảm latency
- [ ] Adaptive resolution based on performance
- [ ] Multi-camera support

---

## 3. Step 2: Person Detection

### 3.1 Mô Tả
Detect người trong frame và trả về bounding box. Đảm bảo chỉ có 1 người được track.

### 3.2 Input / Output

| | Type | Description |
|---|------|-------------|
| **Input** | Frame (H, W, 3) | BGR image |
| **Output** | `BoundingBox(x1, y1, x2, y2, conf)` | Person bounding box |

### 3.3 Model: YOLOv8

```python
class PersonDetector:
    def __init__(self, model_path: str = "yolov8n.pt", conf_threshold: float = 0.5):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.person_class_id = 0  # COCO class
    
    def detect(self, frame: np.ndarray) -> Optional[BoundingBox]:
        results = self.model(frame, verbose=False, conf=self.conf_threshold)
        
        # Lấy person có confidence cao nhất
        best_box = None
        for result in results:
            for box in result.boxes:
                if int(box.cls[0]) == self.person_class_id:
                    if best_box is None or box.conf[0] > best_box.confidence:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        best_box = BoundingBox(x1, y1, x2, y2, float(box.conf[0]))
        
        return best_box
```

### 3.4 Considerations

| Issue | Giải pháp |
|-------|-----------|
| Nhiều người trong frame | Chọn person có bbox lớn nhất hoặc gần center nhất |
| Person bị che khuất | Tracking với DeepSORT để duy trì identity |
| Performance | Có thể skip detection mỗi N frames, interpolate bbox |

---

## 4. Step 3: Pose Estimation

### 4.1 Mô Tả
Estimate 33 body keypoints sử dụng MediaPipe Pose.

### 4.2 Input / Output

| | Type | Description |
|---|------|-------------|
| **Input** | Frame (H, W, 3) | BGR image |
| **Output** | `Dict[str, Landmark]` | 33 keypoints với x, y, z, visibility |

### 4.3 MediaPipe Pose Keypoints

```
Keypoint Index Mapping:
  0: nose               11: left_shoulder    23: left_hip
  1: left_eye_inner     12: right_shoulder   24: right_hip
  2: left_eye           13: left_elbow       25: left_knee
  3: left_eye_outer     14: right_elbow      26: right_knee
  4: right_eye_inner    15: left_wrist       27: left_ankle
  5: right_eye          16: right_wrist      28: right_ankle
  6: right_eye_outer    17: left_pinky       29: left_heel
  7: left_ear           18: right_pinky      30: right_heel
  8: right_ear          19: left_index       31: left_foot_index
  9: mouth_left         20: right_index      32: right_foot_index
  10: mouth_right       21: left_thumb
                        22: right_thumb
```

### 4.4 Implementation

```python
class PoseEstimator:
    # Keypoints quan trọng cho yoga poses
    YOGA_KEYPOINTS = [
        'nose', 
        'left_shoulder', 'right_shoulder',
        'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist',
        'left_hip', 'right_hip',
        'left_knee', 'right_knee',
        'left_ankle', 'right_ankle',
    ]  # 13 keypoints chính
    
    def __init__(self, min_detection_conf: float = 0.5):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # 0=lite, 1=full, 2=heavy
            min_detection_confidence=min_detection_conf,
            min_tracking_confidence=min_detection_conf
        )
    
    def estimate(self, frame: np.ndarray) -> Optional[Dict[str, Landmark]]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        
        if not results.pose_landmarks:
            return None
        
        landmarks = {}
        for name in self.YOGA_KEYPOINTS:
            idx = getattr(self.mp_pose.PoseLandmark, name.upper()).value
            lm = results.pose_landmarks.landmark[idx]
            landmarks[name] = Landmark(lm.x, lm.y, lm.z, lm.visibility)
        
        return landmarks
```

### 4.5 Pose Normalization

Trước khi đưa vào Step 4 & 5, cần normalize pose:

```python
def normalize_pose(landmarks: Dict[str, Landmark]) -> np.ndarray:
    """
    Normalize pose để invariant với:
    - Translation (vị trí trong frame)
    - Scale (khoảng cách tới camera)
    - (Optional) Rotation
    """
    # 1. Extract coordinates
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.values()])
    
    # 2. Center around hip midpoint
    hip_center = (coords[7] + coords[8]) / 2  # left_hip + right_hip
    coords = coords - hip_center
    
    # 3. Scale by torso length (shoulder to hip)
    shoulder_center = (coords[1] + coords[2]) / 2
    torso_length = np.linalg.norm(shoulder_center - np.array([0, 0, 0]))
    coords = coords / (torso_length + 1e-6)
    
    # 4. Flatten to 1D vector
    return coords.flatten()  # Shape: (13 * 3,) = (39,)
```

---

## 5. Step 4: Frame Classification

### 5.1 Mục Tiêu
Phân loại frame thành **KEY_POSE** (tư thế ổn định) hoặc **TRANSITION** (đang chuyển động).

### 5.2 Tại Sao Cần Bước Này?

| Scenario | Không có Step 4 | Có Step 4 |
|----------|-----------------|-----------|
| Đang chuyển động | Match với pose sai, flickering | Skip, không match |
| Giữ tư thế | OK | OK + stable output |
| Performance | Match mọi frame (30 FPS) | Match chỉ key frames (~5-10 FPS) |

### 5.3 Input / Output

| | Type | Description |
|---|------|-------------|
| **Input** | Sequence of normalized poses (window_size, 39) | Last N frames |
| **Output** | `FrameType` enum | KEY_POSE hoặc TRANSITION |

### 5.4 Approach 1: Rule-Based (No AI)

```python
class RuleBasedFrameClassifier:
    """
    Không dùng AI. Dựa vào velocity của joints.
    Ưu điểm: Đơn giản, không cần train
    Nhược điểm: Kém robust với noise
    """
    
    def __init__(self, window_size: int = 5, stability_threshold: float = 0.02):
        self.window_size = window_size
        self.threshold = stability_threshold
        self.history = deque(maxlen=window_size)
    
    def classify(self, normalized_pose: np.ndarray) -> FrameType:
        self.history.append(normalized_pose)
        
        if len(self.history) < 2:
            return FrameType.TRANSITION
        
        # Tính average velocity của tất cả joints
        velocities = []
        for i in range(1, len(self.history)):
            diff = np.abs(self.history[i] - self.history[i-1])
            velocities.append(np.mean(diff))
        
        avg_velocity = np.mean(velocities)
        
        if avg_velocity < self.threshold:
            return FrameType.KEY_POSE
        else:
            return FrameType.TRANSITION
```

### 5.5 Approach 2: LSTM Classifier (AI)

```python
class LSTMFrameClassifier(nn.Module):
    """
    Dùng LSTM để học temporal patterns.
    Ưu điểm: Robust hơn, học được complex patterns
    Nhược điểm: Cần data để train
    """
    
    def __init__(self, input_dim: int = 39, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2)  # KEY_POSE, TRANSITION
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, input_dim)
        _, (h_n, _) = self.lstm(x)
        
        # h_n shape: (num_layers, batch, hidden_dim)
        # Lấy output của layer cuối
        last_hidden = h_n[-1]  # (batch, hidden_dim)
        
        logits = self.classifier(last_hidden)
        return logits
```

### 5.6 Approach 3: Transformer Classifier (AI)

```python
class TransformerFrameClassifier(nn.Module):
    """
    Dùng Transformer để capture long-range dependencies.
    Ưu điểm: Parallel processing, attention mechanism
    Nhược điểm: Cần nhiều data hơn LSTM
    """
    
    def __init__(self, input_dim: int = 39, d_model: int = 64, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Linear(d_model, 2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, input_dim)
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        x = self.transformer(x)        # (batch, seq_len, d_model)
        
        # Global average pooling
        x = x.mean(dim=1)  # (batch, d_model)
        
        logits = self.classifier(x)
        return logits
```

### 5.7 So Sánh Các Approaches

| Approach | AI? | Accuracy | Speed | Data Needed |
|----------|-----|----------|-------|-------------|
| Rule-based | ❌ | 🟡 70-80% | 🟢 Very Fast | None |
| LSTM | ✅ | 🟢 85-90% | 🟢 Fast | ~1000 sequences |
| Transformer | ✅ | 🟢 90-95% | 🟡 Medium | ~5000 sequences |

### 5.8 Khuyến Nghị

> **Bắt đầu với Rule-based** để có baseline nhanh chóng, sau đó **nâng cấp lên LSTM** khi có đủ data.

---

## 6. Step 5: Pose Matching

### 6.1 Mục Tiêu
So khớp KEY_POSE với database để:
1. **Identify**: Đây là pose gì? (Warrior II, Tree Pose, etc.)
2. **Evaluate**: Pose này giống bao nhiêu % với pose chuẩn?
3. **Feedback**: Những điểm nào cần cải thiện?

### 6.2 Input / Output

| | Type | Description |
|---|------|-------------|
| **Input** | Normalized pose (39,) | Single key frame |
| **Output** | `MatchResult` | pose_name, similarity, errors |

```python
@dataclass
class MatchResult:
    pose_name: str           # "warrior_2"
    pose_display_name: str   # "Warrior II"
    similarity: float        # 0.0 - 1.0
    top_k_matches: List[Tuple[str, float]]  # [("warrior_2", 0.87), ("warrior_1", 0.72)]
    form_errors: List[FormError]  # [FormError(joint="left_arm", error="too_low", diff=15)]
    feedback: str            # "Nâng tay trái lên thêm 15°"
```

### 6.3 Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                          POSE MATCHING                                │
├──────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  ┌────────────────┐     ┌────────────────┐     ┌────────────────────┐│
│  │ Pose Encoder   │     │ Vector Search  │     │ Form Evaluator     ││
│  │   (Neural)     │────▶│   (FAISS)      │────▶│   (Rule-based)     ││
│  └────────────────┘     └────────────────┘     └────────────────────┘│
│         │                      │                        │             │
│         ▼                      ▼                        ▼             │
│   128-dim embedding      Top-K matches           Form errors         │
│                          + similarities          + feedback          │
│                                                                        │
└──────────────────────────────────────────────────────────────────────┘
```

### 6.4 Pose Encoder

#### 6.4.1 MLP Encoder (Simple)

```python
class MLPPoseEncoder(nn.Module):
    """Simple MLP để encode pose thành embedding vector."""
    
    def __init__(self, input_dim: int = 39, embed_dim: int = 128):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.Linear(128, embed_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding = self.encoder(x)
        # L2 normalize để dùng cosine similarity
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding
```

#### 6.4.2 Graph Neural Network Encoder (Advanced)

```python
class GNNPoseEncoder(nn.Module):
    """
    Dùng Graph Neural Network để encode skeleton.
    Skeleton được biểu diễn như graph: joints = nodes, bones = edges.
    """
    
    # Adjacency matrix cho skeleton
    SKELETON_EDGES = [
        (0, 1), (0, 2),      # nose - shoulders
        (1, 3), (2, 4),      # shoulders - elbows
        (3, 5), (4, 6),      # elbows - wrists
        (1, 7), (2, 8),      # shoulders - hips
        (7, 8),              # hip - hip
        (7, 9), (8, 10),     # hips - knees
        (9, 11), (10, 12),   # knees - ankles
    ]
    
    def __init__(self, node_features: int = 3, hidden_dim: int = 64, embed_dim: int = 128):
        super().__init__()
        
        self.conv1 = GCNConv(node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        self.fc = nn.Linear(hidden_dim * 13, embed_dim)  # 13 joints
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # x shape: (batch * num_nodes, node_features)
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        
        # Reshape và flatten
        x = x.view(-1, 13 * 64)  # (batch, 13 * hidden_dim)
        
        embedding = self.fc(x)
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding
```

### 6.5 Vector Search

```python
import faiss

class PoseVectorSearch:
    """Tìm kiếm pose tương tự sử dụng FAISS."""
    
    def __init__(self, embed_dim: int = 128):
        self.embed_dim = embed_dim
        self.index = faiss.IndexFlatIP(embed_dim)  # Inner Product (cosine similarity)
        self.pose_names = []
        self.pose_metadata = {}
    
    def add_pose(self, pose_name: str, embedding: np.ndarray, metadata: dict):
        """Thêm pose vào database."""
        self.index.add(embedding.reshape(1, -1).astype('float32'))
        self.pose_names.append(pose_name)
        self.pose_metadata[pose_name] = metadata
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        """Tìm k poses tương tự nhất."""
        query = query_embedding.reshape(1, -1).astype('float32')
        similarities, indices = self.index.search(query, k)
        
        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx < len(self.pose_names):
                results.append((self.pose_names[idx], float(sim)))
        
        return results
    
    def get_metadata(self, pose_name: str) -> dict:
        """Lấy metadata của pose (angles, landmarks, etc.)"""
        return self.pose_metadata.get(pose_name, {})
```

### 6.6 Form Evaluator

```python
class FormEvaluator:
    """Đánh giá form dựa trên pose template."""
    
    # Định nghĩa các góc quan trọng cho từng loại pose
    ANGLE_DEFINITIONS = {
        'left_elbow': ('left_shoulder', 'left_elbow', 'left_wrist'),
        'right_elbow': ('right_shoulder', 'right_elbow', 'right_wrist'),
        'left_shoulder': ('left_elbow', 'left_shoulder', 'left_hip'),
        'right_shoulder': ('right_elbow', 'right_shoulder', 'right_hip'),
        'left_hip': ('left_shoulder', 'left_hip', 'left_knee'),
        'right_hip': ('right_shoulder', 'right_hip', 'right_knee'),
        'left_knee': ('left_hip', 'left_knee', 'left_ankle'),
        'right_knee': ('right_hip', 'right_knee', 'right_ankle'),
    }
    
    def __init__(self, angle_tolerance: float = 15.0):
        self.tolerance = angle_tolerance
    
    def evaluate(self, current_landmarks: Dict, template_metadata: Dict) -> List[FormError]:
        """So sánh current pose với template và tìm errors."""
        errors = []
        
        template_angles = template_metadata.get('angles', {})
        
        for angle_name, joints in self.ANGLE_DEFINITIONS.items():
            if angle_name not in template_angles:
                continue
            
            current_angle = self._calculate_angle(current_landmarks, joints)
            ideal_angle = template_angles[angle_name]
            diff = current_angle - ideal_angle
            
            if abs(diff) > self.tolerance:
                error_type = "too_high" if diff > 0 else "too_low"
                errors.append(FormError(
                    joint=angle_name,
                    error_type=error_type,
                    current_value=current_angle,
                    ideal_value=ideal_angle,
                    difference=diff
                ))
        
        return errors
    
    def _calculate_angle(self, landmarks: Dict, joints: Tuple[str, str, str]) -> float:
        """Tính góc tại joint giữa, tạo bởi 3 điểm."""
        a = np.array([landmarks[joints[0]].x, landmarks[joints[0]].y])
        b = np.array([landmarks[joints[1]].x, landmarks[joints[1]].y])
        c = np.array([landmarks[joints[2]].x, landmarks[joints[2]].y])
        
        ba = a - b
        bc = c - b
        
        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
        
        return angle
    
    def generate_feedback(self, errors: List[FormError]) -> str:
        """Tạo feedback text từ errors."""
        if not errors:
            return "✓ Tư thế hoàn hảo!"
        
        feedback_parts = []
        for error in errors[:3]:  # Top 3 errors
            joint_display = error.joint.replace('_', ' ').title()
            direction = "lên" if error.error_type == "too_low" else "xuống"
            feedback_parts.append(f"{joint_display} cần điều chỉnh {direction} {abs(error.difference):.0f}°")
        
        return " | ".join(feedback_parts)
```

### 6.7 Complete Pose Matcher

```python
class PoseMatcher:
    """Kết hợp tất cả components để match pose."""
    
    def __init__(self, encoder: nn.Module, vector_search: PoseVectorSearch, evaluator: FormEvaluator):
        self.encoder = encoder
        self.vector_search = vector_search
        self.evaluator = evaluator
        self.encoder.eval()
    
    @torch.no_grad()
    def match(self, normalized_pose: np.ndarray, raw_landmarks: Dict) -> MatchResult:
        # 1. Encode pose
        pose_tensor = torch.FloatTensor(normalized_pose).unsqueeze(0)
        embedding = self.encoder(pose_tensor).numpy()
        
        # 2. Search similar poses
        top_k = self.vector_search.search(embedding, k=5)
        
        if not top_k:
            return MatchResult(
                pose_name="unknown",
                pose_display_name="Unknown",
                similarity=0.0,
                top_k_matches=[],
                form_errors=[],
                feedback="Không nhận diện được tư thế"
            )
        
        # 3. Best match
        best_pose, best_similarity = top_k[0]
        pose_metadata = self.vector_search.get_metadata(best_pose)
        
        # 4. Evaluate form
        form_errors = self.evaluator.evaluate(raw_landmarks, pose_metadata)
        feedback = self.evaluator.generate_feedback(form_errors)
        
        return MatchResult(
            pose_name=best_pose,
            pose_display_name=pose_metadata.get('display_name', best_pose),
            similarity=best_similarity,
            top_k_matches=top_k,
            form_errors=form_errors,
            feedback=feedback
        )
```

---

## 7. Pose Database Design

### 7.1 Database Structure

```yaml
# pose_database.yaml
poses:
  warrior_1:
    display_name: "Warrior I (Virabhadrasana I)"
    category: "standing"
    difficulty: "beginner"
    embedding: [0.12, -0.34, ...]  # 128-dim, pre-computed
    reference_landmarks:
      nose: [0.5, 0.2, 0.0]
      left_shoulder: [0.4, 0.35, 0.0]
      # ... all 13 keypoints
    angles:
      left_knee: 90
      right_knee: 170
      left_hip: 120
      right_hip: 160
      left_shoulder: 180  # Arms up
      right_shoulder: 180
    common_errors:
      - id: "front_knee_over_ankle"
        description: "Đầu gối trước vượt quá mắt cá chân"
        severity: "high"
      - id: "hips_not_square"
        description: "Hông không vuông góc với hướng nhìn"
        severity: "medium"
  
  warrior_2:
    display_name: "Warrior II (Virabhadrasana II)"
    category: "standing"
    # ...
  
  tree_pose:
    display_name: "Tree Pose (Vrksasana)"
    category: "balancing"
    # ...
```

### 7.2 Multi-View Templates

```python
# Mỗi pose có thể có nhiều templates cho các góc nhìn khác nhau
pose_templates = {
    "warrior_2": {
        "front": {
            "embedding": [...],
            "angles": {...}
        },
        "side_left": {
            "embedding": [...],
            "angles": {...}
        },
        "side_right": {
            "embedding": [...],
            "angles": {...}
        }
    }
}
```

### 7.3 Database Management

```python
class PoseDatabase:
    def __init__(self, db_path: str = "pose_database.yaml"):
        self.db_path = db_path
        self.poses = self._load_database()
    
    def _load_database(self) -> Dict:
        with open(self.db_path, 'r') as f:
            data = yaml.safe_load(f)
        return data.get('poses', {})
    
    def add_pose(self, pose_name: str, pose_data: Dict):
        """Thêm pose mới vào database."""
        self.poses[pose_name] = pose_data
        self._save_database()
    
    def update_embedding(self, pose_name: str, embedding: np.ndarray):
        """Cập nhật embedding cho pose (sau khi retrain encoder)."""
        if pose_name in self.poses:
            self.poses[pose_name]['embedding'] = embedding.tolist()
            self._save_database()
    
    def get_all_embeddings(self) -> Tuple[List[str], np.ndarray]:
        """Lấy tất cả embeddings để build vector index."""
        names = []
        embeddings = []
        for name, data in self.poses.items():
            if 'embedding' in data:
                names.append(name)
                embeddings.append(data['embedding'])
        return names, np.array(embeddings)
    
    def _save_database(self):
        with open(self.db_path, 'w') as f:
            yaml.dump({'poses': self.poses}, f)
```

---

## 8. Training Strategy

### 8.1 Data Collection

#### 8.1.1 Dataset Requirements

| Component | # Samples Needed | Description |
|-----------|------------------|-------------|
| Frame Classifier | ~2000-5000 sequences | Labeled KEY_POSE/TRANSITION |
| Pose Encoder | ~5000-10000 poses | Multiple people, angles, variations |

#### 8.1.2 Data Sources

1. **Public Datasets**:
   - Yoga-82 Dataset (82 yoga poses)
   - MPII Human Pose Dataset
   - COCO Keypoints

2. **Synthetic Data**:
   - Augmentation từ existing poses
   - Unity/Blender generated

3. **Self-collected**:
   - Record videos của chính mình hoặc volunteers

### 8.2 Training Frame Classifier

```python
def train_frame_classifier(model, train_loader, val_loader, epochs=50):
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            sequences, labels = batch  # (B, seq_len, 39), (B,)
            
            optimizer.zero_grad()
            logits = model(sequences)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_acc = evaluate(model, val_loader)
        print(f"Epoch {epoch+1}: Val Acc = {val_acc:.2%}")
```

### 8.3 Training Pose Encoder

#### 8.3.1 Triplet Loss Training

```python
def train_pose_encoder_triplet(model, train_loader, epochs=100, margin=0.5):
    optimizer = AdamW(model.parameters(), lr=1e-3)
    triplet_loss = nn.TripletMarginLoss(margin=margin)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            anchor, positive, negative = batch  # Same pose, same pose diff person, diff pose
            
            optimizer.zero_grad()
            
            anchor_embed = model(anchor)
            positive_embed = model(positive)
            negative_embed = model(negative)
            
            loss = triplet_loss(anchor_embed, positive_embed, negative_embed)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}: Loss = {total_loss / len(train_loader):.4f}")
```

#### 8.3.2 Contrastive Learning (SimCLR-style)

```python
def train_pose_encoder_contrastive(model, train_loader, epochs=100, temperature=0.07):
    optimizer = AdamW(model.parameters(), lr=1e-3)
    
    for epoch in range(epochs):
        model.train()
        
        for batch in train_loader:
            poses, labels = batch  # poses: (B, 39), labels: pose class IDs
            
            optimizer.zero_grad()
            
            embeddings = model(poses)  # (B, 128)
            
            # Supervised contrastive loss
            loss = supervised_contrastive_loss(embeddings, labels, temperature)
            loss.backward()
            optimizer.step()


def supervised_contrastive_loss(embeddings, labels, temperature=0.07):
    """
    Pull together embeddings of same class, push apart different classes.
    """
    # Similarity matrix
    sim_matrix = torch.mm(embeddings, embeddings.T) / temperature  # (B, B)
    
    # Mask for positive pairs (same label)
    labels = labels.unsqueeze(1)
    positive_mask = (labels == labels.T).float()
    positive_mask.fill_diagonal_(0)  # Exclude self
    
    # Contrastive loss
    exp_sim = torch.exp(sim_matrix)
    log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
    
    # Average over positive pairs
    mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / (positive_mask.sum(dim=1) + 1e-6)
    loss = -mean_log_prob_pos.mean()
    
    return loss
```

### 8.4 Data Augmentation

```python
class PoseAugmentation:
    """Augmentation cho pose data."""
    
    @staticmethod
    def add_noise(pose: np.ndarray, std: float = 0.01) -> np.ndarray:
        """Thêm Gaussian noise."""
        noise = np.random.normal(0, std, pose.shape)
        return pose + noise
    
    @staticmethod
    def scale(pose: np.ndarray, scale_range: Tuple[float, float] = (0.9, 1.1)) -> np.ndarray:
        """Scale pose."""
        scale = np.random.uniform(*scale_range)
        return pose * scale
    
    @staticmethod
    def rotate_2d(pose: np.ndarray, angle_range: Tuple[float, float] = (-15, 15)) -> np.ndarray:
        """Rotate pose trong 2D plane."""
        angle = np.random.uniform(*angle_range)
        rad = np.radians(angle)
        
        # Reshape to (num_joints, 3)
        pose_reshaped = pose.reshape(-1, 3)
        
        # Rotation matrix cho x, y
        rotation = np.array([
            [np.cos(rad), -np.sin(rad)],
            [np.sin(rad), np.cos(rad)]
        ])
        
        pose_reshaped[:, :2] = pose_reshaped[:, :2] @ rotation.T
        
        return pose_reshaped.flatten()
    
    @staticmethod
    def mirror(pose: np.ndarray) -> np.ndarray:
        """Mirror pose (swap left-right)."""
        pose_reshaped = pose.reshape(-1, 3)
        pose_reshaped[:, 0] = -pose_reshaped[:, 0]  # Flip x
        
        # Swap left-right joints
        swap_pairs = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12)]
        for i, j in swap_pairs:
            pose_reshaped[[i, j]] = pose_reshaped[[j, i]]
        
        return pose_reshaped.flatten()
```

---

## 9. Evaluation Metrics

### 9.1 Frame Classification Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Accuracy | Overall correct predictions | > 90% |
| Precision (KEY_POSE) | Khi predict KEY_POSE, đúng bao nhiêu % | > 85% |
| Recall (KEY_POSE) | Trong tất cả KEY_POSE thực, detect được bao nhiêu % | > 90% |
| F1 Score | Harmonic mean of Precision & Recall | > 87% |

### 9.2 Pose Matching Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Top-1 Accuracy | Pose đúng nằm ở vị trí #1 | > 85% |
| Top-5 Accuracy | Pose đúng nằm trong Top 5 | > 95% |
| Mean Reciprocal Rank (MRR) | Average của 1/rank | > 0.9 |
| Similarity Calibration | Similarity score có reflect thực tế không | Correlation > 0.8 |

### 9.3 Form Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Angle Error (MAE) | Mean Absolute Error của predicted angles vs ground truth |
| Error Detection Rate | Đúng khi phát hiện lỗi form |
| False Positive Rate | Báo lỗi khi không có lỗi |

---

## 10. Challenges & Solutions

### 10.1 View Invariance

**Problem**: Cùng một pose nhưng từ góc nhìn khác trông khác hoàn toàn.

**Solutions**:
| Solution | Complexity | Effectiveness |
|----------|------------|---------------|
| Multi-view templates | Low | Medium |
| 3D pose estimation | High | High |
| View-invariant features | Medium | Medium-High |

### 10.2 Pose Similarity (Warrior I vs Warrior II)

**Problem**: Một số poses rất giống nhau, khó phân biệt.

**Solutions**:
- Hard negative mining trong training
- Hierarchical matching (category → specific pose)
- Thêm temporal context (sequence of poses)

### 10.3 Individual Variations

**Problem**: Mỗi người có body proportions khác nhau.

**Solutions**:
- Normalize pose theo torso length
- Dùng angles thay vì absolute positions
- Fine-tune với data của từng user

### 10.4 Real-time Performance

**Problem**: Cần chạy real-time (>20 FPS).

**Solutions**:
| Component | Optimization |
|-----------|--------------|
| Person Detection | Run every N frames, interpolate |
| Pose Estimation | MediaPipe đã optimized |
| Frame Classification | Lightweight LSTM |
| Pose Matching | FAISS với GPU |

---

## 11. Implementation Roadmap

### Phase 1: MVP (2-3 weeks)

- [ ] Refactor existing codebase
- [ ] Implement Rule-based Frame Classifier
- [ ] Create initial Pose Database (10-20 poses)
- [ ] Implement simple MLP Pose Encoder
- [ ] Basic Form Evaluator

**Deliverable**: Working demo với ~20 yoga poses

### Phase 2: AI Integration (3-4 weeks)

- [ ] Collect/prepare training data
- [ ] Train LSTM Frame Classifier
- [ ] Train Pose Encoder với Triplet Loss
- [ ] Integrate FAISS for vector search
- [ ] Expand database to 50+ poses

**Deliverable**: AI-powered matching với 50+ poses

### Phase 3: Production Ready (2-3 weeks)

- [ ] Optimize for real-time performance
- [ ] Add more poses (100+)
- [ ] Implement multi-view matching
- [ ] Add user customization
- [ ] Testing & bug fixes

**Deliverable**: Production-ready application

### Phase 4: Scale (Ongoing)

- [ ] Continuous data collection
- [ ] Model improvements
- [ ] New pose categories (Pilates, PT exercises, etc.)
- [ ] Mobile deployment

---

## 12. References

### Papers
1. BlazePose: On-device Real-time Body Pose tracking (MediaPipe)
2. ST-GCN: Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition
3. SimCLR: A Simple Framework for Contrastive Learning of Visual Representations
4. Deep Metric Learning: A Survey

### Datasets
1. Yoga-82: A Large-Scale Dataset for Fine-grained Yoga Pose Recognition
2. MPII Human Pose Dataset
3. COCO Keypoints

### Libraries
1. MediaPipe: https://mediapipe.dev/
2. FAISS: https://github.com/facebookresearch/faiss
3. PyTorch Geometric: https://pytorch-geometric.readthedocs.io/

---

> **Document Version**: 1.0  
> **Last Updated**: 2026-01-31  
> **Author**: AI Assistant
