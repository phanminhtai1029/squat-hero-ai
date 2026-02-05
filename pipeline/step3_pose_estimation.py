"""
Step 3: Pose Estimation
Estimates body pose using YOLOv8-Pose (replaces MediaPipe + Person Detection)
"""

import cv2
import numpy as np
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field


@dataclass
class Landmark:
    """Single pose landmark."""
    x: float  # Normalized [0, 1]
    y: float  # Normalized [0, 1]
    z: float  # Depth (not used in COCO keypoints, set to 0)
    visibility: float  # Confidence in YOLOv8


@dataclass
class PoseResult:
    """Pose estimation result."""
    landmarks: List[Landmark]
    landmark_dict: Dict[str, Landmark] = field(default_factory=dict)
    raw_landmarks: Optional[object] = None
    bbox: Optional[Tuple[int, int, int, int]] = None  # x1, y1, x2, y2
    detection_confidence: float = 0.0
    
    def to_numpy(self) -> np.ndarray:
        """Convert landmarks to numpy array (N, 4)."""
        return np.array([
            [lm.x, lm.y, lm.z, lm.visibility] 
            for lm in self.landmarks
        ])
    
    def get_landmark(self, name: str) -> Optional[Landmark]:
        """Get landmark by name."""
        return self.landmark_dict.get(name)


class PoseEstimator:
    """Estimate pose using YOLOv8-Pose (combines person detection + pose estimation)."""
    
    # COCO keypoint names (17 keypoints)
    LANDMARK_NAMES = [
        'nose',           # 0
        'left_eye',       # 1
        'right_eye',      # 2
        'left_ear',       # 3
        'right_ear',      # 4
        'left_shoulder',  # 5
        'right_shoulder', # 6
        'left_elbow',     # 7
        'right_elbow',    # 8
        'left_wrist',     # 9
        'right_wrist',    # 10
        'left_hip',       # 11
        'right_hip',      # 12
        'left_knee',      # 13
        'right_knee',     # 14
        'left_ankle',     # 15
        'right_ankle'     # 16
    ]
    
    # Essential keypoints for yoga (only body keypoints, no face details)
    ESSENTIAL_KEYPOINTS = [
        'nose',           # Reference point only
        'left_shoulder', 'right_shoulder',
        'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist',
        'left_hip', 'right_hip',
        'left_knee', 'right_knee',
        'left_ankle', 'right_ankle'
    ]
    
    # Keypoints to draw (exclude eyes and ears for cleaner visualization)
    DRAW_KEYPOINTS = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    
    def __init__(
        self,
        model_path: str = "yolov8s-pose.pt",
        confidence_threshold: float = 0.5,
        device: str = None
    ):
        """
        Initialize YOLOv8-Pose estimator.
        
        Args:
            model_path: Path to YOLOv8-Pose model (nano/small/medium)
            confidence_threshold: Minimum confidence for detection
            device: 'cuda' or 'cpu' (auto-detect if None)
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.model = None
        self._init_yolo()
    
    def _init_yolo(self) -> None:
        """Initialize YOLOv8-Pose model."""
        try:
            from ultralytics import YOLO
            import torch
            
            # Auto-detect device
            if self.device is None:
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            self.model = YOLO(self.model_path)
            print(f"Using YOLOv8-Pose ({self.model_path}) on {self.device.upper()}")
            print(f"  Model: {self.model_path}")
            print(f"  Device: {self.device}")
            print(f"  Keypoints: {len(self.LANDMARK_NAMES)} (COCO format)")
            
        except ImportError:
            print("ERROR: ultralytics not installed. Run: pip install ultralytics")
            self.model = None
        except Exception as e:
            print(f"ERROR: Failed to load YOLOv8-Pose model: {e}")
            self.model = None
    
    def estimate(self, image: np.ndarray) -> Optional[PoseResult]:
        """
        Detect person and estimate pose in one step using YOLOv8-Pose.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            PoseResult or None if no person detected
        """
        if self.model is None:
            return None
        
        # Run YOLOv8-Pose inference
        results = self.model(image, verbose=False, device=self.device)
        
        # Get first person with highest confidence
        best_result = None
        best_conf = 0
        
        for result in results:
            if result.keypoints is None or len(result.keypoints) == 0:
                continue
            
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            
            # Get detection confidence
            conf = float(boxes[0].conf[0])
            
            if conf >= self.confidence_threshold and conf > best_conf:
                best_conf = conf
                best_result = result
        
        if best_result is None:
            return None
        
        # Extract keypoints (shape: [1, 17, 3] where 3 = x, y, conf)
        keypoints = best_result.keypoints.data[0]  # Get first person
        h, w = image.shape[:2]
        
        # Extract bounding box
        box = best_result.boxes[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        
        # Convert to Landmark objects
        landmarks = []
        landmark_dict = {}
        
        for i in range(17):  # COCO has 17 keypoints
            # YOLOv8 returns pixel coordinates, normalize to [0, 1]
            x_norm = float(keypoints[i, 0]) / w
            y_norm = float(keypoints[i, 1]) / h
            conf = float(keypoints[i, 2])
            
            landmark = Landmark(
                x=x_norm,
                y=y_norm,
                z=0.0,  # COCO doesn't have depth
                visibility=conf
            )
            landmarks.append(landmark)
            
            if i < len(self.LANDMARK_NAMES):
                landmark_dict[self.LANDMARK_NAMES[i]] = landmark
        
        return PoseResult(
            landmarks=landmarks,
            landmark_dict=landmark_dict,
            raw_landmarks=keypoints,
            bbox=(x1, y1, x2, y2),
            detection_confidence=best_conf
        )
    
    def get_essential_landmarks(self, pose_result: PoseResult) -> Dict[str, Landmark]:
        """Get only essential landmarks for yoga pose matching."""
        return {
            name: pose_result.landmark_dict[name]
            for name in self.ESSENTIAL_KEYPOINTS
            if name in pose_result.landmark_dict
        }
    
    def draw_pose(
        self, 
        image: np.ndarray, 
        pose_result: PoseResult,
        draw_connections: bool = True,
        draw_bbox: bool = True
    ) -> np.ndarray:
        """Draw pose landmarks and skeleton on image."""
        if pose_result is None:
            return image
        
        annotated = image.copy()
        h, w = image.shape[:2]
        
        # Draw bounding box
        if draw_bbox and pose_result.bbox is not None:
            x1, y1, x2, y2 = pose_result.bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
            # Draw confidence
            conf_text = f"{pose_result.detection_confidence:.2f}"
            cv2.putText(annotated, conf_text, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Draw connections (COCO skeleton)
        if draw_connections:
            # COCO keypoint connections (exclude face connections)
            connections = [
                # (0, 1), (0, 2),  # nose to eyes - SKIP
                # (1, 3), (2, 4),  # eyes to ears - SKIP
                (5, 6),          # shoulders
                (5, 7), (7, 9),  # left arm
                (6, 8), (8, 10), # right arm
                (5, 11), (6, 12), # shoulders to hips
                (11, 12),        # hips
                (11, 13), (13, 15), # left leg
                (12, 14), (14, 16)  # right leg
            ]
            for i, j in connections:
                if i < len(pose_result.landmarks) and j < len(pose_result.landmarks):
                    lm1 = pose_result.landmarks[i]
                    lm2 = pose_result.landmarks[j]
                    
                    # Only draw if both keypoints are visible
                    if lm1.visibility > 0.5 and lm2.visibility > 0.5:
                        x1, y1 = int(lm1.x * w), int(lm1.y * h)
                        x2, y2 = int(lm2.x * w), int(lm2.y * h)
                        cv2.line(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw landmarks (only body keypoints, skip eyes and ears)
        for i in self.DRAW_KEYPOINTS:
            if i < len(pose_result.landmarks):
                lm = pose_result.landmarks[i]
                if lm.visibility > 0.5:  # Only draw visible keypoints
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    # Color based on visibility
                    color = (0, int(255 * lm.visibility), 0)
                    cv2.circle(annotated, (x, y), 4, color, -1)
                    cv2.circle(annotated, (x, y), 4, (255, 255, 255), 1)
        
        return annotated
    
    def close(self) -> None:
        """Release resources."""
        # YOLOv8 handles cleanup automatically
        pass
