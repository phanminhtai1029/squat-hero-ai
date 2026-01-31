"""
Step 3: Pose Estimation
Estimates body pose using MediaPipe Tasks API (v0.10.x+)
"""

import cv2
import numpy as np
from typing import Optional, List, Dict
from dataclasses import dataclass, field


@dataclass
class Landmark:
    """Single pose landmark."""
    x: float  # Normalized [0, 1]
    y: float  # Normalized [0, 1]
    z: float  # Depth
    visibility: float


@dataclass
class PoseResult:
    """Pose estimation result."""
    landmarks: List[Landmark]
    landmark_dict: Dict[str, Landmark] = field(default_factory=dict)
    raw_landmarks: Optional[object] = None
    
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
    """Estimate pose using MediaPipe Tasks API."""
    
    # MediaPipe landmark names (33 landmarks)
    LANDMARK_NAMES = [
        'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
        'right_eye_inner', 'right_eye', 'right_eye_outer',
        'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
        'left_index', 'right_index', 'left_thumb', 'right_thumb',
        'left_hip', 'right_hip', 'left_knee', 'right_knee',
        'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
        'left_foot_index', 'right_foot_index'
    ]
    
    # Essential keypoints for yoga (13 keypoints)
    ESSENTIAL_KEYPOINTS = [
        'left_shoulder', 'right_shoulder',
        'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist',
        'left_hip', 'right_hip',
        'left_knee', 'right_knee',
        'left_ankle', 'right_ankle',
        'nose'
    ]
    
    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ):
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.pose_landmarker = None
        self._use_legacy = False
        self._init_mediapipe()
    
    def _init_mediapipe(self) -> None:
        """Initialize MediaPipe pose."""
        try:
            # Try new Tasks API first (mediapipe >= 0.10.0)
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            import urllib.request
            import os
            
            # Download model if not exists
            model_path = "pose_landmarker.task"
            if not os.path.exists(model_path):
                print("Downloading pose landmarker model...")
                url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
                urllib.request.urlretrieve(url, model_path)
            
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                output_segmentation_masks=False,
                running_mode=vision.RunningMode.IMAGE
            )
            self.pose_landmarker = vision.PoseLandmarker.create_from_options(options)
            self._use_legacy = False
            print("Using MediaPipe Tasks API")
            
        except Exception as e:
            print(f"Tasks API failed ({e}), trying legacy API...")
            try:
                # Fall back to legacy API
                import mediapipe as mp
                self.mp_pose = mp.solutions.pose
                self.mp_drawing = mp.solutions.drawing_utils
                self.pose_landmarker = self.mp_pose.Pose(
                    static_image_mode=self.static_image_mode,
                    model_complexity=self.model_complexity,
                    min_detection_confidence=self.min_detection_confidence,
                    min_tracking_confidence=self.min_tracking_confidence
                )
                self._use_legacy = True
                print("Using MediaPipe Legacy API")
            except Exception as e2:
                print(f"Warning: MediaPipe not available ({e2})")
                self.pose_landmarker = None
    
    def estimate(self, image: np.ndarray) -> Optional[PoseResult]:
        """
        Estimate pose from image.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            PoseResult or None if no pose detected
        """
        if self.pose_landmarker is None:
            return None
        
        if self._use_legacy:
            return self._estimate_legacy(image)
        else:
            return self._estimate_tasks(image)
    
    def _estimate_tasks(self, image: np.ndarray) -> Optional[PoseResult]:
        """Estimate using Tasks API."""
        from mediapipe.tasks.python import vision
        import mediapipe as mp
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create mediapipe image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        # Detect pose
        detection_result = self.pose_landmarker.detect(mp_image)
        
        if not detection_result.pose_landmarks or len(detection_result.pose_landmarks) == 0:
            return None
        
        # Get first person's landmarks
        pose_landmarks = detection_result.pose_landmarks[0]
        
        # Convert to Landmark objects
        landmarks = []
        landmark_dict = {}
        
        for i, lm in enumerate(pose_landmarks):
            landmark = Landmark(
                x=lm.x,
                y=lm.y,
                z=lm.z,
                visibility=lm.visibility if hasattr(lm, 'visibility') else 1.0
            )
            landmarks.append(landmark)
            
            if i < len(self.LANDMARK_NAMES):
                landmark_dict[self.LANDMARK_NAMES[i]] = landmark
        
        return PoseResult(
            landmarks=landmarks,
            landmark_dict=landmark_dict,
            raw_landmarks=pose_landmarks
        )
    
    def _estimate_legacy(self, image: np.ndarray) -> Optional[PoseResult]:
        """Estimate using legacy API."""
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process image
        results = self.pose_landmarker.process(image_rgb)
        
        if not results.pose_landmarks:
            return None
        
        # Convert to Landmark objects
        landmarks = []
        landmark_dict = {}
        
        for i, lm in enumerate(results.pose_landmarks.landmark):
            landmark = Landmark(
                x=lm.x,
                y=lm.y,
                z=lm.z,
                visibility=lm.visibility
            )
            landmarks.append(landmark)
            
            if i < len(self.LANDMARK_NAMES):
                landmark_dict[self.LANDMARK_NAMES[i]] = landmark
        
        return PoseResult(
            landmarks=landmarks,
            landmark_dict=landmark_dict,
            raw_landmarks=results.pose_landmarks
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
        draw_connections: bool = True
    ) -> np.ndarray:
        """Draw pose landmarks on image."""
        if pose_result is None:
            return image
        
        annotated = image.copy()
        h, w = image.shape[:2]
        
        # Draw landmarks
        for lm in pose_result.landmarks:
            x = int(lm.x * w)
            y = int(lm.y * h)
            cv2.circle(annotated, (x, y), 3, (0, 255, 0), -1)
        
        # Draw connections
        if draw_connections:
            connections = [
                (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
                (11, 23), (12, 24), (23, 24),
                (23, 25), (25, 27), (24, 26), (26, 28)
            ]
            for i, j in connections:
                if i < len(pose_result.landmarks) and j < len(pose_result.landmarks):
                    lm1 = pose_result.landmarks[i]
                    lm2 = pose_result.landmarks[j]
                    x1, y1 = int(lm1.x * w), int(lm1.y * h)
                    x2, y2 = int(lm2.x * w), int(lm2.y * h)
                    cv2.line(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        return annotated
    
    def close(self) -> None:
        """Release resources."""
        if self.pose_landmarker and not self._use_legacy:
            self.pose_landmarker.close()
        elif self.pose_landmarker and self._use_legacy:
            self.pose_landmarker.close()
