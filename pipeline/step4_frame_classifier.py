"""
Step 4: Frame Classification
Classifies frames as KEY_POSE or TRANSITION based on movement velocity.
Rule-based approach using velocity threshold.
"""

import numpy as np
from typing import List, Optional, Literal
from dataclasses import dataclass
from enum import Enum


class FrameType(Enum):
    """Frame classification types."""
    KEY_POSE = "KEY_POSE"
    TRANSITION = "TRANSITION"
    UNKNOWN = "UNKNOWN"


@dataclass
class ClassificationResult:
    """Frame classification result."""
    frame_type: FrameType
    velocity: float
    confidence: float
    is_stable: bool


class FrameClassifier:
    """
    Rule-based frame classifier.
    
    Classifies frames as KEY_POSE (stable) or TRANSITION (moving) 
    based on pose landmark velocity over a sliding window.
    
    Uses only essential keypoints for velocity calculation to reduce noise.
    """
    
    # Essential keypoint indices (COCO/YOLOv8-Pose landmark indices)
    # COCO format: 17 keypoints (0-16)
    ESSENTIAL_INDICES = [
        0,      # nose
        5, 6,   # shoulders
        7, 8,   # elbows
        9, 10,  # wrists
        11, 12, # hips
        13, 14, # knees
        15, 16  # ankles
    ]
    
    def __init__(
        self,
        window_size: int = 5,
        velocity_threshold: float = 0.015,
        min_keypoint_visibility: float = 0.5,
        stability_frames: int = 3
    ):
        """
        Initialize frame classifier.
        
        Args:
            window_size: Number of frames to consider for velocity calculation
            velocity_threshold: Movement threshold below which frame is KEY_POSE
            min_keypoint_visibility: Minimum visibility for keypoints to consider
            stability_frames: Number of consecutive stable frames required
        """
        self.window_size = window_size
        self.velocity_threshold = velocity_threshold
        self.min_keypoint_visibility = min_keypoint_visibility
        self.stability_frames = stability_frames
        
        self.history: List[np.ndarray] = []
        self.stable_count = 0
    
    def reset(self) -> None:
        """Reset the classifier state."""
        self.history.clear()
        self.stable_count = 0
    
    def classify(self, landmarks: np.ndarray) -> ClassificationResult:
        """
        Classify current frame based on landmark movement.
        
        Args:
            landmarks: Numpy array of shape (N, 4) where columns are [x, y, z, visibility]
            
        Returns:
            ClassificationResult with frame type and metrics
        """
        # Filter to essential keypoints only (reduces noise from face/hands)
        essential_landmarks = landmarks[self.ESSENTIAL_INDICES]
        
        # Extract positions only (x, y, z)
        positions = essential_landmarks[:, :3].copy()
        
        # Add to history
        self.history.append(positions)
        
        # Keep only window_size frames
        if len(self.history) > self.window_size:
            self.history.pop(0)
        
        # Need at least 2 frames to compute velocity
        if len(self.history) < 2:
            return ClassificationResult(
                frame_type=FrameType.TRANSITION,
                velocity=1.0,
                confidence=0.0,
                is_stable=False
            )
        
        # Compute average velocity
        velocity = self._compute_average_velocity()
        
        # Determine frame type based on velocity
        is_stable = velocity < self.velocity_threshold
        
        if is_stable:
            self.stable_count += 1
        else:
            self.stable_count = 0
        
        # Only classify as KEY_POSE if stable for multiple consecutive frames
        if self.stable_count >= self.stability_frames:
            frame_type = FrameType.KEY_POSE
            confidence = min(1.0, (self.velocity_threshold - velocity) / self.velocity_threshold)
        else:
            frame_type = FrameType.TRANSITION
            confidence = min(1.0, velocity / self.velocity_threshold)
        
        return ClassificationResult(
            frame_type=frame_type,
            velocity=velocity,
            confidence=confidence,
            is_stable=is_stable
        )
    
    def _compute_average_velocity(self) -> float:
        """
        Compute average velocity of landmarks across history window.
        
        Returns:
            Average normalized velocity
        """
        if len(self.history) < 2:
            return 1.0
        
        total_velocity = 0.0
        count = 0
        
        for i in range(1, len(self.history)):
            prev_frame = self.history[i - 1]
            curr_frame = self.history[i]
            
            # Compute per-landmark movement
            diff = np.abs(curr_frame - prev_frame)
            
            # Average movement across all dimensions
            frame_velocity = np.mean(diff)
            total_velocity += frame_velocity
            count += 1
        
        return total_velocity / count if count > 0 else 1.0
    
    def classify_from_pose_result(self, pose_result) -> ClassificationResult:
        """
        Classify from PoseResult object.
        
        Args:
            pose_result: PoseResult from PoseEstimator
            
        Returns:
            ClassificationResult
        """
        if pose_result is None:
            return ClassificationResult(
                frame_type=FrameType.UNKNOWN,
                velocity=0.0,
                confidence=0.0,
                is_stable=False
            )
        
        landmarks = pose_result.to_numpy()
        return self.classify(landmarks)
