"""
Step 2: Person Detection [DEPRECATED]
==================================
⚠️ This module is DEPRECATED and no longer used in the main pipeline.

YOLOv8-Pose in step3_pose_estimation.py now handles both:
  - Person detection (bounding box)
  - Pose estimation (17 keypoints)

This file is kept for backward compatibility with evaluation scripts.

For new code, use: pipeline.step3_pose_estimation.PoseEstimator
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List, NamedTuple
from dataclasses import dataclass


@dataclass
class Detection:
    """Person detection result."""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    cropped_image: Optional[np.ndarray] = None


class PersonDetector:
    """Detect persons using YOLOv8."""
    
    def __init__(
        self, 
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        device: str = "cpu"
    ):
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.model_path = model_path
        self.device = device
        self._load_model()
    
    def _load_model(self) -> None:
        """Load YOLO model."""
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
        except ImportError:
            print("Warning: ultralytics not installed. Person detection disabled.")
            self.model = None
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect persons in frame.
        
        Args:
            frame: Input image (BGR format)
            
        Returns:
            List of Detection objects
        """
        if self.model is None:
            # Return full frame if model not available
            h, w = frame.shape[:2]
            return [Detection(
                bbox=(0, 0, w, h),
                confidence=1.0,
                cropped_image=frame.copy()
            )]
        
        # Run YOLO detection
        results = self.model(frame, verbose=False, device=self.device)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Only keep person class (class 0 in COCO)
                if int(box.cls[0]) != 0:
                    continue
                
                conf = float(box.conf[0])
                if conf < self.confidence_threshold:
                    continue
                
                # Get bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                # Crop image
                cropped = frame[y1:y2, x1:x2].copy()
                
                detections.append(Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    cropped_image=cropped
                ))
        
        # Sort by confidence (highest first)
        detections.sort(key=lambda d: d.confidence, reverse=True)
        
        return detections
    
    def detect_main_person(self, frame: np.ndarray) -> Optional[Detection]:
        """
        Detect the main (largest/most confident) person in frame.
        
        Args:
            frame: Input image
            
        Returns:
            Detection object or None if no person found
        """
        detections = self.detect(frame)
        
        if not detections:
            # Fallback: return full frame
            h, w = frame.shape[:2]
            return Detection(
                bbox=(0, 0, w, h),
                confidence=1.0,
                cropped_image=frame.copy()
            )
        
        return detections[0]
