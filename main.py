"""
Yoga Pose AI - Real-time Pose Recognition Pipeline
===================================================

5-Step Rule-Based Pipeline:
1. Frame Capture - Get frames from webcam/video
2. Person Detection - Detect person using YOLO
3. Pose Estimation - Estimate pose using MediaPipe
4. Frame Classification - Classify KEY_POSE vs TRANSITION
5. Pose Matching - Match pose against database

Usage:
    python main.py                     # Webcam
    python main.py --video path.mp4    # Video file
    python main.py --image path.jpg    # Single image
    
Controls:
    Q - Quit
    R - Reset classifier state
"""

import cv2
import time
import argparse
from pathlib import Path

# Import pipeline modules
from pipeline.step1_frame_capture import WebcamCapture, VideoCapture, ImageCapture
from pipeline.step2_person_detection import PersonDetector
from pipeline.step3_pose_estimation import PoseEstimator
from pipeline.step4_frame_classifier import FrameClassifier, FrameType
from pipeline.step5_pose_matcher import PoseMatcher
from utils.angle_calculator import AngleCalculator


class YogaPosePipeline:
    """Complete 5-step yoga pose recognition pipeline."""
    
    def __init__(
        self,
        database_path: str = "data/pose_database.yaml",
        use_yolo: bool = False,
        velocity_threshold: float = 0.015
    ):
        print("Initializing Yoga Pose Pipeline...")
        
        # Step 2: Person Detector (optional)
        if use_yolo:
            print("  [2/5] Loading YOLO model...")
            self.person_detector = PersonDetector()
        else:
            self.person_detector = None
        
        # Step 3: Pose Estimator
        print("  [3/5] Initializing MediaPipe Pose...")
        self.pose_estimator = PoseEstimator()
        
        # Step 4: Frame Classifier
        print("  [4/5] Initializing Frame Classifier...")
        self.frame_classifier = FrameClassifier(velocity_threshold=velocity_threshold)
        
        # Step 5: Pose Matcher
        print("  [5/5] Loading Pose Database...")
        self.pose_matcher = PoseMatcher(database_path)
        print(f"        Loaded {len(self.pose_matcher.database)} poses")
        
        print("Pipeline ready!")
    
    def process_frame(self, frame):
        """
        Process a single frame through the full pipeline.
        
        Returns:
            dict with: frame_type, pose_name, similarity, annotated_frame
        """
        result = {
            'frame_type': None,
            'pose_name': None,
            'similarity': 0.0,
            'confidence': None,
            'angles': None,
            'annotated_frame': frame.copy()
        }
        
        # Step 2: Person Detection
        if self.person_detector:
            detection = self.person_detector.detect_main_person(frame)
            person_image = detection.cropped_image
        else:
            person_image = frame
        
        # Step 3: Pose Estimation
        pose_result = self.pose_estimator.estimate(person_image)
        if pose_result is None:
            result['frame_type'] = 'NO_POSE'
            return result
        
        # Draw pose on frame
        result['annotated_frame'] = self.pose_estimator.draw_pose(
            result['annotated_frame'], 
            pose_result
        )
        
        # Step 4: Frame Classification
        classification = self.frame_classifier.classify_from_pose_result(pose_result)
        result['frame_type'] = classification.frame_type.value
        
        # Step 5: Pose Matching (only for KEY_POSE frames)
        if classification.frame_type == FrameType.KEY_POSE:
            match_result = self.pose_matcher.match_from_pose_result(pose_result)
            if match_result:
                result['pose_name'] = match_result.display_name
                result['similarity'] = match_result.similarity
                result['confidence'] = match_result.confidence
        
        # Get angles for display
        angle_result = AngleCalculator.from_pose_result(pose_result)
        if angle_result:
            result['angles'] = angle_result.angles_degrees
        
        return result
    
    def run_webcam(self, camera_id: int = 0):
        """Run pipeline on webcam feed."""
        print(f"\nStarting webcam (ID: {camera_id})...")
        print("Press Q to quit, R to reset\n")
        
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        fps_start = time.time()
        frame_count = 0
        fps = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            result = self.process_frame(frame)
            
            # Draw results
            annotated = result['annotated_frame']
            self._draw_overlay(annotated, result, fps)
            
            # Show frame
            cv2.imshow('Yoga Pose AI', annotated)
            
            # Calculate FPS
            frame_count += 1
            if frame_count % 30 == 0:
                fps = 30 / (time.time() - fps_start)
                fps_start = time.time()
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.frame_classifier.reset()
                print("Classifier reset")
        
        cap.release()
        cv2.destroyAllWindows()
    
    def _draw_overlay(self, frame, result, fps):
        """Draw status overlay on frame."""
        h, w = frame.shape[:2]
        
        # Draw FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw frame type
        frame_type = result.get('frame_type', 'UNKNOWN')
        color = (0, 255, 0) if frame_type == 'KEY_POSE' else (0, 165, 255)
        cv2.putText(frame, f"State: {frame_type}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw pose result (if KEY_POSE)
        if result.get('pose_name'):
            pose_name = result['pose_name']
            similarity = result['similarity'] * 100
            confidence = result.get('confidence', 'unknown')
            
            # Background box
            cv2.rectangle(frame, (5, h-80), (w-5, h-5), (0, 0, 0), -1)
            cv2.rectangle(frame, (5, h-80), (w-5, h-5), (0, 255, 0), 2)
            
            # Pose name
            cv2.putText(frame, pose_name, (15, h-50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Similarity
            cv2.putText(frame, f"Match: {similarity:.1f}% ({confidence})", (15, h-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    def close(self):
        """Release resources."""
        self.pose_estimator.close()


def main():
    parser = argparse.ArgumentParser(description='Yoga Pose Recognition')
    parser.add_argument('--video', type=str, help='Path to video file')
    parser.add_argument('--image', type=str, help='Path to image file')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID')
    parser.add_argument('--database', type=str, default='data/pose_database.yaml')
    parser.add_argument('--use-yolo', action='store_true', help='Use YOLO for person detection')
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = YogaPosePipeline(
        database_path=args.database,
        use_yolo=args.use_yolo
    )
    
    try:
        if args.image:
            # Process single image
            frame = cv2.imread(args.image)
            result = pipeline.process_frame(frame)
            print(f"\nResult: {result['pose_name']} ({result['similarity']*100:.1f}%)")
            cv2.imshow('Result', result['annotated_frame'])
            cv2.waitKey(0)
        elif args.video:
            # Process video (similar to webcam)
            print("Video mode not fully implemented yet")
        else:
            # Run webcam
            pipeline.run_webcam(args.camera)
    finally:
        pipeline.close()


if __name__ == '__main__':
    main()
