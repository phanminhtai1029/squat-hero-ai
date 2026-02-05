"""
Yoga Pose AI - Real-time Pose Recognition Pipeline
===================================================

4-Step Pipeline (YOLOv8-Pose):
1. Frame Capture - Get frames from webcam/video
2. Pose Estimation - YOLOv8-Pose (person detection + pose in one step)
3. Frame Classification - Classify KEY_POSE vs TRANSITION
4. Pose Matching - Match pose against database

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
from typing import Dict, Any, Optional

# Import configuration
import config

# Import pipeline modules (4-step pipeline)
from pipeline.step1_frame_capture import WebcamCapture, VideoCapture, ImageCapture
# step2_person_detection - DEPRECATED (replaced by YOLOv8-Pose in step3)
from pipeline.step3_pose_estimation import PoseEstimator  # YOLOv8-Pose (detection + pose)
from pipeline.step4_frame_classifier import FrameClassifier, FrameType
from pipeline.step5_pose_matcher import PoseMatcher
from utils.angle_calculator import AngleCalculator
from utils.visualization import draw_status_overlay, draw_pose_result


class YogaPosePipeline:
    """Complete 4-step yoga pose recognition pipeline using YOLOv8-Pose."""
    
    def __init__(
        self,
        database_path: str = None,
        use_yolo: bool = False,  # Deprecated, kept for backward compatibility
        velocity_threshold: float = None
    ):
        """
        Initialize the pipeline.
        
        Args:
            database_path: Path to pose database YAML (default from config)
            use_yolo: Deprecated (YOLOv8-Pose is always used)
            velocity_threshold: Velocity threshold for frame classifier (default from config)
        """
        # Use config defaults if not specified
        database_path = database_path or config.DATABASE_PATH
        velocity_threshold = velocity_threshold or config.VELOCITY_THRESHOLD
        
        print("Initializing Yoga Pose Pipeline (YOLOv8-Pose)...")
        
        # Step 2: YOLOv8-Pose (combines person detection + pose estimation)
        print("  [2/4] Loading YOLOv8-Pose model...")
        self.pose_estimator = PoseEstimator(
            model_path=config.YOLOV8_POSE_MODEL,
            confidence_threshold=config.YOLOV8_CONFIDENCE,
            device=config.YOLOV8_DEVICE
        )
        
        # Legacy: Person Detector (deprecated, YOLOv8-Pose does this)
        self.person_detector = None
        
        # Step 4: Frame Classifier
        print("  [3/4] Initializing Frame Classifier...")
        self.frame_classifier = FrameClassifier(
            velocity_threshold=velocity_threshold,
            window_size=config.WINDOW_SIZE,
            stability_frames=config.STABILITY_FRAMES
        )
        
        # Step 5: Pose Matcher
        print("  [4/4] Loading Pose Database...")
        self.pose_matcher = PoseMatcher(
            database_path=database_path,
            method=config.MATCH_METHOD
        )
        print(f"        Loaded {len(self.pose_matcher.database)} poses")
        
        print("Pipeline ready!\n")
    
    def process_frame(self, frame) -> Dict[str, Any]:
        """
        Process a single frame through the full pipeline.
        
        Returns:
            dict with: frame_type, pose_name, similarity, confidence, annotated_frame
        """
        result = {
            'frame_type': 'NO_POSE',
            'pose_name': None,
            'similarity': 0.0,
            'confidence': None,
            'angles': None,
            'annotated_frame': frame.copy()
        }
        
        # Step 2: YOLOv8-Pose (person detection + pose estimation in one step)
        pose_result = self.pose_estimator.estimate(frame)
        if pose_result is None:
            return result
        
        # Draw pose on frame
        result['annotated_frame'] = self.pose_estimator.draw_pose(
            result['annotated_frame'], 
            pose_result,
            draw_bbox=True
        )
        
        # Step 3: Frame Classification
        classification = self.frame_classifier.classify_from_pose_result(pose_result)
        result['frame_type'] = classification.frame_type.value
        
        # Step 4: Pose Matching (only for KEY_POSE frames)
        if classification.frame_type == FrameType.KEY_POSE:
            match_result = self.pose_matcher.match_from_pose_result(pose_result)
            if match_result:
                result['pose_name'] = match_result.display_name
                result['similarity'] = match_result.similarity
                result['confidence'] = match_result.confidence
        
        # Get angles for display (optional)
        angle_result = AngleCalculator.from_pose_result(pose_result)
        if angle_result:
            result['angles'] = angle_result.angles_degrees
        
        return result
    
    def run_webcam(self, camera_id: int = None):
        """Run pipeline on webcam feed."""
        camera_id = camera_id if camera_id is not None else config.CAMERA_ID
        
        print(f"Starting webcam (ID: {camera_id})...")
        print("Press Q to quit, R to reset\n")
        
        capture = WebcamCapture(
            camera_id=camera_id,
            width=config.CAMERA_WIDTH,
            height=config.CAMERA_HEIGHT
        )
        
        self._run_capture_loop(capture)
    
    def run_video(self, video_path: str):
        """Run pipeline on video file."""
        print(f"Processing video: {video_path}")
        print("Press Q to quit, R to reset\n")
        
        capture = VideoCapture(video_path)
        self._run_capture_loop(capture)
    
    def run_image(self, image_path: str) -> Dict[str, Any]:
        """Process a single image and return result."""
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Cannot read image {image_path}")
            return None
        
        result = self.process_frame(frame)
        return result
    
    def _run_capture_loop(self, capture):
        """Main capture and processing loop."""
        fps_start = time.time()
        frame_count = 0
        fps = 0.0
        
        while capture.is_opened():
            ret, frame = capture.read()
            if not ret:
                break
            
            # Process frame
            result = self.process_frame(frame)
            
            # Draw overlays
            annotated = result['annotated_frame']
            annotated = draw_status_overlay(annotated, result['frame_type'], fps)
            
            if result['pose_name']:
                annotated = draw_pose_result(
                    annotated,
                    result['pose_name'],
                    result['similarity'],
                    result['confidence']
                )
            
            # Show frame
            cv2.imshow(config.WINDOW_NAME, annotated)
            
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
        
        capture.release()
        cv2.destroyAllWindows()
    
    def close(self):
        """Release resources."""
        self.pose_estimator.close()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Yoga Pose AI - Real-time Pose Recognition',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input source
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument('--video', type=str, help='Path to video file')
    input_group.add_argument('--image', type=str, help='Path to image file')
    
    # Camera settings
    parser.add_argument('--camera', type=int, default=config.CAMERA_ID,
                        help='Camera ID for webcam mode')
    
    # Pipeline settings
    parser.add_argument('--database', type=str, default=config.DATABASE_PATH,
                        help='Path to pose database YAML')
    parser.add_argument('--use-yolo', action='store_true',
                        help='Use YOLO for person detection')
    parser.add_argument('--velocity-threshold', type=float, 
                        default=config.VELOCITY_THRESHOLD,
                        help='Velocity threshold for KEY_POSE detection')
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Initialize pipeline
    pipeline = YogaPosePipeline(
        database_path=args.database,
        use_yolo=args.use_yolo,
        velocity_threshold=args.velocity_threshold
    )
    
    try:
        if args.image:
            # Process single image
            result = pipeline.run_image(args.image)
            if result:
                print(f"\nResult: {result['pose_name']} "
                      f"({result['similarity']*100:.1f}% - {result['confidence']})")
                print("\n📸 Displaying result...")
                print("   Press ANY KEY to close the window")
                cv2.imshow('Result', result['annotated_frame'])
                
                # Save output image
                output_path = 'output_result.jpg'
                cv2.imwrite(output_path, result['annotated_frame'])
                print(f"   Saved result to: {output_path}")
                
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        
        elif args.video:
            # Process video file
            pipeline.run_video(args.video)
        
        else:
            # Run webcam
            pipeline.run_webcam(args.camera)
    
    finally:
        pipeline.close()


if __name__ == '__main__':
    main()
