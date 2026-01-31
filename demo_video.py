"""
Video Analysis Demo
====================

Analyze pose from video file instead of webcam.

Usage:
    python demo_video.py --video path/to/video.mp4
    python demo_video.py --video path/to/video.mp4 --save output.mp4
    
Controls:
    Q - Quit
    SPACE - Pause/Resume
    R - Reset counter
"""

import cv2
import time
import sys
import argparse
from pathlib import Path

# Import pipeline modules
from step2_person_cropping.yolo_cropper import YoloCropper
from step3_pose_detection.pose_detector import PoseDetector
from step4_frame_classification.frame_classifier import FrameClassifier
from step4_pose_classifier import PoseClassifier
from step5_form_scorer import FormScorer
from utils.visualization import draw_status_panel, draw_fps, draw_angle_indicator
import config


class VideoAnalyzer:
    """Analyze poses from video file."""
    
    def __init__(self):
        print("=" * 50)
        print("Initializing Video Analyzer...")
        print("=" * 50)
        
        # Load models
        print("  [1/5] Loading YOLO model...")
        self.cropper = YoloCropper(
            model_path=config.YOLO_MODEL,
            confidence=config.YOLO_CONFIDENCE
        )
        
        print("  [2/5] Initializing MediaPipe Pose...")
        self.pose_detector = PoseDetector(
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE
        )
        
        print("  [3/5] Initializing Frame Classifier...")
        self.frame_classifier = FrameClassifier()
        
        print("  [4/5] Initializing Pose Classifier...")
        self.pose_classifier = PoseClassifier()
        
        print("  [5/5] Initializing Form Scorer...")
        self.form_scorer = FormScorer()
        
        print("=" * 50)
        print("Ready!")
        print("=" * 50)
    
    def process_frame(self, frame):
        """Process single frame through pipeline."""
        result = None
        display_frame = frame.copy()
        h, w = frame.shape[:2]
        
        # Person cropping
        _, bbox = self.cropper.crop_person(frame)
        
        if bbox is not None:
            cv2.rectangle(display_frame, 
                         (bbox.x1, bbox.y1), (bbox.x2, bbox.y2),
                         (255, 0, 0), 2)
        
        # Pose detection
        landmarks = self.pose_detector.detect_pose(frame)
        
        if landmarks is not None:
            # Draw skeleton
            display_frame = self.pose_detector.draw_pose(frame)
            
            # Redraw bbox
            if bbox is not None:
                cv2.rectangle(display_frame, 
                             (bbox.x1, bbox.y1), (bbox.x2, bbox.y2),
                             (255, 0, 0), 2)
            
            # Frame classification
            frame_class = self.frame_classifier.classify(landmarks)
            is_pose_frame = self.frame_classifier.is_pose_frame(frame_class)
            
            if is_pose_frame:
                # Pose classification
                classification = self.pose_classifier.classify(landmarks)
                # Form scoring
                scoring = self.form_scorer.score(landmarks, classification.label)
                
                result = {
                    'rep_count': scoring.rep_count,
                    'phase': classification.label,
                    'feedback': scoring.feedback,
                    'is_good_form': scoring.is_good_form,
                    'pose_label': classification.label,
                    'score': scoring.score,
                    'confidence': classification.confidence,
                    'knee_angle': frame_class.knee_angle,
                }
            else:
                result = {
                    'rep_count': self.form_scorer.rep_count,
                    'phase': 'transition',
                    'feedback': 'Transitioning...',
                    'is_good_form': True,
                    'pose_label': 'transition',
                    'score': None,
                    'confidence': 0.0,
                    'knee_angle': frame_class.knee_angle,
                }
            
            # Draw knee angle indicator
            if 'left_knee' in landmarks:
                knee_pos = self.pose_detector.get_pixel_coords(
                    landmarks['left_knee'], w, h
                )
                display_frame = draw_angle_indicator(
                    display_frame,
                    result['knee_angle'],
                    (knee_pos[0] + 10, knee_pos[1]),
                    label="Knee",
                    good_threshold=config.SQUAT_ANGLE_THRESHOLD
                )
            
            # Draw ONLY status panel (no other text overlays)
            phase_text = result['phase']
            if result.get('score') is not None:
                phase_text = f"{result['phase']} | Score: {result['score']:.0f}%"
            
            display_frame = draw_status_panel(
                display_frame,
                rep_count=result['rep_count'],
                phase=phase_text,
                feedback=result['feedback'],
                is_good_form=result['is_good_form']
            )
        else:
            cv2.putText(display_frame, "No person detected", 
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 0, 255), 2)
        
        return display_frame, result
    
    def analyze_video(self, video_path: str, output_path: str = None, show: bool = True):
        """
        Analyze video file.
        
        Args:
            video_path: Path to input video
            output_path: Optional path to save output video
            show: Whether to display video while processing
        """
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Cannot open video: {video_path}")
            return
        
        # Video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"\nVideo: {Path(video_path).name}")
        print(f"Resolution: {width}x{height}")
        print(f"FPS: {fps:.1f}")
        print(f"Duration: {duration:.1f}s ({total_frames} frames)")
        print("\nControls: SPACE=Pause, Q=Quit, R=Reset\n")
        
        # Video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"Saving to: {output_path}")
        
        # Processing
        frame_count = 0
        paused = False
        prev_time = time.time()
        processing_fps = 0
        
        while True:
            if not paused:
                ret, frame = cap.read()
                
                if not ret:
                    print("\nEnd of video")
                    break
                
                frame_count += 1
                
                # Process frame
                display_frame, result = self.process_frame(frame)
                
                # Calculate processing FPS
                curr_time = time.time()
                processing_fps = 1.0 / (curr_time - prev_time + 1e-6)
                prev_time = curr_time
                
                # Draw progress and FPS
                progress = frame_count / total_frames * 100
                cv2.putText(display_frame, f"FPS: {processing_fps:.0f}", 
                           (width - 100, 25), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (255, 255, 255), 1)
                cv2.putText(display_frame, f"Progress: {progress:.0f}%", 
                           (width - 130, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (255, 255, 255), 1)
                cv2.putText(display_frame, f"Frame: {frame_count}/{total_frames}", 
                           (width - 150, 75), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (200, 200, 200), 1)
                
                # Save frame
                if writer:
                    writer.write(display_frame)
                
                # Print progress
                if frame_count % 30 == 0:
                    reps = result['rep_count'] if result else 0
                    print(f"  Frame {frame_count}/{total_frames} ({progress:.0f}%) - Reps: {reps}")
            
            # Show
            if show:
                # Draw pause indicator
                if paused:
                    cv2.putText(display_frame, "PAUSED", 
                               (width//2 - 60, height//2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                
                cv2.imshow("Video Analysis", display_frame)
                
                # Playback speed control
                wait_time = int(1000 / fps) if not paused else 100
                key = cv2.waitKey(wait_time) & 0xFF
                
                if key == ord('q') or key == ord('Q'):
                    print("\nStopped by user")
                    break
                elif key == ord(' '):  # Space
                    paused = not paused
                    print("Paused" if paused else "Resumed")
                elif key == ord('r') or key == ord('R'):
                    self.form_scorer.reset()
                    print("Counter reset!")
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
            print(f"\nSaved: {output_path}")
        cv2.destroyAllWindows()
        
        # Summary
        final_reps = self.form_scorer.rep_count
        print(f"\n{'='*50}")
        print(f"ANALYSIS COMPLETE")
        print(f"{'='*50}")
        print(f"Total frames analyzed: {frame_count}")
        print(f"Total reps counted: {final_reps}")
        print(f"{'='*50}")
    
    def cleanup(self):
        """Release resources."""
        self.pose_detector.close()


def main():
    parser = argparse.ArgumentParser(description="Video Pose Analysis")
    parser.add_argument("--video", "-v", required=True,
                       help="Path to input video file")
    parser.add_argument("--save", "-s", default=None,
                       help="Path to save output video (optional)")
    parser.add_argument("--no-display", action="store_true",
                       help="Don't show video while processing")
    
    args = parser.parse_args()
    
    # Check input file
    if not Path(args.video).exists():
        print(f"Error: Video not found: {args.video}")
        sys.exit(1)
    
    try:
        analyzer = VideoAnalyzer()
        analyzer.analyze_video(
            video_path=args.video,
            output_path=args.save,
            show=not args.no_display
        )
        analyzer.cleanup()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
