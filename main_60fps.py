"""
Pose AI - Real-time 60 FPS Pipeline
=====================================

Optimized for smooth 60 FPS display with background AI analysis.
- Display thread: 60 FPS real-time camera feed
- Analysis thread: Background pose detection & scoring

Usage:
    python main_60fps.py
    
Controls:
    Q - Quit
    R - Reset counter
"""

import cv2
import time
import sys
import argparse
import threading
from queue import Queue
from collections import deque

# Import pipeline modules
from step1_frame_capture.webcam_capture import WebcamCapture
from step2_person_cropping.yolo_cropper import YoloCropper
from step3_pose_detection.pose_detector import PoseDetector
from step4_frame_classification.frame_classifier import FrameClassifier
from step4_pose_classifier import PoseClassifier
from step5_form_scorer import FormScorer
from utils.visualization import draw_status_panel, draw_fps, draw_angle_indicator
import config


class AsyncPoseAnalyzer:
    """Background thread for AI analysis."""
    
    def __init__(self, pose_detector, cropper, frame_classifier, pose_classifier, form_scorer):
        self.pose_detector = pose_detector
        self.cropper = cropper
        self.frame_classifier = frame_classifier
        self.pose_classifier = pose_classifier
        self.form_scorer = form_scorer
        
        # Thread-safe queues
        self.frame_queue = Queue(maxsize=2)  # Latest frames to analyze
        self.result_queue = Queue(maxsize=1)  # Latest result
        
        # Latest result cache
        self.latest_result = None
        self.latest_landmarks = None
        self.latest_bbox = None
        self.result_lock = threading.Lock()
        
        # Control
        self.running = False
        self.thread = None
        
        # Stats
        self.analysis_fps = 0
    
    def start(self):
        """Start background analysis thread."""
        self.running = True
        self.thread = threading.Thread(target=self._analysis_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop analysis thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
    
    def submit_frame(self, frame):
        """Submit frame for analysis (non-blocking)."""
        # Replace old frame if queue full
        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
            except:
                pass
        try:
            self.frame_queue.put_nowait(frame.copy())
        except:
            pass
    
    def get_latest_result(self):
        """Get latest analysis result (non-blocking)."""
        with self.result_lock:
            return self.latest_result, self.latest_landmarks, self.latest_bbox, self.analysis_fps
    
    def _analysis_loop(self):
        """Main analysis loop running in background."""
        prev_time = time.time()
        
        while self.running:
            try:
                # Get frame with timeout
                frame = self.frame_queue.get(timeout=0.1)
            except:
                continue
            
            # Run analysis
            result, landmarks, bbox = self._analyze_frame(frame)
            
            # Calculate analysis FPS
            curr_time = time.time()
            self.analysis_fps = 1.0 / (curr_time - prev_time + 1e-6)
            prev_time = curr_time
            
            # Update cache
            with self.result_lock:
                self.latest_result = result
                self.latest_landmarks = landmarks
                self.latest_bbox = bbox
    
    def _analyze_frame(self, frame):
        """Analyze single frame."""
        result = None
        landmarks = None
        
        # Person cropping
        _, bbox = self.cropper.crop_person(frame)
        
        # Pose detection
        landmarks = self.pose_detector.detect_pose(frame)
        
        if landmarks is not None:
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
                    'is_pose_frame': True,
                    'frame_type': frame_class.frame_type.value,
                    'score': scoring.score,
                    'confidence': classification.confidence,
                    'knee_angle': frame_class.knee_angle,
                    'angular_velocity': frame_class.angular_velocity,
                }
            else:
                result = {
                    'rep_count': self.form_scorer.rep_count,
                    'phase': 'transition',
                    'feedback': f"Transitioning ({frame_class.frame_type.value.upper()})",
                    'is_good_form': False,
                    'pose_label': 'transition',
                    'is_pose_frame': False,
                    'frame_type': frame_class.frame_type.value,
                    'score': None,
                    'confidence': 0.0,
                    'knee_angle': frame_class.knee_angle,
                    'angular_velocity': frame_class.angular_velocity,
                }
        
        return result, landmarks, bbox


class PoseAIPipeline60FPS:
    """Main pipeline optimized for 60 FPS display."""
    
    TARGET_FPS = 60
    FRAME_TIME = 1.0 / TARGET_FPS  # ~16.67ms
    
    def __init__(self):
        print("=" * 50)
        print("Initializing Pose AI Pipeline (60 FPS Mode)...")
        print("=" * 50)
        
        # Step 1: Webcam (try to set 60 FPS)
        print("  [1/6] Initializing webcam...")
        self.webcam = WebcamCapture(camera_id=config.CAMERA_ID)
        
        # Try to set camera to 60 FPS
        self.webcam.cap.set(cv2.CAP_PROP_FPS, 60)
        actual_fps = self.webcam.cap.get(cv2.CAP_PROP_FPS)
        print(f"        Camera FPS: {actual_fps}")
        
        # Step 2: YOLO Cropper
        print("  [2/6] Loading YOLO model...")
        self.cropper = YoloCropper(
            model_path=config.YOLO_MODEL,
            confidence=config.YOLO_CONFIDENCE
        )
        
        # Step 3: Pose Detector
        print("  [3/6] Initializing MediaPipe Pose...")
        self.pose_detector = PoseDetector(
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE
        )
        
        # Step 4: Frame Classifier
        print("  [4/6] Initializing Frame Classifier...")
        self.frame_classifier = FrameClassifier()
        
        # Step 5: Pose Classifier
        print("  [5/6] Initializing Pose Classifier (MLP)...")
        self.pose_classifier = PoseClassifier()
        
        # Step 6: Form Scorer
        print("  [6/6] Initializing Form Scorer...")
        self.form_scorer = FormScorer()
        
        # Background analyzer
        self.analyzer = AsyncPoseAnalyzer(
            self.pose_detector,
            self.cropper,
            self.frame_classifier,
            self.pose_classifier,
            self.form_scorer
        )
        
        print("=" * 50)
        print("Pipeline ready! Mode: 60 FPS Async")
        print("Display runs at 60 FPS, analysis in background")
        print("Controls: Q=Quit, R=Reset counter")
        print("=" * 50)
    
    def draw_overlay(self, frame, result, landmarks, bbox, display_fps, analysis_fps):
        """Draw all overlays on frame."""
        display_frame = frame.copy()
        h, w = frame.shape[:2]
        
        # Draw bbox
        if bbox is not None:
            cv2.rectangle(display_frame, 
                         (bbox.x1, bbox.y1), (bbox.x2, bbox.y2),
                         (255, 0, 0), 2)
        
        # Draw skeleton
        if landmarks is not None:
            display_frame = self.pose_detector.draw_pose(frame)
            if bbox is not None:
                cv2.rectangle(display_frame, 
                             (bbox.x1, bbox.y1), (bbox.x2, bbox.y2),
                             (255, 0, 0), 2)
        
        # Draw FPS (both display and analysis)
        cv2.putText(display_frame, f"Display: {display_fps:.0f} FPS", 
                   (w - 150, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(display_frame, f"Analysis: {analysis_fps:.0f} FPS", 
                   (w - 150, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        if result is not None:
            # Frame status
            frame_status = result.get('frame_type', 'unknown').upper()
            is_pose = result.get('is_pose_frame', False)
            status_color = (0, 255, 0) if is_pose else (0, 165, 255)
            cv2.putText(display_frame, f"Frame: {frame_status}",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                       0.6, status_color, 2)
            
            # Angular velocity
            velocity = result.get('angular_velocity', 0)
            cv2.putText(display_frame, f"Velocity: {velocity:.1f} deg/f",
                       (10, 85), cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, (200, 200, 200), 1)
            
            # Pose label
            if is_pose and result['pose_label'] != 'transition':
                confidence = result.get('confidence', 0)
                label_text = f"{result['pose_label'].upper()} ({confidence:.0%})"
                cv2.putText(display_frame, label_text,
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                           0.8, (0, 255, 255), 2)
            
            # Knee angle
            if landmarks is not None and 'left_knee' in landmarks:
                knee_pos = self.pose_detector.get_pixel_coords(
                    landmarks['left_knee'], w, h
                )
                display_frame = draw_angle_indicator(
                    display_frame,
                    result.get('knee_angle', 0),
                    (knee_pos[0] + 10, knee_pos[1]),
                    label="Knee",
                    good_threshold=config.SQUAT_ANGLE_THRESHOLD
                )
            
            # Status panel
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
            cv2.putText(display_frame, "Analyzing...", 
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (255, 255, 0), 2)
        
        return display_frame
    
    def run(self):
        """Run 60 FPS display loop with background analysis."""
        # Start background analyzer
        self.analyzer.start()
        
        # FPS tracking
        fps_counter = deque(maxlen=30)
        prev_time = time.time()
        
        print("\nRunning at target 60 FPS...")
        
        while True:
            loop_start = time.time()
            
            # Capture frame
            frame = self.webcam.get_frame()
            if frame is None:
                print("Error: Could not read frame")
                break
            
            # Flip for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Submit to background analyzer
            self.analyzer.submit_frame(frame)
            
            # Get latest analysis result
            result, landmarks, bbox, analysis_fps = self.analyzer.get_latest_result()
            
            # Calculate display FPS
            curr_time = time.time()
            frame_time = curr_time - prev_time
            fps_counter.append(frame_time)
            display_fps = len(fps_counter) / sum(fps_counter) if fps_counter else 0
            prev_time = curr_time
            
            # Draw UI
            display_frame = self.draw_overlay(
                frame, result, landmarks, bbox, display_fps, analysis_fps
            )
            
            # Show frame
            cv2.imshow(config.WINDOW_NAME, display_frame)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print("\nExiting...")
                break
            elif key == ord('r') or key == ord('R'):
                self.form_scorer.reset()
                print("Counter reset!")
            
            # Frame rate limiting for smooth 60 FPS
            elapsed = time.time() - loop_start
            sleep_time = self.FRAME_TIME - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        self.cleanup()
    
    def cleanup(self):
        """Release resources."""
        self.analyzer.stop()
        self.webcam.release()
        self.pose_detector.close()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Pose AI - 60 FPS Mode")
    args = parser.parse_args()
    
    try:
        pipeline = PoseAIPipeline60FPS()
        pipeline.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
