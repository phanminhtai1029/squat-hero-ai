"""
Pose AI - Real-time Pose Analysis Pipeline
===========================================

Pipeline 6 bước với AI:
1. Frame Capture - Lấy frame từ webcam
2. Person Cropping - Crop người bằng YOLOv8
3. Pose Detection - Detect keypoints bằng MediaPipe
4. Frame Classification - Phân biệt pose vs transition (angular velocity)
5. Pose Classifier - Phân loại động tác bằng MLP (AI)
6. Form Scorer - Đánh giá form bằng Cosine Similarity

Hỗ trợ nhiều động tác yoga và fitness.

Usage:
    python main.py                # AI pipeline (6 steps)
    python main.py --legacy       # Legacy pipeline (4 steps)
    
Controls:
    Q - Thoát
    R - Reset counter
"""

import cv2
import time
import sys
import argparse

# Import 5 step modules
from step1_frame_capture.webcam_capture import WebcamCapture
from step2_person_cropping.yolo_cropper import YoloCropper
from step3_pose_detection.pose_detector import PoseDetector
from step4_frame_classification.frame_classifier import FrameClassifier
from step4_pose_classifier import PoseClassifier
from step5_form_scorer import FormScorer
from utils.visualization import draw_status_panel, draw_fps, draw_angle_indicator
import config


class PoseAIPipeline:
    """Main pipeline kết nối 5 bước xử lý với AI."""
    
    def __init__(self, use_legacy: bool = False):
        """
        Khởi tạo pipeline.
        
        Args:
            use_legacy: True để sử dụng pipeline cũ (4 steps)
        """
        self.use_legacy = use_legacy
        print("=" * 50)
        print("Initializing Pose AI Pipeline...")
        print("=" * 50)
        
        # Step 1: Webcam
        print("  [1/5] Initializing webcam...")
        self.webcam = WebcamCapture(camera_id=config.CAMERA_ID)
        
        # Step 2: YOLO Cropper
        print("  [2/5] Loading YOLO model...")
        self.cropper = YoloCropper(
            model_path=config.YOLO_MODEL,
            confidence=config.YOLO_CONFIDENCE
        )
        
        # Step 3: Pose Detector
        print("  [3/5] Initializing MediaPipe Pose...")
        self.pose_detector = PoseDetector(
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE
        )
        
        if use_legacy:
            # Legacy mode: sử dụng PoseComparator cũ
            print("  [4/4] Initializing Legacy Pose Comparator...")
            from step4_legacy.pose_comparator import PoseComparator
            self.legacy_comparator = PoseComparator()
            self.pose_classifier = None
            self.form_scorer = None
            mode_str = "Legacy (4-step)"
        else:
            # New AI mode with 6 steps
            print("  [4/6] Initializing Frame Classifier...")
            self.frame_classifier = FrameClassifier()
            
            print("  [5/6] Initializing Pose Classifier (MLP)...")
            self.pose_classifier = PoseClassifier()
            
            print("  [6/6] Initializing Form Scorer (Cosine Similarity)...")
            self.form_scorer = FormScorer()
            self.legacy_comparator = None
            mode_str = "AI (6-step)"
        
        print("=" * 50)
        print(f"Pipeline ready! Mode: {mode_str}")
        print(f"Supported poses: {', '.join(config.POSE_CLASSES)}")
        print("Controls: Q=Quit, R=Reset counter")
        print("=" * 50)
    
    def process_frame(self, frame):
        """
        Xử lý 1 frame qua pipeline.
        
        Returns:
            processed_frame: Frame đã xử lý với UI overlay
            result: Dict chứa thông tin kết quả hoặc None
        """
        result = None
        display_frame = frame.copy()
        
        # Step 2: Person Cropping
        cropped_frame, bbox = self.cropper.crop_person(frame)
        
        if bbox is not None:
            cv2.rectangle(display_frame, 
                         (bbox.x1, bbox.y1), (bbox.x2, bbox.y2),
                         (255, 0, 0), 2)
        
        # Step 3: Pose Detection
        landmarks = self.pose_detector.detect_pose(frame)
        
        if landmarks is not None:
            # Vẽ skeleton
            display_frame = self.pose_detector.draw_pose(frame)
            
            # Vẽ lại bbox
            if bbox is not None:
                cv2.rectangle(display_frame, 
                             (bbox.x1, bbox.y1), (bbox.x2, bbox.y2),
                             (255, 0, 0), 2)
            
            if self.use_legacy:
                result = self._process_legacy(landmarks, display_frame, frame)
            else:
                result = self._process_ai(landmarks, display_frame, frame)
        
        return display_frame, result
    
    def _process_legacy(self, landmarks, display_frame, frame):
        """Xử lý theo pipeline cũ (4 steps)."""
        comparison = self.legacy_comparator.compare(landmarks)
        
        # Vẽ góc gối
        h, w = frame.shape[:2]
        knee_pos = self.pose_detector.get_pixel_coords(
            landmarks['left_knee'], w, h
        )
        display_frame = draw_angle_indicator(
            display_frame, 
            comparison.knee_angle,
            (knee_pos[0] + 10, knee_pos[1]),
            label="Knee",
            good_threshold=config.SQUAT_ANGLE_THRESHOLD
        )
        
        return {
            'rep_count': comparison.rep_count,
            'phase': comparison.phase.value,
            'feedback': comparison.feedback,
            'is_good_form': comparison.is_good_form,
            'pose_label': 'squat',  # Legacy chỉ hỗ trợ squat
            'score': None,
            'display_frame': display_frame
        }
    
    def _process_ai(self, landmarks, display_frame, frame):
        """Xử lý theo pipeline AI mới (6 steps)."""
        h, w = frame.shape[:2]
        
        # Step 4: Frame Classification - Phân biệt pose vs transition
        frame_class = self.frame_classifier.classify(landmarks)
        is_pose_frame = self.frame_classifier.is_pose_frame(frame_class)
        
        # Hiển thị trạng thái frame (pose/transition)
        frame_status = frame_class.frame_type.value.upper()
        status_color = (0, 255, 0) if is_pose_frame else (0, 165, 255)  # Green for pose, Orange for transition
        cv2.putText(display_frame, f"Frame: {frame_status}",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                   0.6, status_color, 2)
        
        # Hiển thị angular velocity
        cv2.putText(display_frame, f"Velocity: {frame_class.angular_velocity:.1f} deg/f",
                   (10, 85), cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, (200, 200, 200), 1)
        
        # Step 5: Pose Classification - Chỉ classify khi là pose frame
        if is_pose_frame:
            classification = self.pose_classifier.classify(landmarks)
            # Step 6: Form Scoring
            scoring = self.form_scorer.score(landmarks, classification.label)
            
            pose_label = classification.label
            confidence = classification.confidence
            score = scoring.score
            feedback = scoring.feedback
            is_good_form = scoring.is_good_form
            rep_count = scoring.rep_count
        else:
            # Transition frame - không đánh giá
            pose_label = "transition"
            confidence = 0.0
            score = None
            feedback = f"Transitioning ({frame_status})"
            is_good_form = False
            rep_count = self.form_scorer.rep_count  # Giữ nguyên rep count
        
        # Vẽ góc gối
        knee_pos = self.pose_detector.get_pixel_coords(
            landmarks['left_knee'], w, h
        )
        display_frame = draw_angle_indicator(
            display_frame, 
            frame_class.knee_angle,
            (knee_pos[0] + 10, knee_pos[1]),
            label="Knee",
            good_threshold=config.SQUAT_ANGLE_THRESHOLD
        )
        
        # Vẽ pose label và confidence (chỉ khi là pose frame)
        if is_pose_frame and pose_label != "transition":
            label_text = f"{pose_label.upper()} ({confidence:.0%})"
            cv2.putText(display_frame, label_text,
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                       0.8, (0, 255, 255), 2)
        
        return {
            'rep_count': rep_count,
            'phase': pose_label,
            'feedback': feedback,
            'is_good_form': is_good_form,
            'pose_label': pose_label,
            'is_pose_frame': is_pose_frame,
            'frame_type': frame_class.frame_type.value,
            'score': score,
            'confidence': confidence,
            'display_frame': display_frame
        }
    
    def run(self):
        """Chạy pipeline real-time."""
        prev_time = time.time()
        fps = 0
        
        while True:
            # Step 1: Capture frame
            frame = self.webcam.get_frame()
            
            if frame is None:
                print("Error: Could not read frame")
                break
            
            # Flip horizontal để giống gương
            frame = cv2.flip(frame, 1)
            
            # Process qua pipeline
            display_frame, result = self.process_frame(frame)
            
            # Calculate FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time + 1e-6)
            prev_time = curr_time
            
            # Draw UI
            display_frame = draw_fps(display_frame, fps)
            
            if result is not None:
                display_frame = result.get('display_frame', display_frame)
                
                # Hiển thị score nếu có
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
            
            # Show frame
            cv2.imshow(config.WINDOW_NAME, display_frame)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print("\nExiting...")
                break
            elif key == ord('r') or key == ord('R'):
                self.reset()
                print("Counter reset!")
        
        self.cleanup()
    
    def reset(self):
        """Reset counter và state."""
        if self.use_legacy:
            self.legacy_comparator.reset()
        else:
            self.form_scorer.reset()
    
    def cleanup(self):
        """Giải phóng resources."""
        self.webcam.release()
        self.pose_detector.close()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Pose AI - Real-time Analysis")
    parser.add_argument("--legacy", action="store_true", 
                       help="Use legacy 4-step pipeline (squat only)")
    args = parser.parse_args()
    
    try:
        pipeline = PoseAIPipeline(use_legacy=args.legacy)
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
