"""
Squat AI - Real-time Squat Analysis Pipeline
=============================================

Pipeline 4 bước:
1. Frame Capture - Lấy frame từ webcam
2. Person Cropping - Crop người bằng YOLO v8
3. Pose Detection - Detect keypoints bằng MediaPipe
4. Pose Comparison - So sánh và đưa ra feedback

Usage:
    python main.py
    
Controls:
    Q - Thoát
    R - Reset counter
"""

import cv2
import time
import sys

# Import 4 step modules
from step1_frame_capture.webcam_capture import WebcamCapture
from step2_person_cropping.yolo_cropper import YoloCropper
from step3_pose_detection.pose_detector import PoseDetector
from step4_pose_comparison.pose_comparator import PoseComparator
from utils.visualization import draw_status_panel, draw_fps, draw_angle_indicator
import config


class SquatAIPipeline:
    """Main pipeline kết nối 4 bước xử lý."""
    
    def __init__(self):
        print("Initializing Squat AI Pipeline...")
        
        # Step 1: Webcam
        print("  [1/4] Initializing webcam...")
        self.webcam = WebcamCapture(camera_id=config.CAMERA_ID)
        
        # Step 2: YOLO Cropper
        print("  [2/4] Loading YOLO model...")
        self.cropper = YoloCropper(
            model_path=config.YOLO_MODEL,
            confidence=config.YOLO_CONFIDENCE
        )
        
        # Step 3: Pose Detector
        print("  [3/4] Initializing MediaPipe Pose...")
        self.pose_detector = PoseDetector(
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE
        )
        
        # Step 4: Pose Comparator
        print("  [4/4] Initializing Pose Comparator...")
        self.comparator = PoseComparator()
        
        print("Pipeline ready!")
        print("Controls: Q=Quit, R=Reset counter")
        print("-" * 40)
    
    def process_frame(self, frame):
        """
        Xử lý 1 frame qua 4 bước pipeline.
        
        Returns:
            processed_frame: Frame đã xử lý với UI overlay
            result: ComparisonResult hoặc None
        """
        result = None
        display_frame = frame.copy()
        
        # Step 2: Person Cropping
        cropped_frame, bbox = self.cropper.crop_person(frame)
        
        if bbox is not None:
            # Vẽ bounding box lên display frame
            cv2.rectangle(display_frame, 
                         (bbox.x1, bbox.y1), (bbox.x2, bbox.y2),
                         (255, 0, 0), 2)
        
        # Step 3: Pose Detection (trên frame gốc để có tọa độ đúng)
        landmarks = self.pose_detector.detect_pose(frame)
        
        if landmarks is not None:
            # Vẽ skeleton
            display_frame = self.pose_detector.draw_pose(frame)
            
            # Vẽ lại bbox nếu có
            if bbox is not None:
                cv2.rectangle(display_frame, 
                             (bbox.x1, bbox.y1), (bbox.x2, bbox.y2),
                             (255, 0, 0), 2)
            
            # Step 4: Pose Comparison
            result = self.comparator.compare(landmarks)
            
            # Vẽ góc gối lên vị trí đầu gối
            h, w = frame.shape[:2]
            knee_pos = self.pose_detector.get_pixel_coords(
                landmarks['left_knee'], w, h
            )
            display_frame = draw_angle_indicator(
                display_frame, 
                result.knee_angle,
                (knee_pos[0] + 10, knee_pos[1]),
                label="Knee",
                good_threshold=config.SQUAT_ANGLE_THRESHOLD
            )
        
        return display_frame, result
    
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
                display_frame = draw_status_panel(
                    display_frame,
                    rep_count=result.rep_count,
                    phase=result.phase.value,
                    feedback=result.feedback,
                    is_good_form=result.is_good_form
                )
            else:
                # Không detect được người
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
                self.comparator.reset()
                print("Counter reset!")
        
        self.cleanup()
    
    def cleanup(self):
        """Giải phóng resources."""
        self.webcam.release()
        self.pose_detector.close()
        cv2.destroyAllWindows()


def main():
    try:
        pipeline = SquatAIPipeline()
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
