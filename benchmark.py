"""
Benchmark Runner - Chạy pipeline trên video để đánh giá
======================================================

Sử dụng video chuẩn để benchmark sau mỗi lần cải tiến model.
Output: Báo cáo chi tiết về performance và accuracy.

Usage:
    python benchmark.py --video path/to/video.mp4
    python benchmark.py --video path/to/video.mp4 --save-output
"""

import cv2
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Import modules
from step1_frame_capture.video_extractor import VideoFrameExtractor
from step2_person_cropping.yolo_cropper import YoloCropper
from step3_pose_detection.pose_detector import PoseDetector
from step4_pose_comparison.pose_comparator import PoseComparator, SquatPhase
from utils.visualization import draw_status_panel, draw_fps, draw_angle_indicator
import config


class BenchmarkResult:
    """Lưu kết quả benchmark."""
    
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.timestamp = datetime.now().isoformat()
        self.total_frames = 0
        self.processed_frames = 0
        self.detection_success = 0
        self.total_reps = 0
        self.avg_fps = 0
        self.processing_time = 0
        self.phase_history: List[str] = []
        self.angle_history: List[float] = []
        self.form_errors: List[str] = []
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_path": self.video_path,
            "timestamp": self.timestamp,
            "metrics": {
                "total_frames": self.total_frames,
                "processed_frames": self.processed_frames,
                "detection_rate": self.detection_success / max(self.processed_frames, 1) * 100,
                "total_reps": self.total_reps,
                "avg_fps": self.avg_fps,
                "processing_time_seconds": self.processing_time,
            },
            "angle_stats": {
                "min": min(self.angle_history) if self.angle_history else 0,
                "max": max(self.angle_history) if self.angle_history else 0,
                "avg": sum(self.angle_history) / len(self.angle_history) if self.angle_history else 0,
            },
            "form_errors_count": len(self.form_errors),
            "form_errors": self.form_errors[:10],  # Top 10 errors
        }
    
    def print_summary(self):
        data = self.to_dict()
        print("\n" + "=" * 50)
        print("BENCHMARK RESULT")
        print("=" * 50)
        print(f"Video: {self.video_path}")
        print(f"Timestamp: {self.timestamp}")
        print("-" * 50)
        print("METRICS:")
        print(f"  Total Frames: {data['metrics']['total_frames']}")
        print(f"  Detection Rate: {data['metrics']['detection_rate']:.1f}%")
        print(f"  Total Reps Detected: {data['metrics']['total_reps']}")
        print(f"  Avg FPS: {data['metrics']['avg_fps']:.1f}")
        print(f"  Processing Time: {data['metrics']['processing_time_seconds']:.2f}s")
        print("-" * 50)
        print("ANGLE STATS:")
        print(f"  Min Knee Angle: {data['angle_stats']['min']:.1f}°")
        print(f"  Max Knee Angle: {data['angle_stats']['max']:.1f}°")
        print(f"  Avg Knee Angle: {data['angle_stats']['avg']:.1f}°")
        print("-" * 50)
        print(f"Form Errors Detected: {data['form_errors_count']}")
        print("=" * 50)


class BenchmarkRunner:
    """Chạy benchmark trên video."""
    
    def __init__(self, video_path: str):
        self.video_path = video_path
        
        print("Initializing Benchmark Runner...")
        
        # Load video
        print("  Loading video...")
        self.video = VideoFrameExtractor(video_path)
        info = self.video.get_info()
        print(f"  Video: {info['frame_count']} frames, {info['fps']:.1f} FPS, {info['duration_seconds']:.1f}s")
        
        # Load models
        print("  Loading YOLO...")
        self.cropper = YoloCropper()
        
        print("  Loading MediaPipe...")
        self.pose_detector = PoseDetector()
        
        print("  Initializing Comparator...")
        self.comparator = PoseComparator()
        
        print("Ready!\n")
    
    def run(self, save_output: bool = False, output_path: str = None, show_preview: bool = True) -> BenchmarkResult:
        """
        Chạy benchmark trên video.
        
        Args:
            save_output: Có lưu video output không
            output_path: Đường dẫn lưu video output
            show_preview: Có hiển thị preview không
            
        Returns:
            BenchmarkResult
        """
        result = BenchmarkResult(self.video_path)
        result.total_frames = self.video.frame_count
        
        # Setup video writer nếu cần
        writer = None
        if save_output:
            if output_path is None:
                output_path = str(Path(self.video_path).stem) + "_benchmark.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, self.video.fps,
                                    (self.video.width, self.video.height))
        
        # Reset
        self.video.reset()
        self.comparator.reset()
        
        start_time = time.time()
        fps_list = []
        prev_time = start_time
        
        print("Processing video...")
        
        for frame_idx, frame in self.video.frames_generator():
            result.processed_frames += 1
            
            # Process frame
            display_frame, comparison = self._process_frame_pipeline(frame)
            
            # Calculate FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time + 1e-6)
            fps_list.append(fps)
            prev_time = curr_time
            
            # Collect metrics
            if comparison is not None:
                result.detection_success += 1
                result.angle_history.append(comparison.knee_angle)
                result.phase_history.append(comparison.phase.value)
                result.total_reps = comparison.rep_count
                
                if not comparison.is_good_form:
                    for err in comparison.errors:
                        if err.value != "none":
                            result.form_errors.append(f"Frame {frame_idx}: {err.value}")
                
                # Draw UI
                display_frame = draw_status_panel(
                    display_frame,
                    rep_count=comparison.rep_count,
                    phase=comparison.phase.value,
                    feedback=comparison.feedback,
                    is_good_form=comparison.is_good_form
                )
            
            display_frame = draw_fps(display_frame, fps)
            
            # Add frame counter
            cv2.putText(display_frame, f"Frame: {frame_idx}/{self.video.frame_count}",
                       (10, self.video.height - 20), cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, (255, 255, 255), 1)
            
            # Save/show
            if writer:
                writer.write(display_frame)
            
            if show_preview:
                cv2.imshow("Benchmark Preview", display_frame)
                # Nhấn Q để skip preview
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    show_preview = False
                    cv2.destroyAllWindows()
            
            # Progress
            if frame_idx % 30 == 0:
                progress = frame_idx / self.video.frame_count * 100
                print(f"  Progress: {progress:.0f}% ({frame_idx}/{self.video.frame_count})", end='\r')
        
        end_time = time.time()
        result.processing_time = end_time - start_time
        result.avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0
        
        # Cleanup
        if writer:
            writer.release()
            print(f"\nOutput saved to: {output_path}")
        
        cv2.destroyAllWindows()
        
        return result
    
    def _process_frame_pipeline(self, frame):
        """Chạy frame qua 4-step pipeline."""
        display_frame = frame.copy()
        comparison = None
        
        # Step 2: Person cropping
        cropped, bbox = self.cropper.crop_person(frame)
        if bbox:
            cv2.rectangle(display_frame, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), (255, 0, 0), 2)
        
        # Step 3: Pose detection
        landmarks = self.pose_detector.detect_pose(frame)
        
        if landmarks:
            display_frame = self.pose_detector.draw_pose(frame)
            if bbox:
                cv2.rectangle(display_frame, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), (255, 0, 0), 2)
            
            # Step 4: Comparison
            comparison = self.comparator.compare(landmarks)
            
            # Draw angle
            h, w = frame.shape[:2]
            knee_pos = self.pose_detector.get_pixel_coords(landmarks['left_knee'], w, h)
            display_frame = draw_angle_indicator(
                display_frame, comparison.knee_angle,
                (knee_pos[0] + 10, knee_pos[1]),
                label="Knee"
            )
        
        return display_frame, comparison
    
    def cleanup(self):
        self.video.release()
        self.pose_detector.close()


def main():
    parser = argparse.ArgumentParser(description="Squat AI Benchmark Runner")
    parser.add_argument("--video", "-v", required=True, help="Path to benchmark video")
    parser.add_argument("--save-output", "-s", action="store_true", help="Save output video")
    parser.add_argument("--output", "-o", help="Output video path")
    parser.add_argument("--no-preview", action="store_true", help="Disable preview window")
    parser.add_argument("--save-json", "-j", help="Save results to JSON file")
    
    args = parser.parse_args()
    
    try:
        runner = BenchmarkRunner(args.video)
        result = runner.run(
            save_output=args.save_output,
            output_path=args.output,
            show_preview=not args.no_preview
        )
        
        result.print_summary()
        
        # Save JSON if requested
        if args.save_json:
            with open(args.save_json, 'w', encoding='utf-8') as f:
                json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
            print(f"\nResults saved to: {args.save_json}")
        
        runner.cleanup()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
