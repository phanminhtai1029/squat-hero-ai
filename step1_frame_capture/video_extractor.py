"""
Step 1: Video Frame Extractor (Benchmark Mode)
Nhiệm vụ: Đọc video file và trích xuất frames để benchmark
"""

import cv2
import numpy as np
from typing import Iterator, Optional, Tuple


class VideoFrameExtractor:
    """Đọc video file và extract frames cho benchmark testing."""
    
    def __init__(self, video_path: str):
        """
        Khởi tạo video extractor.
        
        Args:
            video_path: Đường dẫn tới file video
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Không thể mở video: {video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.frame_count / self.fps if self.fps > 0 else 0
    
    def get_info(self) -> dict:
        """Lấy thông tin video."""
        return {
            "path": self.video_path,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "width": self.width,
            "height": self.height,
            "duration_seconds": self.duration
        }
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Lấy frame tiếp theo."""
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame
    
    def get_frame_at(self, frame_index: int) -> Optional[np.ndarray]:
        """Lấy frame tại vị trí cụ thể."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame
    
    def frames_generator(self, skip_frames: int = 0) -> Iterator[Tuple[int, np.ndarray]]:
        """
        Generator trả về từng frame.
        
        Args:
            skip_frames: Số frame bỏ qua giữa mỗi frame xử lý (để giảm load)
            
        Yields:
            Tuple[frame_index, frame]
        """
        self.reset()
        frame_idx = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            if frame_idx % (skip_frames + 1) == 0:
                yield frame_idx, frame
            
            frame_idx += 1
    
    def current_frame_index(self) -> int:
        """Lấy index frame hiện tại."""
        return int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
    
    def reset(self):
        """Reset về đầu video."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    def release(self):
        """Giải phóng video."""
        if self.cap.isOpened():
            self.cap.release()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
    
    def __len__(self):
        return self.frame_count


# Test module
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python video_extractor.py <video_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    print(f"Testing VideoFrameExtractor with: {video_path}")
    
    with VideoFrameExtractor(video_path) as extractor:
        info = extractor.get_info()
        print(f"Video info:")
        for k, v in info.items():
            print(f"  {k}: {v}")
        
        # Test lấy vài frame đầu
        print("\nReading first 5 frames...")
        for i, (idx, frame) in enumerate(extractor.frames_generator()):
            print(f"  Frame {idx}: shape={frame.shape}")
            if i >= 4:
                break
    
    print("\nTest completed!")
