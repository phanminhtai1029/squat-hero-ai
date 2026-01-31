"""
Step 1: Frame Capture
Captures frames from webcam or video file.
"""

import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Generator


class FrameCapture(ABC):
    """Abstract base class for frame capture."""
    
    @abstractmethod
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a single frame."""
        pass
    
    @abstractmethod
    def release(self) -> None:
        """Release resources."""
        pass
    
    @abstractmethod
    def is_opened(self) -> bool:
        """Check if capture is opened."""
        pass
    
    def frames(self) -> Generator[np.ndarray, None, None]:
        """Generator that yields frames."""
        while self.is_opened():
            ret, frame = self.read()
            if not ret:
                break
            yield frame
        self.release()


class WebcamCapture(FrameCapture):
    """Capture frames from webcam."""
    
    def __init__(self, camera_id: int = 0, width: int = 640, height: int = 480):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        return self.cap.read()
    
    def release(self) -> None:
        self.cap.release()
    
    def is_opened(self) -> bool:
        return self.cap.isOpened()


class VideoCapture(FrameCapture):
    """Capture frames from video file."""
    
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.current_frame = 0
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        ret, frame = self.cap.read()
        if ret:
            self.current_frame += 1
        return ret, frame
    
    def release(self) -> None:
        self.cap.release()
    
    def is_opened(self) -> bool:
        return self.cap.isOpened()
    
    def seek(self, frame_number: int) -> None:
        """Seek to specific frame."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        self.current_frame = frame_number
    
    def get_progress(self) -> float:
        """Get progress as percentage."""
        if self.total_frames == 0:
            return 0.0
        return (self.current_frame / self.total_frames) * 100


class ImageCapture(FrameCapture):
    """Capture from a single image (for testing)."""
    
    def __init__(self, image_path: str):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        self.read_count = 0
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self.read_count == 0 and self.image is not None:
            self.read_count += 1
            return True, self.image.copy()
        return False, None
    
    def release(self) -> None:
        self.image = None
    
    def is_opened(self) -> bool:
        return self.image is not None and self.read_count == 0
