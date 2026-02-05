"""
Utils: Visualization
Các hàm hỗ trợ vẽ và hiển thị cho Yoga Pose AI
"""

import cv2
import numpy as np
from typing import Tuple, Optional

# Import config
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def draw_angle_indicator(
    frame: np.ndarray, 
    angle: float, 
    position: Tuple[int, int],
    label: str = "Angle",
    good_threshold: float = 90
) -> np.ndarray:
    """
    Vẽ indicator hiển thị góc với màu sắc.
    
    Args:
        frame: Frame ảnh
        angle: Góc cần hiển thị
        position: Vị trí (x, y) để vẽ
        label: Nhãn
        good_threshold: Ngưỡng để đổi màu
    """
    frame_copy = frame.copy()
    
    # Chọn màu dựa vào góc
    color = config.COLOR_GREEN if angle < good_threshold else config.COLOR_ORANGE
    
    text = f"{label}: {angle:.0f}°"
    cv2.putText(frame_copy, text, position, 
                cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE, color, 2)
    
    return frame_copy


def draw_pose_result(
    frame: np.ndarray,
    pose_name: Optional[str],
    similarity: float,
    confidence: str
) -> np.ndarray:
    """
    Vẽ kết quả nhận diện pose.
    
    Args:
        frame: Frame ảnh
        pose_name: Tên pose được nhận diện
        similarity: Độ tương đồng (0-1)
        confidence: Mức độ tin cậy ("high", "medium", "low")
    """
    if pose_name is None:
        return frame
    
    frame_copy = frame.copy()
    h, w = frame_copy.shape[:2]
    
    # Background box
    cv2.rectangle(frame_copy, (5, h-80), (w-5, h-5), config.COLOR_BLACK, -1)
    cv2.rectangle(frame_copy, (5, h-80), (w-5, h-5), config.COLOR_GREEN, 2)
    
    # Pose name
    cv2.putText(frame_copy, pose_name, (15, h-50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, config.COLOR_GREEN, 2)
    
    # Similarity
    similarity_pct = similarity * 100
    cv2.putText(frame_copy, f"Match: {similarity_pct:.1f}% ({confidence})", (15, h-20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.COLOR_WHITE, 1)
    
    return frame_copy


def draw_status_overlay(
    frame: np.ndarray,
    frame_type: str,
    fps: float
) -> np.ndarray:
    """
    Vẽ overlay hiển thị FPS và trạng thái frame.
    
    Args:
        frame: Frame ảnh
        frame_type: Loại frame (KEY_POSE, TRANSITION, NO_POSE)
        fps: Frames per second
    """
    frame_copy = frame.copy()
    
    # Draw FPS
    cv2.putText(frame_copy, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE, config.COLOR_GREEN, 2)
    
    # Draw frame type
    color = config.COLOR_GREEN if frame_type == 'KEY_POSE' else config.COLOR_ORANGE
    cv2.putText(frame_copy, f"State: {frame_type}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE, color, 2)
    
    return frame_copy


def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    """Vẽ FPS counter."""
    frame_copy = frame.copy()
    h, w = frame_copy.shape[:2]
    
    cv2.putText(frame_copy, f"FPS: {fps:.0f}", (w - 100, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.COLOR_WHITE, 1)
    
    return frame_copy
