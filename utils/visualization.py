"""
Utils: Visualization
Các hàm hỗ trợ vẽ và hiển thị
"""

import cv2
import numpy as np
from typing import Tuple


def draw_angle_indicator(frame: np.ndarray, 
                         angle: float, 
                         position: Tuple[int, int],
                         label: str = "Angle",
                         good_threshold: float = 90) -> np.ndarray:
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
    if angle < good_threshold:
        color = (0, 255, 0)  # Xanh lá = tốt
    else:
        color = (0, 165, 255)  # Cam = cần cải thiện
    
    text = f"{label}: {angle:.0f}°"
    cv2.putText(frame_copy, text, position, 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    return frame_copy


def draw_status_panel(frame: np.ndarray,
                      rep_count: int,
                      phase: str,
                      feedback: str,
                      is_good_form: bool) -> np.ndarray:
    """
    Vẽ panel hiển thị trạng thái.
    
    Args:
        frame: Frame ảnh
        rep_count: Số reps
        phase: Phase hiện tại
        feedback: Text feedback
        is_good_form: Form có tốt không
    """
    frame_copy = frame.copy()
    h, w = frame_copy.shape[:2]
    
    # Vẽ background panel
    overlay = frame_copy.copy()
    cv2.rectangle(overlay, (10, 10), (300, 130), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame_copy, 0.4, 0, frame_copy)
    
    # Vẽ text
    cv2.putText(frame_copy, f"REPS: {rep_count}", (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    
    cv2.putText(frame_copy, f"Phase: {phase}", (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    # Feedback với màu
    fb_color = (0, 255, 0) if is_good_form else (0, 165, 255)
    cv2.putText(frame_copy, feedback[:35], (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, fb_color, 1)
    
    return frame_copy


def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    """Vẽ FPS counter."""
    frame_copy = frame.copy()
    h, w = frame_copy.shape[:2]
    
    cv2.putText(frame_copy, f"FPS: {fps:.0f}", (w - 100, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    return frame_copy
