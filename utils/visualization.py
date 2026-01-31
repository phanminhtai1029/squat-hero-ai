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
    Draw status info directly on frame (no black box, no reps).
    
    Args:
        frame: Input frame
        rep_count: Number of reps (ignored)
        phase: Current phase/pose
        feedback: Text feedback
        is_good_form: Whether form is good
    """
    frame_copy = frame.copy()
    h, w = frame_copy.shape[:2]
    
    # No black background - just text with outline for visibility
    def draw_text_with_outline(img, text, pos, scale, color, thickness=2):
        """Draw text with black outline for better visibility."""
        x, y = pos
        # Black outline
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2)
        # Colored text
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)
    
    # Pose/Phase display - truncate if too long
    phase_display = phase[:35] if len(phase) > 35 else phase
    draw_text_with_outline(frame_copy, f"Pose: {phase_display}", (15, 35), 
                          0.6, (255, 255, 255), 1)
    
    # Feedback with color
    fb_color = (0, 255, 0) if is_good_form else (0, 165, 255)
    feedback_display = feedback[:45] if len(feedback) > 45 else feedback
    draw_text_with_outline(frame_copy, feedback_display, (15, 65), 
                          0.55, fb_color, 1)
    
    return frame_copy


def draw_fps(frame: np.ndarray, fps: float) -> np.ndarray:
    """Vẽ FPS counter."""
    frame_copy = frame.copy()
    h, w = frame_copy.shape[:2]
    
    cv2.putText(frame_copy, f"FPS: {fps:.0f}", (w - 100, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    return frame_copy
