"""
Step 4: Frame Classification
Phân loại frame thành tư thế (pose) hoặc chuyển động (transition)
"""

from .frame_classifier import FrameClassifier, FrameType, ClassificationResult

__all__ = ['FrameClassifier', 'FrameType', 'ClassificationResult']
