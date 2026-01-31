"""
Step 5: Form Scorer
====================
Đánh giá form động tác bằng cách so sánh với reference poses.
Sử dụng Cosine Similarity để tính điểm tương đồng.
"""

from .scorer import FormScorer, ScoringResult

__all__ = ['FormScorer', 'ScoringResult']
