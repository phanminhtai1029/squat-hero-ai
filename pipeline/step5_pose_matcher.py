"""
Step 5: Pose Matching (Improved Rule-Based)
Matches detected pose against a database of reference poses.

Improvements over v1:
- Strategy 1: Euclidean distance instead of cosine similarity
- Strategy 3: Weighted angle importance
- Strategy 4: Angle tolerances (min-max ranges)
- Strategy 5: Additional position features
"""

import numpy as np
import yaml
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.angle_calculator import AngleCalculator, AngleResult


@dataclass
class MatchResult:
    """Pose matching result."""
    pose_name: str
    display_name: str
    similarity: float
    confidence: str  # "high", "medium", "low", "no_match"
    method: str = "euclidean"  # Matching method used
    top_matches: List[Tuple[str, float]] = field(default_factory=list)
    angle_differences: Dict[str, float] = field(default_factory=dict)


@dataclass
class PoseReference:
    """Reference pose from database."""
    name: str
    display_name: str
    category: str
    difficulty: str
    angles_degrees: Dict[str, float]
    angles_normalized: Dict[str, float]
    angles_vector: List[float]
    description: str = ""
    # New fields for improved matching
    weights: Dict[str, float] = field(default_factory=dict)
    angle_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)


class PoseMatcher:
    """
    Improved Rule-based pose matcher.
    
    Features:
    - Multiple similarity metrics (cosine, euclidean, weighted)
    - Angle tolerances with min-max ranges
    - Weighted angle importance per pose
    """
    
    # Similarity thresholds (for euclidean-based similarity)
    THRESHOLD_HIGH = 0.85
    THRESHOLD_MEDIUM = 0.70
    THRESHOLD_LOW = 0.55
    
    # Default weights for angles (higher = more important)
    DEFAULT_WEIGHTS = {
        'left_elbow': 0.8,
        'right_elbow': 0.8,
        'left_shoulder': 1.0,
        'right_shoulder': 1.0,
        'left_knee': 1.5,
        'right_knee': 1.5,
        'left_hip': 1.2,
        'right_hip': 1.2,
    }
    
    def __init__(
        self, 
        database_path: Optional[str] = None,
        method: str = "euclidean"  # "cosine", "euclidean", "weighted", "combined"
    ):
        """
        Initialize pose matcher.
        
        Args:
            database_path: Path to YAML database file
            method: Matching method to use
        """
        self.database: Dict[str, PoseReference] = {}
        self.angle_calculator = AngleCalculator()
        self.method = method
        
        if database_path:
            self.load_database(database_path)
    
    def load_database(self, database_path: str) -> None:
        """Load pose database from YAML file."""
        with open(database_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        poses = data.get('poses', {})
        
        for pose_name, pose_data in poses.items():
            angles_deg = pose_data.get('angles_degrees', {})
            angles_norm = pose_data.get('angles_normalized', {})
            
            # Create vector from normalized angles
            if isinstance(angles_norm, list):
                angles_vector = angles_norm
            else:
                angles_vector = [
                    angles_norm.get(name, 1.0)
                    for name in AngleCalculator.ANGLE_ORDER
                ]
            
            # Load weights if available
            weights = pose_data.get('weights', self.DEFAULT_WEIGHTS.copy())
            
            # Load angle ranges if available
            angle_ranges = {}
            if 'angle_ranges' in pose_data:
                for angle_name, range_val in pose_data['angle_ranges'].items():
                    if isinstance(range_val, list) and len(range_val) == 2:
                        angle_ranges[angle_name] = tuple(range_val)
            
            self.database[pose_name] = PoseReference(
                name=pose_name,
                display_name=pose_data.get('display_name', pose_name),
                category=pose_data.get('category', 'unknown'),
                difficulty=pose_data.get('difficulty', 'unknown'),
                angles_degrees=angles_deg,
                angles_normalized=angles_norm if isinstance(angles_norm, dict) else {},
                angles_vector=angles_vector,
                description=pose_data.get('description', ''),
                weights=weights,
                angle_ranges=angle_ranges
            )
    
    # =========================================================================
    # SIMILARITY METRICS
    # =========================================================================
    
    @staticmethod
    def cosine_similarity(a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        a = np.array(a)
        b = np.array(b)
        
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return float(dot_product / (norm_a * norm_b))
    
    @staticmethod
    def euclidean_similarity(a: List[float], b: List[float]) -> float:
        """
        Compute similarity based on Euclidean distance.
        Returns value in [0, 1] where 1 is perfect match.
        """
        a = np.array(a)
        b = np.array(b)
        
        distance = np.linalg.norm(a - b)
        # Convert distance to similarity using exponential decay
        # Max expected distance ~= 1.5 for normalized angles
        similarity = np.exp(-distance * 2)
        
        return float(similarity)
    
    def weighted_euclidean_similarity(
        self, 
        a: List[float], 
        b: List[float],
        weights: Dict[str, float]
    ) -> float:
        """
        Compute weighted Euclidean similarity.
        """
        a = np.array(a)
        b = np.array(b)
        
        # Create weight vector in same order as angles
        weight_vector = np.array([
            weights.get(name, 1.0) 
            for name in AngleCalculator.ANGLE_ORDER
        ])
        
        # Weighted difference
        weighted_diff = (a - b) * weight_vector
        distance = np.linalg.norm(weighted_diff)
        
        # Normalize by sum of weights
        normalized_distance = distance / np.sum(weight_vector)
        
        similarity = np.exp(-normalized_distance * 3)
        
        return float(similarity)
    
    def tolerance_score(
        self,
        current_angles: Dict[str, float],
        angle_ranges: Dict[str, Tuple[float, float]]
    ) -> float:
        """
        Compute score based on whether angles fall within acceptable ranges.
        """
        if not angle_ranges:
            return 1.0  # No ranges defined, full score
        
        total_score = 0.0
        count = 0
        
        for angle_name, (min_val, max_val) in angle_ranges.items():
            if angle_name not in current_angles:
                continue
            
            current = current_angles[angle_name]
            count += 1
            
            if min_val <= current <= max_val:
                total_score += 1.0  # Perfect, within range
            else:
                # Partial credit based on distance from range
                if current < min_val:
                    distance = min_val - current
                else:
                    distance = current - max_val
                
                # Decay: 30 degrees outside = 0 credit
                score = max(0, 1 - distance / 30)
                total_score += score
        
        return total_score / count if count > 0 else 1.0
    
    def combined_similarity(
        self,
        angle_result: AngleResult,
        pose_ref: PoseReference
    ) -> float:
        """
        Combined similarity using multiple metrics.
        
        Formula: 0.4 * euclidean + 0.3 * weighted + 0.3 * tolerance
        """
        current_vector = angle_result.angles_vector
        ref_vector = pose_ref.angles_vector
        
        # Euclidean similarity
        euclidean_sim = self.euclidean_similarity(current_vector, ref_vector)
        
        # Weighted similarity
        weighted_sim = self.weighted_euclidean_similarity(
            current_vector, ref_vector, pose_ref.weights
        )
        
        # Tolerance score
        tolerance_sim = self.tolerance_score(
            angle_result.angles_degrees,
            pose_ref.angle_ranges
        )
        
        # Combine scores
        combined = 0.4 * euclidean_sim + 0.3 * weighted_sim + 0.3 * tolerance_sim
        
        return float(combined)
    
    # =========================================================================
    # MAIN MATCHING METHODS
    # =========================================================================
    
    def match(self, angle_result: AngleResult) -> MatchResult:
        """
        Match current pose angles against database.
        """
        if not self.database:
            return MatchResult(
                pose_name="unknown",
                display_name="Unknown Pose",
                similarity=0.0,
                confidence="no_match"
            )
        
        current_vector = angle_result.angles_vector
        
        # Compare with all poses in database
        matches = []
        for pose_name, pose_ref in self.database.items():
            if self.method == "cosine":
                similarity = self.cosine_similarity(current_vector, pose_ref.angles_vector)
            elif self.method == "euclidean":
                similarity = self.euclidean_similarity(current_vector, pose_ref.angles_vector)
            elif self.method == "weighted":
                similarity = self.weighted_euclidean_similarity(
                    current_vector, pose_ref.angles_vector, pose_ref.weights
                )
            elif self.method == "combined":
                similarity = self.combined_similarity(angle_result, pose_ref)
            else:
                similarity = self.euclidean_similarity(current_vector, pose_ref.angles_vector)
            
            matches.append((pose_name, similarity, pose_ref))
        
        # Sort by similarity (highest first)
        matches.sort(key=lambda x: x[1], reverse=True)
        
        # Get best match
        best_name, best_similarity, best_ref = matches[0]
        
        # Determine confidence level
        if best_similarity >= self.THRESHOLD_HIGH:
            confidence = "high"
        elif best_similarity >= self.THRESHOLD_MEDIUM:
            confidence = "medium"
        elif best_similarity >= self.THRESHOLD_LOW:
            confidence = "low"
        else:
            confidence = "no_match"
        
        # Calculate angle differences for feedback
        angle_diffs = {}
        for angle_name in AngleCalculator.ANGLE_ORDER:
            current = angle_result.angles_degrees.get(angle_name, 180)
            reference = best_ref.angles_degrees.get(angle_name, 180)
            angle_diffs[angle_name] = current - reference
        
        # Top matches
        top_matches = [(name, sim) for name, sim, _ in matches[:5]]
        
        return MatchResult(
            pose_name=best_name,
            display_name=best_ref.display_name,
            similarity=best_similarity,
            confidence=confidence,
            method=self.method,
            top_matches=top_matches,
            angle_differences=angle_diffs
        )
    
    def match_from_landmarks(self, landmark_dict: Dict, use_3d: bool = True) -> MatchResult:
        """Match pose directly from landmarks dictionary."""
        angle_result = AngleCalculator.calculate_all_angles(landmark_dict, use_3d)
        return self.match(angle_result)
    
    def match_from_pose_result(self, pose_result, use_3d: bool = True) -> Optional[MatchResult]:
        """Match pose from PoseResult object."""
        if pose_result is None:
            return None
        
        angle_result = AngleCalculator.from_pose_result(pose_result, use_3d)
        if angle_result is None:
            return None
        
        return self.match(angle_result)
    
    def get_all_poses(self) -> List[str]:
        """Get list of all pose names in database."""
        return list(self.database.keys())
    
    def get_pose_info(self, pose_name: str) -> Optional[PoseReference]:
        """Get information about a specific pose."""
        return self.database.get(pose_name)
