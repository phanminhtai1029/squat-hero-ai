"""
Angle Calculator Utility
Calculates joint angles from pose landmarks.
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class AngleResult:
    """Result of angle calculation."""
    angles_degrees: Dict[str, float]
    angles_normalized: Dict[str, float]
    angles_vector: List[float]


class AngleCalculator:
    """
    Calculate joint angles from pose landmarks.
    
    Uses 8 key angles for yoga pose matching:
    - left_elbow, right_elbow
    - left_shoulder, right_shoulder  
    - left_knee, right_knee
    - left_hip, right_hip
    """
    
    # Angle definitions: (point_a, vertex, point_b)
    ANGLE_DEFINITIONS = {
        'left_elbow': ('left_shoulder', 'left_elbow', 'left_wrist'),
        'right_elbow': ('right_shoulder', 'right_elbow', 'right_wrist'),
        'left_shoulder': ('left_elbow', 'left_shoulder', 'left_hip'),
        'right_shoulder': ('right_elbow', 'right_shoulder', 'right_hip'),
        'left_knee': ('left_hip', 'left_knee', 'left_ankle'),
        'right_knee': ('right_hip', 'right_knee', 'right_ankle'),
        'left_hip': ('left_shoulder', 'left_hip', 'left_knee'),
        'right_hip': ('right_shoulder', 'right_hip', 'right_knee'),
    }
    
    # Order of angles in vector representation
    ANGLE_ORDER = [
        'left_elbow', 'right_elbow',
        'left_shoulder', 'right_shoulder',
        'left_knee', 'right_knee',
        'left_hip', 'right_hip'
    ]
    
    @staticmethod
    def calculate_angle(
        point_a: Tuple[float, float, float],
        vertex: Tuple[float, float, float],
        point_b: Tuple[float, float, float]
    ) -> float:
        """
        Calculate angle at vertex between point_a and point_b.
        
        Args:
            point_a: First point (x, y, z)
            vertex: Vertex point where angle is measured
            point_b: Second point (x, y, z)
            
        Returns:
            Angle in degrees [0, 180]
        """
        # Convert to numpy arrays
        a = np.array(point_a)
        v = np.array(vertex)
        b = np.array(point_b)
        
        # Create vectors from vertex to each point
        va = a - v
        vb = b - v
        
        # Calculate angle using dot product
        cos_angle = np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-6)
        
        # Clamp to valid range for arccos
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
        # Convert to degrees
        angle = np.degrees(np.arccos(cos_angle))
        
        return float(angle)
    
    @classmethod
    def calculate_all_angles(
        cls,
        landmark_dict: Dict[str, object],
        use_3d: bool = True
    ) -> AngleResult:
        """
        Calculate all 8 key angles from landmarks.
        
        Args:
            landmark_dict: Dictionary mapping landmark names to Landmark objects
            use_3d: Whether to use 3D coordinates (x, y, z) or just 2D (x, y)
            
        Returns:
            AngleResult with degrees, normalized and vector representations
        """
        angles_degrees = {}
        
        for angle_name, (pt_a_name, vertex_name, pt_b_name) in cls.ANGLE_DEFINITIONS.items():
            # Get landmarks
            pt_a = landmark_dict.get(pt_a_name)
            vertex = landmark_dict.get(vertex_name)
            pt_b = landmark_dict.get(pt_b_name)
            
            if pt_a is None or vertex is None or pt_b is None:
                angles_degrees[angle_name] = 180.0  # Default to straight
                continue
            
            # Extract coordinates
            if use_3d:
                a = (pt_a.x, pt_a.y, pt_a.z)
                v = (vertex.x, vertex.y, vertex.z)
                b = (pt_b.x, pt_b.y, pt_b.z)
            else:
                a = (pt_a.x, pt_a.y, 0)
                v = (vertex.x, vertex.y, 0)
                b = (pt_b.x, pt_b.y, 0)
            
            angle = cls.calculate_angle(a, v, b)
            angles_degrees[angle_name] = angle
        
        # Normalize angles to [0, 1]
        angles_normalized = {
            name: angle / 180.0 
            for name, angle in angles_degrees.items()
        }
        
        # Create ordered vector
        angles_vector = [
            angles_normalized[name] 
            for name in cls.ANGLE_ORDER
        ]
        
        return AngleResult(
            angles_degrees=angles_degrees,
            angles_normalized=angles_normalized,
            angles_vector=angles_vector
        )
    
    @classmethod
    def from_pose_result(cls, pose_result, use_3d: bool = True) -> Optional[AngleResult]:
        """
        Calculate angles from PoseResult object.
        
        Args:
            pose_result: PoseResult from PoseEstimator
            use_3d: Whether to use 3D coordinates
            
        Returns:
            AngleResult or None if pose_result is None
        """
        if pose_result is None:
            return None
        
        return cls.calculate_all_angles(pose_result.landmark_dict, use_3d)
