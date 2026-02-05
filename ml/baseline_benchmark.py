#!/usr/bin/env python3
"""
Baseline Benchmark: Test current methods on yoga_final test set.
This establishes baseline performance before training GNN encoder.
"""

import os
import sys
from pathlib import Path
import numpy as np
import time
import json
import cv2
from collections import defaultdict
from typing import Dict, List, Tuple

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import torch
from ultralytics import YOLO
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from utils.angle_calculator import AngleCalculator
from ml.models.pose_gnn_encoder import SimplePoseGNN
from pipeline.step5b_vector_matcher import VectorMatcher


class BaselineBenchmark:
    """Benchmark current methods before training."""
    
    def __init__(self, test_dir: Path):
        self.test_dir = test_dir
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load models
        print("Loading models...")
        self.yolo_model = YOLO('yolov8s-pose.pt')
        self.gnn_encoder = SimplePoseGNN().to(self.device)
        self.gnn_encoder.eval()
        
        # Load vector matcher (uses new vector database)
        self.vector_matcher = VectorMatcher(database_path='data/pose_vectors_final.npz')
        
        # Load pose estimator for keypoint extraction
        from pipeline.step3_pose_estimation import PoseEstimator
        self.pose_estimator = PoseEstimator()
        
        print(f"Device: {self.device}")
        print(f"Test directory: {test_dir}")
    
    def extract_keypoints(self, image_path: Path) -> np.ndarray:
        """Extract pose keypoints using YOLOv8."""
        results = self.yolo_model(str(image_path), verbose=False)
        
        if len(results) == 0 or results[0].keypoints is None:
            return None
        
        keypoints = results[0].keypoints.data
        if keypoints.shape[0] == 0:
            return None
        
        # Get first person's keypoints (17 x 3: x, y, confidence)
        kp = keypoints[0].cpu().numpy()
        return kp[:, :2]  # Return only x, y (17, 2)
    
    def calculate_pose_angles(self, keypoints: np.ndarray) -> Dict[str, float]:
        """Calculate key angles for pose (same as step5_pose_matcher)."""
        angles = {}
        
        # Define angle triplets (point1, vertex, point2)
        angle_defs = {
            'left_elbow': (5, 7, 9),      # left_shoulder, left_elbow, left_wrist
            'right_elbow': (6, 8, 10),    # right_shoulder, right_elbow, right_wrist
            'left_shoulder': (7, 5, 11),  # left_elbow, left_shoulder, left_hip
            'right_shoulder': (8, 6, 12), # right_elbow, right_shoulder, right_hip
            'left_hip': (5, 11, 13),      # left_shoulder, left_hip, left_knee
            'right_hip': (6, 12, 14),     # right_shoulder, right_hip, right_knee
            'left_knee': (11, 13, 15),    # left_hip, left_knee, left_ankle
            'right_knee': (12, 14, 16),   # right_hip, right_knee, right_ankle
        }
        
        for name, (p1_idx, vertex_idx, p2_idx) in angle_defs.items():
            if p1_idx < len(keypoints) and vertex_idx < len(keypoints) and p2_idx < len(keypoints):
                p1 = keypoints[p1_idx]
                vertex = keypoints[vertex_idx]
                p2 = keypoints[p2_idx]
                
                # Add z=0 for 2D points
                p1_3d = np.append(p1, 0)
                vertex_3d = np.append(vertex, 0)
                p2_3d = np.append(p2, 0)
                
                angle = AngleCalculator.calculate_angle(p1_3d, vertex_3d, p2_3d)
                angles[name] = angle
        
        return angles
    
    def angle_based_matching(self, test_angles: Dict[str, float], 
                            reference_angles: Dict[str, Dict[str, float]]) -> str:
        """Match pose using angle similarity (cosine similarity on angle vectors)."""
        best_pose = None
        best_similarity = -1
        
        # Convert angles to vectors
        angle_names = sorted(test_angles.keys())
        test_vector = np.array([test_angles.get(name, 0) for name in angle_names])
        
        for pose_name, ref_angles in reference_angles.items():
            ref_vector = np.array([ref_angles.get(name, 0) for name in angle_names])
            
            # Cosine similarity
            dot_product = np.dot(test_vector, ref_vector)
            norm_product = np.linalg.norm(test_vector) * np.linalg.norm(ref_vector)
            
            if norm_product > 0:
                similarity = dot_product / norm_product
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_pose = pose_name
        
        return best_pose
    
    def vector_based_matching(self, keypoints: np.ndarray) -> str:
        """Match pose using GNN encoder + vector similarity."""
        # Encode keypoints
        kp_tensor = torch.FloatTensor(keypoints).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            embedding = self.gnn_encoder(kp_tensor)
        
        embedding_np = embedding.cpu().numpy().flatten()
        
        # Match using vector matcher
        result = self.vector_matcher.match(embedding_np)
        return result.pose_name  # MatchResult object, not dict
    
    def extract_keypoints_from_result(self, image_path: Path) -> np.ndarray:
        """Extract keypoints as PoseEstimator format."""
        image = cv2.imread(str(image_path))
        if image is None:
            return None
        
        result = self.pose_estimator.estimate(image)
        if result is None:
            return None
        
        return result.to_numpy()[:, :2]  # Return (17, 2)
    
    def load_test_data(self) -> Dict[str, List[Path]]:
        """Load test images by pose class."""
        test_images = defaultdict(list)
        
        for pose_folder in self.test_dir.iterdir():
            if not pose_folder.is_dir():
                continue
            
            pose_name = pose_folder.name
            
            for img_file in pose_folder.iterdir():
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    test_images[pose_name].append(img_file)
        
        return dict(test_images)
    
    def build_angle_reference(self, train_dir: Path) -> Dict[str, Dict[str, float]]:
        """Build reference angles from training set (average angles per class)."""
        reference_angles = {}
        
        print("\nBuilding angle reference from training set...")
        
        for pose_folder in train_dir.iterdir():
            if not pose_folder.is_dir():
                continue
            
            pose_name = pose_folder.name
            all_angles = []
            
            # Sample up to 20 images per class for reference
            images = list(pose_folder.glob('*.jpg')) + list(pose_folder.glob('*.png'))
            sample_images = images[:20] if len(images) > 20 else images
            
            for img_path in sample_images:
                keypoints = self.extract_keypoints(img_path)
                if keypoints is not None:
                    angles = self.calculate_pose_angles(keypoints)
                    if angles:
                        all_angles.append(angles)
            
            # Average angles
            if all_angles:
                avg_angles = {}
                angle_names = all_angles[0].keys()
                
                for name in angle_names:
                    values = [a[name] for a in all_angles if name in a]
                    if values:
                        avg_angles[name] = np.mean(values)
                
                reference_angles[pose_name] = avg_angles
        
        print(f"Built reference for {len(reference_angles)} poses")
        return reference_angles
    
    def run_benchmark(self, method: str = 'both') -> Dict:
        """
        Run benchmark on test set.
        
        Args:
            method: 'angle', 'vector', or 'both'
        """
        # Load test data
        test_data = self.load_test_data()
        
        print(f"\n{'='*80}")
        print(f"BASELINE BENCHMARK - Test Set Evaluation")
        print(f"{'='*80}")
        print(f"Test poses: {len(test_data)}")
        print(f"Test images: {sum(len(imgs) for imgs in test_data.values())}")
        
        results = {}
        
        # Build angle reference if needed
        if method in ['angle', 'both']:
            train_dir = self.test_dir.parent / 'train'
            angle_reference = self.build_angle_reference(train_dir)
        
        # Angle-based benchmark
        if method in ['angle', 'both']:
            print(f"\n{'='*80}")
            print("ANGLE-BASED METHOD")
            print(f"{'='*80}")
            
            results['angle_based'] = self._benchmark_method(
                test_data, 
                angle_reference, 
                method_type='angle'
            )
        
        # Vector-based benchmark
        if method in ['vector', 'both']:
            print(f"\n{'='*80}")
            print("VECTOR-BASED METHOD (Untrained GNN)")
            print(f"{'='*80}")
            
            results['vector_based'] = self._benchmark_method(
                test_data,
                None,
                method_type='vector'
            )
        
        return results
    
    def _benchmark_method(self, test_data: Dict[str, List[Path]], 
                         reference_data, method_type: str) -> Dict:
        """Benchmark a single method."""
        y_true = []
        y_pred = []
        inference_times = []
        failed = 0
        
        total_images = sum(len(imgs) for imgs in test_data.values())
        processed = 0
        
        for pose_name, image_paths in test_data.items():
            for img_path in image_paths:
                processed += 1
                if processed % 100 == 0:
                    print(f"  Progress: {processed}/{total_images} ({processed/total_images*100:.1f}%)")
                
                # Extract keypoints using new method
                keypoints = self.extract_keypoints_from_result(img_path)
                
                if keypoints is None:
                    failed += 1
                    continue
                
                # Predict
                start_time = time.time()
                
                try:
                    if method_type == 'angle':
                        angles = self.calculate_pose_angles(keypoints)
                        predicted = self.angle_based_matching(angles, reference_data)
                    else:  # vector
                        predicted = self.vector_based_matching(keypoints)
                    
                    inference_time = time.time() - start_time
                    inference_times.append(inference_time)
                    
                    y_true.append(pose_name)
                    y_pred.append(predicted)
                    
                except Exception as e:
                    failed += 1
                    continue
        
        # Calculate metrics (handle empty arrays)
        if len(y_true) == 0 or len(y_pred) == 0:
            print(f"\n⚠️  WARNING: No successful predictions!")
            print(f"   Total samples attempted: {total_images}")
            print(f"   Failed: {failed}")
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'avg_inference_time_ms': 0.0,
                'fps': 0.0,
                'total_samples': 0,
                'failed_samples': failed,
                'per_class_accuracy': {},
                'y_true': [],
                'y_pred': []
            }
        
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        avg_time = np.mean(inference_times) if inference_times else 0
        fps = 1 / avg_time if avg_time > 0 else 0
        
        # Per-class metrics
        unique_classes = sorted(set(y_true))
        per_class_acc = {}
        
        for cls in unique_classes:
            cls_true = [i for i, gt in enumerate(y_true) if gt == cls]
            if cls_true:
                cls_correct = sum(1 for i in cls_true if y_pred[i] == cls)
                per_class_acc[cls] = cls_correct / len(cls_true)
        
        # Results
        result = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'avg_inference_time_ms': avg_time * 1000,
            'fps': fps,
            'total_samples': len(y_true),
            'failed_samples': failed,
            'per_class_accuracy': per_class_acc,
            'y_true': y_true,
            'y_pred': y_pred
        }
        
        # Print summary
        print(f"\n📊 Results:")
        print(f"  Accuracy:  {accuracy*100:.2f}%")
        print(f"  Precision: {precision*100:.2f}%")
        print(f"  Recall:    {recall*100:.2f}%")
        print(f"  F1-Score:  {f1*100:.2f}%")
        print(f"  Inference: {avg_time*1000:.2f} ms/image ({fps:.0f} FPS)")
        print(f"  Samples:   {len(y_true)} successful, {failed} failed")
        
        # Top/bottom classes
        sorted_classes = sorted(per_class_acc.items(), key=lambda x: x[1], reverse=True)
        print(f"\n🏆 Best 5 classes:")
        for cls, acc in sorted_classes[:5]:
            count = sum(1 for gt in y_true if gt == cls)
            print(f"  {cls:40} {acc*100:5.1f}% ({count} samples)")
        
        print(f"\n📉 Worst 5 classes:")
        for cls, acc in sorted_classes[-5:]:
            count = sum(1 for gt in y_true if gt == cls)
            print(f"  {cls:40} {acc*100:5.1f}% ({count} samples)")
        
        return result


def main():
    # Paths
    test_dir = Path(__file__).parent.parent / "data" / "processed" / "yoga_final" / "test"
    output_file = Path(__file__).parent.parent / "evaluation" / "baseline_benchmark.json"
    
    # Create output directory
    output_file.parent.mkdir(exist_ok=True)
    
    # Run benchmark
    benchmark = BaselineBenchmark(test_dir)
    results = benchmark.run_benchmark(method='both')
    
    # Save results
    # Convert to JSON-serializable format
    save_results = {}
    for method, data in results.items():
        save_results[method] = {
            'accuracy': float(data['accuracy']),
            'precision': float(data['precision']),
            'recall': float(data['recall']),
            'f1_score': float(data['f1_score']),
            'avg_inference_time_ms': float(data['avg_inference_time_ms']),
            'fps': float(data['fps']),
            'total_samples': int(data['total_samples']),
            'failed_samples': int(data['failed_samples']),
            'per_class_accuracy': {k: float(v) for k, v in data['per_class_accuracy'].items()}
        }
    
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'test_set': str(test_dir),
            'model_status': 'untrained',
            'results': save_results
        }, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"RESULTS SAVED")
    print(f"{'='*80}")
    print(f"File: {output_file}")
    print(f"\n✅ Baseline benchmark complete!")
    print(f"   Use this as reference to compare with trained GNN encoder.")


if __name__ == "__main__":
    main()
