#!/usr/bin/env python3
"""
Fast baseline benchmark with live progress - samples only 200 images per method.
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

sys.path.append(str(Path(__file__).parent.parent))

import torch
from ultralytics import YOLO
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from utils.angle_calculator import AngleCalculator
from ml.models.pose_gnn_encoder import SimplePoseGNN
from pipeline.step5b_vector_matcher import VectorMatcher


class FastBenchmark:
    """Fast benchmark with progress tracking."""
    
    def __init__(self, test_dir: Path, max_samples: int = 200):
        self.test_dir = test_dir
        self.max_samples = max_samples
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print("Loading models...")
        from pipeline.step3_pose_estimation import PoseEstimator
        self.pose_estimator = PoseEstimator()
        
        self.gnn_encoder = SimplePoseGNN().to(self.device)
        self.gnn_encoder.eval()
        
        self.vector_matcher = VectorMatcher(database_path='data/pose_vectors_final.npz')
        
        print(f"✅ Models loaded (Device: {self.device})")
    
    def extract_keypoints(self, image_path: Path) -> np.ndarray:
        """Extract keypoints from image."""
        image = cv2.imread(str(image_path))
        if image is None:
            return None
        
        result = self.pose_estimator.estimate(image)
        if result is None:
            return None
        
        return result.to_numpy()[:, :2]
    
    def calculate_angles(self, keypoints: np.ndarray) -> Dict[str, float]:
        """Calculate pose angles."""
        angles = {}
        angle_defs = {
            'left_elbow': (5, 7, 9),
            'right_elbow': (6, 8, 10),
            'left_shoulder': (7, 5, 11),
            'right_shoulder': (8, 6, 12),
            'left_hip': (5, 11, 13),
            'right_hip': (6, 12, 14),
            'left_knee': (11, 13, 15),
            'right_knee': (12, 14, 16),
        }
        
        for name, (p1_idx, vertex_idx, p2_idx) in angle_defs.items():
            if p1_idx < len(keypoints) and vertex_idx < len(keypoints) and p2_idx < len(keypoints):
                p1_3d = np.append(keypoints[p1_idx], 0)
                vertex_3d = np.append(keypoints[vertex_idx], 0)
                p2_3d = np.append(keypoints[p2_idx], 0)
                
                angle = AngleCalculator.calculate_angle(p1_3d, vertex_3d, p2_3d)
                angles[name] = angle
        
        return angles
    
    def angle_match(self, test_angles: Dict[str, float], reference: Dict[str, Dict[str, float]]) -> str:
        """Match by angle similarity."""
        best_pose = None
        best_sim = -1
        
        angle_names = sorted(test_angles.keys())
        test_vec = np.array([test_angles.get(n, 0) for n in angle_names])
        
        for pose_name, ref_angles in reference.items():
            ref_vec = np.array([ref_angles.get(n, 0) for n in angle_names])
            
            dot = np.dot(test_vec, ref_vec)
            norm = np.linalg.norm(test_vec) * np.linalg.norm(ref_vec)
            
            if norm > 0:
                sim = dot / norm
                if sim > best_sim:
                    best_sim = sim
                    best_pose = pose_name
        
        return best_pose
    
    def vector_match(self, keypoints: np.ndarray) -> str:
        """Match by GNN vector."""
        kp_tensor = torch.FloatTensor(keypoints).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            embedding = self.gnn_encoder(kp_tensor)
        
        embedding_np = embedding.cpu().numpy().flatten()
        result = self.vector_matcher.match(embedding_np)
        return result.pose_name  # MatchResult object, not dict
    
    def run(self):
        """Run fast benchmark."""
        # Load test data
        test_data = defaultdict(list)
        for pose_folder in self.test_dir.iterdir():
            if pose_folder.is_dir():
                images = list(pose_folder.glob('*.jpg')) + list(pose_folder.glob('*.png'))
                test_data[pose_folder.name] = images[:5]  # Max 5 per class
        
        # Sample uniformly
        all_samples = []
        for pose_name, images in test_data.items():
            for img in images:
                all_samples.append((pose_name, img))
        
        np.random.seed(42)
        np.random.shuffle(all_samples)
        all_samples = all_samples[:self.max_samples]
        
        print(f"\n{'='*80}")
        print(f"FAST BASELINE BENCHMARK")
        print(f"{'='*80}")
        print(f"Total samples: {len(all_samples)}")
        print(f"Unique poses: {len(test_data)}")
        
        # Build angle reference
        print(f"\nBuilding angle reference...")
        train_dir = self.test_dir.parent / 'train'
        angle_ref = {}
        
        for pose_folder in train_dir.iterdir():
            if not pose_folder.is_dir():
                continue
            
            images = list(pose_folder.glob('*.jpg')) + list(pose_folder.glob('*.png'))
            sample_imgs = images[:10]
            
            all_angles = []
            for img in sample_imgs:
                kp = self.extract_keypoints(img)
                if kp is not None:
                    angles = self.calculate_angles(kp)
                    if angles:
                        all_angles.append(angles)
            
            if all_angles:
                avg_angles = {}
                for key in all_angles[0].keys():
                    vals = [a[key] for a in all_angles if key in a]
                    if vals:
                        avg_angles[key] = np.mean(vals)
                angle_ref[pose_folder.name] = avg_angles
        
        print(f"✅ Reference built for {len(angle_ref)} poses")
        
        # Test both methods
        results = {}
        
        for method_name, method_fn in [('Angle-based', 'angle'), ('Vector-based', 'vector')]:
            print(f"\n{'='*80}")
            print(f"{method_name} Method")
            print(f"{'='*80}")
            
            y_true = []
            y_pred = []
            times = []
            failed = 0
            
            for i, (true_pose, img_path) in enumerate(all_samples):
                if (i+1) % 50 == 0:
                    print(f"  Progress: {i+1}/{len(all_samples)} ({(i+1)/len(all_samples)*100:.1f}%)")
                
                kp = self.extract_keypoints(img_path)
                if kp is None:
                    failed += 1
                    continue
                
                try:
                    start = time.time()
                    
                    if method_fn == 'angle':
                        angles = self.calculate_angles(kp)
                        pred = self.angle_match(angles, angle_ref)
                    else:
                        pred = self.vector_match(kp)
                    
                    elapsed = time.time() - start
                    times.append(elapsed)
                    
                    y_true.append(true_pose)
                    y_pred.append(pred)
                except Exception as e:
                    failed += 1
                    continue
            
            # Metrics
            if len(y_true) > 0:
                acc = accuracy_score(y_true, y_pred)
                prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
                avg_time = np.mean(times) * 1000
                fps = 1000 / avg_time if avg_time > 0 else 0
                
                results[method_name] = {
                    'accuracy': acc,
                    'precision': prec,
                    'recall': rec,
                    'f1': f1,
                    'avg_time_ms': avg_time,
                    'fps': fps,
                    'samples': len(y_true),
                    'failed': failed
                }
                
                print(f"\n📊 Results:")
                print(f"  Accuracy:  {acc*100:.2f}%")
                print(f"  Precision: {prec*100:.2f}%")
                print(f"  Recall:    {rec*100:.2f}%")
                print(f"  F1-Score:  {f1*100:.2f}%")
                print(f"  Time:      {avg_time:.2f} ms ({fps:.0f} FPS)")
                print(f"  Samples:   {len(y_true)} success, {failed} failed")
            else:
                print(f"\n⚠️  No successful predictions!")
                results[method_name] = {'error': 'No successful predictions'}
        
        # Save
        output_file = Path(__file__).parent.parent / "evaluation" / "fast_baseline.json"
        with open(output_file, 'w') as f:
            json.dump({
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'samples': len(all_samples),
                'results': {k: {kk: float(vv) if isinstance(vv, (int, float, np.number)) else vv 
                               for kk, vv in v.items()} for k, v in results.items()}
            }, f, indent=2)
        
        print(f"\n{'='*80}")
        print(f"RESULTS SAVED: {output_file}")
        print(f"{'='*80}")


def main():
    test_dir = Path(__file__).parent.parent / "data" / "processed" / "yoga_final" / "test"
    
    benchmark = FastBenchmark(test_dir, max_samples=200)
    benchmark.run()


if __name__ == "__main__":
    main()
