"""
Evaluation Script for Yoga Pose Recognition Pipeline

Evaluates:
1. Pose Matcher accuracy on test images
2. Pipeline performance (latency)
"""

import sys
import time
import json
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pipeline.step1_frame_capture import ImageCapture
from pipeline.step2_person_detection import PersonDetector
from pipeline.step3_pose_estimation import PoseEstimator
from pipeline.step4_frame_classifier import FrameClassifier
from pipeline.step5_pose_matcher import PoseMatcher, MatchResult
from utils.angle_calculator import AngleCalculator


@dataclass
class EvaluationResults:
    """Container for evaluation results."""
    # Pose matcher metrics
    total_images: int = 0
    total_detected: int = 0
    top1_correct: int = 0
    top3_correct: int = 0
    
    # Similarity scores
    similarities_correct: List[float] = field(default_factory=list)
    similarities_wrong: List[float] = field(default_factory=list)
    
    # Per-pose results
    per_pose_results: Dict[str, Dict] = field(default_factory=dict)
    
    # Confusion matrix
    confusion_matrix: Dict[str, Dict[str, int]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(int)))
    
    # Timing
    latencies_ms: List[float] = field(default_factory=list)
    
    def compute_metrics(self) -> Dict:
        """Compute final metrics."""
        top1_acc = self.top1_correct / self.total_detected if self.total_detected > 0 else 0
        top3_acc = self.top3_correct / self.total_detected if self.total_detected > 0 else 0
        
        mean_sim_correct = np.mean(self.similarities_correct) if self.similarities_correct else 0
        mean_sim_wrong = np.mean(self.similarities_wrong) if self.similarities_wrong else 0
        
        avg_latency = np.mean(self.latencies_ms) if self.latencies_ms else 0
        fps = 1000 / avg_latency if avg_latency > 0 else 0
        
        return {
            'total_images': self.total_images,
            'total_detected': self.total_detected,
            'detection_rate': self.total_detected / self.total_images if self.total_images > 0 else 0,
            'top1_accuracy': top1_acc,
            'top3_accuracy': top3_acc,
            'mean_similarity_correct': mean_sim_correct,
            'mean_similarity_wrong': mean_sim_wrong,
            'avg_latency_ms': avg_latency,
            'avg_fps': fps,
            'per_pose_results': dict(self.per_pose_results),
            'confusion_matrix': {k: dict(v) for k, v in self.confusion_matrix.items()}
        }


class PipelineEvaluator:
    """Evaluates the full yoga pose recognition pipeline."""
    
    def __init__(self, database_path: str, use_yolo: bool = False):
        """
        Initialize evaluator.
        
        Args:
            database_path: Path to pose database YAML
            use_yolo: Whether to use YOLO for person detection
        """
        self.person_detector = PersonDetector() if use_yolo else None
        self.pose_estimator = PoseEstimator(static_image_mode=True)
        self.frame_classifier = FrameClassifier()
        self.pose_matcher = PoseMatcher(database_path)
        
        print(f"Loaded {len(self.pose_matcher.database)} poses from database")
    
    def evaluate_image(self, image_path: str, true_pose: str) -> Optional[Dict]:
        """
        Evaluate a single image.
        
        Args:
            image_path: Path to image file
            true_pose: Ground truth pose name
            
        Returns:
            Evaluation result dict or None if pose not detected
        """
        import cv2
        
        start_time = time.time()
        
        # Step 1: Load image
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        # Step 2: Person detection (skip if not using YOLO)
        if self.person_detector:
            detection = self.person_detector.detect_main_person(image)
            person_image = detection.cropped_image
        else:
            person_image = image
        
        # Step 3: Pose estimation
        pose_result = self.pose_estimator.estimate(person_image)
        if pose_result is None:
            return None
        
        # Step 4 & 5: Pose matching (skip frame classification for static images)
        match_result = self.pose_matcher.match_from_pose_result(pose_result)
        if match_result is None:
            return None
        
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        
        return {
            'predicted_pose': match_result.pose_name,
            'similarity': match_result.similarity,
            'confidence': match_result.confidence,
            'top_matches': match_result.top_matches,
            'latency_ms': latency_ms
        }
    
    def evaluate_dataset(self, dataset_path: str) -> EvaluationResults:
        """
        Evaluate on a dataset organized by pose folders.
        
        Expected structure:
        dataset_path/
            pose1/
                Images/
                    image1.jpg
                    ...
            pose2/
                Images/
                    ...
        """
        results = EvaluationResults()
        dataset = Path(dataset_path)
        
        # Map folder names to database pose names
        folder_to_db = {
            'ArdhaChandrasana': 'ardha_chandrasana',
            'BaddhaKonasana': 'baddha_konasana',
            'Downward_dog': 'downward_dog',
            'Natarajasana': 'natarajasana',
            'Triangle': 'triangle',
            'UtkataKonasana': 'utkata_konasana',
            'Veerabhadrasana': 'veerabhadrasana',
            'Vrukshasana': 'vrukshasana',
        }
        
        for pose_folder in dataset.iterdir():
            if not pose_folder.is_dir():
                continue
            
            pose_name = pose_folder.name
            if pose_name not in folder_to_db:
                continue
            
            db_pose_name = folder_to_db[pose_name]
            
            # Check if pose is in database
            if db_pose_name not in self.pose_matcher.database:
                print(f"Warning: {db_pose_name} not in database, skipping")
                continue
            
            images_dir = pose_folder / 'Images'
            if not images_dir.exists():
                continue
            
            pose_correct = 0
            pose_total = 0
            pose_detected = 0
            
            for image_path in images_dir.glob('*'):
                if image_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                    continue
                
                results.total_images += 1
                pose_total += 1
                
                eval_result = self.evaluate_image(str(image_path), db_pose_name)
                
                if eval_result is None:
                    continue
                
                results.total_detected += 1
                pose_detected += 1
                
                predicted = eval_result['predicted_pose']
                similarity = eval_result['similarity']
                
                # Update confusion matrix
                results.confusion_matrix[db_pose_name][predicted] += 1
                
                # Top-1 accuracy
                if predicted == db_pose_name:
                    results.top1_correct += 1
                    pose_correct += 1
                    results.similarities_correct.append(similarity)
                else:
                    results.similarities_wrong.append(similarity)
                
                # Top-3 accuracy
                top3_poses = [m[0] for m in eval_result['top_matches'][:3]]
                if db_pose_name in top3_poses:
                    results.top3_correct += 1
                
                # Timing
                results.latencies_ms.append(eval_result['latency_ms'])
            
            # Per-pose results
            if pose_detected > 0:
                results.per_pose_results[db_pose_name] = {
                    'total': pose_total,
                    'detected': pose_detected,
                    'correct': pose_correct,
                    'accuracy': pose_correct / pose_detected
                }
            
            print(f"  {pose_name}: {pose_correct}/{pose_detected} correct ({pose_correct/pose_detected*100:.1f}%)" if pose_detected > 0 else f"  {pose_name}: No detections")
        
        return results


def main():
    """Run evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Yoga Pose Pipeline')
    parser.add_argument('--dataset', type=str, 
                       default='data/Yoga_Poses-Dataset/TRAIN',
                       help='Path to dataset')
    parser.add_argument('--database', type=str,
                       default='data/pose_database.yaml',
                       help='Path to pose database')
    parser.add_argument('--use-yolo', action='store_true',
                       help='Use YOLO for person detection')
    parser.add_argument('--output', type=str,
                       default='evaluation/evaluation_report.json',
                       help='Output path for results')
    args = parser.parse_args()
    
    print("="*60)
    print("YOGA POSE RECOGNITION PIPELINE EVALUATION")
    print("="*60)
    print(f"\nDataset: {args.dataset}")
    print(f"Database: {args.database}")
    print(f"Use YOLO: {args.use_yolo}")
    print()
    
    # Initialize evaluator
    evaluator = PipelineEvaluator(
        database_path=args.database,
        use_yolo=args.use_yolo
    )
    
    print("\nEvaluating poses...")
    print("-"*40)
    
    # Run evaluation
    results = evaluator.evaluate_dataset(args.dataset)
    
    # Compute metrics
    metrics = results.compute_metrics()
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"\nTotal Images: {metrics['total_images']}")
    print(f"Detection Rate: {metrics['detection_rate']*100:.1f}%")
    print(f"\n--- Accuracy ---")
    print(f"Top-1 Accuracy: {metrics['top1_accuracy']*100:.1f}%")
    print(f"Top-3 Accuracy: {metrics['top3_accuracy']*100:.1f}%")
    print(f"\n--- Similarity Scores ---")
    print(f"Mean Similarity (Correct): {metrics['mean_similarity_correct']:.3f}")
    print(f"Mean Similarity (Wrong): {metrics['mean_similarity_wrong']:.3f}")
    print(f"\n--- Performance ---")
    print(f"Avg Latency: {metrics['avg_latency_ms']:.1f}ms")
    print(f"Avg FPS: {metrics['avg_fps']:.1f}")
    
    print("\n--- Per-Pose Accuracy ---")
    for pose, data in metrics['per_pose_results'].items():
        print(f"  {pose}: {data['accuracy']*100:.1f}% ({data['correct']}/{data['detected']})")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    # Close resources
    evaluator.pose_estimator.close()
    
    return metrics


if __name__ == '__main__':
    main()
