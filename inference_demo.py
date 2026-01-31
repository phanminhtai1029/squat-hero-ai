"""
Inference Demo Script
======================
Chạy inference trên images và hiển thị kết quả.

Usage:
    python inference_demo.py --images data/raw/DATASET/TEST/plank/img1.jpg data/raw/DATASET/TEST/tree/img2.jpg
    python inference_demo.py --num 5  # Random 5 images từ test set
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
import random
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import mediapipe as mp
from step4_pose_classifier import PoseClassifier
from step5_form_scorer import FormScorer


class InferenceDemo:
    """Demo inference pipeline."""
    
    def __init__(self):
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize classifier and scorer
        print("Loading model...")
        self.classifier = PoseClassifier()
        self.scorer = FormScorer()
        print("Model loaded!")
    
    def process_image(self, image_path: str) -> dict:
        """
        Process một image và trả về kết quả.
        
        Returns:
            dict với keys: image, label, confidence, score, feedback
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return {"error": f"Cannot load image: {image_path}"}
        
        # Convert to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect pose
        results = self.pose.process(rgb_image)
        
        if not results.pose_landmarks:
            return {
                "image": image,
                "error": "No pose detected",
                "image_path": image_path
            }
        
        # Extract landmarks to dict format
        landmarks = self._extract_landmarks(results.pose_landmarks)
        
        # Step 4: Classify
        classification = self.classifier.classify(landmarks)
        
        # Step 5: Score
        scoring = self.scorer.score(landmarks, classification.label)
        
        # Draw skeleton on image
        output_image = image.copy()
        self.mp_drawing.draw_landmarks(
            output_image,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS
        )
        
        # Draw label and score
        label_text = f"{classification.label.upper()} ({classification.confidence:.0%})"
        score_text = f"Score: {scoring.score:.0f}%"
        
        cv2.putText(output_image, label_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(output_image, score_text, (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(output_image, scoring.feedback, (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return {
            "image": output_image,
            "image_path": image_path,
            "label": classification.label,
            "confidence": classification.confidence,
            "score": scoring.score,
            "feedback": scoring.feedback,
            "all_probs": classification.all_probs
        }
    
    def _extract_landmarks(self, pose_landmarks) -> dict:
        """Convert MediaPipe landmarks to dict format."""
        from collections import namedtuple
        Landmark = namedtuple('Landmark', ['x', 'y', 'z', 'visibility'])
        
        # Mapping of landmark indices to names
        landmark_names = {
            11: 'left_shoulder', 12: 'right_shoulder',
            23: 'left_hip', 24: 'right_hip',
            25: 'left_knee', 26: 'right_knee',
            27: 'left_ankle', 28: 'right_ankle',
        }
        
        landmarks = {}
        for idx, name in landmark_names.items():
            lm = pose_landmarks.landmark[idx]
            landmarks[name] = Landmark(lm.x, lm.y, lm.z, lm.visibility)
        
        return landmarks
    
    def run_demo(self, image_paths: list, save_dir: str = "reports/inference_results"):
        """
        Chạy demo trên danh sách images.
        
        Args:
            image_paths: List các đường dẫn ảnh
            save_dir: Thư mục lưu kết quả
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        results = []
        
        print(f"\n{'='*60}")
        print("INFERENCE DEMO RESULTS")
        print(f"{'='*60}\n")
        
        for i, img_path in enumerate(image_paths, 1):
            print(f"[{i}/{len(image_paths)}] Processing: {Path(img_path).name}")
            
            result = self.process_image(img_path)
            
            if "error" in result and result.get("error") != "No pose detected":
                print(f"  ❌ Error: {result['error']}")
                continue
            
            if "error" in result:
                print(f"  ⚠️ No pose detected")
                continue
            
            # Print result
            print(f"  📌 Label: {result['label'].upper()}")
            print(f"  🎯 Confidence: {result['confidence']:.1%}")
            print(f"  📊 Score: {result['score']:.0f}%")
            print(f"  💬 Feedback: {result['feedback']}")
            
            # Get ground truth from folder name
            ground_truth = Path(img_path).parent.name
            is_correct = result['label'].lower() == ground_truth.lower()
            print(f"  ✓ Correct: {is_correct} (Ground Truth: {ground_truth})")
            print()
            
            # Save result image
            output_path = save_path / f"result_{i}_{result['label']}.jpg"
            cv2.imwrite(str(output_path), result['image'])
            
            result['ground_truth'] = ground_truth
            result['correct'] = is_correct
            results.append(result)
        
        # Summary
        correct_count = sum(1 for r in results if r.get('correct', False))
        print(f"{'='*60}")
        print(f"SUMMARY: {correct_count}/{len(results)} correct ({100*correct_count/max(len(results),1):.1f}%)")
        print(f"Results saved to: {save_path}")
        print(f"{'='*60}")
        
        return results
    
    def close(self):
        self.pose.close()


def get_random_test_images(test_dir: str, num: int = 5) -> list:
    """Lấy ngẫu nhiên num images từ test set."""
    test_path = Path(test_dir)
    all_images = []
    
    for class_dir in test_path.iterdir():
        if class_dir.is_dir():
            images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
            all_images.extend(images)
    
    if len(all_images) < num:
        return [str(p) for p in all_images]
    
    selected = random.sample(all_images, num)
    return [str(p) for p in selected]


def main():
    parser = argparse.ArgumentParser(description="Inference demo")
    parser.add_argument("--images", "-i", nargs="+", help="Image paths")
    parser.add_argument("--num", "-n", type=int, default=5, 
                       help="Number of random images from test set")
    parser.add_argument("--test-dir", default="data/raw/DATASET/TEST",
                       help="Test directory")
    parser.add_argument("--output", "-o", default="reports/inference_results",
                       help="Output directory")
    
    args = parser.parse_args()
    
    # Get image paths
    if args.images:
        image_paths = args.images
    else:
        print(f"Selecting {args.num} random images from {args.test_dir}...")
        image_paths = get_random_test_images(args.test_dir, args.num)
    
    if not image_paths:
        print("No images found!")
        return
    
    # Run demo
    demo = InferenceDemo()
    demo.run_demo(image_paths, args.output)
    demo.close()


if __name__ == "__main__":
    main()
