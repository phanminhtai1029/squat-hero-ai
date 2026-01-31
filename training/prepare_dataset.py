"""
Dataset Preparation Script
===========================
Extract keypoints từ images sử dụng MediaPipe,
sau đó lưu thành CSV để train classifier.

Usage:
    python prepare_dataset.py --input data/raw --output data/processed/pose_dataset.csv
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import sys
import os

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mediapipe as mp


class DatasetPreparer:
    """Extract keypoints từ images và tạo dataset."""
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.5
        )
    
    def extract_keypoints(self, image_path: str) -> np.ndarray:
        """
        Extract 33 keypoints từ image.
        
        Returns:
            numpy array (132,) hoặc None nếu không detect được
        """
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        # Convert to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process
        results = self.pose.process(rgb_image)
        
        if not results.pose_landmarks:
            return None
        
        # Extract keypoints
        keypoints = []
        for landmark in results.pose_landmarks.landmark:
            keypoints.extend([
                landmark.x,
                landmark.y,
                landmark.z,
                landmark.visibility
            ])
        
        return np.array(keypoints, dtype=np.float32)
    
    def prepare_from_folders(self, input_dir: str, output_file: str):
        """
        Chuẩn bị dataset từ thư mục có cấu trúc:
        
        input_dir/
        ├── squat/
        │   ├── img1.jpg
        │   └── img2.jpg
        ├── lunge/
        │   └── ...
        └── ...
        
        Output: CSV file với 132 features + label
        """
        input_path = Path(input_dir)
        
        if not input_path.exists():
            print(f"Input directory not found: {input_dir}")
            return
        
        data = []
        labels = []
        
        # Iterate through each class folder
        for class_dir in input_path.iterdir():
            if not class_dir.is_dir():
                continue
            
            class_name = class_dir.name.lower()
            print(f"Processing class: {class_name}")
            
            # Process each image
            image_files = list(class_dir.glob("*.jpg")) + \
                         list(class_dir.glob("*.jpeg")) + \
                         list(class_dir.glob("*.png"))
            
            success_count = 0
            for img_file in image_files:
                keypoints = self.extract_keypoints(str(img_file))
                
                if keypoints is not None:
                    data.append(keypoints)
                    labels.append(class_name)
                    success_count += 1
            
            print(f"  Extracted: {success_count}/{len(image_files)} images")
        
        if not data:
            print("No data extracted!")
            return
        
        # Convert to DataFrame
        columns = [f"kp_{i}" for i in range(132)]
        df = pd.DataFrame(data, columns=columns)
        df['label'] = labels
        
        # Save
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)
        
        print(f"\nDataset saved to: {output_file}")
        print(f"Total samples: {len(df)}")
        print(f"Classes: {df['label'].value_counts().to_dict()}")
    
    def close(self):
        self.pose.close()


def main():
    parser = argparse.ArgumentParser(description="Prepare pose dataset")
    parser.add_argument("--input", "-i", required=True, 
                       help="Input directory with class folders")
    parser.add_argument("--output", "-o", required=True,
                       help="Output CSV file path")
    
    args = parser.parse_args()
    
    preparer = DatasetPreparer()
    preparer.prepare_from_folders(args.input, args.output)
    preparer.close()


if __name__ == "__main__":
    main()
