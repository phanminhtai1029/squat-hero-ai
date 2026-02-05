#!/usr/bin/env python3
"""
Build vector database from yoga_final dataset (simple version).
"""

import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch
import cv2

sys.path.append(str(Path(__file__).parent.parent))

from pipeline.step3_pose_estimation import PoseEstimator
from ml.models.pose_gnn_encoder import SimplePoseGNN


def main():
    # Paths
    train_dir = Path(__file__).parent.parent / "data" / "processed" / "yoga_final" / "train"
    output_file = Path(__file__).parent.parent / "data" / "pose_vectors_final.npz"
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pose_estimator = PoseEstimator(model_path="yolov8s-pose.pt")
    gnn_model = SimplePoseGNN().to(device)
    gnn_model.eval()
    
    print("=" * 80)
    print("BUILDING VECTOR DATABASE (Yoga Final)")
    print("=" * 80)
    print(f"Train directory: {train_dir}")
    print(f"Device: {device}")
    
    # Collect data
    all_embeddings = []
    all_labels = []
    max_per_class = 10
    
    pose_folders = sorted([d for d in train_dir.iterdir() if d.is_dir()])
    print(f"\nFound {len(pose_folders)} pose classes")
    print(f"Max samples per class: {max_per_class}\n")
    
    errors = []
    
    for pose_folder in tqdm(pose_folders, desc="Processing"):
        pose_name = pose_folder.name
        
        # Get images
        images = list(pose_folder.glob('*.jpg')) + list(pose_folder.glob('*.png'))
        sample_images = images[:max_per_class]
        
        for img_path in sample_images:
            try:
                # Load image
                image = cv2.imread(str(img_path))
                if image is None:
                    errors.append(f"Failed to load {img_path.name}")
                    continue
                
                # Estimate pose
                result = pose_estimator.estimate(image)
                if result is None:
                    errors.append(f"No pose detected in {img_path.name}")
                    continue
                
                # Extract keypoints as numpy array
                keypoints_np = result.to_numpy()[:, :2]  # Only x, y (17, 2)
                
                # Extract keypoints as numpy array
                keypoints_np = result.to_numpy()[:, :2]  # Only x, y (17, 2)
                
                # Encode
                kp_tensor = torch.FloatTensor(keypoints_np).unsqueeze(0).to(device)
                with torch.no_grad():
                    embedding = gnn_model(kp_tensor)
                
                embedding_np = embedding.cpu().numpy().flatten()
                
                all_embeddings.append(embedding_np)
                all_labels.append(pose_name)
                
            except Exception as e:
                errors.append(f"Error {img_path.name}: {str(e)}")
                continue
    
    # Print first 10 errors
    if errors:
        print("\nFirst 10 errors:")
        for err in errors[:10]:
            print(f"  - {err}")
    
    # Convert to arrays
    embeddings_array = np.array(all_embeddings)
    labels_array = np.array(all_labels)
    
    # Create metadata in expected format: {pose_name: {display_name: ...}}
    unique_poses = sorted(np.unique(labels_array).tolist())
    metadata = {}
    for pose_name in unique_poses:
        metadata[pose_name] = {
            'display_name': pose_name,
            'count': int(np.sum(labels_array == pose_name))
        }
    
    # Save
    np.savez_compressed(
        output_file,
        embeddings=embeddings_array,
        labels=labels_array,
        metadata=np.array([metadata], dtype=object)
    )
    
    print("\n" + "=" * 80)
    print("DATABASE STATISTICS")
    print("=" * 80)
    print(f"Total samples: {len(labels_array)}")
    print(f"Embedding shape: {embeddings_array.shape}")
    print(f"Unique poses: {len(np.unique(labels_array))}")
    print(f"File size: {output_file.stat().st_size / 1024:.1f} KB")
    print(f"\nSaved to: {output_file}")
    print("✅ Done!")


if __name__ == "__main__":
    main()
