#!/usr/bin/env python3
"""
Verify dataset merge: check if duplicate classes were properly merged.
"""

from pathlib import Path
from collections import defaultdict

# Pose name mapping from prepare_dataset.py
POSE_MAPPING = {
    'ArdhaChandrasana': 'Ardha Chandrasana',
    'BaddhaKonasana': 'Baddha Konasana',
    'Downward_dog': 'Adho Mukha Svanasana',
    'Natarajasana': 'Natarajasana',
    'Triangle': 'Trikonasana',
    'UtkataKonasana': 'Utkatasana',
    'Veerabhadrasana': 'Virabhadrasana Two',
    'Vrukshasana': 'Vrksasana',
}

def count_images(directory: Path) -> int:
    """Count image files in directory."""
    if not directory.exists():
        return 0
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    return len([f for f in directory.iterdir() if f.suffix in image_extensions])

def main():
    base_dir = Path(__file__).parent.parent / "data"
    yoga_poses_dir = base_dir / "Yoga_Poses-Dataset" / "TRAIN"
    posture_dataset_dir = base_dir
    merged_dir = base_dir / "processed" / "yoga_combined" / "train"
    
    print("=" * 80)
    print("DATASET MERGE VERIFICATION")
    print("=" * 80)
    
    # Track which poses came from which dataset
    dataset1_poses = {}  # Yoga_Poses-Dataset
    dataset2_poses = defaultdict(int)  # Yoga Posture Dataset
    merged_poses = {}  # Final merged dataset
    
    # 1. Count from Yoga_Poses-Dataset
    print("\n[1] Counting images from Yoga_Poses-Dataset...")
    for pose_folder in yoga_poses_dir.iterdir():
        if not pose_folder.is_dir():
            continue
        original_name = pose_folder.name
        standard_name = POSE_MAPPING.get(original_name, original_name)
        images_dir = pose_folder / "Images"
        count = count_images(images_dir)
        dataset1_poses[standard_name] = count
        print(f"  {original_name:20} -> {standard_name:30} : {count:3} images")
    
    # 2. Count from Yoga Posture Dataset
    print("\n[2] Counting images from Yoga Posture Dataset...")
    for pose_folder in posture_dataset_dir.iterdir():
        if not pose_folder.is_dir():
            continue
        if pose_folder.name in ['Yoga_Poses-Dataset', 'processed', 'yoga-posture-dataset']:
            continue
        pose_name = pose_folder.name
        count = count_images(pose_folder)
        if count > 0:
            dataset2_poses[pose_name] = count
    
    print(f"  Found {len(dataset2_poses)} poses")
    
    # 3. Count from merged dataset (train only)
    print("\n[3] Counting images from merged dataset (train split)...")
    for pose_folder in merged_dir.iterdir():
        if not pose_folder.is_dir():
            continue
        pose_name = pose_folder.name
        count = count_images(pose_folder)
        merged_poses[pose_name] = count
    
    # 4. Verify merging
    print("\n" + "=" * 80)
    print("MERGE VERIFICATION REPORT")
    print("=" * 80)
    
    print("\n✅ CLASSES THAT WERE MERGED (found in both datasets):")
    print("-" * 80)
    merged_classes = []
    for pose_name in dataset1_poses.keys():
        if pose_name in dataset2_poses:
            merged_classes.append(pose_name)
            ds1_count = dataset1_poses[pose_name]
            ds2_count = dataset2_poses[pose_name]
            total_before = ds1_count + ds2_count
            # Merged count in train set should be ~70% of total
            merged_count_train = merged_poses.get(pose_name, 0)
            # Calculate expected total (train is 70%)
            expected_total = int(merged_count_train / 0.7)
            
            status = "✓" if abs(expected_total - total_before) <= 2 else "✗"
            print(f"{status} {pose_name:30}")
            print(f"    Dataset 1: {ds1_count:3} images")
            print(f"    Dataset 2: {ds2_count:3} images")
            print(f"    Total:     {total_before:3} images")
            print(f"    Train:     {merged_count_train:3} images (70% ≈ {int(total_before*0.7)} expected)")
            print()
    
    print(f"Total merged classes: {len(merged_classes)}")
    
    print("\n📝 CLASSES ONLY IN YOGA_POSES-DATASET:")
    print("-" * 80)
    only_ds1 = [p for p in dataset1_poses.keys() if p not in dataset2_poses]
    if only_ds1:
        for pose_name in only_ds1:
            print(f"  - {pose_name}: {dataset1_poses[pose_name]} images")
    else:
        print("  None")
    
    print("\n📝 CLASSES ONLY IN YOGA POSTURE DATASET:")
    print("-" * 80)
    only_ds2 = [p for p in dataset2_poses.keys() if p not in dataset1_poses]
    print(f"  Total: {len(only_ds2)} unique poses")
    print("  Sample (first 10):")
    for pose_name in sorted(only_ds2)[:10]:
        print(f"    - {pose_name}: {dataset2_poses[pose_name]} images -> train: {merged_poses.get(pose_name, 0)} images")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Dataset 1 (Yoga_Poses): {len(dataset1_poses)} poses, {sum(dataset1_poses.values())} images")
    print(f"Dataset 2 (Posture):    {len(dataset2_poses)} poses, {sum(dataset2_poses.values())} images")
    print(f"Merged classes:         {len(merged_classes)} poses")
    print(f"Unique to Dataset 1:    {len(only_ds1)} poses")
    print(f"Unique to Dataset 2:    {len(only_ds2)} poses")
    print(f"Final unique poses:     {len(merged_poses)} poses")
    print(f"Total train images:     {sum(merged_poses.values())} images")
    
    # Verify math
    expected_unique = len(dataset1_poses) + len(only_ds2)
    if len(merged_poses) == expected_unique:
        print(f"\n✅ MERGE CORRECT: {len(dataset1_poses)} + {len(only_ds2)} = {expected_unique} unique poses")
    else:
        print(f"\n⚠️  WARNING: Expected {expected_unique} poses, got {len(merged_poses)}")

if __name__ == "__main__":
    main()
