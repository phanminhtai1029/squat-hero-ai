#!/usr/bin/env python3
"""
Merge all 3 yoga datasets into one unified dataset with train/val/test split.
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List
import random
from collections import defaultdict

# Normalize pose names to standard format
def normalize_pose_name(name: str) -> str:
    """Normalize pose name to standard format (Title Case)."""
    # Handle special cases
    mappings = {
        'Downward_dog': 'Adho Mukha Svanasana',
        'Triangle': 'Trikonasana',
        'UtkataKonasana': 'Utkatasana',
        'Veerabhadrasana': 'Virabhadrasana Two',
        'Vrukshasana': 'Vrksasana',
        'ArdhaChandrasana': 'Ardha Chandrasana',
        'BaddhaKonasana': 'Baddha Konasana',
        'Natarajasana': 'Natarajasana',
    }
    
    if name in mappings:
        return mappings[name]
    
    # Lowercase dataset names to Title Case
    if name.islower():
        return ' '.join(word.capitalize() for word in name.split())
    
    # Mixed case: add spaces before capitals
    if any(c.isupper() for c in name[1:]):
        spaced = ''.join([' ' + c if c.isupper() else c for c in name]).strip()
        return ' '.join(word.capitalize() for word in spaced.split())
    
    return name


def get_image_files(directory: Path) -> List[Path]:
    """Get all image files in directory."""
    if not directory.exists():
        return []
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    return [f for f in directory.iterdir() if f.suffix in image_extensions]


def merge_all_datasets(
    yoga_poses_dir: Path,
    posture_dataset_dir: Path,
    yoga_107_dir: Path,
    output_dir: Path
) -> Dict[str, List[Path]]:
    """
    Merge all three datasets into unified structure.
    """
    merged_data = defaultdict(list)
    
    print("=" * 80)
    print("MERGING ALL DATASETS")
    print("=" * 80)
    
    # 1. Yoga_Poses-Dataset (8 poses)
    print("\n[1] Processing Yoga_Poses-Dataset...")
    train_dir = yoga_poses_dir / "TRAIN"
    
    for pose_folder in train_dir.iterdir():
        if not pose_folder.is_dir():
            continue
        
        original_name = pose_folder.name
        standard_name = normalize_pose_name(original_name)
        
        images_dir = pose_folder / "Images"
        image_files = get_image_files(images_dir)
        
        merged_data[standard_name].extend(image_files)
        print(f"  {original_name:25} -> {standard_name:35} : {len(image_files):3} images")
    
    # 2. Yoga Posture Dataset (47 poses)
    print("\n[2] Processing Yoga Posture Dataset...")
    posture_poses = [d for d in posture_dataset_dir.iterdir() if d.is_dir() and 
                     d.name not in ['Yoga_Poses-Dataset', 'processed', 'dataset']]
    
    for pose_folder in posture_poses:
        pose_name = pose_folder.name
        image_files = get_image_files(pose_folder)
        
        if image_files:
            merged_data[pose_name].extend(image_files)
    
    print(f"  Found {len(posture_poses)} pose folders")
    
    # 3. Yoga 107 Poses Dataset
    print("\n[3] Processing Yoga 107 Poses Dataset...")
    
    for pose_folder in yoga_107_dir.iterdir():
        if not pose_folder.is_dir():
            continue
        
        original_name = pose_folder.name
        standard_name = normalize_pose_name(original_name)
        
        image_files = get_image_files(pose_folder)
        
        if image_files:
            merged_data[standard_name].extend(image_files)
    
    print(f"  Found {len([d for d in yoga_107_dir.iterdir() if d.is_dir()])} pose folders")
    
    # Summary
    print("\n" + "=" * 80)
    print("MERGE SUMMARY")
    print("=" * 80)
    
    total_images = sum(len(imgs) for imgs in merged_data.values())
    print(f"Total unique poses: {len(merged_data)}")
    print(f"Total images: {total_images}")
    
    # Show top poses by count
    print(f"\n🏆 Top 15 poses by image count:")
    sorted_poses = sorted(merged_data.items(), key=lambda x: len(x[1]), reverse=True)[:15]
    for pose, imgs in sorted_poses:
        print(f"  {pose:40} {len(imgs):4} images")
    
    # Show distribution
    ranges = [(0, 50), (50, 100), (100, 200), (200, float('inf'))]
    print(f"\n📊 Distribution:")
    for min_val, max_val in ranges:
        count = len([p for p, imgs in merged_data.items() if min_val <= len(imgs) < max_val])
        label = f"{min_val}-{max_val if max_val != float('inf') else '∞'}"
        print(f"  {label:10} images: {count:3} poses")
    
    return dict(merged_data)


def split_dataset(
    merged_data: Dict[str, List[Path]],
    output_dir: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    min_images: int = 20
) -> None:
    """Split merged dataset and copy files."""
    random.seed(random_seed)
    
    # Filter out poses with too few images
    filtered_data = {k: v for k, v in merged_data.items() if len(v) >= min_images}
    removed = {k: len(v) for k, v in merged_data.items() if len(v) < min_images}
    
    if removed:
        print(f"\n⚠️  Filtering out {len(removed)} poses with < {min_images} images:")
        for pose, count in sorted(removed.items(), key=lambda x: x[1]):
            print(f"  - {pose:40} {count:3} images")
    
    # Create directories
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    test_dir = output_dir / "test"
    
    for split_dir in [train_dir, val_dir, test_dir]:
        split_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("SPLITTING DATASET")
    print("=" * 80)
    print(f"Ratios: Train={train_ratio:.0%}, Val={val_ratio:.0%}, Test={test_ratio:.0%}")
    print(f"Min images per class: {min_images}")
    print(f"Keeping {len(filtered_data)} / {len(merged_data)} poses")
    
    total_train = 0
    total_val = 0
    total_test = 0
    
    for pose_name, image_paths in filtered_data.items():
        # Shuffle
        shuffled = image_paths.copy()
        random.shuffle(shuffled)
        
        n_images = len(shuffled)
        n_train = int(n_images * train_ratio)
        n_val = int(n_images * val_ratio)
        
        train_images = shuffled[:n_train]
        val_images = shuffled[n_train:n_train + n_val]
        test_images = shuffled[n_train + n_val:]
        
        # Create folders
        for split_dir in [train_dir, val_dir, test_dir]:
            (split_dir / pose_name).mkdir(exist_ok=True)
        
        # Copy files with unique names to avoid collisions
        for idx, img_path in enumerate(train_images):
            dst = train_dir / pose_name / f"{img_path.stem}_{idx}{img_path.suffix}"
            shutil.copy2(img_path, dst)
        
        for idx, img_path in enumerate(val_images):
            dst = val_dir / pose_name / f"{img_path.stem}_{idx}{img_path.suffix}"
            shutil.copy2(img_path, dst)
        
        for idx, img_path in enumerate(test_images):
            dst = test_dir / pose_name / f"{img_path.stem}_{idx}{img_path.suffix}"
            shutil.copy2(img_path, dst)
        
        total_train += len(train_images)
        total_val += len(val_images)
        total_test += len(test_images)
    
    print("\n" + "=" * 80)
    print("FINAL DATASET STATISTICS")
    print("=" * 80)
    total = total_train + total_val + total_test
    avg_per_class = total / len(filtered_data)
    
    print(f"\n📊 Overall:")
    print(f"  Poses:  {len(filtered_data)}")
    print(f"  Train:  {total_train:5} images ({total_train/total:.1%}) - avg {total_train/len(filtered_data):.0f}/class")
    print(f"  Val:    {total_val:5} images ({total_val/total:.1%}) - avg {total_val/len(filtered_data):.0f}/class")
    print(f"  Test:   {total_test:5} images ({total_test/total:.1%}) - avg {total_test/len(filtered_data):.0f}/class")
    print(f"  Total:  {total:5} images - avg {avg_per_class:.0f}/class")
    
    print(f"\n✅ Dataset ready for training!")
    print(f"   Output: {output_dir}")


def main():
    base_dir = Path(__file__).parent.parent / "data"
    
    yoga_poses_dir = base_dir / "Yoga_Poses-Dataset"
    posture_dataset_dir = base_dir
    yoga_107_dir = base_dir / "dataset"
    output_dir = base_dir / "processed" / "yoga_final"
    
    # Merge
    merged_data = merge_all_datasets(
        yoga_poses_dir,
        posture_dataset_dir, 
        yoga_107_dir,
        output_dir
    )
    
    # Split
    split_dataset(
        merged_data,
        output_dir,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        min_images=20  # Filter poses with < 20 images
    )


if __name__ == "__main__":
    main()
