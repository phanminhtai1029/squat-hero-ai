#!/usr/bin/env python3
"""
Analyze dataset sufficiency for deep learning training.
"""

from pathlib import Path
from collections import defaultdict

def analyze_dataset(split_dir: Path, split_name: str):
    """Analyze class distribution for a split."""
    class_counts = {}
    
    for pose_folder in sorted(split_dir.iterdir()):
        if not pose_folder.is_dir():
            continue
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
        count = len([f for f in pose_folder.iterdir() if f.suffix in image_extensions])
        class_counts[pose_folder.name] = count
    
    return class_counts

def main():
    base_dir = Path(__file__).parent.parent / "data" / "processed" / "yoga_combined"
    
    train_counts = analyze_dataset(base_dir / "train", "train")
    val_counts = analyze_dataset(base_dir / "val", "val")
    test_counts = analyze_dataset(base_dir / "test", "test")
    
    print("=" * 80)
    print("DATASET SUFFICIENCY ANALYSIS FOR DEEP LEARNING")
    print("=" * 80)
    
    # Overall stats
    total_train = sum(train_counts.values())
    total_val = sum(val_counts.values())
    total_test = sum(test_counts.values())
    n_classes = len(train_counts)
    
    print(f"\n📊 Overall Statistics:")
    print(f"  Total classes: {n_classes}")
    print(f"  Train:         {total_train} images ({total_train/n_classes:.1f} avg/class)")
    print(f"  Val:           {total_val} images ({total_val/n_classes:.1f} avg/class)")
    print(f"  Test:          {total_test} images ({total_test/n_classes:.1f} avg/class)")
    print(f"  Total:         {total_train + total_val + total_test} images")
    
    # Class distribution analysis
    print(f"\n📈 Class Distribution (Train Set):")
    
    ranges = [
        (0, 20, "❌ Very Small (< 20)", []),
        (20, 50, "⚠️  Small (20-49)", []),
        (50, 100, "✅ Medium (50-99)", []),
        (100, float('inf'), "✅ Large (100+)", [])
    ]
    
    for pose, count in train_counts.items():
        for min_val, max_val, label, items in ranges:
            if min_val <= count < max_val:
                items.append((pose, count))
                break
    
    for min_val, max_val, label, items in ranges:
        print(f"\n{label}: {len(items)} classes")
        if items:
            for pose, count in sorted(items, key=lambda x: x[1], reverse=True)[:5]:
                print(f"    {pose:35} {count:3} images")
            if len(items) > 5:
                print(f"    ... and {len(items) - 5} more")
    
    # Deep learning requirements assessment
    print("\n" + "=" * 80)
    print("DEEP LEARNING REQUIREMENTS ASSESSMENT")
    print("=" * 80)
    
    # Count problematic classes
    very_small = len([c for c in train_counts.values() if c < 20])
    small = len([c for c in train_counts.values() if 20 <= c < 50])
    medium = len([c for c in train_counts.values() if 50 <= c < 100])
    large = len([c for c in train_counts.values() if c >= 100])
    
    print(f"\n✅ SUFFICIENT (50+ train images): {medium + large} / {n_classes} classes ({(medium+large)/n_classes*100:.1f}%)")
    print(f"⚠️  MARGINAL (20-49 train images): {small} / {n_classes} classes ({small/n_classes*100:.1f}%)")
    print(f"❌ INSUFFICIENT (< 20 train images): {very_small} / {n_classes} classes ({very_small/n_classes*100:.1f}%)")
    
    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    if very_small > 0:
        print(f"\n⚠️  CONCERN: {very_small} classes have < 20 training images")
        print("   → Risk: Poor generalization, overfitting")
        print("   → Solution: Remove these classes OR collect more data")
    
    if total_train < 100 * n_classes:
        print(f"\n⚠️  CONCERN: Average {total_train/n_classes:.1f} images/class (recommend 100+)")
        print("   → Risk: Model may not learn robust features")
    
    print("\n💡 IMPROVEMENT OPTIONS:")
    print("-" * 80)
    
    # Option 1: Data augmentation
    print("\n1. ✅ DATA AUGMENTATION (RECOMMENDED)")
    print("   - Rotate ±15°, flip horizontal, scale 0.9-1.1x")
    print("   - Can increase dataset 3-5x → ~6,000-11,000 train images")
    print(f"   - New average: {total_train*4/n_classes:.0f} images/class")
    print("   - Pros: Easy to implement, no new data needed")
    print("   - Cons: Still same base poses, limited variation")
    
    # Option 2: Filter small classes
    min_threshold = 30
    filtered_classes = len([c for c in train_counts.values() if c >= min_threshold])
    filtered_images = sum([c for c in train_counts.values() if c >= min_threshold])
    print(f"\n2. ⚙️  FILTER SMALL CLASSES (>= {min_threshold} images)")
    print(f"   - Keep {filtered_classes} / {n_classes} classes ({filtered_classes/n_classes*100:.0f}%)")
    print(f"   - Total: {filtered_images} train images ({filtered_images/filtered_classes:.0f} avg/class)")
    print("   - Pros: Higher quality training, less class imbalance")
    print("   - Cons: Fewer total classes to recognize")
    
    # Option 3: Collect more data
    print("\n3. 📥 COLLECT MORE DATA")
    print("   - Download from YouTube videos, Google Images, etc.")
    print("   - Target: 100+ images per class → need ~2,600 more images")
    print("   - Pros: Best performance, robust model")
    print("   - Cons: Time-consuming, manual effort")
    
    # Option 4: Use pre-trained features
    print("\n4. 🚀 USE YOLOV8 PRE-TRAINED FEATURES (FASTEST)")
    print("   - Extract features from YOLOv8 backbone (no GNN training needed)")
    print("   - Works with current dataset size")
    print("   - Pros: No training needed, fast to implement")
    print("   - Cons: Lower accuracy than custom-trained GNN")
    
    # Final recommendation
    print("\n" + "=" * 80)
    print("FINAL RECOMMENDATION")
    print("=" * 80)
    print("\n🎯 FOR YOUR DATASET (3,240 images, 48 poses):")
    print("\n   SHORT-TERM (1-2 hours):")
    print("   → Implement data augmentation (rotation, flip, brightness)")
    print("   → Filter out classes with < 20 images (keep ~45 poses)")
    print("   → Start training with triplet loss")
    print("   → Expected accuracy: 60-75% (vs current 28.4%)")
    
    print("\n   LONG-TERM (if needed):")
    print("   → Collect more data for small classes")
    print("   → Target: 80-85% accuracy")
    
    print("\n   ALTERNATIVE (if time-limited):")
    print("   → Use YOLOv8 pre-trained features (no training)")
    print("   → Expected accuracy: 50-65%")
    print("   → Can implement in 30 minutes")
    
    # Export problematic classes
    print("\n" + "=" * 80)
    print("CLASSES TO CONSIDER REMOVING (< 20 train images):")
    print("=" * 80)
    problematic = [(k, v) for k, v in train_counts.items() if v < 20]
    if problematic:
        for pose, count in sorted(problematic, key=lambda x: x[1]):
            print(f"  - {pose:35} {count:3} images")
    else:
        print("  None - all classes have sufficient data!")

if __name__ == "__main__":
    main()
