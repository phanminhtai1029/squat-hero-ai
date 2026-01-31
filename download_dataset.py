"""
Download Dataset Script
========================
Hướng dẫn và script tải dataset từ Kaggle.

Dataset đề xuất cho 5 poses:
1. Yoga-82 Dataset - 82 yoga poses với 28,000+ images
2. Yoga Poses Dataset - 5 yoga poses cơ bản
3. Fitness Exercises Images - Các động tác fitness

Usage:
    1. Cài Kaggle CLI: pip install kaggle
    2. Tạo API key: https://www.kaggle.com/settings -> Create New Token
    3. Đặt kaggle.json vào ~/.kaggle/
    4. Chạy script này: python download_dataset.py
"""

import os
import subprocess
import shutil
from pathlib import Path


# ===============================================================
# DATASET RECOMMENDATIONS
# ===============================================================
#
# 1. YOGA-82 DATASET (Recommended)
#    - URL: https://www.kaggle.com/datasets/shrutisaxena/yoga-pose-image-classification-dataset
#    - Contains: 82 yoga pose classes, 28,000+ images
#    - Includes: Warrior, Tree Pose, and many more
#    - Command: kaggle datasets download -d shrutisaxena/yoga-pose-image-classification-dataset
#
# 2. YOGA POSES DATASET
#    - URL: https://www.kaggle.com/datasets/niharika41298/yoga-poses-dataset
#    - Contains: 5 basic yoga poses
#    - Good for testing
#    - Command: kaggle datasets download -d niharika41298/yoga-poses-dataset
#
# 3. FITNESS EXERCISES IMAGES
#    - URL: https://www.kaggle.com/datasets/hasyimabdillah/workoutfitness-video
#    - Contains: Squat, Lunge, and other fitness moves
#    - Command: kaggle datasets download -d hasyimabdillah/workoutfitness-video
#
# ===============================================================


# Các dataset có thể download
DATASETS = {
    "yoga_poses": {
        "name": "Yoga Poses Dataset (Small, 5 classes)",
        "kaggle_id": "niharika41298/yoga-poses-dataset",
        "url": "https://www.kaggle.com/datasets/niharika41298/yoga-poses-dataset",
        "poses": ["downdog", "goddess", "plank", "tree", "warrior2"],
        "recommended": True
    },
    "yoga_82": {
        "name": "Yoga-82 Dataset (Large, 82 classes)",
        "kaggle_id": "shrutisaxena/yoga-pose-image-classification-dataset",
        "url": "https://www.kaggle.com/datasets/shrutisaxena/yoga-pose-image-classification-dataset",
        "poses": ["82 yoga poses including warrior, tree, etc."],
        "recommended": False
    },
    "exercise_recognition": {
        "name": "Exercise Recognition Dataset",
        "kaggle_id": "sanikamal/exercise-recognition",
        "url": "https://www.kaggle.com/datasets/sanikamal/exercise-recognition",
        "poses": ["squat", "deadlift", "pushup", "pullup"],
        "recommended": True
    }
}


def check_kaggle_installed():
    """Kiểm tra Kaggle CLI đã cài chưa."""
    try:
        result = subprocess.run(["kaggle", "--version"], capture_output=True, text=True)
        print(f"✓ Kaggle CLI installed: {result.stdout.strip()}")
        return True
    except FileNotFoundError:
        print("✗ Kaggle CLI not found")
        print("  Install with: pip install kaggle")
        return False


def check_kaggle_credentials():
    """Kiểm tra Kaggle credentials."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    
    if kaggle_json.exists():
        print(f"✓ Kaggle credentials found: {kaggle_json}")
        return True
    else:
        print("✗ Kaggle credentials not found")
        print("  1. Go to https://www.kaggle.com/settings")
        print("  2. Click 'Create New Token'")
        print(f"  3. Save kaggle.json to {kaggle_json}")
        return False


def download_dataset(dataset_id: str, output_dir: str = "data/raw"):
    """
    Download dataset từ Kaggle.
    
    Args:
        dataset_id: Kaggle dataset ID (e.g., "niharika41298/yoga-poses-dataset")
        output_dir: Thư mục lưu dataset
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nDownloading: {dataset_id}")
    print(f"To: {output_path}")
    
    try:
        cmd = f"kaggle datasets download -d {dataset_id} -p {output_path} --unzip"
        subprocess.run(cmd.split(), check=True)
        print(f"✓ Downloaded successfully to {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Download failed: {e}")
        return False


def list_datasets():
    """Liệt kê các dataset đề xuất."""
    print("\n" + "=" * 60)
    print("RECOMMENDED DATASETS FOR POSE AI")
    print("=" * 60)
    
    for key, info in DATASETS.items():
        rec = "⭐ Recommended" if info["recommended"] else ""
        print(f"\n[{key}] {info['name']} {rec}")
        print(f"  URL: {info['url']}")
        print(f"  Poses: {', '.join(info['poses'][:5])}")
        print(f"  Command: kaggle datasets download -d {info['kaggle_id']}")
    
    print("\n" + "=" * 60)


def manual_download_instructions():
    """Hướng dẫn download thủ công."""
    print("""
╔════════════════════════════════════════════════════════════════╗
║           MANUAL DOWNLOAD INSTRUCTIONS                         ║
╠════════════════════════════════════════════════════════════════╣
║                                                                ║
║  If Kaggle CLI doesn't work, download manually:                ║
║                                                                ║
║  1. Go to one of these URLs:                                   ║
║                                                                ║
║     YOGA POSES (Small, recommended for testing):               ║
║     https://www.kaggle.com/datasets/niharika41298/yoga-poses-dataset
║                                                                ║
║     YOGA-82 (Large, comprehensive):                            ║
║     https://www.kaggle.com/datasets/shrutisaxena/yoga-pose-image-classification-dataset
║                                                                ║
║  2. Click "Download" button                                    ║
║                                                                ║
║  3. Extract ZIP to: data/raw/                                  ║
║                                                                ║
║  4. Organize folders like this:                                ║
║     data/raw/                                                  ║
║     ├── squat/                                                 ║
║     │   ├── img1.jpg                                           ║
║     │   └── ...                                                ║
║     ├── lunge/                                                 ║
║     ├── plank/                                                 ║
║     ├── warrior_i/                                             ║
║     └── tree_pose/                                             ║
║                                                                ║
║  5. Run: python training/prepare_dataset.py                    ║
║          --input data/raw                                      ║
║          --output data/processed/pose_dataset.csv              ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
""")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Download dataset for Pose AI")
    parser.add_argument("--list", "-l", action="store_true", help="List available datasets")
    parser.add_argument("--download", "-d", type=str, help="Dataset key to download")
    parser.add_argument("--output", "-o", default="data/raw", help="Output directory")
    parser.add_argument("--manual", "-m", action="store_true", help="Show manual instructions")
    
    args = parser.parse_args()
    
    if args.manual:
        manual_download_instructions()
        return
    
    if args.list:
        list_datasets()
        return
    
    # Check prerequisites
    print("Checking prerequisites...")
    kaggle_ok = check_kaggle_installed()
    creds_ok = check_kaggle_credentials()
    
    if not (kaggle_ok and creds_ok):
        print("\n⚠️ Please fix the issues above, or use --manual for manual instructions")
        manual_download_instructions()
        return
    
    # Download
    if args.download:
        if args.download in DATASETS:
            dataset_id = DATASETS[args.download]["kaggle_id"]
        else:
            dataset_id = args.download  # Allow direct Kaggle ID
        
        download_dataset(dataset_id, args.output)
    else:
        # Default: download recommended dataset
        print("\nNo dataset specified. Downloading recommended dataset...")
        download_dataset(DATASETS["yoga_poses"]["kaggle_id"], args.output)
    
    print("\nNext steps:")
    print("  1. Check data/raw/ folder")
    print("  2. Rename folders to match: squat, lunge, plank, warrior_i, tree_pose")
    print("  3. Run: python training/prepare_dataset.py --input data/raw --output data/processed/pose_dataset.csv")


if __name__ == "__main__":
    main()
