"""
Train Pose Classifier
======================
Train MLP model để classify 5 poses từ keypoints.

Usage:
    python train_classifier.py --data data/processed/pose_dataset.csv
    python train_classifier.py --data data/processed/pose_dataset.csv --epochs 100
"""

import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import sys
from typing import Tuple

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    import pickle
    TORCH_AVAILABLE = True
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install torch scikit-learn")
    TORCH_AVAILABLE = False


# Pose classes
POSE_CLASSES = ['squat', 'lunge', 'plank', 'warrior_i', 'tree_pose']


class MLPModel(nn.Module):
    """MLP model cho pose classification."""
    
    def __init__(self, input_size: int = 132, num_classes: int = 5):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)


def load_dataset(csv_path: str) -> Tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """Load dataset từ CSV file."""
    df = pd.read_csv(csv_path)
    
    # Features (132 keypoints)
    feature_cols = [col for col in df.columns if col.startswith('kp_')]
    X = df[feature_cols].values.astype(np.float32)
    
    # Labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['label'].values)
    
    print(f"Loaded {len(X)} samples")
    print(f"Classes: {list(label_encoder.classes_)}")
    
    return X, y, label_encoder


def train_model(X: np.ndarray, y: np.ndarray, 
                epochs: int = 50, 
                batch_size: int = 32,
                learning_rate: float = 0.001) -> Tuple[MLPModel, dict]:
    """Train MLP model."""
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}")
    
    # Create DataLoaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.LongTensor(y_val)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Create model
    num_classes = len(np.unique(y))
    model = MLPModel(input_size=X.shape[1], num_classes=num_classes)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # Training loop
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    best_acc = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100 * correct / total
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}] "
                  f"Train Loss: {train_loss:.4f} "
                  f"Val Loss: {val_loss:.4f} "
                  f"Val Acc: {val_acc:.2f}%")
        
        scheduler.step()
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    print(f"\nBest Validation Accuracy: {best_acc:.2f}%")
    
    return model, history


def save_model(model: MLPModel, label_encoder: LabelEncoder, output_dir: str):
    """Save model và label encoder."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save PyTorch model
    model_path = output_path / "pose_classifier.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")
    
    # Save label encoder
    encoder_path = output_path / "label_encoder.pkl"
    with open(encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"Label encoder saved to: {encoder_path}")


def main():
    if not TORCH_AVAILABLE:
        print("PyTorch not available. Cannot train.")
        return
    
    parser = argparse.ArgumentParser(description="Train pose classifier")
    parser.add_argument("--data", "-d", required=True,
                       help="Path to pose_dataset.csv")
    parser.add_argument("--epochs", "-e", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", "-b", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001,
                       help="Learning rate")
    parser.add_argument("--output", "-o", default="step4_pose_classifier/models",
                       help="Output directory for model")
    
    args = parser.parse_args()
    
    # Load data
    print("Loading dataset...")
    X, y, label_encoder = load_dataset(args.data)
    
    # Train
    print("\nTraining model...")
    model, history = train_model(
        X, y, 
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
    
    # Save
    print("\nSaving model...")
    save_model(model, label_encoder, args.output)
    
    print("\nTraining completed!")


if __name__ == "__main__":
    main()
