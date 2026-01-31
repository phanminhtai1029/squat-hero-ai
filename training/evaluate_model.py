"""
Evaluate Pose Classifier
=========================
Script để đánh giá model với các metrics:
- Accuracy
- Precision, Recall, F1-Score (per class & overall)
- Confusion Matrix
- Classification Report

Usage:
    python training/evaluate_model.py --data data/processed/yoga82_dataset.csv
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import json

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_recall_fscore_support
)
import pickle


class MLPModel(torch.nn.Module):
    """MLP model for pose classification."""
    
    def __init__(self, input_size: int = 132, num_classes: int = 5):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_size, 256),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(256),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(128),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)


def load_model(model_dir: str):
    """Load trained model and label encoder."""
    model_path = Path(model_dir) / "pose_classifier.pth"
    encoder_path = Path(model_dir) / "label_encoder.pkl"
    
    # Load label encoder
    with open(encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    
    num_classes = len(label_encoder.classes_)
    
    # Load model
    model = MLPModel(input_size=132, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    return model, label_encoder


def load_dataset(data_path: str):
    """Load and split dataset."""
    df = pd.read_csv(data_path)
    
    # Separate features and labels
    X = df.drop('label', axis=1).values.astype(np.float32)
    y = df['label'].values
    
    return X, y


def evaluate_model(model, X, y, label_encoder):
    """
    Evaluate model on dataset.
    
    Returns:
        dict with all metrics
    """
    # Convert labels to indices
    y_encoded = label_encoder.transform(y)
    
    # Split data (same as training)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Predict on test set
    with torch.no_grad():
        X_tensor = torch.tensor(X_test)
        outputs = model(X_tensor)
        _, y_pred = torch.max(outputs, 1)
        y_pred = y_pred.numpy()
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average=None, zero_division=0
    )
    
    # Overall metrics (weighted)
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted', zero_division=0
    )
    
    # Macro average
    precision_m, recall_m, f1_m, _ = precision_recall_fscore_support(
        y_test, y_pred, average='macro', zero_division=0
    )
    
    # Classification report
    class_report = classification_report(
        y_test, y_pred, 
        target_names=label_encoder.classes_,
        zero_division=0,
        output_dict=True
    )
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Find top confusions
    top_confusions = find_top_confusions(conf_matrix, label_encoder.classes_)
    
    results = {
        'accuracy': accuracy,
        'precision_weighted': precision_w,
        'recall_weighted': recall_w,
        'f1_weighted': f1_w,
        'precision_macro': precision_m,
        'recall_macro': recall_m,
        'f1_macro': f1_m,
        'num_classes': len(label_encoder.classes_),
        'num_samples': len(X),
        'num_test_samples': len(X_test),
        'per_class': {},
        'top_confusions': top_confusions,
        'class_report': class_report
    }
    
    # Per-class metrics
    for i, class_name in enumerate(label_encoder.classes_):
        results['per_class'][class_name] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1': float(f1[i]),
            'support': int(support[i])
        }
    
    return results


def find_top_confusions(conf_matrix, class_names, top_k=10):
    """Find top k most confused class pairs."""
    confusions = []
    n = len(class_names)
    
    for i in range(n):
        for j in range(n):
            if i != j and conf_matrix[i][j] > 0:
                confusions.append({
                    'true': class_names[i],
                    'predicted': class_names[j],
                    'count': int(conf_matrix[i][j])
                })
    
    # Sort by count descending
    confusions.sort(key=lambda x: x['count'], reverse=True)
    
    return confusions[:top_k]


def print_results(results):
    """Print evaluation results."""
    print("\n" + "="*60)
    print("MODEL EVALUATION RESULTS")
    print("="*60)
    
    print(f"\n📊 Overall Metrics:")
    print(f"   Accuracy:          {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"   Precision (macro): {results['precision_macro']:.4f}")
    print(f"   Recall (macro):    {results['recall_macro']:.4f}")
    print(f"   F1-Score (macro):  {results['f1_macro']:.4f}")
    print(f"\n   Precision (weighted): {results['precision_weighted']:.4f}")
    print(f"   Recall (weighted):    {results['recall_weighted']:.4f}")
    print(f"   F1-Score (weighted):  {results['f1_weighted']:.4f}")
    
    print(f"\n📈 Dataset Info:")
    print(f"   Total samples: {results['num_samples']}")
    print(f"   Test samples:  {results['num_test_samples']}")
    print(f"   Num classes:   {results['num_classes']}")
    
    # Top 10 best performing classes
    print(f"\n🏆 Top 10 Best Performing Classes (by F1):")
    sorted_classes = sorted(
        results['per_class'].items(), 
        key=lambda x: x[1]['f1'], 
        reverse=True
    )[:10]
    
    print(f"   {'Class':<35} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>8}")
    print(f"   {'-'*35} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
    for name, metrics in sorted_classes:
        print(f"   {name:<35} {metrics['precision']:>10.3f} {metrics['recall']:>10.3f} {metrics['f1']:>10.3f} {metrics['support']:>8}")
    
    # Top confusions
    print(f"\n⚠️ Top 10 Confusions:")
    print(f"   {'True Class':<30} {'Predicted As':<30} {'Count':>6}")
    print(f"   {'-'*30} {'-'*30} {'-'*6}")
    for conf in results['top_confusions']:
        print(f"   {conf['true']:<30} {conf['predicted']:<30} {conf['count']:>6}")
    
    print("\n" + "="*60)


def save_results(results, output_path: str):
    """Save results to JSON."""
    # Convert numpy types to Python types
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    # Deep copy and convert
    results_clean = json.loads(json.dumps(results, default=convert))
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_clean, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate pose classifier")
    parser.add_argument("--data", "-d", required=True, help="Path to dataset CSV")
    parser.add_argument("--model-dir", "-m", 
                       default="step4_pose_classifier/models",
                       help="Directory containing model files")
    parser.add_argument("--output", "-o", 
                       default="reports/evaluation_results.json",
                       help="Output JSON file")
    
    args = parser.parse_args()
    
    print("Loading model...")
    model, label_encoder = load_model(args.model_dir)
    print(f"Loaded model with {len(label_encoder.classes_)} classes")
    
    print(f"\nLoading dataset from {args.data}...")
    X, y = load_dataset(args.data)
    print(f"Loaded {len(X)} samples")
    
    print("\nEvaluating model...")
    results = evaluate_model(model, X, y, label_encoder)
    
    print_results(results)
    
    # Save results
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_results(results, args.output)


if __name__ == "__main__":
    main()
