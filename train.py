"""
Train DNN-Small / DNN-Base / DNN-Large on 3DYoga90 skeleton dataset.
Reproduces experiments from:
  "3DYoga90: A Hierarchical Video Dataset for Yoga Pose Understanding"
  arXiv:2310.10131

Usage:
  python train.py                    # train DNN-Base (default)
  python train.py --model small      # train DNN-Small
  python train.py --model large      # train DNN-Large
  python train.py --level 3          # train level-3 (90 classes)

Outputs:
  checkpoints/best_<model>_L<level>.pth   -- best checkpoint per validation acc
  checkpoints/label_maps.pkl              -- label encoders (needed for inference)
"""

import argparse
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# ─────────────────────────────────────────────
# Config (matches paper, Section 4.1)
# ─────────────────────────────────────────────
LANDMARKS_DIR = "3DYoga90/data/landmarks/official_dataset"
CSV_PATH      = "3DYoga90/data/3DYoga90.csv"
CKPT_DIR      = "checkpoints"

BATCH_SIZE    = 256         # paper=64; increased safe for RTX3060 6GB (model ~1MB)
LR            = 3.33e-4     # paper: lr = 3.33e-4
LR_FACTOR     = 0.8         # paper: lr reduction factor 0.8
MAX_EPOCHS    = 100         # paper: 100 epochs with early stopping
PATIENCE      = 10          # early stopping patience
VAL_SPLIT     = 0.1         # paper: 10% validation split (from train)
DROPOUT       = 0.4         # paper: dropout rate 0.4
N_LANDMARKS   = 33          # BlazePose: 33 keypoints
N_COORDS      = 3           # (x, y, z)
# Feature vector = mean + std per landmark × coord = 2 × 33 × 3 = 198
FEAT_DIM      = N_LANDMARKS * N_COORDS * 2  # 198


# ─────────────────────────────────────────────
# Feature extraction from parquet  (vectorized + disk cache)
# ─────────────────────────────────────────────
FEAT_CACHE_DIR = "3DYoga90/data/landmarks/feat_cache"
os.makedirs(FEAT_CACHE_DIR, exist_ok=True)

def extract_features(parquet_path: str) -> np.ndarray:
    """
    Convert a skeleton sequence parquet → fixed 198-dim feature vector.
    Uses a .npy disk cache so each file is processed only once.
    """
    seq_id     = os.path.splitext(os.path.basename(parquet_path))[0]
    cache_path = os.path.join(FEAT_CACHE_DIR, f"{seq_id}.npy")

    if os.path.exists(cache_path):
        return np.load(cache_path)

    df = pd.read_parquet(parquet_path, columns=["frame", "landmark_index", "x", "y", "z"])

    # Fast vectorized pivot: no iterrows
    frames_arr    = df["frame"].to_numpy()
    lm_idx_arr    = df["landmark_index"].to_numpy(dtype=np.int32)
    xyz_arr       = df[["x", "y", "z"]].to_numpy(dtype=np.float32)

    unique_frames, fi_arr = np.unique(frames_arr, return_inverse=True)
    n_frames = len(unique_frames)

    arr = np.zeros((n_frames, N_LANDMARKS, N_COORDS), dtype=np.float32)
    arr[fi_arr, lm_idx_arr] = xyz_arr   # vectorized assignment

    # Aggregate across frames → (33, 3)
    mean_feat = arr.mean(axis=0).flatten()   # 99
    std_feat  = arr.std(axis=0).flatten()    # 99
    feat      = np.concatenate([mean_feat, std_feat])  # 198

    np.save(cache_path, feat)
    return feat


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
class YogaDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label_col: str, label_enc: LabelEncoder,
                 landmarks_dir: str):
        """
        Pre-loads ALL features into RAM on init.
        5526 × 198 × float32 ≈ 4.4 MB — negligible.
        Eliminates per-sample disk I/O during training.
        """
        labels = label_enc.transform(df[label_col].values)

        feats = np.empty((len(df), FEAT_DIM), dtype=np.float32)
        for i, seq_id in enumerate(df["sequence_id"].values):
            path = os.path.join(landmarks_dir, f"{int(seq_id)}.parquet")
            feats[i] = extract_features(path)

        # Store as tensors directly — zero-copy in DataLoader
        self.features = torch.from_numpy(feats)
        self.labels   = torch.from_numpy(labels.astype(np.int64))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# ─────────────────────────────────────────────
# DNN Models (paper Fig. 5 & Section 4.1)
# ─────────────────────────────────────────────
def make_block(in_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.BatchNorm1d(out_dim),
        nn.ReLU(),
        nn.Dropout(DROPOUT),
    )


class DNNSmall(nn.Module):
    """Input → Dense(1024) → BN/ReLU/Drop → Dense(512) → BN/ReLU/Drop → Dense(n_classes)"""
    def __init__(self, input_dim: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            make_block(input_dim, 1024),
            make_block(1024, 512),
            nn.Linear(512, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class DNNBase(nn.Module):
    """DNN-Small + Dense(256) → BN/ReLU/Drop before final layer"""
    def __init__(self, input_dim: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            make_block(input_dim, 1024),
            make_block(1024, 512),
            make_block(512, 256),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class DNNLarge(nn.Module):
    """DNN-Base + Dense(128) → BN/ReLU/Drop before final layer"""
    def __init__(self, input_dim: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            make_block(input_dim, 1024),
            make_block(1024, 512),
            make_block(512, 256),
            make_block(256, 128),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.net(x)


MODEL_MAP = {"small": DNNSmall, "base": DNNBase, "large": DNNLarge}


# ─────────────────────────────────────────────
# Training helpers
# ─────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if train:
                optimizer.zero_grad()
            logits = model(x)
            loss   = criterion(logits, y)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            correct    += (logits.argmax(1) == y).sum().item()
            total      += len(y)

    return total_loss / total, correct / total


def train_model(model_name: str, level: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True   # auto-tune CUDA kernels
    print(f"\n{'='*60}")
    print(f"  Model : DNN-{model_name.capitalize()}")
    print(f"  Level : L{level}")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    os.makedirs(CKPT_DIR, exist_ok=True)

    # ── Load metadata ──────────────────────────────────────────
    df = pd.read_csv(CSV_PATH)

    # Keep only rows whose parquet exists
    existing = {int(f.split(".")[0])
                for f in os.listdir(LANDMARKS_DIR) if f.endswith(".parquet")}
    df = df[df["sequence_id"].isin(existing)].reset_index(drop=True)
    print(f"Sequences with parquet: {len(df)}")

    # ── Pre-build feature cache (parallel, runs once) ──────────
    all_parquets = [
        os.path.join(LANDMARKS_DIR, f"{int(sid)}.parquet")
        for sid in df["sequence_id"]
    ]
    cached = sum(1 for p in all_parquets
                 if os.path.exists(os.path.join(
                     FEAT_CACHE_DIR,
                     os.path.splitext(os.path.basename(p))[0] + ".npy")))
    if cached < len(all_parquets):
        need_build = [p for p in all_parquets
                      if not os.path.exists(os.path.join(
                          FEAT_CACHE_DIR,
                          os.path.splitext(os.path.basename(p))[0] + ".npy"))]
        print(f"Building feature cache: {len(need_build)} files "
              f"({cached} already cached) ...")
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import multiprocessing
        n_workers = min(multiprocessing.cpu_count() - 2, 10)  # leave 2 cores for OS
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(extract_features, p): p for p in need_build}
            for i, fut in enumerate(tqdm(as_completed(futs),
                                         total=len(need_build),
                                         desc="  Caching", unit="seq"), 1):
                try:
                    fut.result()
                except Exception as e:
                    print(f"  Cache error {futs[fut]}: {e}")
        print("  Cache ready.")
    else:
        print(f"Feature cache: all {cached} files cached ✓")

    label_col = f"l{level}_pose_id"

    # ── Label encoding ─────────────────────────────────────────
    le = LabelEncoder()
    le.fit(df[label_col])
    n_classes = len(le.classes_)
    print(f"Classes (L{level}): {n_classes}")

    # Save label maps for inference
    maps_path = os.path.join(CKPT_DIR, "label_maps.pkl")
    maps = {}
    if os.path.exists(maps_path):
        with open(maps_path, "rb") as f:
            maps = pickle.load(f)
    maps[level] = le
    with open(maps_path, "wb") as f:
        pickle.dump(maps, f)

    # ── Train / test split (paper uses pre-defined split) ──────
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    test_df  = df[df["split"] == "test"].reset_index(drop=True)

    # 10% validation from train (paper: 10% val split)
    val_size = max(1, int(len(train_df) * VAL_SPLIT))
    val_df   = train_df.sample(n=val_size, random_state=42)
    train_df = train_df.drop(val_df.index).reset_index(drop=True)
    val_df   = val_df.reset_index(drop=True)

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # ── DataLoaders ────────────────────────────────────────────
    # Features pre-loaded into RAM → workers only do tensor indexing (cheap)
    # 6 workers for train (i7-12700H has 14C/20T), 4 for val/test
    print("Loading datasets into RAM...")
    train_ds = YogaDataset(train_df, label_col, le, LANDMARKS_DIR)
    val_ds   = YogaDataset(val_df,   label_col, le, LANDMARKS_DIR)
    test_ds  = YogaDataset(test_df,  label_col, le, LANDMARKS_DIR)
    print(f"  RAM loaded: {len(train_ds)+len(val_ds)+len(test_ds)} samples")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=6, pin_memory=True,
                              persistent_workers=True, prefetch_factor=2)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True,
                              persistent_workers=True, prefetch_factor=2)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True,
                              persistent_workers=True, prefetch_factor=2)

    # ── Model ──────────────────────────────────────────────────
    ModelClass = MODEL_MAP[model_name]
    model      = ModelClass(FEAT_DIM, n_classes).to(device)

    # Multi-GPU if available
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    # ── Optimizer / Scheduler / Loss ───────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=LR_FACTOR, patience=5, verbose=True
    )

    # ── Training loop ──────────────────────────────────────────
    best_val_acc  = 0.0
    patience_cnt  = 0
    ckpt_path     = os.path.join(CKPT_DIR, f"best_{model_name}_L{level}.pth")

    for epoch in range(1, MAX_EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        vl_loss, vl_acc = run_epoch(model, val_loader,   criterion, optimizer, device, train=False)
        scheduler.step(vl_acc)

        print(f"Epoch {epoch:3d}/{MAX_EPOCHS} | "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
              f"val_loss={vl_loss:.4f} val_acc={vl_acc:.4f}")

        # Save best
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            patience_cnt = 0
            torch.save({
                "epoch":      epoch,
                "model_name": model_name,
                "level":      level,
                "n_classes":  n_classes,
                "feat_dim":   FEAT_DIM,
                "state_dict": model.state_dict() if not isinstance(model, nn.DataParallel)
                              else model.module.state_dict(),
            }, ckpt_path)
            print(f"  ✓ Saved best checkpoint (val_acc={best_val_acc:.4f})")
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
                break

    # ── Test evaluation ────────────────────────────────────────
    print(f"\nLoading best checkpoint for test evaluation...")
    ckpt = torch.load(ckpt_path, map_location=device)
    model_cls = MODEL_MAP[model_name]
    eval_model = model_cls(FEAT_DIM, n_classes).to(device)
    eval_model.load_state_dict(ckpt["state_dict"])

    te_loss, te_acc = run_epoch(eval_model, test_loader, criterion, None, device, train=False)
    print(f"\n{'─'*50}")
    print(f"  TEST RESULTS — DNN-{model_name.capitalize()} L{level}")
    print(f"  Test Accuracy : {te_acc:.4f}")
    print(f"  Test Loss     : {te_loss:.4f}")
    print(f"  Best Val Acc  : {best_val_acc:.4f}")
    print(f"  Checkpoint    : {ckpt_path}")
    print(f"{'─'*50}\n")

    return te_acc, best_val_acc


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 3DYoga90 DNN models")
    parser.add_argument("--model", choices=["small", "base", "large"], default="base",
                        help="DNN variant (default: base)")
    parser.add_argument("--level", type=int, choices=[1, 2, 3], default=3,
                        help="Hierarchy level to classify (default: 3 = 90 classes)")
    parser.add_argument("--all", action="store_true",
                        help="Train all 3 models × all 3 levels (reproduces Table 4)")
    args = parser.parse_args()

    if args.all:
        # Reproduce Table 4 in the paper
        results = {}
        for m in ["small", "base", "large"]:
            for lvl in [1, 2, 3]:
                te_acc, _ = train_model(m, lvl)
                results[(m, lvl)] = te_acc

        print("\n" + "="*60)
        print("  TABLE 4 REPRODUCTION")
        print("="*60)
        print(f"{'Model':<12} {'L1':>8} {'L2':>8} {'L3':>8}")
        print("─"*40)
        for m in ["small", "base", "large"]:
            row = " ".join(f"{results[(m, l)]:>8.4f}" for l in [1, 2, 3])
            print(f"DNN-{m.capitalize():<8} {row}")
    else:
        train_model(args.model, args.level)
