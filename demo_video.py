"""
Demo: Yoga pose recognition on any online video (YouTube, etc.)

Steps:
  1. Download video via yt-dlp (or accept a local file path)
  2. Extract BlazePose skeleton frame-by-frame with MediaPipe
  3. Run trained DNN with a rolling window approach
  4. Write annotated output video + print final prediction summary

Usage:
  python demo_video.py --url "https://www.youtube.com/watch?v=XXXX"
  python demo_video.py --url "https://www.youtube.com/watch?v=XXXX" --model base --window 60
  python demo_video.py --file /path/to/local_video.mp4
  python demo_video.py --url "..." --no-video    # skip writing output video (faster)

Output:
  demo_output/<video_id>.mp4   -- annotated video
"""

import argparse
import os
import pickle
import re
import sys
import tempfile
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
CKPT_DIR    = "checkpoints"
OUT_DIR     = "demo_output"
DROPOUT     = 0.4
N_LANDMARKS = 33
N_COORDS    = 3
FEAT_DIM    = N_LANDMARKS * N_COORDS * 2   # 198

# Level labels
LEVEL_NAMES = {1: "Category (L1)", 2: "Group (L2)", 3: "Pose (L3)"}

# Colors (BGR)
CLR_BG      = (20, 20, 20)
CLR_GREEN   = (0, 210, 80)
CLR_YELLOW  = (0, 200, 220)
CLR_BLUE    = (220, 150, 50)
CLR_WHITE   = (240, 240, 240)
CLR_GRAY    = (140, 140, 140)

# ─────────────────────────────────────────────────────────────
# Model definitions (mirror of train.py)
# ─────────────────────────────────────────────────────────────
def make_block(in_dim, out_dim):
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.BatchNorm1d(out_dim),
        nn.ReLU(),
        nn.Dropout(DROPOUT),
    )

class DNNSmall(nn.Module):
    def __init__(self, input_dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            make_block(input_dim, 1024), make_block(1024, 512),
            nn.Linear(512, n_classes))
    def forward(self, x): return self.net(x)

class DNNBase(nn.Module):
    def __init__(self, input_dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            make_block(input_dim, 1024), make_block(1024, 512),
            make_block(512, 256), nn.Linear(256, n_classes))
    def forward(self, x): return self.net(x)

class DNNLarge(nn.Module):
    def __init__(self, input_dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            make_block(input_dim, 1024), make_block(1024, 512),
            make_block(512, 256), make_block(256, 128),
            nn.Linear(128, n_classes))
    def forward(self, x): return self.net(x)

MODEL_MAP = {"small": DNNSmall, "base": DNNBase, "large": DNNLarge}

# ─────────────────────────────────────────────────────────────
# Load models + label encoders
# ─────────────────────────────────────────────────────────────
def _load_pose_names():
    """Return {level: {pose_id_int: name_str}} from pose-index.csv."""
    import pandas as pd
    pose_csv = os.path.join("3DYoga90", "data", "pose-index.csv")
    if not os.path.exists(pose_csv):
        return {}
    df = pd.read_csv(pose_csv)
    # columns: level1_id, level1_pose, level2_id, level2_pose, l3_pose_id, 13_pose
    maps = {}
    # level 1
    sub = df[["level1_id", "level1_pose"]].drop_duplicates()
    maps[1] = dict(zip(sub["level1_id"].astype(int), sub["level1_pose"]))
    # level 2
    sub = df[["level2_id", "level2_pose"]].drop_duplicates()
    maps[2] = dict(zip(sub["level2_id"].astype(int), sub["level2_pose"]))
    # level 3
    sub = df[["l3_pose_id", "13_pose"]].drop_duplicates()
    maps[3] = dict(zip(sub["l3_pose_id"].astype(int), sub["13_pose"]))
    return maps


def load_models(model_name: str, device: torch.device):
    """Load all available level checkpoints.
    Returns {level: (model, le, name_map)} where name_map is {pose_id: name}."""
    maps_path = os.path.join(CKPT_DIR, "label_maps.pkl")
    if not os.path.exists(maps_path):
        print(f"[ERROR] No label_maps.pkl found in {CKPT_DIR}/")
        print("        Please run: python train.py --model base --all")
        sys.exit(1)

    with open(maps_path, "rb") as f:
        label_maps = pickle.load(f)

    pose_names = _load_pose_names()   # {level: {id: name}}

    loaded = {}
    for level in [1, 2, 3]:
        ckpt_path = os.path.join(CKPT_DIR, f"best_{model_name}_L{level}.pth")
        if not os.path.exists(ckpt_path):
            continue
        if level not in label_maps:
            continue

        le       = label_maps[level]
        n_cls    = len(le.classes_)
        ModelCls = MODEL_MAP[model_name]
        model    = ModelCls(FEAT_DIM, n_cls).to(device)

        ckpt = torch.load(ckpt_path, map_location=device)
        state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state)
        model.eval()

        loaded[level] = (model, le, pose_names.get(level, {}))
        print(f"  Loaded L{level}: {n_cls} classes — {ckpt_path}")

    if not loaded:
        print(f"[ERROR] No checkpoints found in {CKPT_DIR}/ for model={model_name}")
        print("        Run: python train.py --model base --level 1 (or 2 or 3)")
        sys.exit(1)

    return loaded

# ─────────────────────────────────────────────────────────────
# Feature extraction from landmark buffer
# ─────────────────────────────────────────────────────────────
def landmarks_to_feature(buffer: list) -> np.ndarray | None:
    """
    buffer: list of MediaPipe NormalizedLandmarkList (one per frame)
    Returns 198-dim feature vector, or None if buffer is empty.
    """
    if not buffer:
        return None

    arr = np.zeros((len(buffer), N_LANDMARKS, N_COORDS), dtype=np.float32)
    for fi, lm_list in enumerate(buffer):
        for li, lm in enumerate(lm_list.landmark):
            arr[fi, li, 0] = lm.x
            arr[fi, li, 1] = lm.y
            arr[fi, li, 2] = lm.z

    mean_f = arr.mean(axis=0).flatten()
    std_f  = arr.std(axis=0).flatten()
    return np.concatenate([mean_f, std_f])


def landmarks_to_feature_new(buffer: list) -> np.ndarray | None:
    """
    buffer: list of MediaPipe Tasks API pose_world_landmarks[0]
            — each element is a plain Python list of NormalizedLandmark objects
              with .x .y .z attributes  (no .landmark sub-attribute)
    Returns 198-dim feature vector, or None if buffer is empty.
    """
    if not buffer:
        return None

    arr = np.zeros((len(buffer), N_LANDMARKS, N_COORDS), dtype=np.float32)
    for fi, lm_list in enumerate(buffer):
        for li, lm in enumerate(lm_list):
            arr[fi, li, 0] = lm.x
            arr[fi, li, 1] = lm.y
            arr[fi, li, 2] = lm.z

    mean_f = arr.mean(axis=0).flatten()
    std_f  = arr.std(axis=0).flatten()
    return np.concatenate([mean_f, std_f])


# ─────────────────────────────────────────────────────────────
# Inference on a feature vector
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def predict(feat: np.ndarray, models_dict: dict, device: torch.device) -> dict:
    """Returns {level: (label_str, confidence_float)}"""
    x = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(device)
    results = {}
    for level, (model, le, name_map) in models_dict.items():
        model.eval()
        logits = model(x)
        probs  = torch.softmax(logits, dim=1)
        conf, idx = probs.max(1)
        pose_id = le.inverse_transform([idx.item()])[0]
        # Map numeric pose_id → human-readable name
        name = name_map.get(int(pose_id), str(pose_id))
        results[level] = (name, round(conf.item() * 100, 1))
    return results

# ─────────────────────────────────────────────────────────────
# Draw overlay on frame
# ─────────────────────────────────────────────────────────────
def draw_overlay(frame: np.ndarray, preds: dict | None,
                 frame_idx: int, total_frames: int,
                 window_frames: int, buffer_len: int) -> np.ndarray:
    h, w = frame.shape[:2]
    panel_w = 320
    panel_h = 30 + len(LEVEL_NAMES) * 55 + 30
    # Semi-transparent background panel
    overlay = frame.copy()
    x0, y0 = 10, 10
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), CLR_BG, -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    y = y0 + 22
    cv2.putText(frame, "Yoga Pose Recognition", (x0 + 8, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, CLR_WHITE, 1, cv2.LINE_AA)
    y += 16
    cv2.line(frame, (x0 + 8, y), (x0 + panel_w - 8, y), CLR_GRAY, 1)
    y += 14

    colors = [CLR_GREEN, CLR_YELLOW, CLR_BLUE]
    if preds:
        for i, level in enumerate([1, 2, 3]):
            if level not in preds:
                continue
            label, conf = preds[level]
            level_name = LEVEL_NAMES[level]
            cv2.putText(frame, level_name, (x0 + 8, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, CLR_GRAY, 1, cv2.LINE_AA)
            y += 16
            # Pose name (clip long names)
            label_str = label if len(label) <= 28 else label[:25] + "..."
            cv2.putText(frame, label_str, (x0 + 8, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, colors[i], 1, cv2.LINE_AA)
            y += 16
            # Confidence bar
            bar_w = int((panel_w - 20) * conf / 100)
            cv2.rectangle(frame, (x0 + 8, y), (x0 + 8 + panel_w - 20, y + 6),
                          (60, 60, 60), -1)
            cv2.rectangle(frame, (x0 + 8, y), (x0 + 8 + bar_w, y + 6),
                          colors[i], -1)
            cv2.putText(frame, f"{conf:.1f}%", (x0 + panel_w - 50, y + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, colors[i], 1, cv2.LINE_AA)
            y += 18
    else:
        cv2.putText(frame, f"Buffering... ({buffer_len}/{window_frames})",
                    (x0 + 8, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    CLR_GRAY, 1, cv2.LINE_AA)
        y += 55 * 3

    # Progress bar (bottom)
    prog = int(w * frame_idx / max(total_frames, 1))
    cv2.rectangle(frame, (0, h - 6), (w, h), (50, 50, 50), -1)
    cv2.rectangle(frame, (0, h - 6), (prog, h), CLR_GREEN, -1)

    return frame

# ─────────────────────────────────────────────────────────────
# Download video
# ─────────────────────────────────────────────────────────────
def download_video(url: str, out_dir: str) -> str:
    """Download video using yt-dlp. Returns local file path."""
    import yt_dlp
    os.makedirs(out_dir, exist_ok=True)
    out_template = os.path.join(out_dir, "%(id)s.%(ext)s")

    ydl_opts = {
        "format": "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best[height<=720]",
        "outtmpl": out_template,
        "quiet": False,
        "noplaylist": True,
        "merge_output_format": "mp4",
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        video_id = info.get("id", "video")
        ext = info.get("ext", "mp4")
        path = os.path.join(out_dir, f"{video_id}.{ext}")
        if not os.path.exists(path):
            # fallback: look for any mp4 with video_id
            for f in os.listdir(out_dir):
                if f.startswith(video_id):
                    path = os.path.join(out_dir, f)
                    break
    return path

# ─────────────────────────────────────────────────────────────
# Main processing  (MediaPipe Tasks API — v0.10+)
# ─────────────────────────────────────────────────────────────
POSE_MODEL_PATH = "pose_landmarker_full.task"

def process_video(video_path: str, models_dict: dict, device: torch.device,
                  window: int, write_video: bool, out_path: str,
                  max_sec: int = 0):
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30
    w            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\nVideo: {os.path.basename(video_path)}")
    print(f"  {w}×{h} @ {fps:.1f} fps | {total_frames} frames | window={window}")

    max_frames = int(fps * max_sec) if max_sec > 0 else total_frames
    if max_sec > 0:
        print(f"  Capped at first {max_sec}s ({max_frames} frames)")

    writer = None
    if write_video:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    # ── MediaPipe Tasks PoseLandmarker ────────────────────────
    base_opts = mp_tasks.BaseOptions(model_asset_path=POSE_MODEL_PATH)
    options   = mp_vision.PoseLandmarkerOptions(
        base_options=base_opts,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )
    landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    lm_buffer    = []
    preds        = None
    frame_idx    = 0
    pred_history = {level: [] for level in models_dict}
    detected     = 0

    print("Processing frames...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_ms = int(frame_idx * 1000 / fps)
        rgb          = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img       = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result       = landmarker.detect_for_video(mp_img, timestamp_ms)

        if result.pose_world_landmarks:
            detected += 1
            lm_buffer.append(result.pose_world_landmarks[0])   # first person
            if len(lm_buffer) > window:
                lm_buffer.pop(0)

            # Draw skeleton on frame if writing video
            if writer and result.pose_landmarks:
                # Draw connections manually using pixel coordinates
                h_px, w_px = frame.shape[:2]
                lms_2d = result.pose_landmarks[0]
                CONNECTIONS = [
                    (11,13),(13,15),(12,14),(14,16),(11,12),
                    (23,25),(25,27),(24,26),(26,28),(23,24),
                    (11,23),(12,24),(0,11),(0,12),
                ]
                for (a, b) in CONNECTIONS:
                    if a < len(lms_2d) and b < len(lms_2d):
                        x1 = int(lms_2d[a].x * w_px)
                        y1 = int(lms_2d[a].y * h_px)
                        x2 = int(lms_2d[b].x * w_px)
                        y2 = int(lms_2d[b].y * h_px)
                        cv2.line(frame, (x1, y1), (x2, y2), CLR_GREEN, 2)

        # Update prediction every 15 frames
        if len(lm_buffer) >= min(window, 15) and frame_idx % 15 == 0:
            feat = landmarks_to_feature_new(lm_buffer)
            if feat is not None:
                preds = predict(feat, models_dict, device)
                for level, (label, conf) in preds.items():
                    pred_history[level].append((label, conf))

        if writer:
            frame = draw_overlay(frame, preds, frame_idx, max_frames, window,
                                 len(lm_buffer))
            writer.write(frame)

        frame_idx += 1
        if frame_idx >= max_frames:
            break
        if frame_idx % 200 == 0:
            print(f"  Frame {frame_idx}/{max_frames}  pose_detected={detected}",
                  flush=True)

    cap.release()
    landmarker.close()
    if writer:
        writer.release()
        print(f"\nOutput video saved: {out_path}")

    print(f"  Frames processed: {frame_idx} | Pose detected in: {detected} frames")

    print("\n" + "="*55)
    print("PREDICTION SUMMARY")
    print("="*55)
    for level, history in pred_history.items():
        if not history:
            continue
        from collections import Counter
        label_counts = Counter(label for label, _ in history)
        top_label    = label_counts.most_common(1)[0][0]
        avg_conf     = np.mean([c for l, c in history if l == top_label])
        level_tag = {1: "L1_CATEGORY", 2: "L2_GROUP", 3: "L3_POSE"}.get(level, f"L{level}")
        print(f"  {level_tag}: {top_label} | conf={avg_conf:.1f}%")
    print("="*55)


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Demo yoga pose recognition on a video")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--url",  type=str, help="YouTube or any video URL")
    src.add_argument("--file", type=str, help="Path to local video file")

    ap.add_argument("--model",     choices=["small", "base", "large"], default="base")
    ap.add_argument("--window",    type=int, default=60,
                    help="Rolling window size in frames (default 60 ≈ 2s @ 30fps)")
    ap.add_argument("--no-video",  action="store_true",
                    help="Skip writing annotated output video (faster)")
    ap.add_argument("--max-sec",   type=int, default=0,
                    help="Only process first N seconds (0=full video, default 0)")
    ap.add_argument("--out",       type=str, default=OUT_DIR,
                    help=f"Output directory (default: {OUT_DIR})")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load models
    print(f"\nLoading DNN-{args.model.capitalize()} checkpoints...")
    models_dict = load_models(args.model, device)
    print(f"Loaded {len(models_dict)} level(s): L{list(models_dict.keys())}")

    # Get video
    if args.file:
        video_path = args.file
    else:
        print(f"\nDownloading video: {args.url}")
        video_path = download_video(args.url, os.path.join(args.out, "_cache"))
        print(f"Downloaded: {video_path}")

    # Output path
    stem     = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(args.out, f"{stem}_annotated.mp4")

    process_video(
        video_path   = video_path,
        models_dict  = models_dict,
        device       = device,
        window       = args.window,
        write_video  = not args.no_video,
        out_path     = out_path,
        max_sec      = args.max_sec,
    )

if __name__ == "__main__":
    main()
