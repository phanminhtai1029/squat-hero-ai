"""
Real-time yoga pose recognition from webcam using MediaPipe + trained DNN.

Usage:
  python realtime_webcam.py                          # default: all 3 levels, DNN-Base
  python realtime_webcam.py --model small --level 3  # specific model/level
  python realtime_webcam.py --camera 1               # use camera index 1

Controls:
  Q / ESC  — quit
  S        — save screenshot
"""

import argparse
import os
import pickle
import time
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp

# ─── reuse model definitions from train.py ───────────────────
DROPOUT     = 0.4
N_LANDMARKS = 33
N_COORDS    = 3
FEAT_DIM    = N_LANDMARKS * N_COORDS * 2   # 198
CKPT_DIR    = "checkpoints"

def make_block(in_dim: int, out_dim: int) -> nn.Sequential:
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

# ─── POSE NAME MAP from CSV ───────────────────────────────────
import pandas as pd

def build_pose_names():
    """Build {pose_id: pose_name} dicts for L1/L2/L3 from CSV."""
    csv_path = "3DYoga90/data/3DYoga90.csv"
    if not os.path.exists(csv_path):
        return {1: {}, 2: {}, 3: {}}
    df = pd.read_csv(csv_path)
    names = {}
    for lvl, id_col, name_col in [(1, "l1_pose_id", "l1_pose"),
                                   (2, "l2_pose_id", "l2_pose"),
                                   (3, "l3_pose_id", "l3_pose")]:
        names[lvl] = dict(zip(df[id_col], df[name_col]))
    return names


# ─── Feature extraction from live landmarks ──────────────────
def landmarks_to_feature(landmark_buffer: list) -> np.ndarray:
    """
    landmark_buffer: list of mediapipe pose_world_landmarks results
    Returns feature vector of shape (FEAT_DIM,) = (198,)
    """
    arr = np.array([
        [[lm.x, lm.y, lm.z] for lm in frame.landmark]
        for frame in landmark_buffer
    ], dtype=np.float32)   # (T, 33, 3)

    mean_feat = arr.mean(axis=0).flatten()   # 99
    std_feat  = arr.std(axis=0).flatten()    # 99
    return np.concatenate([mean_feat, std_feat])   # 198


# ─── Load model ───────────────────────────────────────────────
def load_model(model_name: str, level: int, device: torch.device):
    ckpt_path = os.path.join(CKPT_DIR, f"best_{model_name}_L{level}.pth")
    maps_path = os.path.join(CKPT_DIR, "label_maps.pkl")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Run `python train.py --model {model_name} --level {level}` first."
        )
    if not os.path.exists(maps_path):
        raise FileNotFoundError(f"Label maps not found: {maps_path}")

    ckpt       = torch.load(ckpt_path, map_location=device)
    n_classes  = ckpt["n_classes"]
    ModelClass = MODEL_MAP[model_name]
    model      = ModelClass(FEAT_DIM, n_classes).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    with open(maps_path, "rb") as f:
        maps = pickle.load(f)
    label_enc = maps[level]

    return model, label_enc


# ─── Drawing helpers ──────────────────────────────────────────
COLORS = {1: (0, 255, 100), 2: (0, 200, 255), 3: (255, 165, 0)}

def draw_prediction(frame: np.ndarray, predictions: dict, fps: float,
                    buffer_size: int, pose_names: dict, conf: dict):
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Semi-transparent background panel
    cv2.rectangle(overlay, (0, 0), (w, 160), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}  Frames: {buffer_size}",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    # Predictions for each level
    y_offset = 50
    for lvl in sorted(predictions.keys()):
        pid        = predictions[lvl]
        pose_label = pose_names.get(lvl, {}).get(pid, f"ID {pid}")
        confidence = conf.get(lvl, 0.0)
        color      = COLORS.get(lvl, (255, 255, 255))

        text = f"L{lvl}: {pose_label}  ({confidence:.1%})"
        cv2.putText(frame, text, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        y_offset += 36

    # Collecting indicator
    if buffer_size < 10:
        cv2.putText(frame, "Collecting frames...", (w // 2 - 100, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 100, 255), 1)

    return frame


# ─── Main inference loop ──────────────────────────────────────
def run(model_name: str, levels: list, camera_idx: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load models for each requested level
    models = {}
    label_encs = {}
    for lvl in levels:
        print(f"Loading DNN-{model_name.capitalize()} L{lvl}...")
        models[lvl], label_encs[lvl] = load_model(model_name, lvl, device)

    pose_names = build_pose_names()

    # MediaPipe pose
    mp_pose    = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_styles  = mp.solutions.drawing_styles

    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_idx}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print(f"Camera {camera_idx} opened — {int(cap.get(3))}x{int(cap.get(4))}")
    print("Press Q or ESC to quit, S to save screenshot")

    landmark_buffer = []   # rolling window of pose landmarks
    WINDOW_SIZE     = 30   # accumulate 30 frames before first prediction
    predictions     = {}
    confidences     = {}

    prev_time = time.time()
    os.makedirs("screenshots", exist_ok=True)
    screenshot_cnt = 0

    with mp_pose.Pose(model_complexity=1,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera read failed")
                break

            # FPS
            cur_time  = time.time()
            fps       = 1.0 / max(cur_time - prev_time, 1e-6)
            prev_time = cur_time

            # Pose detection
            rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            result = pose.process(rgb)
            rgb.flags.writeable = True
            frame = cv2.cvtColor(cv2.flip(rgb, 1), cv2.COLOR_RGB2BGR)

            if result.pose_world_landmarks:
                landmark_buffer.append(result.pose_world_landmarks)
                if len(landmark_buffer) > WINDOW_SIZE:
                    landmark_buffer.pop(0)

                # Draw skeleton
                mp_drawing.draw_landmarks(
                    frame,
                    result.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                )

                # Run inference once we have enough frames
                if len(landmark_buffer) >= 10:
                    feat = landmarks_to_feature(landmark_buffer)
                    x    = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(device)

                    with torch.no_grad():
                        for lvl in levels:
                            logits = models[lvl](x)
                            probs  = torch.softmax(logits, dim=1)
                            top1   = probs.argmax(1).item()
                            conf   = probs[0, top1].item()

                            pose_id           = int(label_encs[lvl].classes_[top1])
                            predictions[lvl]  = pose_id
                            confidences[lvl]  = conf
            else:
                # No person detected — clear buffer
                landmark_buffer.clear()
                predictions = {}
                confidences = {}
                cv2.putText(frame, "No person detected",
                            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            # Draw UI
            frame = draw_prediction(frame, predictions, fps,
                                    len(landmark_buffer), pose_names, confidences)

            cv2.imshow("3DYoga90 — Real-time Pose Recognition", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):   # Q or ESC
                break
            elif key == ord("s"):
                path = f"screenshots/yoga_{screenshot_cnt:04d}.jpg"
                cv2.imwrite(path, frame)
                print(f"Screenshot saved: {path}")
                screenshot_cnt += 1

    cap.release()
    cv2.destroyAllWindows()


# ─── Entry point ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time yoga pose recognition")
    parser.add_argument("--model",  choices=["small", "base", "large"], default="base")
    parser.add_argument("--level",  type=int, choices=[1, 2, 3], default=None,
                        help="Level to show (default: show all available levels)")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera device index (default: 0)")
    args = parser.parse_args()

    # Auto-detect available checkpoints
    if args.level is not None:
        levels = [args.level]
    else:
        levels = [lvl for lvl in [1, 2, 3]
                  if os.path.exists(os.path.join(
                      CKPT_DIR, f"best_{args.model}_L{lvl}.pth"))]
        if not levels:
            print(f"No checkpoints found in {CKPT_DIR}/")
            print(f"Run `python train.py --model {args.model} --all` first.")
            exit(1)

    print(f"Loaded levels: {levels}")
    run(args.model, levels, args.camera)
