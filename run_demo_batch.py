"""
Batch demo: download 5 yoga videos and run pose recognition on each.
Run: python run_demo_batch.py
"""
import os, sys, re, io, json, time, subprocess
from pathlib import Path
from collections import Counter

# 5 short yoga videos covering different pose categories (verified available)
VIDEOS = [
    {
        "url":   "https://www.youtube.com/watch?v=sTANio_2E0Q",
        "label": "Full Body Stretch & Yoga (20min)",
        "expected_l1": "reclining/standing",
    },
    {
        "url":   "https://www.youtube.com/watch?v=VaoV1PrYft4",
        "label": "Morning Yoga for Beginners (10min)",
        "expected_l1": "standing",
    },
    {
        "url":   "https://www.youtube.com/watch?v=j7rKKpwdXNE",
        "label": "10-Min Yoga for Beginners",
        "expected_l1": "standing/sitting",
    },
    {
        "url":   "https://www.youtube.com/watch?v=4vTJHUDB5ak",
        "label": "Yoga Neck & Shoulders (15min)",
        "expected_l1": "sitting/reclining",
    },
    {
        "url":   "https://www.youtube.com/watch?v=KWBfQjuwp4E",
        "label": "Yoga for Beginners Full Body",
        "expected_l1": "standing/sitting",
    },
]

CACHE_DIR = "demo_output/_cache"
OUT_DIR   = "demo_output"
MODEL     = "small"          # DNN-Small best per our Table 4
WINDOW    = 60
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

def download(url: str) -> str | None:
    import yt_dlp
    ydl_opts = {
        "format": "bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/best[height<=480][ext=mp4]/best[height<=480]",
        "outtmpl": os.path.join(CACHE_DIR, "%(id)s.%(ext)s"),
        "quiet": True,
        "noplaylist": True,
        "merge_output_format": "mp4",
        "socket_timeout": 30,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            vid_id = info.get("id", "video")
            # find file
            for ext in ["mp4", "webm", "mkv"]:
                p = os.path.join(CACHE_DIR, f"{vid_id}.{ext}")
                if os.path.exists(p):
                    return p
            for f in os.listdir(CACHE_DIR):
                if f.startswith(vid_id):
                    return os.path.join(CACHE_DIR, f)
    except Exception as e:
        print(f"  Download error: {e}")
    return None

def run_inference(video_path: str) -> dict | None:
    """Run demo_video.py with --no-video and parse summary output."""
    cmd = [
        sys.executable, "demo_video.py",
        "--file", video_path,
        "--model", MODEL,
        "--window", str(WINDOW),
        "--max-sec", "120",   # first 2 minutes only — plenty for pose detection
        "--no-video",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        output = result.stdout + result.stderr
        # Parse PREDICTION SUMMARY block
        # Format: "  L1_CATEGORY: standing | conf=85.2%"
        preds = {}
        for line in output.splitlines():
            m = re.search(r"(L\d)_(\w+):\s*(.+?)\s*\|\s*conf=([\d.]+)%", line)
            if m:
                level_map = {"L1": 1, "L2": 2, "L3": 3}
                level     = level_map.get(m.group(1), 0)
                pose_name = m.group(3).strip()
                conf      = float(m.group(4))
                if level:
                    preds[level] = {"pose": pose_name, "conf": conf}
        return preds if preds else None
    except subprocess.TimeoutExpired:
        print("  Timeout (>10 min) — skipping")
        return None
    except Exception as e:
        print(f"  Inference error: {e}")
        return None

def main():
    results = []
    print("=" * 65)
    print("  3DYoga90 Batch Demo  —  DNN-Small  —  5 videos")
    print("=" * 65)

    for i, vid in enumerate(VIDEOS, 1):
        print(f"\n[{i}/5] {vid['label']}")
        print(f"       {vid['url']}")

        # Download
        print("  Downloading (≤480p)...", end=" ", flush=True)
        t0 = time.time()
        path = download(vid["url"])
        if not path:
            print("FAILED — skipping")
            results.append({**vid, "status": "download_failed"})
            continue
        size_mb = os.path.getsize(path) / 1e6
        print(f"OK ({size_mb:.1f} MB, {time.time()-t0:.1f}s)")

        # Inference
        print("  Running inference...", end=" ", flush=True)
        t1 = time.time()
        preds = run_inference(path)
        elapsed = time.time() - t1

        if preds:
            print(f"OK ({elapsed:.1f}s)")
            print(f"  L1 Category : {preds.get(1,{}).get('pose','—'):30s}  conf={preds.get(1,{}).get('conf',0):.1f}%")
            print(f"  L2 Group    : {preds.get(2,{}).get('pose','—'):30s}  conf={preds.get(2,{}).get('conf',0):.1f}%")
            print(f"  L3 Pose     : {preds.get(3,{}).get('pose','—'):30s}  conf={preds.get(3,{}).get('conf',0):.1f}%")
            results.append({**vid, "status": "ok", "preds": preds})
        else:
            print(f"No prediction ({elapsed:.1f}s) — possibly no pose detected")
            results.append({**vid, "status": "no_pred"})

    # ── Summary table ────────────────────────────────────────
    print("\n\n" + "=" * 65)
    print("  RESULTS SUMMARY")
    print("=" * 65)
    print(f"{'#':<3} {'Video':<30} {'L1 Category':<16} {'L2 Group':<26} {'L3 Pose (conf%)'}")
    print("-" * 65)
    for i, r in enumerate(results, 1):
        label = r["label"][:29]
        if r.get("status") == "ok":
            p = r["preds"]
            l1 = f"{p.get(1,{}).get('pose','—')[:14]} ({p.get(1,{}).get('conf',0):.0f}%)"
            l2 = f"{p.get(2,{}).get('pose','—')[:22]} ({p.get(2,{}).get('conf',0):.0f}%)"
            l3 = f"{p.get(3,{}).get('pose','—')[:25]} ({p.get(3,{}).get('conf',0):.0f}%)"
            print(f"{i:<3} {label:<30} {l1:<16} {l2:<26} {l3}")
        else:
            print(f"{i:<3} {label:<30} [{r['status']}]")
    print("=" * 65)

    # Save JSON
    out_json = os.path.join(OUT_DIR, "batch_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nDetailed results saved to: {out_json}")

if __name__ == "__main__":
    main()
