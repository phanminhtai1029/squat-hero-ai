"""
Download 3DYoga90 skeleton parquet files from Google Drive.

Strategy:
  1. Capture gdown's stdout while listing the folder (skip_download=True) to
     collect (file_id, filename) pairs — works for up to ~50 at a time.
  2. Download each file individually using gdown, skipping already-present ones.
  3. Repeat step-1 once per run; already-downloaded files are skipped each time,
     so successive runs make forward progress as Drive's rate-limit resets.

Usage:  python download_skeleton.py [--delay SECS] [--batch-pause SECS]
"""

import os, re, sys, time, io, argparse, subprocess
from pathlib import Path
from contextlib import redirect_stdout
import gdown

FOLDER_ID = "11SOWVJ5CF5pbkftMqogVP5Pkyg88hbau"
OUT_DIR   = Path("3DYoga90/data/landmarks/official_dataset")

def enumerate_batch(folder_id: str) -> list[tuple[str, str]]:
    """Return up to 50 (file_id, filename) pairs by capturing gdown stdout."""
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            gdown.download_folder(
                id=folder_id,
                skip_download=True,
                remaining_ok=True,
                quiet=False,
            )
    except Exception:
        pass
    out = buf.getvalue()
    pairs = re.findall(r"Processing file ([A-Za-z0-9_-]{25,}) (\d+\.parquet)", out)
    return pairs

def download_one(file_id: str, filename: str, out_dir: Path, delay: float) -> bool:
    out_path = out_dir / filename
    if out_path.exists():
        return True
    url = f"https://drive.google.com/uc?id={file_id}"
    try:
        gdown.download(url, str(out_path), quiet=True, fuzzy=True)
    except Exception as e:
        print(f"  ✗ {filename}: {e}")
        return False
    time.sleep(delay)
    return out_path.exists()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delay",       type=float, default=0.8,
                    help="Seconds between file downloads (default 0.8)")
    ap.add_argument("--batch-pause", type=float, default=90.0,
                    help="Seconds to pause between listing batches (default 90)")
    ap.add_argument("--max-batches", type=int,   default=200,
                    help="Max listing+download cycles (default 200)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for batch_num in range(1, args.max_batches + 1):
        on_disk = len(list(OUT_DIR.glob("*.parquet")))
        print(f"\n=== Batch {batch_num} | on-disk: {on_disk} ===")

        pairs = enumerate_batch(FOLDER_ID)
        if not pairs:
            print("Folder listing returned nothing — rate limited. Pausing...")
            time.sleep(args.batch_pause)
            continue

        todo = [(fid, name) for fid, name in pairs if not (OUT_DIR / name).exists()]
        print(f"Listed {len(pairs)} files, {len(todo)} need downloading.")

        if not todo:
            print("All listed files already on disk.")
            # If we've plateaued, just keep cycling until Drive returns new IDs
            print(f"Pausing {args.batch_pause}s before next listing attempt...")
            time.sleep(args.batch_pause)
            continue

        failed = []
        for i, (fid, name) in enumerate(todo, 1):
            ok = download_one(fid, name, OUT_DIR, args.delay)
            if not ok:
                failed.append((fid, name))
            if i % 10 == 0:
                print(f"  {i}/{len(todo)} done (failed={len(failed)})", flush=True)

        on_disk_after = len(list(OUT_DIR.glob("*.parquet")))
        print(f"Batch done. on-disk: {on_disk} → {on_disk_after} | failed: {len(failed)}")

        if on_disk_after >= 5526:
            print("\n✓ All 5526 files downloaded!")
            return

        if failed:
            print(f"Retrying {len(failed)} failed after 30s...")
            time.sleep(30)
            for fid, name in failed:
                download_one(fid, name, OUT_DIR, delay=2.0)

        if todo:
            print(f"Pausing {args.batch_pause}s for quota reset before next batch...")
            time.sleep(args.batch_pause)

    print(f"\nReached max batches. Final count: {len(list(OUT_DIR.glob('*.parquet')))}")

if __name__ == "__main__":
    main()
