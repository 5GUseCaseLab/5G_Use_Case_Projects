import argparse
import gc
import logging
import os
import shutil
from importlib.resources import as_file, files
from pathlib import Path

import cv2
import numpy as np
import vitallens.ssd
from vitallens import VitalLens

# Prevent thread collisions and silence harmless 1-pixel crop warnings
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
logging.getLogger().setLevel(logging.ERROR)

# Python 3.13 context manager fix
vitallens.ssd.files = lambda *args, **kwargs: as_file(files(*args, **kwargs))

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def resolve_video_paths(videos: list[str]) -> list[str]:
    if videos:
        resolved = []
        for video in videos:
            if not os.path.exists(video):
                raise FileNotFoundError(f"Video not found at: {video}")
            resolved.append(video)
        return resolved

    discovered = sorted(
        (p for p in Path.cwd().iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS),
        key=lambda p: p.name,
    )
    if not discovered:
        raise FileNotFoundError(f"No video files found in: {Path.cwd()}")
    return [str(p) for p in discovered]


def load_video_rgb(video_path: str) -> tuple[np.ndarray, float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames or fps <= 0:
        raise ValueError(f"Failed to read frames or FPS from: {video_path}")
    return np.asarray(frames, dtype=np.uint8), float(fps)


def analyze_video(vl: VitalLens, video_path: str, expected_bpm: float | None = None) -> None:
    print(f"[INFO] Analyzing {video_path}...")
    if shutil.which("ffprobe"):
        results = vl(video_path)
    else:
        frames_rgb, fps = load_video_rgb(video_path)
        results = vl(frames_rgb, fps=fps)

    print("\n" + "=" * 55)
    print("             VITAL SIGNS & TRIAGE REPORT")
    print("=" * 55)

    if not results:
        print("  ⚠️  No face detected in the video.")
    else:
        for idx, person in enumerate(results, 1):
            vitals = person.get("vitals", {})
            hr_info = vitals.get("heart_rate", {})

            bpm = hr_info.get("value")
            conf = hr_info.get("confidence", 0.0)

            if bpm is not None:
                if bpm > 120 or bpm < 40:
                    tag = "RED (Immediate Attention)"
                elif bpm > 100 or bpm < 60:
                    tag = "YELLOW (Abnormal / High)"
                else:
                    tag = "GREEN (Normal / Stable)"
            else:
                tag = "UNKNOWN"

            print(f"\nSubject #{idx} ({os.path.basename(video_path)}):")
            if bpm is not None:
                print(f"  ❤️  Estimated HR:    {bpm:5.1f} BPM  (Confidence: {conf:.2f})")
                if expected_bpm is not None:
                    print(f"  🎯  Expected HR:     {expected_bpm:5.1f} BPM  (Error: {abs(bpm - expected_bpm):.1f} BPM)")
                print(f"  🏷️  START Tag:       [{tag}]")
            else:
                print("  ⚠️  Could not detect reliable pulse from video.")

            if "respiratory_rate" in vitals and vitals["respiratory_rate"].get("value"):
                resp = vitals["respiratory_rate"]["value"]
                resp_conf = vitals["respiratory_rate"].get("confidence", 0.0)
                print(f"  🫁  Respiratory:     {resp:5.1f} RPM  (Confidence: {resp_conf:.2f})")

    print("\n" + "=" * 55)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VitalLens POS analysis on local video files.")
    parser.add_argument("videos", nargs="*", help="Optional video file paths or filenames in the current directory.")
    parser.add_argument("--expected-bpm", type=float, default=None, help="Optional expected heart rate in BPM.")
    args = parser.parse_args()

    video_paths = resolve_video_paths(args.videos)

    print("[INFO] Initializing VitalLens (POS algorithm)...")
    vl = VitalLens(method="pos")

    for video_path in video_paths:
        analyze_video(vl, video_path, expected_bpm=args.expected_bpm)

    gc.collect()


if __name__ == "__main__":
    main()
