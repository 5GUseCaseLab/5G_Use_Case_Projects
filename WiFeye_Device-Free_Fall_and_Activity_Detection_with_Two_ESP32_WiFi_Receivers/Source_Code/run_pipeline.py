"""
WiFeye: one-command reproducible data pipeline
===================================================
Runs merge -> preprocess -> feature extraction with the canonical flags of a named
pipeline version, so every result can be regenerated exactly.

  v1  original handover pipeline (all frames, all 64 subcarriers, 0.5 s moving-average
      static removal, 0.5 s windows, 22 features)
  v2  dominant (mac, sig_mode) source per trial, drop subcarriers 0 & 27-37,
      median static removal, 2 s windows + scale-free per-RX features
  v3  (DEFAULT, recommended) as v2 but keeps only HT frames whose mac field is the
      receiver's own MAC = router -> that ESP32 (AGENT_BRIDGE.md A7)

Outputs: data/aligned_<v>, data/processed_<v>, data/features_<v> (dual-RX),
data/features_<v>_rx1 and data/features_<v>_rx2 (single-receiver feature sets).
v1 writes to the original folders (data/aligned, data/processed, data/features).

Usage:
  python run_pipeline.py                 # v3, all stages
  python run_pipeline.py --version v2 --stages features
"""

import os
import sys
import argparse
import subprocess

PY = sys.executable

VERSIONS = {
    "v1": {"merge": [], "preprocess": ["--dc-mode", "moving"],
           "features": ["--win-len", "50", "--stride", "25"], "suffix": "", "single_rx": False},
    "v2": {"merge": ["--source-filter", "dominant", "--drop-subcarriers", "0,27-37"],
           "preprocess": ["--dc-mode", "median"],
           "features": ["--win-len", "200", "--stride", "25", "--extended"], "suffix": "_v2", "single_rx": True},
    "v3": {"merge": ["--source-filter", "own", "--drop-subcarriers", "0,27-37"],
           "preprocess": ["--dc-mode", "median"],
           "features": ["--win-len", "200", "--stride", "25", "--extended"], "suffix": "_v3", "single_rx": True},
}


def run(cmd):
    print("  $ " + " ".join(cmd), flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(f"  [FAIL] exit {r.returncode}: {' '.join(cmd)}")


def main():
    ap = argparse.ArgumentParser(description="Reproducible WiFeye pipeline runner")
    ap.add_argument("--version", choices=sorted(VERSIONS), default="v3")
    ap.add_argument("--stages", default="merge,preprocess,features",
                    help="comma list of: merge, preprocess, features")
    ap.add_argument("--raw-dir", default="data/raw")
    args = ap.parse_args()
    cfg, sfx = VERSIONS[args.version], VERSIONS[args.version]["suffix"]
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    aligned, processed, feats = f"data/aligned{sfx}", f"data/processed{sfx}", f"data/features{sfx}"

    print(f"\n  WiFeye pipeline {args.version}: stages {stages}")
    if "merge" in stages:
        run([PY, "merge_dual_rx.py", "--raw-dir", args.raw_dir, "--out-dir", aligned, "--fs", "100.0"] + cfg["merge"])
    if "preprocess" in stages:
        run([PY, "csi_preprocess.py", "--in-dir", aligned, "--out-dir", processed,
             "--fs", "100.0", "--cutoff", "10.0"] + cfg["preprocess"])
    if "features" in stages:
        base = [PY, "feature_extraction.py", "--in-dir", processed, "--fs", "100.0"] + cfg["features"]
        run(base + ["--out-dir", feats])
        if cfg["single_rx"]:
            run(base + ["--out-dir", f"{feats}_rx1", "--rx1-only"])
            run(base + ["--out-dir", f"{feats}_rx2", "--rx2-only"])
    print(f"  [DONE] pipeline {args.version}\n")


if __name__ == "__main__":
    main()
