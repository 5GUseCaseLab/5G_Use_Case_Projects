"""
WiFeye: Fall-trial augmentation (training-fold-only)
==========================================================
Subject 3 recorded no fall trials. This module creates SYNTHETIC fall trials from the
real fall trials of the other subjects, to be used ONLY as extra training data inside
each cross-validation fold. They are never test data and are never reported as
subject 3's falls.

Provenance is kept for every synthetic trial (augmented_manifest.csv):
  aug_trial_id, source_trial (real trial it was derived from), source_subject,
  style_subject (subject whose motion-energy scale it was rescaled to, or ""),
  kind ("generic" or "style"), and the time mapping (warp w, centre c, shift delta)
  so action-interval labels can be moved with the signal:
      t_aug = c + w * (t_src - c) + delta

train_classifier.py enforces the leakage rules per fold:
  * a synthetic trial is used only if its source_trial is in that fold's training set;
  * "style" trials for subject S are dropped from the fold whose test subject is S
    (their scale was estimated from S's data);
  * synthetic trials are never added to any test set.

Transforms (amplitude only; phase columns get the same time mapping, no value change):
  * time shift delta ~ U(-0.4, 0.4) s (edge-hold padding, no circular wrap) and a mild
    time warp w ~ U(0.95, 1.05) around c = 4 s (kept small: warping changes apparent
    body speed, which is itself a fall cue; Iwana & Uchida, PLOS ONE 2021)
  * scaling of the dynamic part (signal minus per-subcarrier median) by a global factor
    ("generic": U(0.8, 1.25); "style": subject-energy ratio x U(0.9, 1.1))
  * smooth gain curve across subcarriers on the dynamic part (sd 0.08)
  * additive Gaussian noise at 0.15 x the trial's own packet-to-packet noise level
No mixup and no frequency/Doppler shifting (could turn a fast fall into a slow sit-down).

Usage:
  python augment_csi.py --aligned-dir data/aligned --out-dir data/augmented \
         --k-generic 2 --style-subject 3 --seed 42
then run csi_preprocess.py and feature_extraction.py on the output directory.
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd

ADL = ("walk", "sit", "stand", "hand")
CENTER_S = 4.0


def _amp_cols(df, rx):
    return [c for c in df.columns if c.startswith(f"{rx}_amp_")]


def dynamic_scale(df, t_lo=2.0, t_hi=6.0):
    """Motion-energy proxy per RX: mean over subcarriers of the std over time (2-6 s)."""
    t = df["time_rel_s"].values
    m = (t >= t_lo) & (t < t_hi)
    out = {}
    for rx in ("rx1", "rx2"):
        cols = _amp_cols(df, rx)
        if not cols:
            continue
        x = df.loc[m, cols].values
        out[rx] = float(np.mean(np.std(x, axis=0)))
    return out


def subject_energy_table(aligned_dir):
    """Median ADL motion-energy proxy per (subject, rx), from aligned trials."""
    rows = []
    for act in ADL:
        for f in glob.glob(os.path.join(aligned_dir, act, "*_aligned.csv")):
            subj = os.path.basename(f).split("_")[0]
            for rx, v in dynamic_scale(pd.read_csv(f)).items():
                rows.append({"subject": subj, "rx": rx, "act": act, "scale": v})
    df = pd.DataFrame(rows)
    # per-activity medians, then the median over activities (robust to one odd class)
    per_act = df.groupby(["subject", "rx", "act"])["scale"].median().reset_index()
    return per_act.groupby(["subject", "rx"])["scale"].median().to_dict()


def _time_map(df, w, delta, rng=None):
    """Resample every signal column so that content at t_src appears at t_aug."""
    t = df["time_rel_s"].values
    t_src = CENTER_S + (t - delta - CENTER_S) / w          # inverse mapping
    t_src = np.clip(t_src, t[0], t[-1])                     # edge-hold outside range
    out = df.copy()
    for c in df.columns:
        if "amp_" in c or c.endswith("_rssi"):
            out[c] = np.interp(t_src, t, df[c].values)
        elif "phase_" in c:
            un = np.unwrap(df[c].values)
            out[c] = (np.interp(t_src, t, un) + np.pi) % (2 * np.pi) - np.pi
    return out


def augment_trial(df, rng, amp_scale_by_rx, noise_frac=0.15, gain_sd=0.08,
                  shift_s=0.4, warp=0.05):
    w = float(rng.uniform(1 - warp, 1 + warp))
    delta = float(rng.uniform(-shift_s, shift_s))
    out = _time_map(df, w, delta)
    for rx in ("rx1", "rx2"):
        cols = _amp_cols(out, rx)
        if not cols:
            continue
        x = out[cols].values
        base = np.median(x, axis=0, keepdims=True)
        dyn = x - base
        # smooth gain curve across subcarriers (independent per-subcarrier gains are less
        # physical): 1 + gain_sd * random combination of 3 low-order cosines
        k = np.linspace(0, np.pi, x.shape[1])
        coef = rng.normal(0.0, 1.0, size=3) / np.sqrt(3)
        gain = (1.0 + gain_sd * sum(c * np.cos(j * k) for j, c in enumerate(coef)))[None, :]
        scale = amp_scale_by_rx.get(rx, 1.0)
        src_noise = np.median(np.abs(np.diff(df[cols].values, axis=0)), axis=0, keepdims=True)
        noise = rng.normal(0.0, 1.0, size=x.shape) * noise_frac * src_noise
        y = base + scale * gain * dyn + noise
        y[:, (np.abs(x) < 1e-9).all(axis=0)] = 0.0          # keep null subcarriers null
        out[cols] = np.maximum(y, 0.0)                         # amplitudes stay non-negative
    return out, w, delta


def main():
    ap = argparse.ArgumentParser(description="Generate training-only synthetic fall trials")
    ap.add_argument("--aligned-dir", default="data/aligned")
    ap.add_argument("--out-dir", default="data/augmented")
    ap.add_argument("--activity", default="fall")
    ap.add_argument("--k-generic", type=int, default=2, help="generic variants per real trial")
    ap.add_argument("--style-subject", default="3",
                    help="subject whose energy scale 'style' variants mimic ('' to disable)")
    ap.add_argument("--k-style", type=int, default=1, help="style variants per real trial")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    src_files = sorted(glob.glob(os.path.join(args.aligned_dir, args.activity, "*_aligned.csv")))
    out_act = os.path.join(args.out_dir, args.activity)
    os.makedirs(out_act, exist_ok=True)

    energy = subject_energy_table(args.aligned_dir) if args.style_subject else {}
    rows = []
    print("\n=======================================================")
    print("  SAFESENSE-5G FALL AUGMENTATION (training-only data)")
    print("=======================================================")
    print(f"  Real source trials: {len(src_files)} | generic/trial: {args.k_generic} | "
          f"style(subject {args.style_subject or '-'})/trial: {args.k_style if args.style_subject else 0}")
    if energy:
        print("  ADL motion-energy proxy per subject (rx1, rx2):")
        for s in sorted({k[0] for k in energy}):
            print(f"    subject {s}: {energy.get((s, 'rx1'), float('nan')):.3f}, {energy.get((s, 'rx2'), float('nan')):.3f}")

    for f in src_files:
        src_trial = os.path.basename(f).replace("_aligned.csv", "")
        src_subj = src_trial.split("_")[0]
        if src_subj == args.style_subject:
            continue
        df = pd.read_csv(f)
        jobs = [("generic", j) for j in range(args.k_generic)]
        if args.style_subject:
            jobs += [("style", j) for j in range(args.k_style)]
        for kind, j in jobs:
            if kind == "generic":
                s = float(rng.uniform(0.8, 1.25))
                scales = {"rx1": s, "rx2": s}
                style = ""
            else:
                scales = {}
                for rx in ("rx1", "rx2"):
                    ratio = energy.get((args.style_subject, rx), np.nan) / energy.get((src_subj, rx), np.nan)
                    scales[rx] = float(np.clip(ratio if np.isfinite(ratio) else 1.0, 0.5, 2.0)
                                       * rng.uniform(0.9, 1.1))
                style = args.style_subject
            aug_df, w, delta = augment_trial(df, rng, scales)
            tag = f"g{j}" if kind == "generic" else f"s{args.style_subject}{j}"
            aug_id = f"aug_{src_trial}_{tag}"
            aug_df.to_csv(os.path.join(out_act, f"{aug_id}_aligned.csv"), index=False)
            rows.append({"aug_trial_id": aug_id, "source_trial": src_trial, "source_subject": src_subj,
                         "style_subject": style, "kind": kind, "activity": args.activity,
                         "warp_w": round(w, 4), "warp_center_s": CENTER_S, "shift_delta_s": round(delta, 4),
                         "scale_rx1": round(scales["rx1"], 4), "scale_rx2": round(scales["rx2"], 4)})
    man = pd.DataFrame(rows)
    man.to_csv(os.path.join(args.out_dir, "augmented_manifest.csv"), index=False)
    print(f"\n  Wrote {len(man)} synthetic trials ({(man.kind == 'generic').sum()} generic, "
          f"{(man.kind == 'style').sum()} style) to {out_act}")
    print(f"  Provenance: {os.path.join(args.out_dir, 'augmented_manifest.csv')}")
    print("  NOTE: training-only data. Never evaluate on it or report it as subject "
          f"{args.style_subject}'s falls.")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
