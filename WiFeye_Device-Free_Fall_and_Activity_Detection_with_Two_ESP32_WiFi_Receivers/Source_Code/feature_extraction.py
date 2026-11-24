"""
WiFeye: Multi-Tier Windowing & Feature Extraction Engine
=============================================================
Segments preprocessed CSI streams into 0.5s sliding windows (50 samples, 50% overlap).
Extracts:
  1. Tier 1 & 2 (Edge Node / Arduino UNO Q):
     - Signal Energy Variance (σ²)
     - Motion Entropy (H)
     - Fall-Burst Derivative (dE/dt)
     - Decisions: Dormant Skip vs. Fall Fast-Path vs. Routine MEC
  2. Tier 3 (MEC Server / DANN Classifier):
     - Statistical: Mean, Variance, Skewness, Kurtosis, Peak-to-Peak Range
     - Spectral: Power Spectral Density (PSD) energy bands (0.5-2Hz, 2-5Hz, 5-10Hz),
       Spectral Entropy, Dominant Doppler Shift
     - Spatial: Cross-RX correlation between RX1 and RX2
Outputs feature matrix X, labels y, and window manifest.
"""

import os
import glob
import re
import argparse
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, entropy
from scipy.signal import welch


def extract_window_features(window_df, rx1_cols, rx2_cols=None, fs=100.0,
                            dormant_thresh=1.5, fall_fast_thresh=15.0):
    """
    Extracts Tier 1/2 edge metrics and Tier 3 deep features from a 0.5s window.
    """
    feats = {}

    # 1. Separate amplitude matrices
    mat_rx1 = window_df[rx1_cols].values # (W, 64)
    has_rx2 = rx2_cols is not None and len(rx2_cols) > 0
    mat_rx2 = window_df[rx2_cols].values if has_rx2 else None

    # Total instantaneous energy over time in window
    energy_rx1 = np.sum(mat_rx1 ** 2, axis=1) # (W,)
    total_energy = energy_rx1 + (np.sum(mat_rx2 ** 2, axis=1) if has_rx2 else 0)

    # -------------------------------------------------------------
    # TIER 1 & TIER 2: Edge Node Metrics (Arduino UNO Q fast logic)
    # -------------------------------------------------------------
    energy_var = float(np.var(total_energy))
    energy_mean = float(np.mean(total_energy))
    energy_max = float(np.max(total_energy))

    # Fall-burst derivative: max rate of energy rise (dE/dt)
    dE_dt = float(np.max(np.diff(total_energy))) if len(total_energy) > 1 else 0.0

    # Motion entropy across subcarrier variances
    subcarrier_vars = np.var(mat_rx1, axis=0) + 1e-9
    subcarrier_probs = subcarrier_vars / np.sum(subcarrier_vars)
    motion_entropy = float(entropy(subcarrier_probs))

    feats['tier_energy_var'] = energy_var
    feats['tier_energy_mean'] = energy_mean
    feats['tier_dE_dt'] = dE_dt
    feats['tier_motion_entropy'] = motion_entropy

    # Decision flags
    is_dormant = int(energy_var < dormant_thresh)
    is_fall_fastpath = int(dE_dt > fall_fast_thresh and energy_var > (dormant_thresh * 3))
    
    feats['flag_dormant'] = is_dormant
    feats['flag_fall_fastpath'] = is_fall_fastpath
    feats['routing_tier'] = 1 if is_dormant else (2 if is_fall_fastpath else 3)

    # -------------------------------------------------------------
    # TIER 3: MEC Server Features (DANN Classifier)
    # -------------------------------------------------------------
    # Statistical features across RX1
    feats['stat_mean_rx1'] = float(np.mean(mat_rx1))
    feats['stat_std_rx1'] = float(np.std(mat_rx1))
    feats['stat_skew_rx1'] = float(skew(mat_rx1.flatten()))
    feats['stat_kurt_rx1'] = float(kurtosis(mat_rx1.flatten()))
    feats['stat_ptp_rx1'] = float(np.ptp(mat_rx1))

    if has_rx2:
        feats['stat_mean_rx2'] = float(np.mean(mat_rx2))
        feats['stat_std_rx2'] = float(np.std(mat_rx2))
        feats['stat_skew_rx2'] = float(skew(mat_rx2.flatten()))
        feats['stat_kurt_rx2'] = float(kurtosis(mat_rx2.flatten()))
        feats['stat_ptp_rx2'] = float(np.ptp(mat_rx2))

        # Spatial cross-correlation between RX1 and RX2
        r_rx1_flat = mat_rx1.flatten() - np.mean(mat_rx1)
        r_rx2_flat = mat_rx2.flatten() - np.mean(mat_rx2)
        norm_prod = (np.linalg.norm(r_rx1_flat) * np.linalg.norm(r_rx2_flat)) + 1e-9
        feats['spatial_cross_corr'] = float(np.dot(r_rx1_flat, r_rx2_flat) / norm_prod)
    else:
        feats['spatial_cross_corr'] = 0.0

    # Spectral features via Welch PSD on energy signal
    try:
        freqs, psd = welch(total_energy, fs=fs, nperseg=min(len(total_energy), 32))
        # Band powers: 0.5-2Hz (walk), 2-5Hz (transitions), 5-10Hz (fall impact)
        b1 = np.sum(psd[(freqs >= 0.5) & (freqs < 2.0)])
        b2 = np.sum(psd[(freqs >= 2.0) & (freqs < 5.0)])
        b3 = np.sum(psd[(freqs >= 5.0) & (freqs <= 10.0)])
        total_psd = np.sum(psd) + 1e-9

        feats['spec_band_05_2Hz'] = float(b1 / total_psd)
        feats['spec_band_2_5Hz'] = float(b2 / total_psd)
        feats['spec_band_5_10Hz'] = float(b3 / total_psd)
        feats['spec_dominant_freq'] = float(freqs[np.argmax(psd)])
        feats['spec_entropy'] = float(entropy(psd / total_psd))
    except Exception:
        feats['spec_band_05_2Hz'] = 0.0
        feats['spec_band_2_5Hz'] = 0.0
        feats['spec_band_5_10Hz'] = 0.0
        feats['spec_dominant_freq'] = 0.0
        feats['spec_entropy'] = 0.0

    return feats


def extended_window_features(window_df, rx_cols_by_rx, noise_by_rx, fs=100.0):
    """
    Scale-free posture & motion features per receiver (requires preprocessing with
    --dc-mode median so the window mean still carries the posture level).
      level      : mean |window mean| over subcarriers / noise   (posture offset)
      level_chg  : mean |mean(last 25%) - mean(first 25%)|/noise (posture transition)
      motion     : mean within-window std over subcarriers / noise
      motion_pk  : peak 0.25 s short-time std / noise            (burst sharpness)
    """
    feats = {}
    for rx, cols in rx_cols_by_rx.items():
        if not cols:
            continue
        x = window_df[cols].values
        n = noise_by_rx.get(rx, 1.0) + 1e-9
        q = max(len(x) // 4, 1)
        k = max(int(0.25 * fs), 2)
        st = [x[i:i + k].std(axis=0).mean() for i in range(0, max(len(x) - k + 1, 1), max(k // 2, 1))]
        feats[f"ext_level_{rx}"] = float(np.abs(x.mean(axis=0)).mean() / n)
        feats[f"ext_level_chg_{rx}"] = float(np.abs(x[-q:].mean(axis=0) - x[:q].mean(axis=0)).mean() / n)
        feats[f"ext_motion_{rx}"] = float(x.std(axis=0).mean() / n)
        feats[f"ext_motion_pk_{rx}"] = float(max(st) / n)
    return feats


def trial_noise_scale(df, cols):
    """Robust per-trial noise level: median over subcarriers of the MAD of the signal."""
    if not cols:
        return 1.0
    x = df[cols].values
    mad = np.median(np.abs(x - np.median(x, axis=0)), axis=0)
    return float(np.median(mad) + 1e-9)


def process_trial_windowing(csv_path, activity_label, subject="Devansh",
                            window_len=50, stride=25, fs=100.0,
                            dormant_thresh=1.5, fall_fast_thresh=15.0, rx1_only=False,
                            extended=False):
    """
    Segments a processed trial into sliding windows and extracts feature vectors.
    rx1_only=True ignores RX2 channels (for the single-receiver live model).
    extended=True adds scale-free per-RX posture/motion features (ext_*).
    """
    df = pd.read_csv(csv_path)
    rx1_cols = [c for c in df.columns if c.startswith('rx1_amp_')]
    rx2_cols = [] if rx1_only else [c for c in df.columns if c.startswith('rx2_amp_')]
    if rx1_only == "rx2":   # single-receiver run on RX2 (features keep the rx1_ slot names)
        rx1_cols, rx2_cols = [c for c in df.columns if c.startswith('rx2_amp_')], []
    if not rx1_cols:
        rx1_cols = [c for c in df.columns if 'amp_' in c]
    rx_cols_by_rx = {"rx1": rx1_cols, "rx2": rx2_cols}
    noise_by_rx = {rx: trial_noise_scale(df, cols) for rx, cols in rx_cols_by_rx.items()} if extended else {}

    n_samples = len(df)
    windows_data = []

    for start_idx in range(0, n_samples - window_len + 1, stride):
        end_idx = start_idx + window_len
        sub_df = df.iloc[start_idx:end_idx]

        t_start = sub_df['time_rel_s'].iloc[0] if 'time_rel_s' in sub_df else start_idx / fs
        t_end = sub_df['time_rel_s'].iloc[-1] if 'time_rel_s' in sub_df else end_idx / fs

        feats = extract_window_features(
            sub_df, rx1_cols, rx2_cols, fs=fs,
            dormant_thresh=dormant_thresh, fall_fast_thresh=fall_fast_thresh
        )
        if extended:
            feats.update(extended_window_features(sub_df, rx_cols_by_rx, noise_by_rx, fs=fs))

        feats['window_start_s'] = round(t_start, 3)
        feats['window_end_s'] = round(t_end, 3)
        feats['activity'] = activity_label
        feats['subject'] = subject
        feats['source_file'] = os.path.basename(csv_path)
        windows_data.append(feats)

    return windows_data


def process_dataset_feature_extraction(in_dir="data/processed", out_dir="data/features",
                                       window_len=50, stride=25, fs=100.0,
                                       dormant_thresh=1.5, fall_fast_thresh=15.0, rx1_only=False,
                                       extended=False):
    os.makedirs(out_dir, exist_ok=True)
    all_files = glob.glob(os.path.join(in_dir, "**", "*.csv"), recursive=True)
    all_files = [f for f in all_files if "manifest" not in os.path.basename(f)]

    print(f"\n=======================================================")
    print(f"  SAFESENSE-5G MULTI-TIER FEATURE EXTRACTION")
    print(f"=======================================================")
    print(f"  Input Files    : {len(all_files)}")
    print(f"  Window Length  : {window_len} samples ({window_len/fs:.2f}s)")
    print(f"  Window Stride  : {stride} samples ({stride/fs:.2f}s, 50% overlap)")
    print(f"  Dormant Thresh : {dormant_thresh} | Fall Fast-Path Thresh: {fall_fast_thresh}\n")

    all_windows = []
    for f in sorted(all_files):
        # Extract activity and subject from file path or name
        parts = os.path.normpath(f).split(os.sep)
        act = parts[-2] if len(parts) >= 2 else "unknown"
        base = os.path.basename(f)
        subj = base.split('_')[0] if '_' in base else "Devansh"

        trial_wins = process_trial_windowing(
            f, activity_label=act, subject=subj,
            window_len=window_len, stride=stride, fs=fs,
            dormant_thresh=dormant_thresh, fall_fast_thresh=fall_fast_thresh,
            rx1_only=rx1_only, extended=extended
        )
        all_windows.extend(trial_wins)
        print(f"  [EXTRACTED] {base} ({act.upper()}) -> {len(trial_wins)} windows")

    if all_windows:
        df_feats = pd.DataFrame(all_windows)
        csv_out = os.path.join(out_dir, "features_all_windows.csv")
        df_feats.to_csv(csv_out, index=False)

        # Build clean X matrix and y vector for ML
        meta_cols = ['window_start_s', 'window_end_s', 'activity', 'subject', 'source_file', 'routing_tier']
        feature_cols = [c for c in df_feats.columns if c not in meta_cols]

        X = np.array(df_feats[feature_cols].values, dtype=np.float32)
        y = np.array(df_feats['activity'].values, dtype='<U32')

        np.save(os.path.join(out_dir, "X.npy"), X)
        np.save(os.path.join(out_dir, "y.npy"), y)
        with open(os.path.join(out_dir, "feature_names.txt"), "w") as f:
            f.write("\n".join(feature_cols))

        print(f"\n  Summary:")
        print(f"  Total Windows   : {len(df_feats)}")
        print(f"  Feature Dim (D) : {len(feature_cols)} features per window")
        print(f"  Routing Stats   :")
        tier_counts = df_feats['routing_tier'].value_counts().to_dict()
        print(f"    - Tier 1 (Dormant Skip) : {tier_counts.get(1, 0)} windows")
        print(f"    - Tier 2 (Fall FastPath): {tier_counts.get(2, 0)} windows")
        print(f"    - Tier 3 (MEC Normal Q) : {tier_counts.get(3, 0)} windows")
        print(f"  Saved to: {csv_out}")
        print(f"  Numpy arrays saved to: X.npy and y.npy")
    else:
        print("  No files available for feature extraction.")

    print(f"=======================================================\n")
    return all_windows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-tier feature extraction for WiFeye")
    parser.add_argument("--in-dir", default="data/processed", help="Input directory of denoised trials")
    parser.add_argument("--out-dir", default="data/features", help="Output directory for feature matrices")
    parser.add_argument("--win-len", type=int, default=50, help="Window length in samples (default 50 = 0.5s)")
    parser.add_argument("--stride", type=int, default=25, help="Window stride in samples (default 25 = 50% overlap)")
    parser.add_argument("--fs", type=float, default=100.0, help="Sampling frequency (Hz)")
    parser.add_argument("--dormant-th", type=float, default=1.5, help="Variance threshold for dormant tier")
    parser.add_argument("--fall-th", type=float, default=15.0, help="dE/dt threshold for fall fast-path")
    parser.add_argument("--rx1-only", action="store_true", help="Ignore RX2 channels (single-RX live model)")
    parser.add_argument("--rx2-only", action="store_true",
                        help="Use only RX2 channels (stored in the rx1_ feature slots) for 1-vs-2 RX comparisons")
    parser.add_argument("--extended", action="store_true",
                        help="Add scale-free per-RX posture/motion features (use with --dc-mode median data)")
    args = parser.parse_args()

    process_dataset_feature_extraction(
        in_dir=args.in_dir, out_dir=args.out_dir,
        window_len=args.win_len, stride=args.stride, fs=args.fs,
        dormant_thresh=args.dormant_th, fall_fast_thresh=args.fall_th,
        rx1_only=("rx2" if args.rx2_only else args.rx1_only), extended=args.extended
    )
