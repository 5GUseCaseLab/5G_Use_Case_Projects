"""
WiFeye: 5-Stage CSI Signal Preprocessing & Denoising Engine
================================================================
Stages:
  1. Hampel Filter: Outlier and glitch suppression using median absolute deviation (MAD).
  2. Subcarrier Selection: Removes DC (subcarrier 0) and low-SNR guard band subcarriers.
  3. Butterworth Filter: 4th-order zero-phase low-pass (cutoff 10 Hz) isolates human motion.
  4. Dynamic Component Extraction: Baseline/DC subtraction isolates moving body reflections.
  5. Phase Sanitization: Linear regression unwrapping cancels CFO and packet detection delay.
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt


def hampel_filter(data_1d, window_size=7, n_sigmas=3.0):
    """
    Applies Hampel filter to 1D time series to replace impulsive spikes with rolling median.
    """
    return hampel_filter_2d(np.asarray(data_1d, dtype=float)[:, None],
                            window_size=window_size, n_sigmas=n_sigmas)[:, 0]


def hampel_filter_2d(mat, window_size=7, n_sigmas=3.0):
    """
    Vectorized Hampel filter applied independently to every column of a (N, C) matrix.
    Standard (non-recursive) form: medians/MADs use the original samples. Edge samples
    (first/last window_size//2) are left unchanged.
    """
    x = np.array(mat, dtype=float, copy=True)
    n = x.shape[0]
    k = window_size // 2
    if n < window_size:
        return x
    win = np.lib.stride_tricks.sliding_window_view(x, window_size, axis=0)  # (n-2k, C, W)
    med = np.median(win, axis=-1)
    mad = np.median(np.abs(win - med[..., None]), axis=-1)
    center = x[k:n - k]
    outlier = np.abs(center - med) > n_sigmas * 1.4826 * mad
    center[outlier] = med[outlier]
    return x


def butter_lowpass_filter(data_1d, cutoff=10.0, fs=100.0, order=4):
    """
    Applies zero-phase 4th-order Butterworth low-pass filter.
    """
    nyq = 0.5 * fs
    normal_cutoff = min(cutoff / nyq, 0.99)
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    # Use filtfilt for zero phase distortion
    if len(data_1d) > 3 * order:
        return filtfilt(b, a, data_1d)
    return data_1d


def remove_static_dc_component(data_1d, window_size=50):
    """
    Subtracts moving average to remove static reflection from static walls and furniture,
    leaving only the dynamic multipath reflections from human body motion.
    """
    # Moving average baseline
    k = min(window_size, len(data_1d) // 2 or 1)
    kernel = np.ones(k) / k
    # Pad edges to keep same length
    pad = np.pad(data_1d, (k // 2, k - 1 - k // 2), mode='edge')
    baseline = np.convolve(pad, kernel, mode='valid')
    return data_1d - baseline


def sanitize_phase(raw_phases_across_subcarriers):
    """
    Performs linear transformation to sanitize phase across subcarriers:
    phase_sanitized(k) = phase_unwrapped(k) - a * k - b
    where a and b are linear fit parameters to eliminate CFO and PDD.
    """
    unwrapped = np.unwrap(raw_phases_across_subcarriers)
    k = np.arange(len(unwrapped))
    # Linear fit
    p = np.polyfit(k, unwrapped, deg=1)
    linear_trend = np.polyval(p, k)
    sanitized = unwrapped - linear_trend
    return sanitized


def preprocess_csi_dataframe(df, fs=100.0, cutoff_hz=10.0, dc_mode="moving"):
    """
    Processes all subcarrier amplitude and phase channels in an aligned DataFrame.
    dc_mode:
      "moving" (original): subtract a 0.5 s moving average. Acts as a ~2 Hz high-pass, so
               slow posture changes (sit/stand/fall end-state) are removed.
      "median": subtract the per-subcarrier median of the whole stream. Removes the static
               room/furniture level but keeps posture shifts and slow motion.
    """
    proc_df = df.copy()

    # Identify amplitude columns (rx1_amp_X, rx2_amp_X, etc.)
    amp_cols = [c for c in df.columns if 'amp_' in c]
    phase_cols = [c for c in df.columns if 'phase_' in c]

    # Clean Amplitudes
    if amp_cols:
        # 1. Hampel outlier removal (vectorized over all subcarriers)
        step1 = hampel_filter_2d(df[amp_cols].values, window_size=7, n_sigmas=3.0)
        for idx, col in enumerate(amp_cols):
            # 2. Butterworth lowpass filter
            step2 = butter_lowpass_filter(step1[:, idx], cutoff=cutoff_hz, fs=fs, order=4)
            # 3. Static component removal
            if dc_mode == "median":
                proc_df[col] = step2 - np.median(step2)
            else:
                proc_df[col] = remove_static_dc_component(step2, window_size=int(fs * 0.5))

    # Clean Phases across subcarriers for each time slice, per receiver
    # (RX1 and RX2 have independent CFO/PDD, so each gets its own linear fit).
    if phase_cols:
        groups = {}
        for c in phase_cols:
            groups.setdefault(c.split('phase_')[0], []).append(c)
        for cols in groups.values():
            unwrapped = np.unwrap(df[cols].values, axis=1)  # (N, K)
            k = np.arange(unwrapped.shape[1], dtype=float)
            A = np.vstack([k, np.ones_like(k)]).T            # (K, 2)
            coef, *_ = np.linalg.lstsq(A, unwrapped.T, rcond=None)  # (2, N)
            sanitized = unwrapped - (A @ coef).T
            for idx, col in enumerate(cols):
                proc_df[col] = sanitized[:, idx]

    return proc_df


def process_dataset_preprocessing(in_dir="data/aligned", out_dir="data/processed", fs=100.0, cutoff=10.0,
                                  dc_mode="moving"):
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, "manifest_processed.csv")

    csv_files = glob.glob(os.path.join(in_dir, "**", "*.csv"), recursive=True)
    csv_files = [f for f in csv_files if "manifest" not in os.path.basename(f)]

    print(f"\n=======================================================")
    print(f"  SAFESENSE-5G CSI PREPROCESSING & DENOISING")
    print(f"=======================================================")
    print(f"  Found {len(csv_files)} aligned files to denoise.")
    print(f"  Cutoff Frequency: {cutoff} Hz | Sampling Rate: {fs} Hz\n")

    manifest_rows = []
    for f in sorted(csv_files):
        rel_subpath = os.path.relpath(f, in_dir)
        out_fpath = os.path.join(out_dir, rel_subpath)
        os.makedirs(os.path.dirname(out_fpath), exist_ok=True)

        df = pd.read_csv(f)
        proc_df = preprocess_csi_dataframe(df, fs=fs, cutoff_hz=cutoff, dc_mode=dc_mode)
        proc_df.to_csv(out_fpath, index=False)

        base = os.path.basename(f)
        manifest_rows.append({
            'source_file': f,
            'processed_file': out_fpath,
            'samples': len(proc_df),
            'channels': len([c for c in proc_df.columns if 'amp_' in c]),
            'status': 'DENOISED'
        })
        print(f"  [DENOISED] {base} -> Saved to {out_fpath}")

    if manifest_rows:
        pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
        print(f"\n  Preprocessing manifest saved: {manifest_path}")

    print(f"=======================================================\n")
    return manifest_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess and denoise CSI signals")
    parser.add_argument("--in-dir", default="data/aligned", help="Input directory of aligned trials")
    parser.add_argument("--out-dir", default="data/processed", help="Output directory for cleaned trials")
    parser.add_argument("--fs", type=float, default=100.0, help="Sampling frequency (Hz)")
    parser.add_argument("--cutoff", type=float, default=10.0, help="Butterworth low-pass cutoff (Hz)")
    parser.add_argument("--dc-mode", choices=["moving", "median"], default="moving",
                        help="static removal: 'moving' 0.5 s moving average (original) or 'median' (keeps posture)")
    args = parser.parse_args()

    process_dataset_preprocessing(in_dir=args.in_dir, out_dir=args.out_dir, fs=args.fs, cutoff=args.cutoff,
                                  dc_mode=args.dc_mode)

