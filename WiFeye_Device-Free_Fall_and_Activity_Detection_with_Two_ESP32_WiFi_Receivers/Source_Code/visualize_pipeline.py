"""
WiFeye: Pipeline Visual Diagnostics & Spectrogram Inspector
=================================================================
Generates visual figures of:
  1. Raw vs. Filtered CSI Amplitude Waveforms
  2. Doppler Spectrogram (STFT showing motion velocity over time)
  3. Tier-1 & Tier-2 Edge Trigger Signals (Variance, Energy Derivative, Decision Path)
  4. Subcarrier Spatial Profiles
Saves high-resolution plots to data/visualizations/
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Headless rendering to file
import matplotlib.pyplot as plt
from scipy.signal import spectrogram


def plot_pipeline_diagnostics(raw_csv_path, proc_csv_path, out_png_path, fs=100.0):
    df_raw = pd.read_csv(raw_csv_path)
    df_proc = pd.read_csv(proc_csv_path)

    # Pick subcarrier 20, 30, 40 for display
    subcarriers = [20, 30, 40]
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)
    fig.suptitle(f"WiFeye CSI Pipeline Diagnostics: {os.path.basename(raw_csv_path)}", fontsize=14, fontweight='bold')

    # Panel 1: Raw vs Denoised Waveforms
    ax1 = axes[0]
    t_raw = df_raw['timestamp_rel'].values if 'timestamp_rel' in df_raw else np.arange(len(df_raw))/fs
    t_proc = df_proc['time_rel_s'].values if 'time_rel_s' in df_proc else np.arange(len(df_proc))/fs

    for sc in subcarriers:
        raw_col = f"amp_{sc}"
        proc_col = f"rx1_amp_{sc}" if f"rx1_amp_{sc}" in df_proc else (f"amp_{sc}" if f"amp_{sc}" in df_proc else None)
        if raw_col in df_raw:
            ax1.plot(t_raw, df_raw[raw_col], alpha=0.35, label=f"Raw Subcarrier {sc}")
        if proc_col and proc_col in df_proc:
            ax1.plot(t_proc, df_proc[proc_col], linewidth=1.8, label=f"Denoised Subcarrier {sc}")

    ax1.set_ylabel("Amplitude")
    ax1.set_title("1. Raw vs. Filtered CSI Waveforms (Butterworth 10 Hz + Static DC Removal)")
    ax1.legend(loc="upper right", ncol=3, fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Panel 2: Doppler Spectrogram (STFT)
    ax2 = axes[1]
    # Sum of processed amplitudes for Doppler analysis
    amp_cols = [c for c in df_proc.columns if 'amp_' in c]
    total_signal = df_proc[amp_cols].sum(axis=1).values
    
    nperseg = min(len(total_signal), 64)
    if nperseg >= 8:
        f_spec, t_spec, Sxx = spectrogram(total_signal, fs=fs, nperseg=nperseg, noverlap=nperseg//2)
        im = ax2.pcolormesh(t_spec, f_spec, 10 * np.log10(Sxx + 1e-9), shading='gouraud', cmap='inferno')
        ax2.set_ylabel("Doppler Freq (Hz)")
        ax2.set_ylim(0, min(25, fs/2))
        ax2.set_title("2. Doppler Spectrogram (Dynamic Motion Intensity & Velocity)")
        fig.colorbar(im, ax=ax2, label="Power (dB)")
    else:
        ax2.text(0.5, 0.5, "Insufficient samples for spectrogram", ha="center")
    ax2.grid(True, linestyle="--", alpha=0.4)

    # Panel 3: Tier 1 & Tier 2 Edge Trigger Analysis
    ax3 = axes[2]
    energy = np.sum(df_proc[amp_cols].values ** 2, axis=1)
    
    # Calculate rolling energy variance (0.5s window = 50 samples)
    win_samples = 50
    rolling_var = pd.Series(energy).rolling(win_samples, min_periods=10).var().fillna(0).values
    dE_dt = np.gradient(energy)

    ax3.plot(t_proc, rolling_var, color="crimson", linewidth=1.8, label="Energy Variance σ² (Tier 1/2 Check)")
    ax3.plot(t_proc, dE_dt, color="darkorange", linestyle=":", linewidth=1.5, label="Burst Acceleration dE/dt (Fall Fast-Path)")
    ax3.axhline(1.5, color="black", linestyle="--", alpha=0.7, label="Dormant Threshold (1.5)")
    ax3.set_xlabel("Relative Time (seconds)")
    ax3.set_ylabel("Metric Magnitude")
    ax3.set_title("3. Tier 1 & Tier 2 Edge Node Routing Signals (Arduino UNO Q Decision Logic)")
    ax3.legend(loc="upper right", fontsize=9)
    ax3.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png_path), exist_ok=True)
    plt.savefig(out_png_path, dpi=200)
    plt.close()
    print(f"  [SAVED PLOT] {out_png_path}")


def generate_all_visualizations(raw_dir="data/raw", proc_dir="data/processed", out_dir="data/visualizations"):
    os.makedirs(out_dir, exist_ok=True)
    raw_files = glob.glob(os.path.join(raw_dir, "**", "*.csv"), recursive=True)
    raw_files = [f for f in raw_files if "manifest" not in os.path.basename(f)]

    print(f"\n=======================================================")
    print(f"  SAFESENSE-5G VISUALIZATION & DIAGNOSTICS")
    print(f"=======================================================")

    count = 0
    for raw_f in raw_files:
        base = os.path.basename(raw_f)
        subj = base.split('_')[0]
        act = base.split('_')[1]
        
        # Look for matching processed file
        candidates = glob.glob(os.path.join(proc_dir, "**", f"*{subj}*{act}*.csv"), recursive=True)
        if not candidates:
            continue
        proc_f = candidates[0]
        out_png = os.path.join(out_dir, f"diag_{os.path.splitext(base)[0]}.png")
        plot_pipeline_diagnostics(raw_f, proc_f, out_png)
        count += 1

    print(f"\n  Generated {count} diagnostic reports in {out_dir}")
    print(f"=======================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate pipeline diagnostic reports")
    parser.add_argument("--raw-dir", default="data/raw", help="Raw CSV directory")
    parser.add_argument("--proc-dir", default="data/processed", help="Processed CSV directory")
    parser.add_argument("--out-dir", default="data/visualizations", help="Output directory for plots")
    args = parser.parse_args()

    generate_all_visualizations(raw_dir=args.raw_dir, proc_dir=args.proc_dir, out_dir=args.out_dir)

