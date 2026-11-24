"""
WiFeye: Dual-RX CSI Merging & Time-Grid Alignment Engine
=============================================================
Fuses asynchronous CSI streams from Laptop A (RX1) and Laptop B (Worker / RX2).
Interpolates packet arrivals onto an exact uniform 100 Hz time grid (t = 0.00, 0.01, ..., T).
Outputs synchronized 128-channel paired matrices (64 from RX1 + 64 from RX2).
Also supports single-RX standalone trial normalization.
"""

import os
import glob
import re
import argparse
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


KNOWN_ACTIVITIES = ("empty_room", "fall", "walk", "sit", "stand", "hand")


def parse_filename_meta(filename):
    """
    Extracts (subject, activity, trial_num, rx_role) from filenames like:
    Devansh_fall_t01_RX1.csv, sub01_walk_t03_RX2.csv, 6_empty_room_t01_RX1.csv

    Activity names may contain underscores (empty_room), so a greedy split is
    ambiguous. Resolution order:
      1. Parent directory name, if it is a suffix of the stem before _t<N>_RX<k>
         (data/raw/<activity>/<subject>_<activity>_t<N>_RX<k>.csv layout).
      2. Longest known activity token matching the end of that stem.
      3. Legacy fallback: last underscore splits subject/activity.
    """
    base = os.path.splitext(os.path.basename(filename))[0]
    m = re.match(r"^(.+)_[tT]?(\d+)_(RX\d)$", base)
    if not m:
        return None, None, None, None
    stem, t_num, rx_role = m.groups()

    parent = os.path.basename(os.path.dirname(os.path.abspath(filename)))
    candidates = [parent] + sorted(KNOWN_ACTIVITIES, key=len, reverse=True)
    for act in candidates:
        if act and stem.endswith("_" + act) and len(stem) > len(act) + 1:
            return stem[: -(len(act) + 1)], act, int(t_num), rx_role

    if "_" in stem:
        subject, activity = stem.rsplit("_", 1)
        return subject, activity, int(t_num), rx_role
    return None, None, None, None


def parse_index_list(spec):
    """'0,27-37' -> {0, 27, ..., 37}"""
    out = set()
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


RX_OWN_MACS = {"RX1": "C0:CD:D6:85:65:EC", "RX2": "1C:C3:AB:B4:A2:48"}  # AGENT_BRIDGE.md A7


def clean_rx_frame(df, source_filter="none", drop_subcarriers=(), own_mac=None):
    """
    Pre-alignment cleaning of one receiver's raw stream.
      source_filter="dominant": keep only packets from the most frequent (mac, sig_mode)
        source. Raw ESP32 files interleave CSI from several transmitters and from HT and
        legacy frames; switching sources causes amplitude jumps ~4x larger than true
        packet-to-packet variation.
      drop_subcarriers: subcarrier indices whose amp_/phase_ columns are removed
        (ESP32 LLTF: index 0 is not channel data, 27-37 are null/guard).
    Returns (cleaned_df, info dict).
    """
    info = {"src_frac": 1.0, "src": ""}
    if source_filter in ("dominant", "own") and "mac" in df.columns and "sig_mode" in df.columns:
        key = df["mac"].astype(str) + "|" + df["sig_mode"].astype(str)
        if source_filter == "own" and own_mac:
            # frames whose mac field is the receiver's own MAC, HT only: router->this ESP32
            top = f"{own_mac}|1"
        else:
            top = key.value_counts().index[0]
        keep = (key == top).values
        info = {"src_frac": float(keep.mean()), "src": top}
        df = df[keep]
    if drop_subcarriers:
        cols = [c for c in df.columns
                if (c.startswith("amp_") or c.startswith("phase_"))
                and int(c.split("_")[1]) in drop_subcarriers]
        df = df.drop(columns=cols)
    return df.reset_index(drop=True), info


def stream_quality(df, t0, lo=2.0, hi=6.0):
    """Packet rate over the trial and the largest gap (ms) inside [lo, hi] s after t0."""
    t = np.sort(df["epoch_timestamp"].values) - t0
    rate = len(t) / max(t[-1] - t[0], 1e-6) if len(t) > 1 else 0.0
    seg = t[(t >= lo) & (t <= hi)]
    edges = np.concatenate([[lo], seg, [hi]])
    return float(rate), float(np.max(np.diff(edges)) * 1000.0)


def align_trial_pair(rx1_df, rx2_df, sampling_rate_hz=100.0, target_duration=None):
    """
    Interpolates RX1 and RX2 data onto a uniform time grid.
    Returns aligned DataFrame with 128 subcarrier channels.
    """
    t_min = max(rx1_df['epoch_timestamp'].min(), rx2_df['epoch_timestamp'].min())
    t_max = min(rx1_df['epoch_timestamp'].max(), rx2_df['epoch_timestamp'].max())

    if target_duration:
        t_max = min(t_max, t_min + target_duration)

    duration = t_max - t_min
    if duration <= 0.1:
        raise ValueError(f"Insufficient overlapping duration between RX1 and RX2: {duration:.3f}s")

    num_samples = int(np.floor(duration * sampling_rate_hz))
    t_grid = np.linspace(t_min, t_min + (num_samples - 1) / sampling_rate_hz, num_samples)
    t_rel_grid = t_grid - t_min

    aligned_dict = {
        'time_rel_s': t_rel_grid,
        'epoch_timestamp': t_grid
    }

    def _interp_df(df, prefix):
        t_orig = df['epoch_timestamp'].values
        idx = np.argsort(t_orig)
        t_orig = t_orig[idx]
        t_orig, unique_idx = np.unique(t_orig, return_index=True)
        orig_indices = idx[unique_idx]

        f_rssi = interp1d(t_orig, df['rssi'].values[orig_indices], kind='linear', fill_value='extrapolate')
        aligned_dict[f'{prefix}_rssi'] = f_rssi(t_grid)

        amp_cols = [c for c in df.columns if c.startswith('amp_')]
        for col in amp_cols:
            f_amp = interp1d(t_orig, df[col].values[orig_indices], kind='linear', fill_value='extrapolate')
            aligned_dict[f'{prefix}_{col}'] = f_amp(t_grid)

        phase_cols = [c for c in df.columns if c.startswith('phase_')]
        for col in phase_cols:
            raw_phase = df[col].values[orig_indices]
            unwrapped = np.unwrap(raw_phase)
            f_phase = interp1d(t_orig, unwrapped, kind='linear', fill_value='extrapolate')
            interp_phase = (f_phase(t_grid) + np.pi) % (2 * np.pi) - np.pi
            aligned_dict[f'{prefix}_{col}'] = interp_phase

    _interp_df(rx1_df, 'rx1')
    _interp_df(rx2_df, 'rx2')

    return pd.DataFrame(aligned_dict), duration, num_samples


def align_single_rx(rx_df, sampling_rate_hz=100.0, target_duration=None):
    """
    Interpolates a single-RX trial onto a uniform 100 Hz grid.
    """
    t_min = rx_df['epoch_timestamp'].min()
    t_max = rx_df['epoch_timestamp'].max()

    if target_duration:
        t_max = min(t_max, t_min + target_duration)

    duration = t_max - t_min
    num_samples = int(np.floor(duration * sampling_rate_hz))
    if num_samples < 2:
        num_samples = len(rx_df)

    t_grid = np.linspace(t_min, t_max, num_samples)
    t_rel_grid = t_grid - t_min

    aligned_dict = {
        'time_rel_s': t_rel_grid,
        'epoch_timestamp': t_grid
    }

    t_orig = rx_df['epoch_timestamp'].values
    idx = np.argsort(t_orig)
    t_orig = t_orig[idx]
    t_orig, unique_idx = np.unique(t_orig, return_index=True)
    orig_indices = idx[unique_idx]

    f_rssi = interp1d(t_orig, rx_df['rssi'].values[orig_indices], kind='linear', fill_value='extrapolate')
    aligned_dict['rx1_rssi'] = f_rssi(t_grid)

    for col in [c for c in rx_df.columns if c.startswith('amp_')]:
        f_amp = interp1d(t_orig, rx_df[col].values[orig_indices], kind='linear', fill_value='extrapolate')
        aligned_dict[f'rx1_{col}'] = f_amp(t_grid)

    for col in [c for c in rx_df.columns if c.startswith('phase_')]:
        raw_phase = rx_df[col].values[orig_indices]
        unwrapped = np.unwrap(raw_phase)
        f_phase = interp1d(t_orig, unwrapped, kind='linear', fill_value='extrapolate')
        aligned_dict[f'rx1_{col}'] = (f_phase(t_grid) + np.pi) % (2 * np.pi) - np.pi

    return pd.DataFrame(aligned_dict), duration, num_samples


def process_dataset_merging(raw_dir="data/raw", out_dir="data/aligned", sampling_rate_hz=100.0,
                            source_filter="none", drop_subcarriers=()):
    os.makedirs(out_dir, exist_ok=True)
    aligned_manifest_path = os.path.join(out_dir, "manifest_aligned.csv")

    csv_files = glob.glob(os.path.join(raw_dir, "**", "*.csv"), recursive=True)
    pairs = {}
    for f in csv_files:
        if "manifest" in os.path.basename(f):
            continue
        subj, act, t_num, role = parse_filename_meta(f)
        if subj is None:
            continue
        key = (subj, act, t_num)
        if key not in pairs:
            pairs[key] = {}
        pairs[key][role] = f

    manifest_rows = []
    print(f"\n=======================================================")
    print(f"  SAFESENSE-5G DUAL-RX MERGING & ALIGNMENT")
    print(f"=======================================================")
    print(f"  Found {len(pairs)} trial sets to process.\n")

    for (subj, act, t_num), role_dict in sorted(pairs.items()):
        act_out_dir = os.path.join(out_dir, act)
        os.makedirs(act_out_dir, exist_ok=True)
        aligned_fname = f"{subj}_{act}_t{t_num:02d}_aligned.csv"
        aligned_fpath = os.path.join(act_out_dir, aligned_fname)

        if 'RX1' in role_dict and 'RX2' in role_dict:
            df_rx1, info1 = clean_rx_frame(pd.read_csv(role_dict['RX1']), source_filter, drop_subcarriers,
                                           RX_OWN_MACS["RX1"])
            df_rx2, info2 = clean_rx_frame(pd.read_csv(role_dict['RX2']), source_filter, drop_subcarriers,
                                           RX_OWN_MACS["RX2"])
            if len(df_rx1) < 10 or len(df_rx2) < 10:
                print(f"  [SKIP] {subj} {act} T{t_num:02d}: too few packets after source filter "
                      f"({len(df_rx1)}/{len(df_rx2)})")
                continue
            try:
                aligned_df, dur, n_samples = align_trial_pair(df_rx1, df_rx2, sampling_rate_hz)
                aligned_df.to_csv(aligned_fpath, index=False)
                n_ch = len([c for c in aligned_df.columns if '_amp_' in c])
                t0 = aligned_df['epoch_timestamp'].iloc[0]
                r1, g1 = stream_quality(df_rx1, t0)
                r2, g2 = stream_quality(df_rx2, t0)
                manifest_rows.append({
                    'trial_id': f"{subj}_{act}_t{t_num:02d}",
                    'subject': subj,
                    'activity': act,
                    'trial_num': t_num,
                    'mode': 'Dual-RX',
                    'duration_s': round(dur, 2),
                    'samples': n_samples,
                    'channels': n_ch,
                    'filename': aligned_fname,
                    'rx1_src': info1['src'], 'rx1_src_frac': round(info1['src_frac'], 3),
                    'rx2_src': info2['src'], 'rx2_src_frac': round(info2['src_frac'], 3),
                    'rx1_rate_hz': round(r1, 1), 'rx2_rate_hz': round(r2, 1),
                    'rx1_maxgap_2_6s_ms': round(g1, 1), 'rx2_maxgap_2_6s_ms': round(g2, 1),
                })
                print(f"  [ALIGNED DUAL-RX] {subj} | {act.upper()} T{t_num:02d} -> {n_samples} samples "
                      f"({dur:.2f}s, {n_ch} channels, src {info1['src_frac']:.2f}/{info2['src_frac']:.2f})")
            except Exception as e:
                print(f"  [ERROR] Failed to align dual pair {subj} {act} T{t_num}: {e}")

        elif 'RX1' in role_dict:
            df_rx1 = pd.read_csv(role_dict['RX1'])
            if len(df_rx1) < 10:
                print(f"  [SKIP] Skipping {subj} {act} T{t_num:02d}: Insufficient frames ({len(df_rx1)})")
                continue
            aligned_df, dur, n_samples = align_single_rx(df_rx1, sampling_rate_hz)
            aligned_df.to_csv(aligned_fpath, index=False)
            manifest_rows.append({
                'trial_id': f"{subj}_{act}_t{t_num:02d}",
                'subject': subj,
                'activity': act,
                'trial_num': t_num,
                'mode': 'Single-RX',
                'duration_s': round(dur, 2),
                'samples': n_samples,
                'channels': 64,
                'filename': aligned_fname
            })
            print(f"  [ALIGNED SINGLE-RX] {subj} | {act.upper()} T{t_num:02d} -> {n_samples} samples ({dur:.2f}s, 64 channels)")

    if manifest_rows:
        df_manifest = pd.DataFrame(manifest_rows)
        df_manifest.to_csv(aligned_manifest_path, index=False)
        print(f"\n  Manifest written to: {aligned_manifest_path}")
        print(f"  All trials successfully aligned onto uniform {sampling_rate_hz} Hz grid!")
    else:
        print("  No matching trials found in input directory.")

    print(f"=======================================================\n")
    return manifest_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge and align dual-RX CSI trials")
    parser.add_argument("--raw-dir", default="data/raw", help="Directory containing raw CSV trials")
    parser.add_argument("--out-dir", default="data/aligned", help="Directory to save aligned CSV trials")
    parser.add_argument("--fs", type=float, default=100.0, help="Target uniform sampling rate in Hz")
    parser.add_argument("--source-filter", choices=["none", "dominant", "own"], default="none",
                        help="'dominant': most frequent (mac, sig_mode) source per file; "
                             "'own': only HT frames whose mac is the receiver's own MAC (router->ESP32)")
    parser.add_argument("--drop-subcarriers", default="",
                        help="subcarrier indices to drop, e.g. '0,27-37' (ESP32 LLTF non-data/null)")
    args = parser.parse_args()

    process_dataset_merging(raw_dir=args.raw_dir, out_dir=args.out_dir, sampling_rate_hz=args.fs,
                            source_filter=args.source_filter,
                            drop_subcarriers=parse_index_list(args.drop_subcarriers))

