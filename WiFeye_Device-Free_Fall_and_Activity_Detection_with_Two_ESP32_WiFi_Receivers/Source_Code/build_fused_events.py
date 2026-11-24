"""
WiFeye: event features with the fused, reliability-weighted localiser
==========================================================================
1. For every processed v3 trial, recover each receiver's real packet times (raw files, router->ESP32
   HT frames) on the trial's time base, so interpolated packet gaps can be masked.
2. Calibrate the per-receiver noise reference on the empty-room trials: the 95th percentile of the
   log-peak a receiver reaches with NO person present.
3. Localise every trial (onset / peak / offset) and compute segment-based event features.
Output: data/event_features_v3_fused.csv  (+ data/models/localizer_ref.json)

Usage: python build_fused_events.py
"""

import os
import glob
import json
import numpy as np
import pandas as pd

from event_features import event_features_fused, trial_matrices
from event_localizer import localize

OWN = {"rx1": ("RX1", "C0:CD:D6:85:65:EC"), "rx2": ("RX2", "1C:C3:AB:B4:A2:48")}


def packet_times(tid, act, t0, raw_dir="data/raw"):
    out = {}
    for rx, (role, mac) in OWN.items():
        f = os.path.join(raw_dir, act, f"{tid}_{role}.csv")
        if os.path.exists(f):
            d = pd.read_csv(f, usecols=["epoch_timestamp", "mac", "sig_mode"])
            d = d[(d["mac"].str.upper() == mac) & (d["sig_mode"] == 1)]
            out[rx] = d["epoch_timestamp"].values - t0
    return out


def load(f):
    df = pd.read_csv(f)
    tid = os.path.basename(f).replace("_aligned.csv", "")
    act = os.path.basename(os.path.dirname(f))
    return tid, act, df["time_rel_s"].values, trial_matrices(df), packet_times(tid, act, df["epoch_timestamp"].iloc[0])


def main():
    files = sorted(glob.glob("data/processed_v3/*/*_aligned.csv"))
    # pass 1: noise reference from empty-room trials (no person present)
    peaks = {"rx1": [], "rx2": []}
    for f in [f for f in files if os.sep + "empty_room" + os.sep in f or "/empty_room/" in f]:
        _, _, t, mats, pk = load(f)
        loc = localize(t, mats, pk, ref={})
        for rx in peaks:
            peaks[rx].append(loc[f"peak_{rx}"])
    ref = {rx: float(np.percentile(v, 95)) for rx, v in peaks.items()}
    os.makedirs("data/models", exist_ok=True)
    with open("data/models/localizer_ref.json", "w") as fh:
        json.dump({"noise_ref_logpeak_p95": ref, "n_empty_trials": len(peaks["rx1"])}, fh, indent=2)
    print(f"  noise reference (95th pct log-peak, empty room): {ref}")
    # pass 2: all trials
    rows = []
    for i, f in enumerate(files):
        tid, act, t, mats, pk = load(f)
        r = {"trial_id": tid, "activity": act, "subject": tid.split("_")[0]}
        r.update(event_features_fused(t, mats, pk, ref))
        rows.append(r)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(files)}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv("data/event_features_v3_fused.csv", index=False)
    print(f"  wrote {len(out)} trials x {out.shape[1] - 3} columns to data/event_features_v3_fused.csv")
    print("  receiver weights: median w_rx1 %.2f | trials where one receiver carries >90%% of the weight: %.1f%%"
          % (out.loc_w_rx1.median(), 100 * np.mean((out.loc_w_rx1 > 0.9) | (out.loc_w_rx2 > 0.9))))
    print("  share of curve points masked for packet gaps: rx1 %.1f%%, rx2 %.1f%%"
          % (100 * (1 - out.loc_rel_rx1.mean()), 100 * (1 - out.loc_rel_rx2.mean())))


if __name__ == "__main__":
    main()
