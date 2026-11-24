"""
WiFeye: In-fold calibration and measurement of the Tier 1 / Tier 2 gates
=============================================================================
The original thresholds (dormant 1.5, fall fast-path 15.0) were placeholders from the
synthetic test suite (AGENT_BRIDGE.md A11); on real data 99.96% of windows hit the
fast-path. This script fits both gates inside each LOSO fold on TRAINING trials only
and measures them on the held-out subject:

  Tier 1 (dormant skip): window skipped if motion metric < T1.
      T1 = the q1-quantile (default 2.5%) of the per-trial MAX motion of training fall
      trials, so >= ~97.5% of falls keep at least one window above the gate.
  Tier 2 (fall fast-path): window escalated if burst metric > T2.
      T2 = the (1 - q2)-quantile (default 95%) of the burst metric over training ADL
      event windows (walk/sit/stand/hand), so <= ~5% of ADL event windows escalate.
  Tier 3: everything else goes to the MEC/edge classifier.

Metrics (scale-free ext_* features by default; --metric raw uses the original
tier_energy_var / tier_dE_dt):
  * fraction of all windows exiting at each tier, per class
  * falls lost at Tier 1 (fall trials with every window dormant)
  * fall trials reaching the fast-path at least once; ADL event windows on the fast-path
Note: a fallen, motionless person looks dormant, so deployments need a post-fall latch
(stay alert after any Tier 2 trigger) rather than returning to Tier 1.

Usage: python calibrate_tiers.py --features-dir data/features_v2 --out data/models_exp/tiers_v2
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import train_classifier as tc

ADL = ["walk", "sit", "stand", "hand"]


def metrics_for(df, kind):
    if kind == "raw":
        return df["tier_energy_var"].values, df["tier_dE_dt"].values
    mot = df[[c for c in df.columns if c.startswith("ext_motion_") and not c.startswith("ext_motion_pk_")]].sum(axis=1)
    pk = df[[c for c in df.columns if c.startswith("ext_motion_pk_")]].sum(axis=1)
    return mot.values, pk.values


def main():
    ap = argparse.ArgumentParser(description="Calibrate Tier 1/2 gates in-fold (LOSO)")
    ap.add_argument("--features-dir", default="data/features_v2")
    ap.add_argument("--out", default="data/models_exp/tiers_v2")
    ap.add_argument("--metric", choices=["ext", "raw"], default="ext")
    ap.add_argument("--q1", type=float, default=0.025)
    ap.add_argument("--q2", type=float, default=0.05)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df, _, _ = tc.load_features(os.path.join(args.features_dir, "features_all_windows.csv"))
    df = tc.apply_event_labels(df, 1.0)
    motion, burst = metrics_for(df, args.metric)
    df["m"], df["b"] = motion, burst
    folds = tc.loso_folds(df)

    rows, thr = [], []
    for fold, test_ids in folds:
        te = df["trial_id"].isin(test_ids).values
        tr = df[~te]
        fall_max = tr[tr.activity == "fall"].groupby("trial_id")["m"].max()
        T1 = float(np.quantile(fall_max, args.q1))
        adl_ev = tr[tr.activity.isin(ADL) & tr.is_action]
        T2 = float(np.quantile(adl_ev["b"], 1 - args.q2))
        thr.append({"fold": fold, "T1": T1, "T2": T2})
        t = df[te].copy()
        t["tier"] = np.where(t["m"] < T1, 1, np.where(t["b"] > T2, 2, 3))
        rows.append(t[["trial_id", "activity", "subject", "is_action", "tier"]].assign(fold=fold))
    R = pd.concat(rows)

    print("\n=======================================================")
    print(f"  SAFESENSE-5G TIER GATE CALIBRATION (LOSO, metric={args.metric})")
    print("=======================================================")
    for t in thr:
        print(f"  {t['fold']:<11} T1={t['T1']:.3f}  T2={t['T2']:.3f}")
    share = R.groupby("activity")["tier"].value_counts(normalize=True).unstack().fillna(0).round(3)
    share.columns = [f"tier{c}" for c in share.columns]
    print("\n  Share of ALL windows per tier, by class (held-out subjects):")
    print(share.to_string())
    overall = R["tier"].value_counts(normalize=True).sort_index().round(3).to_dict()
    print(f"\n  Overall: {overall}   (original placeholder thresholds: 99.96% to tier 2)")

    by_trial = R.groupby(["trial_id", "activity"])["tier"].agg(lambda s: set(s)).reset_index()
    falls = by_trial[by_trial.activity == "fall"]
    lost = int(sum(s == {1} for s in falls["tier"]))
    reach2 = int(sum(2 in s for s in falls["tier"]))
    adl_ev = R[R.activity.isin(ADL) & R.is_action]
    adl_fp = float(np.mean(adl_ev["tier"] == 2))
    adl_trials = by_trial[by_trial.activity.isin(ADL)]
    adl_trial_fp = float(np.mean([2 in s for s in adl_trials["tier"]]))
    empty = R[R.activity == "empty_room"]
    print(f"\n  Falls lost at Tier 1 (all windows dormant): {lost}/{len(falls)}")
    print(f"  Fall trials reaching the fast-path: {reach2}/{len(falls)} ({reach2/len(falls):.1%})")
    print(f"  ADL event windows on the fast-path: {adl_fp:.1%} | ADL trials with any fast-path window: {adl_trial_fp:.1%}")
    print(f"  Empty-room windows skipped at Tier 1: {np.mean(empty['tier'] == 1):.1%}")

    summary = {"metric": args.metric, "q1": args.q1, "q2": args.q2, "thresholds": thr,
               "share_by_class": share.to_dict(), "overall": {int(k): v for k, v in overall.items()},
               "falls_lost_tier1": lost, "n_falls": int(len(falls)), "fall_trials_reach_fastpath": reach2,
               "adl_event_windows_fastpath": adl_fp, "adl_trials_any_fastpath": adl_trial_fp,
               "empty_windows_skipped": float(np.mean(empty["tier"] == 1))}
    with open(os.path.join(args.out, f"tier_calibration_{args.metric}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved to {args.out}/")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
