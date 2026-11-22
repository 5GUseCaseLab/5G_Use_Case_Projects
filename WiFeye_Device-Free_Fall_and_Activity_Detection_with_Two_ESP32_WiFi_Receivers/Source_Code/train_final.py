"""
WiFeye: Train and package the final deployable models
==========================================================
Trains the final window-level classifiers on ALL trials of a pipeline version (default
v3) and writes one bundle for the live demo engine:

  data/models/safesense_bundle.pkl   joblib dict with:
      models["rx1"], models["rx2"]   single-receiver RandomForest pipelines
      models["dual"]                 feature-level (early) fusion pipeline, for reference
      fusion                         "late": mean of rx1 and rx2 class probabilities
                                     (falls back to the available receiver alone)
      feature_names[...]             exact feature order per model
      tiers[...]                     Tier 1/2 gate thresholds per receiver set
      config                         pipeline flags, window, event and fs parameters
  data/models/model_meta.json        human-readable copy of everything except the models
  data/models/feature_importance_<set>.csv

Why late fusion and RF (data/models_exp/rx_compare_v3, 5 seeds, LOSO): no fusion beat
the best single receiver, but late fusion keeps the AUC of RX1 while lowering ADL false
alarms and degrades gracefully if one ESP32 stream drops, which suits a live 2-ESP32
demo. Features from the original placeholder tier logic (flag_*) are excluded.

Usage: python train_final.py --version v3
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

import train_classifier as tc

ADL = ["walk", "sit", "stand", "hand"]
PIPELINE_FLAGS = {
    "v2": {"source_filter": "dominant", "drop_subcarriers": "0,27-37", "dc_mode": "median"},
    "v3": {"source_filter": "own", "drop_subcarriers": "0,27-37", "dc_mode": "median"},
}


def load_set(feat_dir, half_width):
    df, X, cols = tc.load_features(os.path.join(feat_dir, "features_all_windows.csv"))
    keep = [i for i, c in enumerate(cols) if not c.startswith("flag_")]
    cols = [cols[i] for i in keep]
    X = X[:, keep]
    df = tc.apply_event_labels(df, half_width)
    return df, X, cols


def gate_metrics(df):
    mot = df[[c for c in df.columns if c.startswith("ext_motion_") and not c.startswith("ext_motion_pk_")]].sum(axis=1)
    pk = df[[c for c in df.columns if c.startswith("ext_motion_pk_")]].sum(axis=1)
    return mot.values, pk.values


def calibrate_tiers(df, q1=0.025, q2=0.05):
    m, b = gate_metrics(df)
    d = df.assign(m=m, b=b)
    fall_max = d[d.activity == "fall"].groupby("trial_id")["m"].max()
    t1 = float(np.quantile(fall_max, q1))
    t2 = float(np.quantile(d[d.activity.isin(ADL) & d.is_action]["b"], 1 - q2))
    return {"T1_dormant_motion": t1, "T2_fastpath_burst": t2, "q1": q1, "q2": q2,
            "motion_metric": "sum of ext_motion_rx*", "burst_metric": "sum of ext_motion_pk_rx*"}


CL5 = ["fall", "walk", "posture", "hand", "empty_room"]
MERGE5 = {"sit": "posture", "stand": "posture"}


def build_adopted(out_dir, n_estimators=500):
    bundle = make_adopted_bundle(n_estimators=n_estimators, verbose=True)
    joblib.dump(bundle, os.path.join(out_dir, "safesense_live_bundle.pkl"), compress=3)
    meta = {k: bundle[k] for k in ("format", "head_features", "classes", "selected_domain_features",
                                   "localizer_ref", "fall_threshold", "tiers", "config")}
    with open(os.path.join(out_dir, "model_meta_adopted.json"), "w") as fh:
        json.dump(meta, fh, indent=2, default=float)
    print(f"   saved {os.path.join(out_dir, 'safesense_live_bundle.pkl')} (two-head, 5 classes)")


def make_adopted_bundle(exclude_subject=None, n_estimators=500, verbose=False):
    """Adopted system (validated in validate_adopted.py): fused localiser, two heads, 5 classes.
      fall head     = A2 features (fused-localiser event features + per-receiver window aggregates)
      activity head = A2 + CSI domain features chosen by the acceptance test + redundancy pruning
    One Random Forest per receiver per head; late fusion; P(fall) from the fall head.
    F11_phase_std_ratio is excluded from the candidates (it failed every acceptance test and the live
    engine does not stream phase)."""
    from validate_features import select
    from domain_features import foldfit_params
    seg = pd.read_csv("data/event_features_v3_fused.csv", dtype={"subject": str}).set_index("trial_id")
    import evaluate_event_model as em
    agg = em.window_aggregates("data/features_v3_rx1", "w1").join(em.window_aggregates("data/features_v3_rx2", "w2"))
    base = seg[["activity", "subject"] + [c for c in seg.columns if c.startswith("ev_rx")]].join(agg)
    dom = pd.read_csv("data/models_exp/eda_domain_v3/domain_features.csv", dtype={"subject": str}).set_index("trial_id")
    cand = [c for c in dom.columns if c.startswith("F") and not c.startswith("F11_phase")]
    T = base.join(dom[cand]).dropna(subset=[c for c in base.columns if c not in ("activity", "subject")])
    T = T.replace([np.inf, -np.inf], np.nan)
    fill = T[cand].median()
    T[cand] = T[cand].fillna(fill)
    if exclude_subject is not None:                        # honest replay: retrain without one person
        T = T[T.subject != str(exclude_subject)]
    base_cols = [c for c in base.columns if c not in ("activity", "subject") and T[c].nunique() > 1]
    sel = select(T, cand, base_cols)                       # F1/F4 values here are LOSO out-of-fold
    y5 = T.activity.map(lambda a: MERGE5.get(a, a)).values
    rx_of = lambda c: "rx1" if c.endswith("_rx1") or c.startswith(("ev_rx1_", "w1_")) else (
        "rx2" if c.endswith("_rx2") or c.startswith(("ev_rx2_", "w2_")) else "shared")
    heads, feats = {}, {}
    for head, cols in (("fall", base_cols), ("activity", base_cols + sel)):
        heads[head], feats[head] = {}, {}
        for rx in ("rx1", "rx2"):
            cs = [c for c in cols if rx_of(c) in (rx, "shared")]
            heads[head][rx] = RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=2,
                                                     class_weight="balanced", n_jobs=-1, random_state=42).fit(
                np.nan_to_num(T[cs].values.astype(float)), y5)
            feats[head][rx] = cs
        if verbose:
            print(f"   [{head} head] features per receiver: rx1 {len(feats[head]['rx1'])}, rx2 {len(feats[head]['rx2'])}")
    z = np.load("data/models_exp/eda_domain_v3/foldfit_material.npz", allow_pickle=True)
    ffp = foldfit_params(list(T.index), T.activity.to_dict(), z["deltas"].item(), z["profs"].item())
    tau = {"target_adl_far": 0.05, "tau": 0.30, "source": "default"}
    oof_path = "data/models_exp/adopted_v3/adopted_oof.csv"
    if os.path.exists(oof_path):
        oof = pd.read_csv(oof_path)
        adl_p = oof.loc[oof["true5"].isin(["walk", "posture", "hand"]), "p_fall"].values
        for tg in (0.03, 0.05, 0.10):
            tau[f"tau_far_{int(tg * 100)}"] = float(np.quantile(adl_p, 1 - tg))
        tau["tau"], tau["source"] = tau["tau_far_5"], oof_path
    with open("data/models/localizer_ref.json") as fh:
        ref = json.load(fh)["noise_ref_logpeak_p95"]
    # Tier-1/2 gates per receiver on the SCALE-FREE window motion (ext_motion = window std / the
    # buffer's own noise scale). An absolute empty-room floor does not work here: in the recorded data
    # it would pass only ~2.5 % of falls (absolute CSI motion barely separates a person from an empty room).
    tiers = {}
    for rx in ("rx1", "rx2"):
        wdf = load_set(f"data/features_v3_{rx}", 1.0)[0]
        if exclude_subject is not None:
            wdf = wdf[wdf.trial_id.str.split("_").str[0] != str(exclude_subject)]
        tiers[rx] = calibrate_tiers(wdf)
    bundle = {"format": "two_head_v1", "heads": heads, "head_features": feats, "classes": CL5,
              "tiers": tiers,
              "selected_domain_features": sel, "foldfit_params": ffp, "localizer_ref": ref,
              "fall_threshold": tau,
              "feature_fill": {c: float(fill[c]) for c in sel},   # training median for NaN / inf inputs
              "config": {"pipeline_version": "v3", "source_filter": "own", "drop_subcarriers": "0,27-37",
                         "dc_mode": "median", "fs_hz": 100.0, "lowpass_hz": 10.0, "window_len": 200, "stride": 25,
                         "localiser": "fused (reliability-weighted, gap-masked, protocol prior)",
                         "rx_own_macs": {"RX1": "C0:CD:D6:85:65:EC", "RX2": "1C:C3:AB:B4:A2:48"},
                         "classes": CL5, "merged": MERGE5}}
    if verbose:
        print(f"   selected domain features ({len(sel)}): {sel}")
        print(f"   fall threshold tau (ADL FAR 5%): {tau['tau']:.3f}  [{tau['source']}]")
    return bundle


def main():
    ap = argparse.ArgumentParser(description="Train final WiFeye models")
    ap.add_argument("--version", default="v3", choices=sorted(PIPELINE_FLAGS))
    ap.add_argument("--out", default="data/models")
    ap.add_argument("--n-estimators", type=int, default=300)
    ap.add_argument("--event-half-width", type=float, default=1.0)
    ap.add_argument("--adopted-only", action="store_true",
                    help="only (re)build the adopted two-head live bundle")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.adopted_only:
        build_adopted(args.out)
        return
    v = args.version
    sets = {"rx1": f"data/features_{v}_rx1", "rx2": f"data/features_{v}_rx2", "dual": f"data/features_{v}"}

    models, feature_names, tiers, counts = {}, {}, {}, {}
    print(f"\n  Training final models on pipeline {v}")
    for name, path in sets.items():
        df, X, cols = load_set(path, args.event_half_width)
        tr = df["is_action"].values
        pipe = make_pipeline(StandardScaler(), RandomForestClassifier(
            n_estimators=args.n_estimators, max_depth=15, class_weight="balanced", n_jobs=-1, random_state=42))
        pipe.fit(X[tr], df["activity"].values[tr])
        models[name], feature_names[name] = pipe, cols
        tiers[name] = calibrate_tiers(df)
        counts[name] = {"windows": int(len(df)), "training_windows": int(tr.sum()), "trials": int(df.trial_id.nunique())}
        rf = pipe[-1]
        pd.DataFrame({"feature": cols, "importance": rf.feature_importances_}).sort_values(
            "importance", ascending=False).to_csv(os.path.join(args.out, f"feature_importance_{name}.csv"), index=False)
        print(f"   [{name}] {len(cols)} features | {tr.sum()} training windows | classes {list(rf.classes_)} | "
              f"T1={tiers[name]['T1_dormant_motion']:.3f} T2={tiers[name]['T2_fastpath_burst']:.3f}")

    # ---- event-level (two-stage) hybrid models: event features + window aggregates -------
    import evaluate_event_model as em
    ev = pd.read_csv(f"data/event_features_{v}.csv", dtype={"subject": str})
    T = ev.set_index("trial_id").join(em.window_aggregates(sets["rx1"], "w1")) \
          .join(em.window_aggregates(sets["rx2"], "w2")).reset_index().dropna()
    event_models, event_features = {}, {}
    for rx, w in (("rx1", "w1"), ("rx2", "w2")):
        cs = [c for c in T.columns if c.startswith(f"ev_{rx}_")] + [c for c in T.columns if c.startswith(f"{w}_")]
        clf = RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced",
                                     n_jobs=-1, random_state=42).fit(T[cs].values, T["activity"].values)
        event_models[rx], event_features[rx] = clf, cs
        print(f"   [event {rx}] {len(cs)} features | {len(T)} trials")
    # fall threshold from out-of-fold LOSO predictions (fall_operating_point.py), ADL FAR target 5 %
    tau = {"target_adl_far": 0.05, "tau": 0.30, "source": "default"}
    oof_path = f"data/models_exp/event_{v}/hyb_late_oof.csv"
    if os.path.exists(oof_path):
        oof = pd.read_csv(oof_path)
        adl_p = oof.loc[oof["true"].isin(ADL), "p_fall"].values
        for tgt in (0.03, 0.05, 0.10):
            tau[f"tau_far_{int(tgt * 100)}"] = float(np.quantile(adl_p, 1 - tgt))
        tau["tau"], tau["source"] = tau["tau_far_5"], oof_path
    print(f"   fall threshold tau (ADL FAR 5%): {tau['tau']:.3f}")

    ref = {}
    ref_path = f"data/models_exp/rx_compare_{v}/rx_comparison.json"
    if os.path.exists(ref_path):
        with open(ref_path) as f:
            ref = json.load(f)["results"]

    config = {"pipeline_version": v, **PIPELINE_FLAGS[v], "fs_hz": 100.0, "lowpass_hz": 10.0,
              "hampel": {"window": 7, "n_sigmas": 3.0}, "window_len": 200, "stride": 25,
              "event_half_width_s": args.event_half_width, "event_search_s": [1.0, 8.5],
              "rx_own_macs": {"RX1": "C0:CD:D6:85:65:EC", "RX2": "1C:C3:AB:B4:A2:48"},
              "classes": tc.CLASSES}
    bundle = {"models": models, "feature_names": feature_names, "tiers": tiers, "fusion": "late",
              "event_models": event_models, "event_feature_names": event_features, "fall_threshold": tau,
              "config": config, "loso_reference": ref, "counts": counts}
    joblib.dump(bundle, os.path.join(args.out, "safesense_bundle.pkl"))
    meta = {k: bundle[k] for k in ("feature_names", "tiers", "fusion", "event_feature_names", "fall_threshold",
                                   "config", "loso_reference", "counts")}
    with open(os.path.join(args.out, "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    # lightweight bundle with only what live_demo_engine.py needs (small enough for a repository)
    live = {k: bundle[k] for k in ("event_models", "event_feature_names", "fall_threshold", "tiers", "config")}
    joblib.dump(live, os.path.join(args.out, "safesense_live_bundle_legacy.pkl"), compress=3)
    print(f"\n  Saved legacy bundle + meta to {args.out}/; building the adopted two-head live bundle ...")
    build_adopted(args.out)


if __name__ == "__main__":
    main()
