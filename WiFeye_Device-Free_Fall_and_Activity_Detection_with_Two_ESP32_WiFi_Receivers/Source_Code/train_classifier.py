"""
WiFeye: Classifier Training & Evaluation
===============================================
Input : data/features/features_all_windows.csv (from feature_extraction.py)
Output: data/models/{safesense_model,scaler,label_encoder}.pkl, model_meta.json,
        metrics_summary.json

Labeling (trial protocol, AGENT_BRIDGE.md A1):
  0-2 s static baseline, 2-6 s action (walk 2-7.5 s, hand 2-7 s), then rest.
  Training/window-level eval use only "action windows" (window midpoint inside the
  class's action interval); empty_room uses every window.

Evaluation:
  * Window-level   : action windows of test trials.
  * Trial-level    : (primary) per test trial, pool class probabilities over windows
                     whose midpoint lies in the class-agnostic region [2.0, 7.5] s;
                     argmax of the mean, with a fall override if any window's
                     P(fall) >= fall_tau.
  * Schemes        : StratifiedGroupKFold(5, groups=trial_id) and LOSO over subjects
                     1-5, with empty_room trials (subject 6) spread 2 bare + 2 chair +
                     2 mattress per test fold.
  * Fall metrics   : sensitivity = recall(fall); false-alarm rate = share of ADL
                     trials/windows (walk, sit, stand, hand) predicted fall.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

CLASSES = ["fall", "walk", "sit", "stand", "hand", "empty_room"]
ADL = ["walk", "sit", "stand", "hand"]
ACTION_INTERVALS = {  # seconds, window-midpoint must fall inside
    "fall": (2.0, 6.0), "sit": (2.0, 6.0), "stand": (2.0, 6.0),
    "walk": (2.0, 7.5), "hand": (2.0, 7.0), "empty_room": (0.0, 1e9),
}
TRIAL_REGION = (2.0, 7.5)
META_COLS = ["window_start_s", "window_end_s", "activity", "subject", "source_file",
             "routing_tier", "trial_id", "mid_s", "is_action",
             "source_trial", "source_subject", "style_subject", "kind"]
SEED = 42


def build_models():
    return {
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=100, max_depth=15, class_weight="balanced", n_jobs=-1, random_state=SEED),
        "hist_gradient_boosting": lambda: HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.08, class_weight="balanced", random_state=SEED),
        "svm_rbf": lambda: SVC(kernel="rbf", C=10.0, gamma="scale", probability=True,
                               class_weight="balanced", random_state=SEED),
        "mlp": lambda: MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-3, max_iter=600,
                                     early_stopping=True, random_state=SEED),
    }


def load_features(features_csv):
    df = pd.read_csv(features_csv)
    df["trial_id"] = df["source_file"].str.replace(r"_aligned\.csv$", "", regex=True)
    df["subject"] = df["subject"].astype(str)
    df["mid_s"] = (df["window_start_s"] + df["window_end_s"]) / 2.0
    lo = df["activity"].map(lambda a: ACTION_INTERVALS[a][0])
    hi = df["activity"].map(lambda a: ACTION_INTERVALS[a][1])
    df["is_action"] = (df["mid_s"] >= lo) & (df["mid_s"] <= hi)
    feature_cols = [c for c in df.columns if c not in META_COLS]
    X = np.nan_to_num(df[feature_cols].values.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    return df, X, feature_cols


def load_augmented(aug_dir, feature_cols):
    """Synthetic training-only trials from augment_csi.py (features + provenance)."""
    df = pd.read_csv(os.path.join(aug_dir, "features_all_windows.csv"))
    man = pd.read_csv(os.path.join(aug_dir, "augmented_manifest.csv"), dtype=str, keep_default_na=False)
    man = man.drop(columns=["activity"], errors="ignore")   # features file already carries the label
    df["trial_id"] = df["source_file"].str.replace(r"_aligned\.csv$", "", regex=True)
    df = df.drop(columns=["subject"]).merge(man, left_on="trial_id", right_on="aug_trial_id", how="inner")
    df["subject"] = df["source_subject"]
    df["mid_s"] = (df["window_start_s"] + df["window_end_s"]) / 2.0
    w = df["warp_w"].astype(float); c = df["warp_center_s"].astype(float); d = df["shift_delta_s"].astype(float)
    lo = df["activity"].map(lambda a: ACTION_INTERVALS[a][0]); hi = df["activity"].map(lambda a: ACTION_INTERVALS[a][1])
    df["is_action"] = (df["mid_s"] >= c + w * (lo - c) + d) & (df["mid_s"] <= c + w * (hi - c) + d)
    X = np.nan_to_num(df[feature_cols].values.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    return df, X


def apply_event_labels(df, half_width=1.0, search=(1.0, 8.5)):
    """
    Label-free event localisation (protocol timing was not followed tightly: falls peak
    anywhere from ~1 s to ~5 s). Per trial, the event centre is the window with the largest
    motion score inside `search`; windows within +-half_width s of it are 'action' windows
    (training labels) and form the trial-pooling region. empty_room keeps all windows as
    training windows. Uses features only, never the label, so it is valid at test time.
    """
    score_cols = [c for c in df.columns if c.startswith("ext_motion_pk_")] or ["tier_energy_var"]
    score = df[score_cols].sum(axis=1).values
    df = df.copy()
    df["event_mid_s"] = np.nan
    for _, idx in df.groupby("trial_id").indices.items():
        mids = df["mid_s"].values[idx]
        ok = (mids >= search[0]) & (mids <= search[1])
        cand = idx[ok] if ok.any() else idx
        df.loc[df.index[idx], "event_mid_s"] = df["mid_s"].values[cand[np.argmax(score[cand])]]
    near = (df["mid_s"] - df["event_mid_s"]).abs() <= half_width
    df["in_region"] = near
    df["is_action"] = near | (df["activity"] == "empty_room")
    return df


def aug_mask_for_fold(df_aug, train_trials, scheme, fold_name, aug_kinds):
    """Leakage rules: source trial must be in this fold's training set; 'style' variants for
    subject S are dropped when S is the LOSO test subject (and not used in grouped CV)."""
    m = df_aug["source_trial"].isin(train_trials).values & df_aug["is_action"].values
    m &= df_aug["kind"].isin(aug_kinds).values
    if scheme == "loso":
        test_subj = fold_name.replace("subject_", "")
        m &= (df_aug["style_subject"] != test_subj).values
    else:
        m &= (df_aug["kind"] != "style").values
    return m


def loso_folds(df):
    """Yield (name, test_trial_ids) for LOSO over subjects 1-5 + spread empty_room."""
    trials = df.drop_duplicates("trial_id")[["trial_id", "subject", "activity"]]
    empty = trials[trials["activity"] == "empty_room"].copy()
    empty["tnum"] = empty["trial_id"].str.extract(r"_t(\d+)$")[0].astype(int)
    human_subjects = sorted(s for s in trials["subject"].unique()
                            if s not in set(empty["subject"]))
    folds = []
    for i, subj in enumerate(human_subjects):
        test = set(trials.loc[trials["subject"] == subj, "trial_id"])
        for base in (0, 10, 20):  # bare 1-10, chair 11-20, mattress 21-30
            wanted = {base + 2 * i + 1, base + 2 * i + 2}
            test |= set(empty.loc[empty["tnum"].isin(wanted), "trial_id"])
        folds.append((f"subject_{subj}", test))
    return folds


def grouped_folds(df, n_splits=5):
    trials = df.drop_duplicates("trial_id")[["trial_id", "activity"]].reset_index(drop=True)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    folds = []
    for k, (_, te) in enumerate(sgkf.split(trials, trials["activity"], groups=trials["trial_id"])):
        folds.append((f"fold_{k + 1}", set(trials.loc[te, "trial_id"])))
    return folds


def trial_predict(proba, df_test, classes, fall_tau):
    """Pool window probabilities into one prediction per trial."""
    fall_idx = classes.index("fall")
    out = []
    if "in_region" in df_test.columns:
        in_region = df_test["in_region"].astype(bool)
    else:
        in_region = (df_test["mid_s"] >= TRIAL_REGION[0]) & (df_test["mid_s"] <= TRIAL_REGION[1])
    for tid, idx in df_test.groupby("trial_id").indices.items():
        sel = idx[in_region.values[idx]]
        if len(sel) == 0:
            sel = idx
        p = proba[sel]
        pred_pool = classes[int(np.argmax(p.mean(axis=0)))]
        max_fall = float(p[:, fall_idx].max())
        pred = "fall" if max_fall >= fall_tau else pred_pool
        out.append({"trial_id": tid, "true": df_test["activity"].values[idx[0]],
                    "pred": pred, "pred_pool": pred_pool, "max_p_fall": max_fall,
                    "subject": df_test["subject"].values[idx[0]]})
    return pd.DataFrame(out)


def fall_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    falls = y_true == "fall"
    adl = np.isin(y_true, ADL)
    empty = y_true == "empty_room"
    return {
        "fall_sensitivity": float(np.mean(y_pred[falls] == "fall")) if falls.any() else None,
        "fall_false_alarm_rate_adl": float(np.mean(y_pred[adl] == "fall")) if adl.any() else None,
        "fall_specificity_adl": float(np.mean(y_pred[adl] != "fall")) if adl.any() else None,
        "fall_false_alarm_rate_empty": float(np.mean(y_pred[empty] == "fall")) if empty.any() else None,
        "n_fall": int(falls.sum()), "n_adl": int(adl.sum()),
    }


def summarize(y_true, y_pred, classes):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)),
        **fall_metrics(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=classes).tolist(),
        "per_class": {k: v for k, v in classification_report(
            y_true, y_pred, labels=classes, output_dict=True, zero_division=0).items()
            if k in classes},
    }


def evaluate_scheme(name, folds, df, X, make_model, classes, fall_tau, aug=None, aug_kinds=()):
    win_true, win_pred, trial_frames, per_fold = [], [], [], []
    for fold_name, test_ids in folds:
        te_mask = df["trial_id"].isin(test_ids).values
        tr_mask = (~te_mask) & df["is_action"].values
        X_tr, y_tr = X[tr_mask], df["activity"].values[tr_mask]
        n_aug = 0
        if aug is not None and aug_kinds:
            df_aug, X_aug = aug
            train_trials = set(df.loc[~te_mask, "trial_id"])
            am = aug_mask_for_fold(df_aug, train_trials, name, fold_name, aug_kinds)
            n_aug = int(df_aug.loc[am, "trial_id"].nunique())
            X_tr = np.vstack([X_tr, X_aug[am]])
            y_tr = np.concatenate([y_tr, df_aug["activity"].values[am]])
        model = make_pipeline(StandardScaler(), make_model())
        model.fit(X_tr, y_tr)
        model_classes = list(model.classes_)

        df_te = df[te_mask].reset_index(drop=True)
        proba_raw = model.predict_proba(X[te_mask])
        proba = np.zeros((len(df_te), len(classes)))
        for j, c in enumerate(model_classes):
            proba[:, classes.index(c)] = proba_raw[:, j]

        act = df_te["is_action"].values
        wp = np.array(classes)[np.argmax(proba[act], axis=1)]
        win_true.extend(df_te["activity"].values[act]); win_pred.extend(wp)

        tp = trial_predict(proba, df_te, classes, fall_tau)
        tp["fold"] = fold_name
        trial_frames.append(tp)
        fm = fall_metrics(tp["true"], tp["pred"])
        per_fold.append({"fold": fold_name, "n_test_trials": len(tp), "n_aug_train_trials": n_aug,
                         "trial_accuracy": float(np.mean(tp["true"] == tp["pred"])),
                         "trial_macro_f1": float(f1_score(tp["true"], tp["pred"], average="macro",
                                                          zero_division=0)),
                         **fm})
        print(f"    [{name}] {fold_name:<12} trials={len(tp):3d} "
              f"acc={per_fold[-1]['trial_accuracy']:.3f} "
              f"fallSens={fm['fall_sensitivity'] if fm['fall_sensitivity'] is not None else float('nan'):.3f} "
              f"FAR={fm['fall_false_alarm_rate_adl']:.3f}")

    trials_all = pd.concat(trial_frames, ignore_index=True)
    # fall-threshold sweep on pooled out-of-fold trial predictions
    sweep = []
    for tau in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.01):
        pred = np.where(trials_all["max_p_fall"] >= tau, "fall", trials_all["pred_pool"])
        sweep.append({"fall_tau": tau, **{k: v for k, v in fall_metrics(trials_all["true"], pred).items()
                                          if k.startswith("fall_")},
                      "trial_accuracy": float(np.mean(pred == trials_all["true"]))})
    return {
        "window_level": summarize(win_true, win_pred, classes),
        "trial_level": summarize(trials_all["true"], trials_all["pred"], classes),
        "trial_level_pool_only": summarize(trials_all["true"], trials_all["pred_pool"], classes),
        "fall_tau_sweep": sweep,
        "per_fold": per_fold,
    }, trials_all


def main():
    ap = argparse.ArgumentParser(description="Train & evaluate WiFeye classifiers")
    ap.add_argument("--features-dir", default="data/features")
    ap.add_argument("--model-dir", default="data/models")
    ap.add_argument("--fall-tau", type=float, default=0.5,
                    help="Trial-level fall override: any window P(fall) >= tau -> fall")
    ap.add_argument("--models", default="random_forest,hist_gradient_boosting,svm_rbf,mlp")
    ap.add_argument("--schemes", default="stratified_group_5fold,loso")
    ap.add_argument("--aug-dir", default=None,
                    help="features dir of augment_csi.py output (contains augmented_manifest.csv)")
    ap.add_argument("--aug-kinds", default="",
                    help="comma list of synthetic kinds to add to TRAINING folds: generic,style")
    ap.add_argument("--label-mode", choices=["timeline", "event"], default="timeline",
                    help="timeline: fixed protocol intervals; event: label-free motion-peak localisation")
    ap.add_argument("--event-half-width", type=float, default=1.0)
    ap.add_argument("--quality-manifest", default=None,
                    help="aligned manifest with v2 quality columns (merge_dual_rx.py --source-filter)")
    ap.add_argument("--exclude-flagged", action="store_true",
                    help="drop trials with <9 s RX overlap, a >1 s gap in 2-6 s, or dominant-source share <0.5")
    args = ap.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
    df, X, feature_cols = load_features(os.path.join(args.features_dir, "features_all_windows.csv"))
    if args.quality_manifest and args.exclude_flagged:
        q = pd.read_csv(args.quality_manifest)
        flag = ((q["duration_s"] < 9.0) | (q[["rx1_maxgap_2_6s_ms", "rx2_maxgap_2_6s_ms"]].max(axis=1) > 1000)
                | (q[["rx1_src_frac", "rx2_src_frac"]].min(axis=1) < 0.5))
        bad = set(q.loc[flag, "trial_id"])
        keep = ~df["trial_id"].isin(bad).values
        print(f"  Quality filter: excluding {len(bad)} flagged trials ({int((~keep).sum())} windows)")
        df, X = df[keep].reset_index(drop=True), X[keep]
    aug_kinds = tuple(k.strip() for k in args.aug_kinds.split(",") if k.strip())
    aug = load_augmented(args.aug_dir, feature_cols) if (args.aug_dir and aug_kinds) else None
    if args.label_mode == "event":
        df = apply_event_labels(df, args.event_half_width)
        if aug is not None:
            aug = (apply_event_labels(aug[0], args.event_half_width), aug[1])
    META_COLS.extend(["event_mid_s", "in_region"])
    if aug is not None:
        print(f"  Augmented (training-only) trials loaded: {aug[0]['trial_id'].nunique()} "
              f"| kinds used: {aug_kinds}")
    classes = [c for c in CLASSES if c in set(df["activity"])]
    n_trials = df["trial_id"].nunique()

    print("\n=======================================================")
    print("  SAFESENSE-5G CLASSIFIER TRAINING & EVALUATION")
    print("=======================================================")
    print(f"  Windows: {len(df)} (action windows used for training: {int(df['is_action'].sum())})")
    print(f"  Trials : {n_trials} | Features: {len(feature_cols)} | Classes: {classes}")
    print(f"  Trial counts: {df.drop_duplicates('trial_id')['activity'].value_counts().to_dict()}\n")

    all_schemes = {"stratified_group_5fold": grouped_folds, "loso": loso_folds}
    schemes = {s.strip(): all_schemes[s.strip()](df) for s in args.schemes.split(",") if s.strip()}
    all_models = build_models()
    chosen = [m.strip() for m in args.models.split(",") if m.strip()]
    results, oof = {}, {}
    for mname in chosen:
        results[mname] = {}
        for sname, folds in schemes.items():
            print(f"  >> {mname} / {sname}")
            res, trials_df = evaluate_scheme(sname, folds, df, X, all_models[mname], classes, args.fall_tau,
                                             aug=aug, aug_kinds=aug_kinds)
            results[mname][sname] = res
            oof[(mname, sname)] = trials_df
            tl = res["trial_level"]
            print(f"     TRIAL  acc={tl['accuracy']:.3f} macroF1={tl['macro_f1']:.3f} "
                  f"fallSens={tl['fall_sensitivity']:.3f} FAR(ADL)={tl['fall_false_alarm_rate_adl']:.3f}")
            wl = res["window_level"]
            print(f"     WINDOW acc={wl['accuracy']:.3f} macroF1={wl['macro_f1']:.3f}\n")

    sel_scheme = "loso" if "loso" in schemes else next(iter(schemes))
    best = max(chosen, key=lambda m: results[m][sel_scheme]["trial_level"]["macro_f1"])
    print(f"  Selected model (best {sel_scheme} trial-level macro F1): {best}")

    # Final fit on all real action windows (+ synthetic training trials of the chosen kinds)
    train_mask = df["is_action"].values
    X_fin, y_fin = X[train_mask], df["activity"].values[train_mask]
    if aug is not None:
        am = aug[0]["is_action"].values & aug[0]["kind"].isin(aug_kinds).values
        X_fin = np.vstack([X_fin, aug[1][am]]); y_fin = np.concatenate([y_fin, aug[0]["activity"].values[am]])
    le = LabelEncoder().fit(classes)
    scaler = StandardScaler().fit(X_fin)
    model = all_models[best]()
    model.fit(scaler.transform(X_fin), le.transform(y_fin))

    joblib.dump(model, os.path.join(args.model_dir, "safesense_model.pkl"))
    joblib.dump(scaler, os.path.join(args.model_dir, "scaler.pkl"))
    joblib.dump(le, os.path.join(args.model_dir, "label_encoder.pkl"))
    for (m_, s_), tdf in oof.items():
        tdf.to_csv(os.path.join(args.model_dir, f"oof_trials_{m_}_{s_}.csv"), index=False)

    importances = None
    if hasattr(model, "feature_importances_"):
        importances = dict(sorted(zip(feature_cols, map(float, model.feature_importances_)),
                                  key=lambda kv: -kv[1]))
    else:  # always provide RF importances for the figure
        rf = all_models["random_forest"]().fit(scaler.transform(X_fin), y_fin)
        importances = dict(sorted(zip(feature_cols, map(float, rf.feature_importances_)),
                                  key=lambda kv: -kv[1]))

    meta = {
        "selected_model": best, "classes": classes, "feature_names": feature_cols,
        "window_len": 50, "stride": 25, "fs": 100.0, "fall_tau": args.fall_tau,
        "action_intervals_s": ACTION_INTERVALS, "trial_region_s": TRIAL_REGION,
        "features_dir": args.features_dir,
    }
    with open(os.path.join(args.model_dir, "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    summary = {
        "dataset": {"windows": int(len(df)), "action_windows": int(train_mask.sum()),
                    "trials": int(n_trials), "n_features": len(feature_cols),
                    "trial_counts": {k: int(v) for k, v in
                                     df.drop_duplicates("trial_id")["activity"].value_counts().items()}},
        "protocol": {"fall_tau": args.fall_tau, "trial_region_s": TRIAL_REGION,
                     "action_intervals_s": ACTION_INTERVALS,
                     "loso_folds": [n for n, _ in schemes.get("loso", [])],
                     "augmentation": {"aug_dir": args.aug_dir, "kinds": list(aug_kinds),
                                      "rule": "training folds only; source trial must be in training set; "
                                              "style variants excluded from their style subject's LOSO fold"},
                     "targets": {"fall_sensitivity": ">=0.95", "fall_false_alarm_rate_adl": "<=0.03"}},
        "selected_model": best,
        "feature_importance": importances,
        "results": results,
    }
    with open(os.path.join(args.model_dir, "metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n  Comparison (trial level)")
    for m in chosen:
        parts = []
        for s in schemes:
            a = results[m][s]["trial_level"]
            parts.append(f"{s}: acc={a['accuracy']:.3f} F1={a['macro_f1']:.3f} "
                         f"sens={a['fall_sensitivity']:.3f} FAR={a['fall_false_alarm_rate_adl']:.3f}")
        print(f"   {m:<24} " + " | ".join(parts))
    print(f"\n  Artifacts saved in {args.model_dir}/")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
