"""
WiFeye: Controlled 1-ESP32 vs 2-ESP32 receiver comparison (LOSO, trial level)
===================================================================================
Conditions (same trials, same LOSO folds, same model family and budget, same seeds):
  RX1       single receiver 1 (features from RX1 only)
  RX2       single receiver 2 (features from RX2 only)
  SELECTED  per fold, the single receiver (RX1 or RX2) with the better inner-LOSO
            macro-F1 on the training subjects only: the honest "1 ESP32" bar
  EARLY     feature-level fusion (dual-RX feature set incl. cross-RX terms)
  LATE      decision fusion: mean of the RX1 and RX2 trial probabilities
Pre-registered primary hypotheses (from the survey report, P1-6):
  H1  best fusion (EARLY or LATE) beats SELECTED on trial-level macro-F1 / correctness
  H2  fusion lowers ADL false alarms at matched fall sensitivity (reported via AUC)
Statistics: exact McNemar on pooled trial-level correctness (seed-averaged
probabilities), Holm-adjusted; cluster bootstrap (trials resampled within subjects)
for Delta macro-F1 and Delta fall-vs-ADL AUC; per-seed SD.

Usage:
  python compare_receivers.py --dual-dir data/features_v2 --rx1-dir data/features_v2_rx1 \
      --rx2-dir data/features_v2_rx2 --out data/models_exp/rx_compare_v2 --seeds 5
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

import train_classifier as tc

CLASSES = tc.CLASSES


def load_condition(feat_dir, half_width):
    df, X, cols = tc.load_features(os.path.join(feat_dir, "features_all_windows.csv"))
    df = tc.apply_event_labels(df, half_width)
    return df, X, cols


def fit_predict_trials(df, X, train_ids, test_ids, seed, n_estimators):
    """Train on action windows of train trials; return per-test-trial (prob vector, max p_fall)."""
    tr = df["trial_id"].isin(train_ids).values & df["is_action"].values
    te = df["trial_id"].isin(test_ids).values
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=15, class_weight="balanced",
                                 n_jobs=-1, random_state=seed)
    clf.fit(X[tr], df["activity"].values[tr])
    p_raw = clf.predict_proba(X[te])
    p = np.zeros((te.sum(), len(CLASSES)))
    for j, c in enumerate(clf.classes_):
        p[:, CLASSES.index(c)] = p_raw[:, j]
    sub = df[te].reset_index(drop=True)
    out = {}
    fi = CLASSES.index("fall")
    for tid, idx in sub.groupby("trial_id").indices.items():
        reg = idx[sub["in_region"].values[idx]]
        reg = reg if len(reg) else idx
        out[tid] = (p[reg].mean(axis=0), float(p[reg, fi].max()))
    return out


def inner_select(conds, train_ids, trials, seed, n_estimators):
    """Pick RX1 or RX2 by inner LOSO macro-F1 over the training subjects only."""
    subj = trials.set_index("trial_id")["subject"]
    tr_subjects = sorted(s for s in subj[list(train_ids)].unique() if s != "6")
    scores = {}
    for name in ("RX1", "RX2"):
        df, X = conds[name][0], conds[name][1]
        y_true, y_pred = [], []
        for s in tr_subjects:
            inner_test = {t for t in train_ids if subj[t] == s}
            inner_train = set(train_ids) - inner_test
            res = fit_predict_trials(df, X, inner_train, inner_test, seed, n_estimators)
            for tid, (pv, _) in res.items():
                y_true.append(trials.set_index("trial_id").loc[tid, "activity"])
                y_pred.append(CLASSES[int(np.argmax(pv))])
        scores[name] = f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0)
    return max(scores, key=scores.get), scores


def summarize(trials, probs, pfall):
    y = trials["activity"].values
    pred = np.array([CLASSES[int(np.argmax(probs[t]))] for t in trials["trial_id"]])
    adl = np.isin(y, ["fall", "walk", "sit", "stand", "hand"])
    s = np.array([pfall[t] for t in trials["trial_id"]])
    falls, nonfall = y == "fall", np.isin(y, ["walk", "sit", "stand", "hand"])
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=CLASSES, average="macro", zero_division=0)),
        "fall_auc": float(roc_auc_score(y[adl] == "fall", s[adl])),
        "fall_sensitivity": float(np.mean(pred[falls] == "fall")),
        "adl_false_alarm": float(np.mean(pred[nonfall] == "fall")),
    }, pred


def cluster_bootstrap(trials, a, b, n=2000, seed=0):
    """Delta (a - b) for macro-F1 and fall AUC; trials resampled within subjects."""
    rng = np.random.default_rng(seed)
    groups = [g.index.values for _, g in trials.reset_index(drop=True).groupby("subject")]
    T = trials.reset_index(drop=True)
    d_f1, d_auc = [], []
    for _ in range(n):
        idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        tb = T.loc[idx]
        try:
            ma = summarize(tb, a[0], a[1])[0]; mb = summarize(tb, b[0], b[1])[0]
        except ValueError:
            continue
        d_f1.append(ma["macro_f1"] - mb["macro_f1"]); d_auc.append(ma["fall_auc"] - mb["fall_auc"])
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return {"d_macro_f1_ci": q(d_f1), "d_fall_auc_ci": q(d_auc)}


def mcnemar(y, pred_a, pred_b):
    a_ok, b_ok = pred_a == y, pred_b == y
    n01, n10 = int(np.sum(a_ok & ~b_ok)), int(np.sum(~a_ok & b_ok))
    p = binomtest(n01, n01 + n10, 0.5).pvalue if (n01 + n10) else 1.0
    return {"a_right_b_wrong": n01, "a_wrong_b_right": n10, "p": float(p)}


def holm(pvals):
    order = np.argsort(pvals); m = len(pvals); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * pvals[i]); adj[i] = min(run, 1.0)
    return adj


def main():
    ap = argparse.ArgumentParser(description="1 vs 2 ESP32 receiver comparison")
    ap.add_argument("--dual-dir", required=True)
    ap.add_argument("--rx1-dir", required=True)
    ap.add_argument("--rx2-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n-estimators", type=int, default=200)
    ap.add_argument("--event-half-width", type=float, default=1.0)
    ap.add_argument("--no-inner-select", action="store_true", help="skip SELECTED (faster)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    conds = {"EARLY": load_condition(args.dual_dir, args.event_half_width),
             "RX1": load_condition(args.rx1_dir, args.event_half_width),
             "RX2": load_condition(args.rx2_dir, args.event_half_width)}
    ids = [set(c[0]["trial_id"]) for c in conds.values()]
    common = set.intersection(*ids)
    if any(len(s) != len(common) for s in ids):
        print(f"  WARNING: trial sets differ; using the {len(common)} common trials")
    trials = conds["RX1"][0].drop_duplicates("trial_id")[["trial_id", "activity", "subject"]]
    trials = trials[trials["trial_id"].isin(common)].sort_values("trial_id").reset_index(drop=True)
    folds = tc.loso_folds(trials.assign(trial_id=trials["trial_id"]))

    print("\n=======================================================")
    print("  SAFESENSE-5G: 1 vs 2 ESP32 RECEIVERS (LOSO, trial level)")
    print("=======================================================")
    print(f"  Trials: {len(trials)} | seeds: {args.seeds} | folds: {[f for f, _ in folds]}")
    print(f"  Features: EARLY {len(conds['EARLY'][2])}, RX1 {len(conds['RX1'][2])}, RX2 {len(conds['RX2'][2])}\n")

    names = ["RX1", "RX2", "EARLY", "LATE"] + ([] if args.no_inner_select else ["SELECTED"])
    acc_p = {n: {} for n in names}; acc_f = {n: {} for n in names}
    per_seed = {n: [] for n in names}; selections = []
    for seed in range(args.seeds):
        seed_p = {n: {} for n in names}; seed_f = {n: {} for n in names}
        for fold_name, test_ids in folds:
            test_ids = set(test_ids) & common
            train_ids = common - test_ids
            res = {n: fit_predict_trials(conds[n][0], conds[n][1], train_ids, test_ids, seed, args.n_estimators)
                   for n in ("RX1", "RX2", "EARLY")}
            for tid in test_ids:
                for n in ("RX1", "RX2", "EARLY"):
                    seed_p[n][tid], seed_f[n][tid] = res[n][tid]
                seed_p["LATE"][tid] = (res["RX1"][tid][0] + res["RX2"][tid][0]) / 2
                seed_f["LATE"][tid] = (res["RX1"][tid][1] + res["RX2"][tid][1]) / 2
            if not args.no_inner_select:
                pick, sc = inner_select(conds, train_ids, trials, seed, args.n_estimators)
                selections.append({"seed": seed, "fold": fold_name, "picked": pick,
                                   "inner_macro_f1": {k: round(v, 3) for k, v in sc.items()}})
                for tid in test_ids:
                    seed_p["SELECTED"][tid], seed_f["SELECTED"][tid] = seed_p[pick][tid], seed_f[pick][tid]
        for n in names:
            per_seed[n].append(summarize(trials, seed_p[n], seed_f[n])[0])
            for tid in seed_p[n]:
                acc_p[n][tid] = acc_p[n].get(tid, 0) + seed_p[n][tid] / args.seeds
                acc_f[n][tid] = acc_f[n].get(tid, 0) + seed_f[n][tid] / args.seeds
        print(f"  seed {seed}: " + " | ".join(f"{n} F1={per_seed[n][-1]['macro_f1']:.3f} "
                                             f"AUC={per_seed[n][-1]['fall_auc']:.3f}" for n in names))

    y = trials["activity"].values
    final, preds = {}, {}
    for n in names:
        final[n], preds[n] = summarize(trials, acc_p[n], acc_f[n])
        final[n]["seed_sd_macro_f1"] = float(np.std([s["macro_f1"] for s in per_seed[n]]))
        final[n]["seed_sd_fall_auc"] = float(np.std([s["fall_auc"] for s in per_seed[n]]))

    bar = "SELECTED" if "SELECTED" in names else "RX1"
    pairs = [("EARLY", bar), ("LATE", bar), ("EARLY", "LATE"), ("RX1", "RX2")]
    tests = []
    for a, b in pairs:
        t = {"a": a, "b": b, **mcnemar(y, preds[a], preds[b]),
             **cluster_bootstrap(trials, (acc_p[a], acc_f[a]), (acc_p[b], acc_f[b]))}
        tests.append(t)
    adj = holm([t["p"] for t in tests])
    for t, pa in zip(tests, adj):
        t["p_holm"] = float(pa)

    print("\n  Seed-averaged results (trial level):")
    print(f"   {'cond':<9} {'acc':>6} {'macroF1':>8} {'fallAUC':>8} {'sens':>6} {'FAR':>6} {'sd(F1)':>7} {'sd(AUC)':>8}")
    for n in names:
        r = final[n]
        print(f"   {n:<9} {r['accuracy']:6.3f} {r['macro_f1']:8.3f} {r['fall_auc']:8.3f} "
              f"{r['fall_sensitivity']:6.3f} {r['adl_false_alarm']:6.3f} {r['seed_sd_macro_f1']:7.3f} {r['seed_sd_fall_auc']:8.3f}")
    print("\n  Paired comparisons (McNemar on trial correctness, Holm-adjusted; cluster-bootstrap 95% CIs):")
    for t in tests:
        print(f"   {t['a']:>8} vs {t['b']:<8} a>b {t['a_right_b_wrong']:3d} / b>a {t['a_wrong_b_right']:3d} "
              f"p={t['p']:.3f} p_holm={t['p_holm']:.3f} | dF1 CI [{t['d_macro_f1_ci'][0]:+.3f},{t['d_macro_f1_ci'][1]:+.3f}] "
              f"dAUC CI [{t['d_fall_auc_ci'][0]:+.3f},{t['d_fall_auc_ci'][1]:+.3f}]")
    if selections:
        picks = pd.Series([s["picked"] for s in selections]).value_counts().to_dict()
        print(f"\n  Inner-CV single-receiver picks: {picks}")

    with open(os.path.join(args.out, "rx_comparison.json"), "w") as f:
        json.dump({"args": vars(args), "n_trials": len(trials), "results": final, "per_seed": per_seed,
                   "tests": tests, "selections": selections}, f, indent=2)
    pd.DataFrame({"trial_id": trials["trial_id"], "subject": trials["subject"], "true": y,
                  **{f"pred_{n}": preds[n] for n in names},
                  **{f"pfall_{n}": [acc_f[n][t] for t in trials["trial_id"]] for n in names}}
                 ).to_csv(os.path.join(args.out, "rx_comparison_trials.csv"), index=False)
    print(f"\n  Saved to {args.out}/")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
