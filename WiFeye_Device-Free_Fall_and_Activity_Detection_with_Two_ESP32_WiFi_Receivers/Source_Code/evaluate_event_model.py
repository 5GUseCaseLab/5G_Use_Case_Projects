"""
WiFeye: Phase-2 evaluation of event-level (two-stage) models
=================================================================
Trial-level LOSO (subjects 1-5, empty-room trials spread 2/2/2 per fold), several seeds,
same folds and trials as compare_receivers.py so results are paired with the window-level
baseline (data/models_exp/rx_compare_<v>/rx_comparison_trials.csv).

Conditions
  EV_RX1 / EV_RX2   event features of one receiver
  EV_DUAL           both receivers' event features + cross-receiver features (early)
  EV_LATE           mean of EV_RX1 and EV_RX2 probabilities
  HYB_RX1 / HYB_RX2 event features + that receiver's window features averaged (mean, max)
                    over the event region (no model outputs are stacked -> no leakage)
  HYB_LATE          mean of HYB_RX1 and HYB_RX2 probabilities
  *_NOABS           ablation without the absolute (session-dependent) log_noise features
Extra analyses
  --kshot           calibration curve: k = 0,1,2,3 labelled trials per class from the held-out
                    subject added to training (weight 5), tested on that subject's remaining
                    trials; k = 0 is scored on the same remaining trials (paired)
  hand-vs-empty AUC and a subject-identity probe are always reported

Usage:
  python evaluate_event_model.py --version v3 --seeds 5 --kshot
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold

import train_classifier as tc
import compare_receivers as cr

CLASSES = tc.CLASSES


def window_aggregates(feat_dir, prefix):
    df, X, cols = tc.load_features(os.path.join(feat_dir, "features_all_windows.csv"))
    df = tc.apply_event_labels(df, 1.0)
    keep = [c for c in cols if not c.startswith("flag_")]
    W = pd.DataFrame(X[:, [cols.index(c) for c in keep]], columns=keep)
    W["trial_id"] = df["trial_id"].values
    W = W[df["in_region"].values]
    agg = W.groupby("trial_id").agg(["mean", "max"])
    agg.columns = [f"{prefix}_{a}_{b}" for a, b in agg.columns]
    return agg


def fit_proba(Xtr, ytr, Xte, seed, wtr=None):
    clf = RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced",
                                 n_jobs=-1, random_state=seed)
    clf.fit(Xtr, ytr, sample_weight=wtr)
    p_raw = clf.predict_proba(Xte)
    p = np.zeros((len(Xte), len(CLASSES)))
    for j, c in enumerate(clf.classes_):
        p[:, CLASSES.index(c)] = p_raw[:, j]
    return p


def main():
    ap = argparse.ArgumentParser(description="Evaluate event-level models (Phase 2)")
    ap.add_argument("--version", default="v3")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--kshot", action="store_true")
    ap.add_argument("--kshot-draws", type=int, default=5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    v = args.version
    out_dir = args.out or f"data/models_exp/event_{v}"
    os.makedirs(out_dir, exist_ok=True)

    ev = pd.read_csv(f"data/event_features_{v}.csv", dtype={"subject": str})
    agg1 = window_aggregates(f"data/features_{v}_rx1", "w1")
    agg2 = window_aggregates(f"data/features_{v}_rx2", "w2")
    T = ev.set_index("trial_id").join(agg1).join(agg2).reset_index()
    T = T.dropna().sort_values("trial_id").reset_index(drop=True)
    trials = T[["trial_id", "activity", "subject"]]
    folds = tc.loso_folds(trials)

    col = lambda pre: [c for c in T.columns if c.startswith(pre)]
    noabs = lambda cs: [c for c in cs if not c.endswith("log_noise")]
    sets = {
        "EV_RX1": col("ev_rx1_"), "EV_RX2": col("ev_rx2_"),
        "EV_DUAL": col("ev_rx1_") + col("ev_rx2_") + col("ev_x_"),
        "HYB_RX1": col("ev_rx1_") + col("w1_"), "HYB_RX2": col("ev_rx2_") + col("w2_"),
        "EV_RX1_NOABS": noabs(col("ev_rx1_")), "HYB_RX1_NOABS": noabs(col("ev_rx1_")) + col("w1_"),
    }
    late = {"EV_LATE": ("EV_RX1", "EV_RX2"), "HYB_LATE": ("HYB_RX1", "HYB_RX2")}
    names = list(sets) + list(late)
    y = trials["activity"].values

    print("\n=======================================================")
    print(f"  PHASE 2: EVENT-LEVEL MODELS ({v}, LOSO, {args.seeds} seeds, {len(T)} trials)")
    print("=======================================================")
    acc_p = {n: np.zeros((len(T), len(CLASSES))) for n in names}
    per_seed = {n: [] for n in names}
    for seed in range(args.seeds):
        sp = {n: np.zeros((len(T), len(CLASSES))) for n in names}
        for _, test_ids in folds:
            te = trials["trial_id"].isin(test_ids).values
            for n, cs in sets.items():
                sp[n][te] = fit_proba(T.loc[~te, cs].values, y[~te], T.loc[te, cs].values, seed)
        for n, (a, b) in late.items():
            sp[n] = (sp[a] + sp[b]) / 2
        for n in names:
            acc_p[n] += sp[n] / args.seeds
            probs = {t: sp[n][i] for i, t in enumerate(trials["trial_id"])}
            pf = {t: sp[n][i, CLASSES.index("fall")] for i, t in enumerate(trials["trial_id"])}
            per_seed[n].append(cr.summarize(trials, probs, pf)[0])
        print(f"  seed {seed} done")

    final, preds, P, PF = {}, {}, {}, {}
    for n in names:
        P[n] = {t: acc_p[n][i] for i, t in enumerate(trials["trial_id"])}
        PF[n] = {t: acc_p[n][i, CLASSES.index("fall")] for i, t in enumerate(trials["trial_id"])}
        final[n], preds[n] = cr.summarize(trials, P[n], PF[n])
        final[n]["seed_sd_macro_f1"] = float(np.std([s["macro_f1"] for s in per_seed[n]]))
        final[n]["per_class_f1"] = dict(zip(CLASSES, f1_score(y, preds[n], labels=CLASSES, average=None,
                                                              zero_division=0).round(3).tolist()))
        hm = np.isin(y, ["hand", "empty_room"])
        final[n]["hand_vs_empty_auc"] = float(roc_auc_score(y[hm] == "hand", acc_p[n][hm, CLASSES.index("hand")]))
        final[n]["confusion"] = confusion_matrix(y, preds[n], labels=CLASSES).tolist()

    # paired comparison with the window-level baseline (same trials, same folds)
    base_path = f"data/models_exp/rx_compare_{v}/rx_comparison_trials.csv"
    tests = []
    if os.path.exists(base_path):
        B = pd.read_csv(base_path).set_index("trial_id").loc[trials["trial_id"]]
        for bname in ("RX1", "LATE"):
            bp = {t: np.eye(len(CLASSES))[CLASSES.index(p)] for t, p in B[f"pred_{bname}"].items()}
            bf = B[f"pfall_{bname}"].to_dict()
            final[f"WIN_{bname}"] = cr.summarize(trials, bp, bf)[0]
            preds[f"WIN_{bname}"] = B[f"pred_{bname}"].values
            P[f"WIN_{bname}"], PF[f"WIN_{bname}"] = bp, bf
        for a, b in [("HYB_LATE", "WIN_LATE"), ("HYB_RX1", "WIN_RX1"), ("EV_LATE", "WIN_LATE"),
                     ("HYB_LATE", "HYB_RX1"), ("HYB_RX1", "HYB_RX1_NOABS")]:
            t = {"a": a, "b": b, **cr.mcnemar(y, preds[a], preds[b]),
                 **cr.cluster_bootstrap(trials, (P[a], PF[a]), (P[b], PF[b]), n=2000)}
            tests.append(t)
        for t, pa in zip(tests, cr.holm([t["p"] for t in tests])):
            t["p_holm"] = float(pa)

    print(f"\n   {'condition':<15} {'acc':>6} {'macroF1':>8} {'fallAUC':>8} {'sens':>6} {'FAR':>6} {'hand/empty AUC':>15}")
    for n in [k for k in final]:
        r = final[n]
        print(f"   {n:<15} {r['accuracy']:6.3f} {r['macro_f1']:8.3f} {r['fall_auc']:8.3f} "
              f"{r['fall_sensitivity']:6.3f} {r['adl_false_alarm']:6.3f} {r.get('hand_vs_empty_auc', float('nan')):15.3f}")
    print("\n   per-class F1 (HYB_LATE):", final["HYB_LATE"]["per_class_f1"])
    print("   confusion HYB_LATE (rows true, cols pred):", CLASSES)
    for row, c in zip(final["HYB_LATE"]["confusion"], CLASSES):
        print(f"     {c:<11} {row}")
    if tests:
        print("\n   Paired tests (McNemar Holm-adjusted; cluster-bootstrap 95% CI):")
        for t in tests:
            print(f"   {t['a']:>14} vs {t['b']:<14} p_holm={t['p_holm']:.3f} | dF1 [{t['d_macro_f1_ci'][0]:+.3f},"
                  f"{t['d_macro_f1_ci'][1]:+.3f}] dAUC [{t['d_fall_auc_ci'][0]:+.3f},{t['d_fall_auc_ci'][1]:+.3f}]")

    # subject-identity probe (human trials, trial-grouped 5-fold): how much identity do features carry?
    H = T[T.activity != "empty_room"].reset_index(drop=True)
    probe = {}
    for n in ("EV_RX1", "HYB_RX1", "EV_RX1_NOABS"):
        cs = sets[n]; correct = []
        for tr_i, te_i in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(H, H["subject"], H["trial_id"]):
            p = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1).fit(
                H.loc[tr_i, cs], H.loc[tr_i, "subject"]).predict(H.loc[te_i, cs])
            correct.extend(p == H.loc[te_i, "subject"].values)
        probe[n] = float(np.mean(correct))
    print(f"\n   Subject-ID probe accuracy (chance 0.20): {probe}")

    kshot = None
    if args.kshot:
        kshot = run_kshot(T, trials, folds, sets, late, args)
    with open(os.path.join(out_dir, "event_results.json"), "w") as f:
        json.dump({"results": final, "tests": tests, "subject_probe": probe, "kshot": kshot,
                   "n_trials": int(len(T)), "seeds": args.seeds}, f, indent=2)
    pd.DataFrame({"trial_id": trials["trial_id"], "subject": trials["subject"], "true": y,
                  **{f"pred_{n}": preds[n] for n in preds},
                  **{f"pfall_{n}": [PF[n][t] for t in trials["trial_id"]] for n in PF}}).to_csv(
        os.path.join(out_dir, "event_trials.csv"), index=False)
    print(f"\n  Saved to {out_dir}/")
    print("=======================================================\n")


def run_kshot(T, trials, folds, sets, late, args, weight=5.0):
    """Per-person calibration curve for HYB_LATE: k labelled trials/class from the held-out subject."""
    y = trials["activity"].values
    rng = np.random.default_rng(123)
    rows = []
    for fold_name, test_ids in folds:
        subj = fold_name.replace("subject_", "")
        s_mask = (trials["subject"] == subj).values
        te_all = trials["trial_id"].isin(test_ids).values
        for draw in range(args.kshot_draws):
            # one nested draw per repetition: the k=3 set contains the k=2 set contains k=1
            order = {c: rng.permutation(np.where(s_mask & (y == c))[0]) for c in CLASSES}
            hold = set(np.concatenate([o[:3] for o in order.values() if len(o)]).tolist())
            eval_mask = te_all.copy()
            eval_mask[list(hold)] = False          # same evaluation trials for every k
            for k in (0, 1, 2, 3):
                cal = np.concatenate([o[:k] for o in order.values() if len(o)]).astype(int) if k else np.array([], int)
                tr = ~te_all
                tr_idx = np.concatenate([np.where(tr)[0], cal])
                w = np.concatenate([np.ones(tr.sum()), np.full(len(cal), weight)])
                probs = []
                for n in late["HYB_LATE"]:
                    cs = sets[n]
                    probs.append(fit_proba(T.loc[tr_idx, cs].values, y[tr_idx], T.loc[eval_mask, cs].values,
                                           draw, wtr=w))
                p = (probs[0] + probs[1]) / 2
                pred = np.array(CLASSES)[p.argmax(1)]
                ye = y[eval_mask]
                adl = np.isin(ye, ["fall", "walk", "sit", "stand", "hand"])
                auc = roc_auc_score(ye[adl] == "fall", p[adl, CLASSES.index("fall")]) if (ye[adl] == "fall").any() else np.nan
                ss = np.isin(ye, ["sit", "stand"])
                rows.append({"fold": fold_name, "draw": draw, "k": k,
                             "macro_f1": f1_score(ye, pred, labels=CLASSES, average="macro", zero_division=0),
                             "accuracy": float(np.mean(pred == ye)), "fall_auc": auc,
                             "sit_stand_confusions": int(np.sum(ss & (pred != ye) & np.isin(pred, ["sit", "stand"])))})
    K = pd.DataFrame(rows)
    summ = K.groupby("k")[["macro_f1", "accuracy", "fall_auc", "sit_stand_confusions"]].mean().round(3)
    print("\n   k-shot calibration curve (HYB_LATE; held-out subject's remaining trials):")
    print(summ.to_string())
    base = K[K.k == 0].set_index(["fold", "draw"])["macro_f1"]
    deltas = {}
    for k in (1, 2, 3):
        d = (K[K.k == k].set_index(["fold", "draw"])["macro_f1"] - base).values
        boots = [np.mean(rng.choice(d, len(d))) for _ in range(5000)]
        deltas[k] = {"mean": float(d.mean()), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}
        print(f"   k={k}: dMacroF1 vs k=0 {d.mean():+.3f}  95% CI [{deltas[k]['ci95'][0]:+.3f}, {deltas[k]['ci95'][1]:+.3f}]")
    return {"curve": summ.reset_index().to_dict("records"), "delta_vs_k0": deltas}


if __name__ == "__main__":
    main()
