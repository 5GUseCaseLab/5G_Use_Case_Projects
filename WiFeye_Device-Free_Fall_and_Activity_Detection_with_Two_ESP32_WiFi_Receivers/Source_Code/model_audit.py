"""
WiFeye: critical audit of the deployed model + fusion choice
=================================================================
Question: is "Random Forest per receiver + equal-weight late fusion" (the deployed system) actually
the best choice for this data? Everything below is evaluated on the SAME 509 trials and the SAME
leave-one-subject-out folds (subjects 1-5, empty-room trials spread 2/2/2), averaged over seeds, on
the final two-stage hybrid features (event features + window aggregates, 52 per receiver).

Learners   RF (deployed), ExtraTrees, HistGradientBoosting (hyper-parameters chosen by inner LOSO on
           the training subjects only), logistic regression, SVM-RBF
Fusion     RX1 only, RX2 only, EARLY (both receivers' features + cross-receiver features),
           LATE_MEAN (deployed: equal-weight mean), LATE_CONF (each receiver weighted by its own
           max-probability), LATE_QUAL (weighted by each receiver's packet rate in that trial),
           LATE_LEARNED (RX1 weight w chosen by inner LOSO), ORACLE (either receiver right; ceiling)
           Adaptive (per-trial) fusion:
             LATE_SELECT  use the receiver that is more confident on this trial (hard selection)
             LATE_ADAPT   weight = confidence x packet-rate share of each receiver on this trial
             STACK        logistic-regression combiner on both receivers' probabilities + per-trial
                          link quality (log packet rate, log largest gap in 2-6 s, both receivers),
                          trained on inner out-of-fold predictions of the training data only
Metrics    accuracy, macro-F1, balanced accuracy, Cohen's kappa, per-class precision / recall / F1,
           fall-vs-ADL AUC, fall sensitivity + precision at 5 % ADL false alarms (ROC read-out, the same
           rule for every model), Brier score and expected calibration error of P(fall), receiver error
           correlation, feature correlation between receivers
Tests      paired vs the deployed configuration: exact McNemar (Holm-adjusted) + cluster bootstrap
           (trials resampled within subjects) for delta macro-F1 and delta AUC

Usage: python model_audit.py --seeds 3
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (f1_score, precision_recall_fscore_support, accuracy_score, balanced_accuracy_score,
                             cohen_kappa_score, roc_auc_score, roc_curve, brier_score_loss)

import train_classifier as tc
import evaluate_event_model as em
import compare_receivers as cr

CLASSES = tc.CLASSES
ADL = ["walk", "sit", "stand", "hand"]
LEARNERS = ["RF", "ET", "HGB", "LR", "SVM"]
HGB_GRID = [dict(max_depth=2, learning_rate=0.1, max_iter=150, l2_regularization=0.5),
            dict(max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=1.0),
            dict(max_leaf_nodes=15, min_samples_leaf=10, learning_rate=0.05, max_iter=300, l2_regularization=1.0)]


def make(name, seed, params=None):
    if name == "RF":
        return RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced",
                                      n_jobs=-1, random_state=seed)
    if name == "ET":
        return ExtraTreesClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced",
                                    n_jobs=-1, random_state=seed)
    if name == "HGB":
        return HistGradientBoostingClassifier(class_weight="balanced", random_state=seed, **(params or {}))
    if name == "LR":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=5000, class_weight="balanced"))
    if name == "SVM":
        return make_pipeline(StandardScaler(), SVC(C=3.0, gamma="scale", probability=True,
                                                   class_weight="balanced", random_state=seed))
    raise ValueError(name)


def fit_proba(name, Xtr, ytr, Xte, seed, params=None):
    m = make(name, seed, params).fit(Xtr, ytr)
    raw = m.predict_proba(Xte)
    p = np.zeros((len(Xte), len(CLASSES)))
    for j, c in enumerate(m.classes_):
        p[:, CLASSES.index(c)] = raw[:, j]
    return p


def inner_folds(group, idx):
    """Inner folds mirror the outer scheme: leave one of the remaining outer groups out
    (LOSO -> inner leave-one-subject-out; BLOCK -> inner leave-one-chunk-out)."""
    return [idx[group[idx] == k] for k in np.unique(group[idx])]


def block_folds(trials, n_chunks=5):
    """Seen-person validation: every recording block (subject x activity, 20 consecutive trials;
    empty room: bare / chair / mattress sets of 10) is cut into n_chunks contiguous chunks in recording
    order; fold k tests chunk k of every block, so each test trial's people and session are seen in
    training but its neighbouring trials are mostly not."""
    tnum = trials["trial_id"].str.extract(r"_t(\d+)$")[0].astype(int).values
    furn = pd.Series(np.where(tnum <= 10, "bare", np.where(tnum <= 20, "chair", "mattress")), index=trials.index)
    block = np.where(trials["activity"] == "empty_room", "6_" + furn,
                     trials["subject"].astype(str) + "_" + trials["activity"]).astype(str)
    g = np.zeros(len(trials), int)
    for b in np.unique(block):
        idx = np.where(block == b)[0]
        order = idx[np.argsort(tnum[idx])]
        g[order] = (np.arange(len(order)) * n_chunks // len(order))
    return [(f"chunk_{k + 1}", set(trials["trial_id"].values[g == k])) for k in range(n_chunks)]


def inner_oof(name, X, y, group, tr_idx, seed, params=None):
    oof = np.zeros((len(group), len(CLASSES)))
    for te in inner_folds(group, tr_idx):
        trn = np.setdiff1d(tr_idx, te)
        oof[te] = fit_proba(name, X[trn], y[trn], X[te], seed, params)
    return oof


def mf1(y, p):
    return f1_score(y, np.array(CLASSES)[p.argmax(1)], labels=CLASSES, average="macro", zero_division=0)


def metrics(y, p, subj):
    pred = np.array(CLASSES)[p.argmax(1)]
    pr, rc, f1, sup = precision_recall_fscore_support(y, pred, labels=CLASSES, zero_division=0)
    adl = np.isin(y, ["fall"] + ADL)
    yf, sf = (y[adl] == "fall"), p[adl, CLASSES.index("fall")]
    fpr, tpr, thr = roc_curve(yf, sf)
    ok = fpr <= 0.05
    i = np.argmax(tpr * ok) if ok.any() else 0
    t5 = thr[i]
    flag = sf >= t5
    prec5 = float(yf[flag].mean()) if flag.any() else 0.0
    pf_all = p[:, CLASSES.index("fall")]
    yb = (y == "fall").astype(float)
    bins = np.clip((pf_all * 10).astype(int), 0, 9)
    ece = sum(abs(yb[bins == b].mean() - pf_all[bins == b].mean()) * (bins == b).mean()
              for b in range(10) if (bins == b).any())
    per_subj = {s: float(f1_score(y[subj == s], pred[subj == s], labels=CLASSES, average="macro", zero_division=0))
                for s in ["1", "2", "3", "4", "5"]}
    return {"accuracy": float(accuracy_score(y, pred)), "macro_f1": float(f1_score(y, pred, labels=CLASSES,
                                                                                     average="macro", zero_division=0)),
            "balanced_acc": float(balanced_accuracy_score(y, pred)), "kappa": float(cohen_kappa_score(y, pred)),
            "fall_auc": float(roc_auc_score(yf, sf)), "fall_sens_at_far5": float(tpr[i]), "fall_far_at_op": float(fpr[i]),
            "fall_prec_at_far5": prec5, "fall_brier": float(brier_score_loss(yb, pf_all)), "fall_ece": float(ece),
            "per_class": {c: {"precision": float(pr[k]), "recall": float(rc[k]), "f1": float(f1[k]), "n": int(sup[k])}
                          for k, c in enumerate(CLASSES)},
            "per_subject_macro_f1": per_subj}


def main():
    ap = argparse.ArgumentParser(description="Critical audit of model and fusion choice")
    ap.add_argument("--version", default="v3")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--scheme", choices=["loso", "block"], default="loso",
                    help="loso: new person (headline); block: seen people, within-block time split")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    args.out = args.out or f"data/models_exp/model_audit_{args.version}_{args.scheme}"
    os.makedirs(args.out, exist_ok=True)
    v = args.version

    ev = pd.read_csv(f"data/event_features_{v}.csv", dtype={"subject": str})
    T = ev.set_index("trial_id").join(em.window_aggregates(f"data/features_{v}_rx1", "w1")) \
          .join(em.window_aggregates(f"data/features_{v}_rx2", "w2")).reset_index().dropna()
    T = T.sort_values("trial_id").reset_index(drop=True)
    man = pd.read_csv(f"data/aligned_{v}/manifest_aligned.csv").set_index("trial_id")
    rate = man.loc[T["trial_id"], ["rx1_rate_hz", "rx2_rate_hz"]].values
    qual = np.log(np.clip(man.loc[T["trial_id"], ["rx1_rate_hz", "rx2_rate_hz", "rx1_maxgap_2_6s_ms",
                                                  "rx2_maxgap_2_6s_ms"]].values.astype(float), 1e-3, None))
    trials = T[["trial_id", "activity", "subject"]]
    y, subj = trials["activity"].values, trials["subject"].values
    col = lambda pre: [c for c in T.columns if c.startswith(pre)]
    sets = {"RX1": col("ev_rx1_") + col("w1_"), "RX2": col("ev_rx2_") + col("w2_")}
    sets["EARLY"] = sets["RX1"] + sets["RX2"] + col("ev_x_")
    Xs = {k: np.nan_to_num(T[c].values.astype(float)) for k, c in sets.items()}
    folds = tc.loso_folds(trials) if args.scheme == "loso" else block_folds(trials)
    group = np.full(len(T), -1)
    for k, (_, ids) in enumerate(folds):
        group[trials["trial_id"].isin(ids).values] = k
    assert (group >= 0).all(), "every trial must belong to exactly one outer fold"
    names = [f"{L}_{f}" for L in LEARNERS for f in ("RX1", "RX2", "EARLY", "LATE_MEAN", "LATE_CONF", "LATE_QUAL",
                                                   "LATE_LEARNED", "LATE_SELECT", "LATE_ADAPT", "STACK")]
    P = {n: np.zeros((len(T), len(CLASSES))) for n in names}
    chosen = {"HGB_params": [], "late_weights": {L: [] for L in LEARNERS}}

    print(f"\n  MODEL & FUSION AUDIT ({v}, scheme={args.scheme}): {len(T)} trials, {args.seeds} seeds, "
          f"folds {[f for f, _ in folds]}, learners {LEARNERS}")
    for seed in range(args.seeds):
        for fold_name, test_ids in folds:
            te = np.where(trials["trial_id"].isin(test_ids).values)[0]
            tr = np.setdiff1d(np.arange(len(T)), te)
            outer, inner = {}, {}
            for L in LEARNERS:
                for s in ("RX1", "RX2", "EARLY"):
                    params = None
                    if L == "HGB":       # choose HGB settings by inner LOSO macro-F1 (training subjects only)
                        best = (-1, None, None)
                        for g in HGB_GRID:
                            o = inner_oof("HGB", Xs[s], y, group, tr, seed, g)
                            sc = mf1(y[tr], o[tr])
                            if sc > best[0]:
                                best = (sc, g, o)
                        params = best[1]
                        chosen["HGB_params"].append({"seed": seed, "fold": fold_name, "set": s, "params": params})
                        if s != "EARLY":
                            inner[(L, s)] = best[2]
                    elif s != "EARLY":
                        inner[(L, s)] = inner_oof(L, Xs[s], y, group, tr, seed)
                    outer[(L, s)] = fit_proba(L, Xs[s][tr], y[tr], Xs[s][te], seed, params)
                p1, p2 = outer[(L, "RX1")], outer[(L, "RX2")]
                # learned weight on inner out-of-fold predictions
                i1, i2 = inner[(L, "RX1")][tr], inner[(L, "RX2")][tr]
                w = max(np.round(np.arange(0, 1.01, 0.1), 1), key=lambda ww: mf1(y[tr], ww * i1 + (1 - ww) * i2))
                chosen["late_weights"][L].append(float(w))
                c1, c2 = p1.max(1, keepdims=True), p2.max(1, keepdims=True)
                q = rate[te] / rate[te].sum(1, keepdims=True)
                a1, a2 = c1 * q[:, :1], c2 * q[:, 1:]
                # stacked combiner: trained on inner out-of-fold probabilities + link quality (training data only)
                meta = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000,
                                                                          class_weight="balanced"))
                meta.fit(np.hstack([i1, i2, qual[tr]]), y[tr])
                raw = meta.predict_proba(np.hstack([p1, p2, qual[te]]))
                p_stack = np.zeros_like(p1)
                for j, cl in enumerate(meta.classes_):
                    p_stack[:, CLASSES.index(cl)] = raw[:, j]
                fused = {"RX1": p1, "RX2": p2, "EARLY": outer[(L, "EARLY")], "LATE_MEAN": (p1 + p2) / 2,
                         "LATE_CONF": (c1 * p1 + c2 * p2) / (c1 + c2), "LATE_QUAL": q[:, :1] * p1 + q[:, 1:] * p2,
                         "LATE_LEARNED": w * p1 + (1 - w) * p2,
                         "LATE_SELECT": np.where(c1 >= c2, p1, p2),
                         "LATE_ADAPT": (a1 * p1 + a2 * p2) / (a1 + a2),
                         "STACK": p_stack}
                for f, pf in fused.items():
                    P[f"{L}_{f}"][te] += pf / args.seeds
            print(f"  seed {seed} {fold_name} done", flush=True)

    R = {n: metrics(y, P[n], subj) for n in names}
    r1, r2 = P["RF_RX1"].argmax(1), P["RF_RX2"].argmax(1)
    ok1, ok2 = np.array(CLASSES)[r1] == y, np.array(CLASSES)[r2] == y
    R["RF_ORACLE"] = {"accuracy": float(np.mean(ok1 | ok2))}

    # correlation diagnostics
    fall_i = CLASSES.index("fall")
    diag = {"error_corr_phi_RF": float(np.corrcoef(ok1.astype(float), ok2.astype(float))[0, 1]),
            "pfall_corr_RF_rx1_rx2": float(np.corrcoef(P["RF_RX1"][:, fall_i], P["RF_RX2"][:, fall_i])[0, 1]),
            "both_wrong_RF": float(np.mean(~ok1 & ~ok2)), "rx2_only_right_RF": float(np.mean(~ok1 & ok2))}
    pairs = [(a, a.replace("ev_rx1_", "ev_rx2_")) for a in sets["RX1"] if a.startswith("ev_rx1_")] + \
            [(a, a.replace("w1_", "w2_", 1)) for a in sets["RX1"] if a.startswith("w1_")]
    rhos = {a: float(spearmanr(T[a], T[b]).correlation) for a, b in pairs if b in T.columns}
    diag["feature_corr_rx1_rx2_median"] = float(np.nanmedian(list(rhos.values())))
    diag["feature_corr_top"] = dict(sorted(rhos.items(), key=lambda kv: -abs(kv[1]))[:6])
    diag["feature_corr_share_above_0.7"] = float(np.mean([abs(r) > 0.7 for r in rhos.values()]))

    # paired tests vs the deployed configuration
    base = "RF_LATE_MEAN"
    probs = {n: {t: P[n][i] for i, t in enumerate(trials["trial_id"])} for n in names}
    pfs = {n: {t: P[n][i, fall_i] for i, t in enumerate(trials["trial_id"])} for n in names}
    preds = {n: np.array(CLASSES)[P[n].argmax(1)] for n in names}
    rank = sorted(names, key=lambda n: -R[n]["macro_f1"])
    contenders = [n for n in rank if n != base][:6] + [n for n in ("RF_RX1", "HGB_LATE_MEAN", "RF_EARLY")
                                                        if n not in rank[:7] and n != base]
    tests = []
    for n in contenders:
        t = {"a": n, "b": base, **cr.mcnemar(y, preds[n], preds[base]),
             **cr.cluster_bootstrap(trials, (probs[n], pfs[n]), (probs[base], pfs[base]), n=1500)}
        tests.append(t)
    for t, pa in zip(tests, cr.holm([t["p"] for t in tests])):
        t["p_holm"] = float(pa)

    # ---- report
    print("\n  Ranked by macro-F1 (trial level, LOSO, seed-averaged)")
    print(f"   {'configuration':<17} {'acc':>6} {'mF1':>6} {'balAcc':>6} {'kappa':>6} {'AUC':>6} {'sens@5%':>8} "
          f"{'prec@5%':>8} {'Brier':>6} {'ECE':>6}")
    for n in rank:
        r = R[n]
        star = " <- deployed" if n == base else ""
        print(f"   {n:<17} {r['accuracy']:6.3f} {r['macro_f1']:6.3f} {r['balanced_acc']:6.3f} {r['kappa']:6.3f} "
              f"{r['fall_auc']:6.3f} {r['fall_sens_at_far5']:8.3f} {r['fall_prec_at_far5']:8.3f} "
              f"{r['fall_brier']:6.3f} {r['fall_ece']:6.3f}{star}")
    print(f"   RF_ORACLE (either receiver right) accuracy {R['RF_ORACLE']['accuracy']:.3f}")
    print("\n  Per-class precision / recall / F1:")
    for n in [base, rank[0]] if rank[0] != base else [base]:
        print(f"   {n}")
        for c in CLASSES:
            pc = R[n]["per_class"][c]
            print(f"     {c:<11} P {pc['precision']:.2f}  R {pc['recall']:.2f}  F1 {pc['f1']:.2f}  (n={pc['n']})")
        print(f"     per-subject macro-F1: {R[n]['per_subject_macro_f1']}")
    print("\n  Correlation diagnostics:", json.dumps({k: v for k, v in diag.items() if k != 'feature_corr_top'}))
    print("   most correlated RX1/RX2 feature pairs:", {k: round(v, 2) for k, v in diag["feature_corr_top"].items()})
    print("\n  Learned RX1 weight (inner LOSO), mean over folds/seeds:",
          {L: round(float(np.mean(w)), 2) for L, w in chosen["late_weights"].items()})
    hp = pd.Series([json.dumps(c["params"], sort_keys=True) for c in chosen["HGB_params"]]).value_counts()
    print("  HGB settings chosen by inner LOSO (count):", hp.to_dict())
    print("\n  Paired tests vs deployed RF_LATE_MEAN (McNemar Holm; cluster-bootstrap 95% CI):")
    for t in tests:
        print(f"   {t['a']:<17} better/worse trials {t['a_right_b_wrong']:3d}/{t['a_wrong_b_right']:3d} "
              f"p_holm={t['p_holm']:.3f} | dF1 [{t['d_macro_f1_ci'][0]:+.3f},{t['d_macro_f1_ci'][1]:+.3f}] "
              f"dAUC [{t['d_fall_auc_ci'][0]:+.3f},{t['d_fall_auc_ci'][1]:+.3f}]")
    with open(os.path.join(args.out, "audit_results.json"), "w") as f:
        json.dump({"results": R, "diagnostics": diag, "tests": tests, "chosen": chosen,
                   "seeds": args.seeds, "n_trials": int(len(T))}, f, indent=2)
    print(f"\n  Saved to {args.out}/")


if __name__ == "__main__":
    main()
