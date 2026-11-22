"""
WiFeye: Honest fall-alarm operating point (nested threshold selection)
===========================================================================
The fall alarm fires when the fused P(fall) of the event model (HYB_LATE: event features +
window aggregates, late fusion of RX1 and RX2) exceeds a threshold tau. Choosing tau on the
test data would be optimistic, so tau is chosen INSIDE each outer LOSO fold from inner-LOSO
out-of-fold predictions on the training subjects only, to meet a target ADL false-alarm rate.
The chosen tau is then applied to the held-out subject.

Also reports: the pooled out-of-fold ROC (for figures), Wilson 95% intervals, and 6-class vs
5-class (sit + stand merged into "sit/stand") accuracy from the same predictions.

Usage: python fall_operating_point.py --version v3 --seeds 3
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score, f1_score

import train_classifier as tc
import evaluate_event_model as em

CLASSES = tc.CLASSES
ADL = ["walk", "sit", "stand", "hand"]


def wilson(k, n, z=1.96):
    if n == 0:
        return [float("nan")] * 2
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def late_proba(T, y, tr_idx, te_idx, sets, seed):
    ps = [em.fit_proba(T.loc[tr_idx, sets[n]].values, y[tr_idx], T.loc[te_idx, sets[n]].values, seed)
          for n in ("HYB_RX1", "HYB_RX2")]
    return (ps[0] + ps[1]) / 2


def main():
    ap = argparse.ArgumentParser(description="Nested fall-threshold selection")
    ap.add_argument("--version", default="v3")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--targets", default="0.03,0.05,0.10", help="target ADL false-alarm rates")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    v = args.version
    out_dir = args.out or f"data/models_exp/event_{v}"
    os.makedirs(out_dir, exist_ok=True)
    targets = [float(x) for x in args.targets.split(",")]

    ev = pd.read_csv(f"data/event_features_{v}.csv", dtype={"subject": str})
    T = ev.set_index("trial_id").join(em.window_aggregates(f"data/features_{v}_rx1", "w1")) \
          .join(em.window_aggregates(f"data/features_{v}_rx2", "w2")).reset_index()
    T = T.dropna().sort_values("trial_id").reset_index(drop=True)
    trials = T[["trial_id", "activity", "subject"]]
    y = trials["activity"].values
    col = lambda pre: [c for c in T.columns if c.startswith(pre)]
    sets = {"HYB_RX1": col("ev_rx1_") + col("w1_"), "HYB_RX2": col("ev_rx2_") + col("w2_")}
    folds = tc.loso_folds(trials)
    fi = CLASSES.index("fall")

    oof = np.zeros((len(T), len(CLASSES)))
    decisions = {t: np.zeros(len(T), bool) for t in targets}
    taus = {t: [] for t in targets}
    for seed in range(args.seeds):
        for fold_name, test_ids in folds:
            te = trials["trial_id"].isin(test_ids).values
            tr_idx, te_idx = np.where(~te)[0], np.where(te)[0]
            # inner LOSO over the training subjects -> out-of-fold P(fall) for threshold choice
            inner = trials.loc[tr_idx]
            inner_scores, inner_y = [], []
            for s in sorted(inner.loc[inner.activity != "empty_room", "subject"].unique()):
                i_te = tr_idx[(inner["subject"] == s).values]
                i_tr = np.setdiff1d(tr_idx, i_te)
                p = late_proba(T, y, i_tr, i_te, sets, seed)
                inner_scores.append(p[:, fi]); inner_y.append(y[i_te])
            sc, iy = np.concatenate(inner_scores), np.concatenate(inner_y)
            adl_sc = sc[np.isin(iy, ADL)]
            p_out = late_proba(T, y, tr_idx, te_idx, sets, seed)
            oof[te_idx] += p_out / args.seeds
            for tgt in targets:
                tau = float(np.quantile(adl_sc, 1 - tgt))      # inner ADL FAR ~= target
                taus[tgt].append({"seed": seed, "fold": fold_name, "tau": tau})
                decisions[tgt][te_idx] |= False
                # majority vote over seeds is applied after the loop via stored scores
                decisions.setdefault(f"score_{tgt}_{seed}", np.zeros(len(T)))
                decisions[f"score_{tgt}_{seed}"][te_idx] = (p_out[:, fi] >= tau).astype(float)
        print(f"  seed {seed} done")

    res = {"targets": {}}
    falls, adl = y == "fall", np.isin(y, ADL)
    print("\n=======================================================")
    print(f"  FALL ALARM OPERATING POINTS (HYB_LATE, nested tau, {args.seeds} seeds)")
    print("=======================================================")
    for tgt in targets:
        votes = np.mean([decisions[f"score_{tgt}_{s}"] for s in range(args.seeds)], axis=0) >= 0.5
        k_s, n_s = int(votes[falls].sum()), int(falls.sum())
        k_f, n_f = int(votes[adl].sum()), int(adl.sum())
        per_subj = {s: float(votes[falls & (trials.subject == s).values].mean())
                    for s in ["1", "2", "4", "5"]}
        res["targets"][str(tgt)] = {"sensitivity": k_s / n_s, "sens_ci95": wilson(k_s, n_s),
                                    "adl_far": k_f / n_f, "far_ci95": wilson(k_f, n_f),
                                    "per_subject_sensitivity": per_subj,
                                    "median_tau": float(np.median([d["tau"] for d in taus[tgt]]))}
        r = res["targets"][str(tgt)]
        print(f"  target FAR {tgt:.0%}: sensitivity {r['sensitivity']:.1%} [{r['sens_ci95'][0]:.1%}-{r['sens_ci95'][1]:.1%}]"
              f" | ADL FAR {r['adl_far']:.1%} [{r['far_ci95'][0]:.1%}-{r['far_ci95'][1]:.1%}] | median tau {r['median_tau']:.3f}")
        print(f"      per-subject sensitivity: {per_subj}")

    s_adl = oof[np.isin(y, ["fall"] + ADL), fi]; y_adl = y[np.isin(y, ["fall"] + ADL)] == "fall"
    fpr, tpr, thr = roc_curve(y_adl, s_adl)
    res["pooled_auc"] = float(roc_auc_score(y_adl, s_adl))
    pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thr}).to_csv(os.path.join(out_dir, "fall_roc_hyb_late.csv"), index=False)
    pred = np.array(CLASSES)[oof.argmax(1)]
    merge = lambda a: np.where(np.isin(a, ["sit", "stand"]), "sit/stand", a)
    res["six_class"] = {"accuracy": float(np.mean(pred == y)),
                        "macro_f1": float(f1_score(y, pred, average="macro"))}
    res["five_class_sit_stand_merged"] = {"accuracy": float(np.mean(merge(pred) == merge(y))),
                                          "macro_f1": float(f1_score(merge(y), merge(pred), average="macro"))}
    print(f"\n  pooled OOF fall AUC {res['pooled_auc']:.3f}")
    print(f"  6-class: acc {res['six_class']['accuracy']:.3f}, macro-F1 {res['six_class']['macro_f1']:.3f} | "
          f"5-class (sit+stand merged): acc {res['five_class_sit_stand_merged']['accuracy']:.3f}, "
          f"macro-F1 {res['five_class_sit_stand_merged']['macro_f1']:.3f}")
    with open(os.path.join(out_dir, "fall_operating_points.json"), "w") as f:
        json.dump(res, f, indent=2)
    pd.DataFrame({"trial_id": trials["trial_id"], "subject": trials["subject"], "true": y, "pred": pred,
                  **{f"p_{c}": oof[:, i] for i, c in enumerate(CLASSES)}}).to_csv(
        os.path.join(out_dir, "hyb_late_oof.csv"), index=False)
    print(f"  Saved to {out_dir}/")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
