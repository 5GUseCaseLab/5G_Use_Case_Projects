"""
WiFeye: two-head model and smaller new-feature set (leak-free, both validation levels)
==========================================================================================
  BASE        current A2 features (fall and activity from one model)
  SEL         BASE + new domain features passing the fold-internal acceptance test + pruning (~45)
  SMALL       BASE + top-12 of those, ranked by Random-Forest importance on the fold's training data
  2H_SEL      two heads: activity head = SEL, fall head = BASE (P(fall) from BASE, the other classes
              re-scaled from SEL to fill 1 - P(fall))
  2H_SMALL    two heads: activity head = SMALL, fall head = BASE
Late fusion (mean of RX1 and RX2 models) everywhere, as deployed. Fold-fit features (F1, F4) and the
selection are refitted inside every fold; training trials get inner out-of-fold fold-fit values.
Usage: python validate_twohead.py --seeds 2 --topk 12
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import train_classifier as tc
import evaluate_event_model as em
import compare_receivers as cr
import model_audit as ma
from validate_features import raw_material, foldfit, select, rf_proba

CLASSES = tc.CLASSES
FALL = CLASSES.index("fall")
OUT = "data/models_exp/twohead_v3"
os.makedirs(OUT, exist_ok=True)


def two_head(p_act, p_fallhead):
    out = p_act.copy()
    rest = np.delete(np.arange(len(CLASSES)), FALL)
    s = p_act[:, rest].sum(1, keepdims=True) + 1e-12
    out[:, rest] = p_act[:, rest] / s * (1 - p_fallhead[:, [FALL]])
    out[:, FALL] = p_fallhead[:, FALL]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--topk", type=int, default=12)
    args = ap.parse_args()
    seg = pd.read_csv("data/event_features_v3_fused.csv", dtype={"subject": str}).set_index("trial_id")
    agg = em.window_aggregates("data/features_v3_rx1", "w1").join(em.window_aggregates("data/features_v3_rx2", "w2"))
    base = seg[["activity", "subject"] + [c for c in seg.columns if c.startswith("ev_rx")]].join(agg)
    dom = pd.read_csv("data/models_exp/eda_domain_v3/domain_features.csv", dtype={"subject": str}).set_index("trial_id")
    labelfree = [c for c in dom.columns if c.startswith("F") and not c.startswith(("F1_", "F4_"))]
    T = base.join(dom[labelfree]).dropna(subset=[c for c in base.columns if c not in ("activity", "subject")])
    T = T.replace([np.inf, -np.inf], np.nan)
    T[labelfree] = T[labelfree].fillna(T[labelfree].median())
    base_cols = [c for c in base.columns if c not in ("activity", "subject") and T[c].nunique() > 1]
    T = T.sort_index()
    act = T.activity.to_dict()
    deltas, profs = raw_material(seg)
    trials = T.reset_index()[["trial_id", "activity", "subject"]]
    y, subj = trials.activity.values, trials.subject.values
    rx_of = lambda c: "rx1" if c.endswith("_rx1") or c.startswith(("ev_rx1_", "w1_")) else (
        "rx2" if c.endswith("_rx2") or c.startswith(("ev_rx2_", "w2_")) else "shared")
    names = ["BASE", "SEL", "SMALL", "2H_SEL", "2H_SMALL"]
    results, chosen = {}, {}
    for sname, folds in {"block (seen people)": ma.block_folds(trials), "LOSO (new person)": tc.loso_folds(trials)}.items():
        group = np.full(len(trials), -1)
        for k, (_, ids) in enumerate(folds):
            group[trials.trial_id.isin(ids).values] = k
        P = {n: np.zeros((len(T), len(CLASSES))) for n in names}
        chosen[sname] = []
        for seed in range(args.seeds):
            for k, (fname, test_ids) in enumerate(folds):
                te = group == k
                tr_ids, te_ids = trials.trial_id[~te].tolist(), trials.trial_id[te].tolist()
                ff = foldfit(tr_ids, te_ids, act, deltas, profs)
                for g in np.unique(group[~te]):
                    ia = trials.trial_id[group == g].tolist()
                    ff.update(foldfit([t for t in tr_ids if t not in set(ia)], ia, act, deltas, profs))
                TT = T.join(pd.DataFrame.from_dict(ff, orient="index").reindex(T.index))
                new_all = labelfree + [c for c in TT.columns if c.startswith(("F1_", "F4_"))]
                if seed == 0:
                    sel = select(TT.loc[tr_ids], new_all, base_cols)
                    imp_cols = base_cols + sel
                    imp = RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced",
                                                 n_jobs=-1, random_state=0).fit(
                        np.nan_to_num(TT.loc[tr_ids, imp_cols].values.astype(float)), y[~te]).feature_importances_
                    rank = sorted(sel, key=lambda c: -imp[imp_cols.index(c)])
                    small = rank[:args.topk]
                    chosen[sname].append({"fold": fname, "selected": sel, "small": small})
                else:
                    sel, small = chosen[sname][k]["selected"], chosen[sname][k]["small"]
                X = lambda cs: np.nan_to_num(TT[cs].values.astype(float))
                late = {}
                for v, cols in (("BASE", base_cols), ("SEL", base_cols + sel), ("SMALL", base_cols + small)):
                    cr_ = {r: [c for c in cols if rx_of(c) in (r, "shared")] for r in ("rx1", "rx2")}
                    p1 = rf_proba(X(cr_["rx1"])[~te], y[~te], X(cr_["rx1"])[te], seed)
                    p2 = rf_proba(X(cr_["rx2"])[~te], y[~te], X(cr_["rx2"])[te], seed)
                    late[v] = (p1 + p2) / 2
                late["2H_SEL"] = two_head(late["SEL"], late["BASE"])
                late["2H_SMALL"] = two_head(late["SMALL"], late["BASE"])
                for n in names:
                    P[n][te] += late[n] / args.seeds
                print(f"  {sname} seed {seed} {fname}: sel {len(sel)} small {len(small)}", flush=True)
        res = {n: ma.metrics(y, P[n], subj) for n in names}
        tid = trials.trial_id.values
        tests = []
        for n in names[1:]:
            a = ({t: P[n][i] for i, t in enumerate(tid)}, {t: P[n][i, FALL] for i, t in enumerate(tid)})
            b = ({t: P["BASE"][i] for i, t in enumerate(tid)}, {t: P["BASE"][i, FALL] for i, t in enumerate(tid)})
            tests.append({"variant": n, **cr.mcnemar(y, np.array(CLASSES)[P[n].argmax(1)], np.array(CLASSES)[P["BASE"].argmax(1)]),
                          **cr.cluster_bootstrap(trials, a, b, n=1500)})
        for t_, ph in zip(tests, cr.holm([t_["p"] for t_ in tests])):
            t_["p_holm"] = float(ph)
        results[sname] = {"metrics": res, "tests": tests}
        print(f"\n  ===== {sname} =====")
        print(f"   {'config':<9} {'acc':>6} {'mF1':>6} {'AUC':>6} {'sens@5%':>8} {'prec@5%':>8}   F1: fall  walk  sit  stand hand  empty   per-subject mF1")
        for n in names:
            m = res[n]; pc = m["per_class"]
            print(f"   {n:<9} {m['accuracy']:6.3f} {m['macro_f1']:6.3f} {m['fall_auc']:6.3f} {m['fall_sens_at_far5']:8.3f} "
                  f"{m['fall_prec_at_far5']:8.3f}       {pc['fall']['f1']:.2f}  {pc['walk']['f1']:.2f}  {pc['sit']['f1']:.2f}  "
                  f"{pc['stand']['f1']:.2f}  {pc['hand']['f1']:.2f}  {pc['empty_room']['f1']:.2f}   "
                  f"{[round(v, 2) for v in m['per_subject_macro_f1'].values()]}")
        for t_ in tests:
            print(f"   {t_['variant']:<9} vs BASE: better/worse {t_['a_right_b_wrong']}/{t_['a_wrong_b_right']} p_holm={t_['p_holm']:.3f} "
                  f"| dF1 [{t_['d_macro_f1_ci'][0]:+.3f},{t_['d_macro_f1_ci'][1]:+.3f}] dAUC [{t_['d_fall_auc_ci'][0]:+.3f},{t_['d_fall_auc_ci'][1]:+.3f}]")
        sc = pd.Series([f for c in chosen[sname] for f in c["small"]]).value_counts()
        print(f"   top-{args.topk} features chosen in >= 4 of 5 folds: {sc[sc >= 4].index.tolist()}")
    with open(os.path.join(OUT, "twohead_results.json"), "w") as fh:
        json.dump({"results": results, "chosen": chosen}, fh, indent=2, default=float)
    print(f"\n  Saved to {OUT}/")


if __name__ == "__main__":
    main()
