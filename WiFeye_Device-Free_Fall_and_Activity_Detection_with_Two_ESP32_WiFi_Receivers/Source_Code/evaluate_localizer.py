"""
WiFeye: old (argmax) vs fused (reliability-weighted, prior-informed, gap-masked) localisation
================================================================================================
A. Label-free reliability: each person repeated the same action 20 times in one block with a fairly
   consistent personal timing, so a good localiser puts the events of a block at consistent times.
   Metric: within-block standard deviation of the event time (median over blocks, per class).
B. Downstream: Random Forest (deployed settings) on trial-level event features + window aggregates,
   old vs fused, under BOTH validation levels (seen people: within-block time split; new person: LOSO),
   same folds, paired tests (McNemar Holm + cluster bootstrap).
Usage: python evaluate_localizer.py --seeds 2
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

CLASSES = tc.CLASSES


def segment_aggregates(feat_dir, prefix, seg):
    """Window features averaged / maxed over windows whose centre lies inside the fused segment."""
    df, X, cols = tc.load_features(os.path.join(feat_dir, "features_all_windows.csv"))
    keep = [c for c in cols if not c.startswith("flag_")]
    W = pd.DataFrame(X[:, [cols.index(c) for c in keep]], columns=keep)
    W["trial_id"], W["mid_s"] = df["trial_id"].values, df["mid_s"].values
    rows = {}
    for tid, g in W.groupby("trial_id"):
        if tid not in seg.index:
            continue
        on, off, pk = seg.loc[tid, ["event_onset", "event_offset", "event_t"]]
        sel = g[(g.mid_s >= on - 0.25) & (g.mid_s <= off + 0.25)]
        if sel.empty:
            sel = g.iloc[[int(np.argmin(np.abs(g.mid_s.values - pk)))]]
        f = sel[keep]
        rows[tid] = {**{f"{prefix}_{c}_mean": v for c, v in f.mean().items()},
                     **{f"{prefix}_{c}_max": v for c, v in f.max().items()}}
    return pd.DataFrame.from_dict(rows, orient="index")


def build(kind):
    if kind == "old":
        ev = pd.read_csv("data/event_features_v3.csv", dtype={"subject": str})
        a1, a2 = em.window_aggregates("data/features_v3_rx1", "w1"), em.window_aggregates("data/features_v3_rx2", "w2")
    else:
        ev = pd.read_csv("data/event_features_v3_fused.csv", dtype={"subject": str})
        seg = ev.set_index("trial_id")
        a1, a2 = segment_aggregates("data/features_v3_rx1", "w1", seg), segment_aggregates("data/features_v3_rx2", "w2", seg)
    T = ev.set_index("trial_id").join(a1).join(a2).reset_index().rename(columns={"index": "trial_id"})
    T = T.dropna(subset=[c for c in T.columns if c.startswith(("ev_rx", "w1_", "w2_"))]).sort_values("trial_id")
    return T.reset_index(drop=True)


def consistency(T, name):
    tnum = T.trial_id.str.extract(r"_t(\d+)$")[0].astype(int)
    blk = np.where(T.activity == "empty_room", "6_" + pd.Series(np.where(tnum <= 10, "b", np.where(tnum <= 20, "c", "m"))),
                   T.subject + "_" + T.activity)
    d = T.assign(block=blk)
    d = d[d.activity != "empty_room"]
    sd = d.groupby(["activity", "block"])["event_t"].std().groupby("activity").median()
    inside = ((d.event_t >= 0.5) & (d.event_t <= 7.5)).mean()
    return {"median_within_block_sd_s": sd.round(2).to_dict(), "overall": float(sd.median()),
            "share_events_in_protocol_window": float(inside)}


def run(T, folds, seeds):
    col = lambda pre: [c for c in T.columns if c.startswith(pre)]
    sets = {"RX1": col("ev_rx1_") + col("w1_"), "RX2": col("ev_rx2_") + col("w2_")}
    sets["EARLY"] = sets["RX1"] + sets["RX2"] + col("ev_x_") + col("loc_")
    y = T.activity.values
    P = {k: np.zeros((len(T), len(CLASSES))) for k in ("RX1", "RX2", "EARLY", "LATE_MEAN")}
    for seed in range(seeds):
        for _, ids in folds:
            te = T.trial_id.isin(ids).values
            out = {}
            for s, cs in sets.items():
                X = np.nan_to_num(T[cs].values.astype(float))
                m = RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced",
                                           n_jobs=-1, random_state=seed).fit(X[~te], y[~te])
                p = np.zeros((te.sum(), len(CLASSES)))
                for j, c in enumerate(m.classes_):
                    p[:, CLASSES.index(c)] = m.predict_proba(X[te])[:, j]
                out[s] = p
            out["LATE_MEAN"] = (out["RX1"] + out["RX2"]) / 2
            for k in P:
                P[k][te] += out[k] / seeds
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=2)
    args = ap.parse_args()
    out_dir = "data/models_exp/localizer_v3"
    os.makedirs(out_dir, exist_ok=True)
    T_old, T_new = build("old"), build("fused")
    common = sorted(set(T_old.trial_id) & set(T_new.trial_id))
    T_old = T_old[T_old.trial_id.isin(common)].reset_index(drop=True)
    T_new = T_new[T_new.trial_id.isin(common)].reset_index(drop=True)
    res = {"consistency": {"old_argmax": consistency(T_old, "old"), "fused": consistency(T_new, "fused")}}
    print("\n  A. Label-free consistency (lower within-block SD = more consistent event timing)")
    for k, v in res["consistency"].items():
        print(f"   {k:<11} overall {v['overall']:.2f} s | by class {v['median_within_block_sd_s']} | "
              f"in protocol window {v['share_events_in_protocol_window']:.1%}")
    trials = T_new[["trial_id", "activity", "subject"]]
    y, subj = trials.activity.values, trials.subject.values
    schemes = {"block (seen people)": ma.block_folds(trials), "LOSO (new person)": tc.loso_folds(trials)}
    res["downstream"] = {}
    print("\n  B. Random Forest, old vs fused localisation")
    for sname, folds in schemes.items():
        Po, Pn = run(T_old, folds, args.seeds), run(T_new, folds, args.seeds)
        res["downstream"][sname] = {}
        print(f"\n   {sname}")
        print(f"   {'config':<10} {'':<6} {'acc':>6} {'mF1':>6} {'AUC':>6} {'sens@5%':>8} {'F1 fall/sit/stand/hand':>24}")
        for k in ("RX1", "RX2", "EARLY", "LATE_MEAN"):
            mo, mn = ma.metrics(y, Po[k], subj), ma.metrics(y, Pn[k], subj)
            res["downstream"][sname][k] = {"old": mo, "fused": mn}
            for tag, m in (("old", mo), ("fused", mn)):
                pc = m["per_class"]
                print(f"   {k:<10} {tag:<6} {m['accuracy']:6.3f} {m['macro_f1']:6.3f} {m['fall_auc']:6.3f} "
                      f"{m['fall_sens_at_far5']:8.3f}   {pc['fall']['f1']:.2f}/{pc['sit']['f1']:.2f}/"
                      f"{pc['stand']['f1']:.2f}/{pc['hand']['f1']:.2f}")
        tid = trials.trial_id.values
        tests = []
        for k in ("LATE_MEAN", "RX1", "EARLY"):
            a = ({t: Pn[k][i] for i, t in enumerate(tid)}, {t: Pn[k][i, 0] for i, t in enumerate(tid)})
            b = ({t: Po[k][i] for i, t in enumerate(tid)}, {t: Po[k][i, 0] for i, t in enumerate(tid)})
            pa, pb = np.array(CLASSES)[Pn[k].argmax(1)], np.array(CLASSES)[Po[k].argmax(1)]
            tests.append({"config": k, **cr.mcnemar(y, pa, pb), **cr.cluster_bootstrap(trials, a, b, n=1500)})
        for t, ph in zip(tests, cr.holm([t["p"] for t in tests])):
            t["p_holm"] = float(ph)
            print(f"   fused vs old {t['config']:<10} better/worse {t['a_right_b_wrong']}/{t['a_wrong_b_right']} "
                  f"p_holm={t['p_holm']:.3f} dF1 [{t['d_macro_f1_ci'][0]:+.3f},{t['d_macro_f1_ci'][1]:+.3f}] "
                  f"dAUC [{t['d_fall_auc_ci'][0]:+.3f},{t['d_fall_auc_ci'][1]:+.3f}]")
        res["downstream"][sname]["tests"] = tests
    with open(os.path.join(out_dir, "localizer_eval.json"), "w") as fh:
        json.dump(res, fh, indent=2, default=float)
    print(f"\n  Saved to {out_dir}/")


if __name__ == "__main__":
    main()
