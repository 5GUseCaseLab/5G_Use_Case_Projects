"""
WiFeye: adopted model (two-head, selected features, 5 classes with sit+stand merged into
"posture") + subject-dependent feature pruning + per-person calibration, under LOSO (new person).
  ADOPTED     activity head = BASE + fold-selected new features; fall head = BASE; 5 classes
  PRUNED      as ADOPTED, but the activity head drops features with eta2(subject) >= 0.5 * eta2(activity)
              (computed on the fold's training data only; a non-neural version of DANN's goal)
  CALIBRATED  ADOPTED activity head retrained with k=2 trials per daily activity (walk, posture, hand;
              no falls) from the test person, weight 5, 3 draws; scored on that person's remaining trials,
              paired with ADOPTED on the same trials. Fall head stays generic.
All selection / fold-fit features are computed inside each fold (training data only).
Usage: python validate_adopted.py
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, accuracy_score

import train_classifier as tc
import evaluate_event_model as em
import compare_receivers as cr
from validate_features import raw_material, foldfit, select

CL5 = ["fall", "walk", "posture", "hand", "empty_room"]
FALL = 0
MERGE = {"sit": "posture", "stand": "posture"}
OUT = "data/models_exp/adopted_v3"
os.makedirs(OUT, exist_ok=True)


def rf5(Xtr, ytr, Xte, seed=0, w=None):
    m = RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced", n_jobs=-1,
                               random_state=seed).fit(Xtr, ytr, sample_weight=w)
    p = np.zeros((len(Xte), len(CL5)))
    for j, c in enumerate(m.classes_):
        p[:, CL5.index(c)] = m.predict_proba(Xte)[:, j]
    return p


def two_head(pa, pf):
    out = pa.copy(); rest = [1, 2, 3, 4]
    out[:, rest] = pa[:, rest] / (pa[:, rest].sum(1, keepdims=True) + 1e-12) * (1 - pf[:, [FALL]])
    out[:, FALL] = pf[:, FALL]
    return out


def eta2(x, g):
    x = np.asarray(x, float); m = x.mean(); sst = ((x - m) ** 2).sum() + 1e-12
    return sum((g == k).sum() * (x[g == k].mean() - m) ** 2 for k in np.unique(g)) / sst


def metrics5(y, p, subj):
    pred = np.array(CL5)[p.argmax(1)]
    adl = y != "empty_room"
    yf, sf = (y[adl] == "fall"), p[adl, FALL]
    fpr, tpr, _ = roc_curve(yf, sf); ok = fpr <= 0.05
    return {"accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, labels=CL5, average="macro", zero_division=0)),
            "fall_auc": float(roc_auc_score(yf, sf)), "fall_sens_at_far5": float(tpr[ok].max() if ok.any() else 0),
            "per_class_f1": dict(zip(CL5, np.round(f1_score(y, pred, labels=CL5, average=None, zero_division=0), 3).tolist())),
            "per_subject_macro_f1": {s: round(float(f1_score(y[subj == s], pred[subj == s], labels=CL5, average="macro",
                                                             zero_division=0)), 3) for s in ["1", "2", "3", "4", "5"]}}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-extras", action="store_true", help="skip PRUNED and calibration (already reported)")
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
    act6 = T.activity.to_dict()
    deltas, profs = raw_material(seg)
    trials = T.reset_index()[["trial_id", "activity", "subject"]]
    y6, subj = trials.activity.values, trials.subject.values
    y = np.array([MERGE.get(a, a) for a in y6])
    rx_of = lambda c: "rx1" if c.endswith("_rx1") or c.startswith(("ev_rx1_", "w1_")) else (
        "rx2" if c.endswith("_rx2") or c.startswith(("ev_rx2_", "w2_")) else "shared")
    folds = tc.loso_folds(trials)
    group = np.full(len(trials), -1)
    for k, (_, ids) in enumerate(folds):
        group[trials.trial_id.isin(ids).values] = k
    P = {n: np.zeros((len(T), len(CL5))) for n in ("BASE", "ADOPTED", "PRUNED")}
    PRX = {n: np.zeros((len(T), len(CL5))) for n in ("RX1", "RX2")}
    TAU, ALARM = {}, {tg: np.zeros(len(T), bool) for tg in (0.03, 0.05, 0.10)}
    cal_rows, info = [], []
    rng = np.random.default_rng(7)
    for k, (fname, _) in enumerate(folds):
        te = group == k
        tr_ids, te_ids = trials.trial_id[~te].tolist(), trials.trial_id[te].tolist()
        ff = foldfit(tr_ids, te_ids, act6, deltas, profs)
        for g in np.unique(group[~te]):
            ia = trials.trial_id[group == g].tolist()
            ff.update(foldfit([t for t in tr_ids if t not in set(ia)], ia, act6, deltas, profs))
        TT = T.join(pd.DataFrame.from_dict(ff, orient="index").reindex(T.index))
        new_all = labelfree + [c for c in TT.columns if c.startswith(("F1_", "F4_"))]
        sel = select(TT.loc[tr_ids], new_all, base_cols)
        act_cols = base_cols + sel
        # subject-dependent pruning on training data only (humans for subject, 5-class labels for activity)
        trm = ~te & (y6 != "empty_room")
        drop = [c for c in act_cols if eta2(TT[c].values[trm], subj[trm]) >= 0.5 * eta2(TT[c].values[~te], y[~te])]
        pruned_cols = [c for c in act_cols if c not in set(drop)]
        X = lambda cs: np.nan_to_num(TT[cs].values.astype(float))
        late = {}
        variants = (("BASE", base_cols), ("ACT", act_cols)) + (() if args.no_extras else (("PRU", pruned_cols),))
        # nested fall-alarm threshold: inner leave-one-group-out P(fall) of the fall head on training data
        b1 = [c for c in base_cols if rx_of(c) in ("rx1", "shared")]
        b2 = [c for c in base_cols if rx_of(c) in ("rx2", "shared")]
        inner_pf, inner_y = [], []
        for g in np.unique(group[~te]):
            gi = group == g
            ti = ~te & ~gi
            pf = (rf5(X(b1)[ti], y[ti], X(b1)[gi]) + rf5(X(b2)[ti], y[ti], X(b2)[gi])) / 2
            inner_pf.append(pf[:, FALL]); inner_y.append(y[gi])
        ipf, iy = np.concatenate(inner_pf), np.concatenate(inner_y)
        adl_scores = ipf[(iy != "fall") & (iy != "empty_room")]
        TAU[k] = {tg: float(np.quantile(adl_scores, 1 - tg)) for tg in (0.03, 0.05, 0.10)}
        per_rx = {}
        for v, cols in variants:
            p1 = rf5(X([c for c in cols if rx_of(c) in ("rx1", "shared")])[~te], y[~te], X([c for c in cols if rx_of(c) in ("rx1", "shared")])[te])
            p2 = rf5(X([c for c in cols if rx_of(c) in ("rx2", "shared")])[~te], y[~te], X([c for c in cols if rx_of(c) in ("rx2", "shared")])[te])
            late[v] = (p1 + p2) / 2
            per_rx[v] = (p1, p2)
        P["BASE"][te] = late["BASE"]
        P["ADOPTED"][te] = two_head(late["ACT"], late["BASE"])
        # 1 vs 2 ESP32 with the adopted system: the same two heads from ONE receiver only
        PRX["RX1"][te] = two_head(per_rx["ACT"][0], per_rx["BASE"][0])
        PRX["RX2"][te] = two_head(per_rx["ACT"][1], per_rx["BASE"][1])
        if "PRU" in late:
            P["PRUNED"][te] = two_head(late["PRU"], late["BASE"])
        for tg, tau in TAU[k].items():
            ALARM[tg][te] = late["BASE"][:, FALL] >= tau
        info.append({"fold": fname, "selected": len(sel), "pruned_away": len(drop), "pruned_examples": drop[:8]})
        # calibration: k=2 trials per daily activity from the test person (no falls), 3 draws
        te_idx = np.where(te)[0]
        for draw in range(0 if args.no_extras else 3):
            cal = []
            for c in ("walk", "posture", "hand"):
                cand = te_idx[y[te_idx] == c]
                if len(cand) >= 3:
                    cal.extend(rng.choice(cand, 2, replace=False).tolist())
            if not cal:
                continue
            ev_idx = np.setdiff1d(te_idx, cal)
            tr_idx = np.concatenate([np.where(~te)[0], cal])
            w = np.concatenate([np.ones((~te).sum()), np.full(len(cal), 5.0)])
            pc = []
            for r in ("rx1", "rx2"):
                cs = [c for c in act_cols if rx_of(c) in (r, "shared")]
                pc.append(rf5(X(cs)[tr_idx], y[tr_idx], X(cs)[ev_idx], w=w))
            p_cal = two_head((pc[0] + pc[1]) / 2, late["BASE"][np.searchsorted(te_idx, ev_idx)])
            p_ref = P["ADOPTED"][ev_idx]
            f_cal = f1_score(y[ev_idx], np.array(CL5)[p_cal.argmax(1)], labels=CL5, average="macro", zero_division=0)
            f_ref = f1_score(y[ev_idx], np.array(CL5)[p_ref.argmax(1)], labels=CL5, average="macro", zero_division=0)
            cal_rows.append({"fold": fname, "draw": draw, "f1_ref": f_ref, "f1_cal": f_cal,
                             "acc_ref": float(np.mean(np.array(CL5)[p_ref.argmax(1)] == y[ev_idx])),
                             "acc_cal": float(np.mean(np.array(CL5)[p_cal.argmax(1)] == y[ev_idx]))})
        print(f"  {fname}: selected {len(sel)}, pruned away {len(drop)}", flush=True)

    names_run = ("BASE", "ADOPTED") if args.no_extras else ("BASE", "ADOPTED", "PRUNED")
    res = {n: metrics5(y, P[n], subj) for n in names_run}
    # nested operating points of the deployed fall head
    def wilson(kk, nn, z=1.96):
        p_ = kk / nn; d_ = 1 + z * z / nn; c_ = (p_ + z * z / (2 * nn)) / d_
        h_ = z * np.sqrt(p_ * (1 - p_) / nn + z * z / (4 * nn * nn)) / d_
        return [float(c_ - h_), float(c_ + h_)]
    falls, adl = y == "fall", (y != "fall") & (y != "empty_room")
    ops = {}
    for tg, al in ALARM.items():
        ks, kf = int(al[falls].sum()), int(al[adl].sum())
        ops[str(tg)] = {"sensitivity": ks / falls.sum(), "sens_ci95": wilson(ks, falls.sum()),
                        "adl_far": kf / adl.sum(), "far_ci95": wilson(kf, adl.sum())}
    pd.DataFrame({"trial_id": trials.trial_id, "subject": subj, "true6": y6, "true5": y,
                  "pred_adopted": np.array(CL5)[P["ADOPTED"].argmax(1)], "p_fall": P["BASE"][:, FALL],
                  **{f"p_{c}": P["ADOPTED"][:, i] for i, c in enumerate(CL5)},
                  **{f"alarm_far{int(tg * 100)}": ALARM[tg] for tg in ALARM}}).to_csv(os.path.join(OUT, "adopted_oof.csv"), index=False)
    print("  nested fall-alarm operating points (deployed fall head):",
          {k_: f"FAR {v['adl_far']:.1%} -> sens {v['sensitivity']:.1%} [{v['sens_ci95'][0]:.0%}-{v['sens_ci95'][1]:.0%}]" for k_, v in ops.items()})
    tid = trials.trial_id.values
    tr5 = trials.assign(activity=y)
    tests = []
    for a, b in ((("ADOPTED", "BASE"),) if args.no_extras else (("ADOPTED", "BASE"), ("PRUNED", "ADOPTED"), ("PRUNED", "BASE"))):
        A = ({t: P[a][i] for i, t in enumerate(tid)}, {t: P[a][i, FALL] for i, t in enumerate(tid)})
        B = ({t: P[b][i] for i, t in enumerate(tid)}, {t: P[b][i, FALL] for i, t in enumerate(tid)})
        old_classes = cr.CLASSES
        cr.CLASSES = CL5                       # compare_receivers helpers use the 5-class list here
        bs = cr.cluster_bootstrap(tr5, A, B, n=1500)
        cr.CLASSES = old_classes
        tests.append({"a": a, "b": b, **cr.mcnemar(y, np.array(CL5)[P[a].argmax(1)], np.array(CL5)[P[b].argmax(1)]), **bs})
    for t_, ph in zip(tests, cr.holm([t_["p"] for t_ in tests])):
        t_["p_holm"] = float(ph)
    if cal_rows:
        C = pd.DataFrame(cal_rows)
        d = (C.f1_cal - C.f1_ref).values
        boots = [np.mean(rng.choice(d, len(d))) for _ in range(5000)]
        cal = {"mean_dF1": float(d.mean()), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
               "mean_dAcc": float((C.acc_cal - C.acc_ref).mean()), "n_runs": int(len(d)),
               "significant": bool(np.percentile(boots, 2.5) > 0)}
    else:
        cal = {"mean_dF1": float("nan"), "ci95": [float("nan")] * 2, "mean_dAcc": float("nan"), "n_runs": 0, "significant": False}

    print("\n  LOSO (new person), 5 classes (sit + stand = posture)")
    print(f"   {'config':<8} {'acc':>6} {'mF1':>6} {'AUC':>6} {'sens@5%':>8}  per-class F1 (fall/walk/posture/hand/empty)   per-subject mF1")
    for n in names_run:
        m = res[n]; pc = m["per_class_f1"]
        print(f"   {n:<8} {m['accuracy']:6.3f} {m['macro_f1']:6.3f} {m['fall_auc']:6.3f} {m['fall_sens_at_far5']:8.3f}  "
              f"{pc['fall']:.2f}/{pc['walk']:.2f}/{pc['posture']:.2f}/{pc['hand']:.2f}/{pc['empty_room']:.2f}"
              f"   {list(m['per_subject_macro_f1'].values())}")
    for t_ in tests:
        print(f"   {t_['a']} vs {t_['b']}: better/worse {t_['a_right_b_wrong']}/{t_['a_wrong_b_right']} p_holm={t_['p_holm']:.3f} "
              f"| dF1 [{t_['d_macro_f1_ci'][0]:+.3f},{t_['d_macro_f1_ci'][1]:+.3f}]")
    print(f"   pruning: {[(i['fold'], i['pruned_away']) for i in info]} features removed per fold; e.g. {info[0]['pruned_examples']}")
    print(f"   calibration (k=2 per daily activity): dF1 {cal['mean_dF1']:+.3f} 95% CI [{cal['ci95'][0]:+.3f},{cal['ci95'][1]:+.3f}] "
          f"dAcc {cal['mean_dAcc']:+.3f} -> significant: {cal['significant']}")
    with open(os.path.join(OUT, "adopted_results.json"), "w") as fh:
        json.dump({"metrics": res, "tests": tests, "calibration": cal, "fold_info": info, "fall_operating_points": ops,
                   "tau_per_fold": {str(k_): v for k_, v in TAU.items()}}, fh, indent=2, default=float)

    # ---- 1 vs 2 ESP32 with the adopted system (own Holm family: late fusion vs each single receiver)
    rxres = {"LATE (2 ESP32)": res["ADOPTED"], "RX1 only": metrics5(y, PRX["RX1"], subj),
             "RX2 only": metrics5(y, PRX["RX2"], subj)}
    rtests = []
    for b in ("RX1", "RX2"):
        A = ({t: P["ADOPTED"][i] for i, t in enumerate(tid)}, {t: P["ADOPTED"][i, FALL] for i, t in enumerate(tid)})
        B = ({t: PRX[b][i] for i, t in enumerate(tid)}, {t: PRX[b][i, FALL] for i, t in enumerate(tid)})
        old_classes = cr.CLASSES
        cr.CLASSES = CL5
        bs = cr.cluster_bootstrap(tr5, A, B, n=1500)
        cr.CLASSES = old_classes
        rtests.append({"a": "LATE", "b": b, **cr.mcnemar(y, np.array(CL5)[P["ADOPTED"].argmax(1)],
                                                         np.array(CL5)[PRX[b].argmax(1)]), **bs})
    for t_, ph in zip(rtests, cr.holm([t_["p"] for t_ in rtests])):
        t_["p_holm"] = float(ph)
    print("\n  1 vs 2 ESP32 (adopted two-head system, LOSO, 5 classes)")
    for n, m in rxres.items():
        print(f"   {n:<15} acc {m['accuracy']:.3f}  mF1 {m['macro_f1']:.3f}  fall AUC {m['fall_auc']:.3f}  "
              f"sens@5% {m['fall_sens_at_far5']:.3f}  per-class {m['per_class_f1']}")
    for t_ in rtests:
        print(f"   LATE vs {t_['b']}: better/worse {t_['a_right_b_wrong']}/{t_['a_wrong_b_right']} p={t_['p']:.3f} "
              f"p_holm={t_['p_holm']:.3f} | dF1 [{t_['d_macro_f1_ci'][0]:+.3f},{t_['d_macro_f1_ci'][1]:+.3f}]")
    pd.DataFrame({"trial_id": tid, "true5": y,
                  **{f"pred_{n.lower()}": np.array(CL5)[PRX[n].argmax(1)] for n in PRX},
                  **{f"p_fall_{n.lower()}": PRX[n][:, FALL] for n in PRX}}).to_csv(
        os.path.join(OUT, "adopted_rx_oof.csv"), index=False)
    with open(os.path.join(OUT, "adopted_rx_compare.json"), "w") as fh:
        json.dump({"results": rxres, "tests": rtests}, fh, indent=2, default=float)


if __name__ == "__main__":
    main()
