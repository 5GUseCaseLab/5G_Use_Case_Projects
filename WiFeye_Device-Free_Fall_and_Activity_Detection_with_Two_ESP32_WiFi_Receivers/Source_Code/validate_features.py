"""
WiFeye: do the domain-specific features improve recognition? (validation ladder, leak-free)
================================================================================================
Feature sets (trial level, per receiver, late fusion = deployed configuration, plus RX1 and EARLY):
  BASE      A2: fused-localiser event features + per-receiver window aggregates (constant columns dropped)
  BASE+SEL  BASE + new domain features that pass the acceptance test and redundancy pruning, where the
            selection is done INSIDE each fold on the training data only
  BASE+ALL  BASE + all 85 new domain features (no selection; checks that selection is not what helps)
Fold-fit features (F1 signed posture signature, F4 DTW template distances) are fitted on the training
data of each fold; training trials receive inner out-of-fold values (no in-sample fitted features).
Schemes: seen people (within-block time split) and new person (LOSO). Random Forest (deployed settings).
Usage: python validate_features.py --seeds 2
"""

import os
import glob
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score

import train_classifier as tc
import evaluate_event_model as em
import compare_receivers as cr
import model_audit as ma
from event_localizer import receiver_curve

CLASSES = tc.CLASSES
ADL = ["walk", "sit", "stand", "hand"]
OUT = "data/models_exp/feature_validation_v3"
os.makedirs(OUT, exist_ok=True)
PAIRS = {"sit_stand": ("sit", ["stand"]), "hand_empty": ("hand", ["empty_room"]), "fall_sit": ("fall", ["sit"]),
         "fall_adl": ("fall", ADL)}


# ---------------------------------------------------------------- raw material for fold-fit features
def raw_material(seg):
    cache = "data/models_exp/eda_domain_v3/foldfit_material.npz"
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return z["deltas"].item(), z["profs"].item()
    deltas, profs = {"rx1": {}, "rx2": {}}, {"rx1": {}, "rx2": {}}
    for f in sorted(glob.glob("data/processed_v3/*/*_aligned.csv")):
        tid = os.path.basename(f).replace("_aligned.csv", "")
        if tid not in seg.index:
            continue
        df = pd.read_csv(f); t = df["time_rel_s"].values
        on, off = seg.loc[tid, "event_onset"], seg.loc[tid, "event_offset"]
        pre, post = (t >= on - 2.2) & (t <= on - 0.2), (t >= off + 0.2) & (t <= off + 2.2)
        for rx in ("rx1", "rx2"):
            X = df[[c for c in df.columns if c.startswith(f"{rx}_amp_")]].values
            if pre.any() and post.any():
                noise = np.sqrt((X[pre].var(0) + X[post].var(0)) / 2).mean() + 1e-9
                deltas[rx][tid] = (np.median(X[post], 0) - np.median(X[pre], 0)) / noise
            else:
                deltas[rx][tid] = np.zeros(X.shape[1])
            tc_, curve, _ = receiver_curve(t, X)
            cm = (tc_ >= on - 1.0) & (tc_ <= off + 2.0)
            cv = curve[cm] if cm.sum() >= 4 else curve
            p = np.interp(np.linspace(0, 1, 40), np.linspace(0, 1, len(cv)), cv)
            profs[rx][tid] = (p - p.mean()) / (p.std() + 1e-9)
    np.savez(cache, deltas=deltas, profs=profs)
    return deltas, profs


def dtw(a, b, band=8):
    n, m = len(a), len(b)
    D = np.full((n + 1, m + 1), np.inf); D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(max(1, i - band), min(m, i + band) + 1):
            D[i, j] = (a[i - 1] - b[j - 1]) ** 2 + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(np.sqrt(D[n, m]))


def foldfit(fit_ids, apply_ids, act, deltas, profs):
    """F1 + F4 values for apply_ids, learned from fit_ids only."""
    out = {t: {} for t in apply_ids}
    for rx in ("rx1", "rx2"):
        mean = lambda ids: np.mean([deltas[rx][t] for t in ids], 0) if ids else np.zeros_like(next(iter(deltas[rx].values())))
        d_ss = mean([t for t in fit_ids if act[t] == "sit"]) - mean([t for t in fit_ids if act[t] == "stand"])
        d_fa = mean([t for t in fit_ids if act[t] == "fall"]) - mean([t for t in fit_ids if act[t] in ADL])
        u_ss, u_fa = d_ss / (np.linalg.norm(d_ss) + 1e-9), d_fa / (np.linalg.norm(d_fa) + 1e-9)
        tmpl = {c: np.mean([profs[rx][t] for t in fit_ids if act[t] == c], 0)
                for c in set(act[t] for t in fit_ids)}
        for t in apply_ids:
            v = deltas[rx][t]
            out[t][f"F1_sitstand_proj_{rx}"] = float(v @ u_ss)
            out[t][f"F1_sitstand_cos_{rx}"] = float(v @ u_ss / (np.linalg.norm(v) + 1e-9))
            out[t][f"F1_fall_proj_{rx}"] = float(v @ u_fa)
            ds = {c: dtw(profs[rx][t], tm) for c, tm in tmpl.items()}
            for c, val in ds.items():
                out[t][f"F4_dtw_{c}_{rx}"] = val
            out[t][f"F4_dtw_fall_margin_{rx}"] = (min(v_ for c, v_ in ds.items() if c != "fall") - ds.get("fall", 0.0))
    return out


# ---------------------------------------------------------------- fold-internal acceptance + pruning
def eta2(x, g):
    x = np.asarray(x, float); m = x.mean(); sst = ((x - m) ** 2).sum() + 1e-12
    return sum((g == k).sum() * (x[g == k].mean() - m) ** 2 for k in np.unique(g)) / sst


def select(Ttr, new_cols, base_cols):
    y = Ttr.activity.values
    H = Ttr[Ttr.activity != "empty_room"]
    mi = dict(zip(new_cols, mutual_info_classif(np.nan_to_num(Ttr[new_cols].values.astype(float)), y, random_state=0)))
    subs_all = sorted(H.subject.unique())
    keep = []
    for f in new_cols:
        best, best_auc, best_dir = None, 0.5, 0
        for p, (a, bs) in PAIRS.items():
            d = Ttr[Ttr.activity.isin([a] + bs)]
            if (d.activity == a).nunique() < 2:
                continue
            v = roc_auc_score(d.activity == a, d[f])
            if max(v, 1 - v) > best_auc:
                best, best_auc, best_dir = p, max(v, 1 - v), np.sign(v - 0.5)
        if best is None or not ((best_auc >= 0.60) or (mi[f] >= 0.05)):
            continue
        if eta2(Ttr[f], y) <= eta2(H[f], H.subject.values):
            continue
        a, bs = PAIRS[best]
        dirs = []
        for s in subs_all:
            d = Ttr[Ttr.activity.isin([a] + bs)]
            d = d[(d.subject == s) | (d.activity == "empty_room")] if bs == ["empty_room"] else d[d.subject == s]
            if (d.activity == a).nunique() == 2:
                dirs.append(np.sign(roc_auc_score(d.activity == a, d[f]) - 0.5))
        if len(dirs) == 0 or np.sum(np.array(dirs) == best_dir) < len(dirs) - 1:
            continue
        keep.append((f, best_auc))
    # redundancy vs base, then greedy pruning among the new ones (best AUC first)
    R = Ttr[[k for k, _ in keep] + base_cols].rank()
    C = R.corr().abs()
    chosen = []
    for f, _ in sorted(keep, key=lambda kv: -kv[1]):
        if C.loc[f, base_cols].max() >= 0.85:
            continue
        if any(C.loc[f, g] >= 0.85 for g in chosen):
            continue
        chosen.append(f)
    return chosen


# ---------------------------------------------------------------- evaluation
def rf_proba(Xtr, ytr, Xte, seed):
    m = RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced", n_jobs=-1,
                               random_state=seed).fit(Xtr, ytr)
    p = np.zeros((len(Xte), len(CLASSES)))
    for j, c in enumerate(m.classes_):
        p[:, CLASSES.index(c)] = m.predict_proba(Xte)[:, j]
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=2)
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
    dropped = [c for c in base.columns if c not in ("activity", "subject") and c not in base_cols]
    T = T.sort_index()
    act = T.activity.to_dict()
    deltas, profs = raw_material(seg)
    trials = T.reset_index()[["trial_id", "activity", "subject"]]
    y, subj = trials.activity.values, trials.subject.values
    rx_of = lambda c: "rx1" if c.endswith("_rx1") or c.startswith(("ev_rx1_", "w1_")) else ("rx2" if c.endswith("_rx2") or c.startswith(("ev_rx2_", "w2_")) else "shared")
    print(f"  trials {len(T)} | base features {len(base_cols)} (dropped constant: {dropped}) | label-free new {len(labelfree)}")

    schemes = {"block (seen people)": ma.block_folds(trials), "LOSO (new person)": tc.loso_folds(trials)}
    results, selections = {}, {}
    for sname, folds in schemes.items():
        group = np.full(len(trials), -1)
        for k, (_, ids) in enumerate(folds):
            group[trials.trial_id.isin(ids).values] = k
        P = {v: {c: np.zeros((len(T), len(CLASSES))) for c in ("RX1", "RX2", "LATE_MEAN", "EARLY")}
             for v in ("BASE", "BASE+SEL", "BASE+ALL")}
        selections[sname] = []
        for seed in range(args.seeds):
            for k, (fname, test_ids) in enumerate(folds):
                te = group == k
                tr_ids, te_ids = trials.trial_id[~te].tolist(), trials.trial_id[te].tolist()
                # fold-fit features: test from full training fit; training via inner out-of-fold
                ff = foldfit(tr_ids, te_ids, act, deltas, profs)
                for g in np.unique(group[~te]):
                    inner_apply = trials.trial_id[group == g].tolist()
                    inner_fit = [t for t in tr_ids if t not in set(inner_apply)]
                    ff.update(foldfit(inner_fit, inner_apply, act, deltas, profs))
                FF = pd.DataFrame.from_dict(ff, orient="index").reindex(T.index)
                TT = T.join(FF)
                new_all = labelfree + list(FF.columns)
                Ttr = TT.loc[tr_ids]
                if seed == 0:
                    sel = select(Ttr, new_all, base_cols)
                    selections[sname].append({"fold": fname, "selected": sel})
                else:
                    sel = selections[sname][k]["selected"]
                sets = {"BASE": base_cols, "BASE+SEL": base_cols + sel, "BASE+ALL": base_cols + new_all}
                for v, cols in sets.items():
                    cols_rx = {r: [c for c in cols if rx_of(c) in (r, "shared")] for r in ("rx1", "rx2")}
                    X = lambda cs: np.nan_to_num(TT[cs].values.astype(float))
                    p1 = rf_proba(X(cols_rx["rx1"])[~te], y[~te], X(cols_rx["rx1"])[te], seed)
                    p2 = rf_proba(X(cols_rx["rx2"])[~te], y[~te], X(cols_rx["rx2"])[te], seed)
                    pe = rf_proba(X(cols)[~te], y[~te], X(cols)[te], seed)
                    for c, pp in (("RX1", p1), ("RX2", p2), ("LATE_MEAN", (p1 + p2) / 2), ("EARLY", pe)):
                        P[v][c][te] += pp / args.seeds
                print(f"  {sname} seed {seed} {fname}: {len(sel)} new features selected", flush=True)
        results[sname] = {v: {c: ma.metrics(y, P[v][c], subj) for c in P[v]} for v in P}
        # paired tests vs BASE (late fusion and early fusion)
        tid = trials.trial_id.values
        tests = []
        for v in ("BASE+SEL", "BASE+ALL"):
            for c in ("LATE_MEAN", "EARLY"):
                a = ({t: P[v][c][i] for i, t in enumerate(tid)}, {t: P[v][c][i, 0] for i, t in enumerate(tid)})
                b = ({t: P["BASE"][c][i] for i, t in enumerate(tid)}, {t: P["BASE"][c][i, 0] for i, t in enumerate(tid)})
                pa, pb = np.array(CLASSES)[P[v][c].argmax(1)], np.array(CLASSES)[P["BASE"][c].argmax(1)]
                tests.append({"variant": v, "fusion": c, **cr.mcnemar(y, pa, pb), **cr.cluster_bootstrap(trials, a, b, n=1500)})
        for t_, ph in zip(tests, cr.holm([t_["p"] for t_ in tests])):
            t_["p_holm"] = float(ph)
        results[sname]["tests"] = tests

    # ---- report
    for sname in schemes:
        print(f"\n  ===== {sname} =====")
        print(f"   {'set':<9} {'fusion':<9} {'acc':>6} {'mF1':>6} {'AUC':>6} {'sens@5%':>8}   F1: fall  walk  sit  stand hand  empty")
        for v in ("BASE", "BASE+SEL", "BASE+ALL"):
            for c in ("LATE_MEAN", "EARLY", "RX1"):
                m = results[sname][v][c]; pc = m["per_class"]
                print(f"   {v:<9} {c:<9} {m['accuracy']:6.3f} {m['macro_f1']:6.3f} {m['fall_auc']:6.3f} {m['fall_sens_at_far5']:8.3f}"
                      f"       {pc['fall']['f1']:.2f}  {pc['walk']['f1']:.2f}  {pc['sit']['f1']:.2f}  {pc['stand']['f1']:.2f}"
                      f"  {pc['hand']['f1']:.2f}  {pc['empty_room']['f1']:.2f}")
        for t_ in results[sname]["tests"]:
            print(f"   {t_['variant']} vs BASE ({t_['fusion']}): better/worse {t_['a_right_b_wrong']}/{t_['a_wrong_b_right']} "
                  f"p_holm={t_['p_holm']:.3f} | dF1 [{t_['d_macro_f1_ci'][0]:+.3f},{t_['d_macro_f1_ci'][1]:+.3f}] "
                  f"dAUC [{t_['d_fall_auc_ci'][0]:+.3f},{t_['d_fall_auc_ci'][1]:+.3f}]")
        sel_counts = pd.Series([f for s in selections[sname] for f in s["selected"]]).value_counts()
        print(f"   features selected in every fold ({len(selections[sname])}): "
              f"{sel_counts[sel_counts == len(selections[sname])].index.tolist()}")
    with open(os.path.join(OUT, "feature_validation.json"), "w") as fh:
        json.dump({"results": results, "selections": selections, "dropped_constant": dropped}, fh, indent=2, default=float)
    print(f"\n  Saved to {OUT}/")


if __name__ == "__main__":
    main()
