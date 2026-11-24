"""
Which part of the fused localiser hurt classification? Swap one component at a time.
  A1 old event features + old per-receiver window aggregates (each receiver centred on ITS OWN peak)
  A2 fused segment event features + old per-receiver aggregates
  A3 old event features + aggregates over the shared fused segment
  A4 union: A1 + fused segment event features and localisation-quality features (RF chooses)
Random Forest, late fusion (mean) and RX1, both validation levels, 2 seeds.
"""

import numpy as np
import pandas as pd
import train_classifier as tc
import evaluate_event_model as em
import model_audit as ma
import compare_receivers as cr
from evaluate_localizer import segment_aggregates, run

CLASSES = tc.CLASSES
old = pd.read_csv("data/event_features_v3.csv", dtype={"subject": str}).set_index("trial_id")
new = pd.read_csv("data/event_features_v3_fused.csv", dtype={"subject": str}).set_index("trial_id")
agg_old = em.window_aggregates("data/features_v3_rx1", "w1").join(em.window_aggregates("data/features_v3_rx2", "w2"))
agg_seg = segment_aggregates("data/features_v3_rx1", "w1", new).join(segment_aggregates("data/features_v3_rx2", "w2", new))
meta = old[["activity", "subject"]]
evc = lambda d: d[[c for c in d.columns if c.startswith("ev_rx")]]
newf = new[[c for c in new.columns if c.startswith(("ev_rx", "loc_"))]].add_prefix("f_")
newf.columns = [c.replace("f_ev_rx1_", "ev_rx1_fz_").replace("f_ev_rx2_", "ev_rx2_fz_").replace("f_loc_", "loc_")
                for c in newf.columns]
variants = {"A1 old+old": meta.join(evc(old)).join(agg_old),
            "A2 fusedEV+oldAGG": meta.join(evc(new)).join(agg_old),
            "A3 oldEV+segAGG": meta.join(evc(old)).join(agg_seg),
            "A4 union": meta.join(evc(old)).join(agg_old).join(newf)}
common = sorted(set.intersection(*[set(v.dropna().index) for v in variants.values()]))
for k in variants:
    variants[k] = variants[k].loc[common].reset_index().rename(columns={"index": "trial_id"})
trials = variants["A1 old+old"][["trial_id", "activity", "subject"]]
y, subj = trials.activity.values, trials.subject.values
for sname, folds in {"block (seen)": ma.block_folds(trials), "LOSO (new person)": tc.loso_folds(trials)}.items():
    print(f"\n  {sname}")
    base = None
    for k, T in variants.items():
        P = run(T, folds, 2)
        for cfg in ("LATE_MEAN", "RX1"):
            m = ma.metrics(y, P[cfg], subj)
            line = (f"   {k:<18} {cfg:<9} acc {m['accuracy']:.3f}  mF1 {m['macro_f1']:.3f}  AUC {m['fall_auc']:.3f}  "
                    f"sens@5% {m['fall_sens_at_far5']:.3f}  F1 fall/sit/stand {m['per_class']['fall']['f1']:.2f}/"
                    f"{m['per_class']['sit']['f1']:.2f}/{m['per_class']['stand']['f1']:.2f}")
            if cfg == "LATE_MEAN":
                if k.startswith("A1"):
                    base = P[cfg]
                else:
                    tid = trials.trial_id.values
                    a = ({t: P[cfg][i] for i, t in enumerate(tid)}, {t: P[cfg][i, 0] for i, t in enumerate(tid)})
                    b = ({t: base[i] for i, t in enumerate(tid)}, {t: base[i, 0] for i, t in enumerate(tid)})
                    bs = cr.cluster_bootstrap(trials, a, b, n=1000)
                    line += f"   vs A1 dF1 [{bs['d_macro_f1_ci'][0]:+.3f},{bs['d_macro_f1_ci'][1]:+.3f}]"
            print(line, flush=True)
