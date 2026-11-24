"""
Acceptance test for the domain-specific features (reads the cached data/models_exp/eda_domain_v3/domain_features.csv).
Fixes vs the first pass: per-subject consistency for pairs involving empty_room compares each subject's trials
with the (shared) empty-room trials, and fall-vs-ADL is evaluated as a binary split per subject.
Criteria: target-pair AUC >= 0.60 (direction-free; signed for fold-fit features) or MI >= 0.05,
eta2(activity) > eta2(subject), direction consistent for >= 4/5 subjects (>= 3/4 for fall pairs),
max |Spearman| with the current feature set < 0.85.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score
import evaluate_event_model as em

OUT = "data/models_exp/eda_domain_v3"
ADL = ["walk", "sit", "stand", "hand"]
PAIRS = {"sit_stand": ("sit", ["stand"]), "hand_empty": ("hand", ["empty_room"]), "fall_sit": ("fall", ["sit"]),
         "fall_adl": ("fall", ADL)}
SIGNED = {"F1_sitstand": "sit_stand", "F1_fall": "fall_adl", "F4_dtw_fall_margin": "fall_adl"}

D = pd.read_csv(os.path.join(OUT, "domain_features.csv"), dtype={"subject": str}).set_index("trial_id")
feats = [c for c in D.columns if c.startswith("F")]
D = D.replace([np.inf, -np.inf], np.nan)
D[feats] = D[feats].fillna(D[feats].median())
new = pd.read_csv("data/event_features_v3_fused.csv", dtype={"subject": str}).set_index("trial_id")
agg = em.window_aggregates("data/features_v3_rx1", "w1").join(em.window_aggregates("data/features_v3_rx2", "w2"))
E = new[[c for c in new.columns if c.startswith("ev_rx")]].join(agg)
E = E[[c for c in E.columns if E[c].nunique() > 1]].reindex(D.index)
y = D.activity.values
H = D[D.activity != "empty_room"]
mi = dict(zip(feats, mutual_info_classif(D[feats].values, y, random_state=0)))
Er, Dr = E.rank(), D[feats].rank()


def eta2(x, g):
    x = np.asarray(x, float); m = x.mean(); sst = ((x - m) ** 2).sum() + 1e-12
    return sum((g == k).sum() * (x[g == k].mean() - m) ** 2 for k in np.unique(g)) / sst


def pair_data(pair, subject=None):
    a, bs = PAIRS[pair]
    d = D[D.activity.isin([a] + bs)]
    if subject is not None:
        if bs == ["empty_room"]:              # empty room has no human subject: compare with all empty trials
            d = d[(d.subject == subject) | (d.activity == "empty_room")]
        else:
            d = d[d.subject == subject]
    return d, (d.activity == a)


def auc(pair, f, signed, subject=None):
    d, lab = pair_data(pair, subject)
    if lab.nunique() < 2:
        return np.nan
    v = roc_auc_score(lab, d[f])
    return v if signed else max(v, 1 - v)


rows = []
for f in feats:
    sp = next((v for k, v in SIGNED.items() if f.startswith(k)), None)
    aucs = {p: auc(p, f, sp is not None) for p in PAIRS}
    best = sp or max(aucs, key=lambda p: aucs[p])
    d, lab = pair_data(best)
    pooled_dir = np.sign(roc_auc_score(lab, d[f]) - 0.5)
    subs = ["1", "2", "4", "5"] if best.startswith("fall") else ["1", "2", "3", "4", "5"]
    dirs = [np.sign(roc_auc_score(*(lambda dd, ll: (ll, dd[f]))(*pair_data(best, s))) - 0.5)
            for s in subs if pair_data(best, s)[1].nunique() == 2]
    consistent = int(np.sum(np.array(dirs) == pooled_dir))
    need = 3 if best.startswith("fall") else 4
    red = float(np.nanmax(np.abs(np.corrcoef(Dr[f].values, Er.values.T)[0, 1:])))
    e_a, e_s = eta2(D[f], y), eta2(H[f], H.subject.values)
    checks = {"separates": (aucs[best] >= 0.60) or (mi[f] >= 0.05), "activity>subject": e_a > e_s,
              "consistent": consistent >= need, "not_redundant": red < 0.85}
    rows.append({"feature": f, "family": f.split("_")[0], "best_pair": best, "auc_best": aucs[best],
                 **{f"auc_{p}": aucs[p] for p in PAIRS}, "MI": mi[f], "eta2_act": e_a, "eta2_subj": e_s,
                 "consistent": f"{consistent}/{len(dirs)}", "max_rho_existing": red, "PASS": all(checks.values()),
                 "fails": ",".join(k for k, v in checks.items() if not v)})
A = pd.DataFrame(rows)
A.to_csv(os.path.join(OUT, "acceptance_test.csv"), index=False)

ref = {}
EJ = E.join(D[["activity", "subject"]])
for p in PAIRS:
    a, bs = PAIRS[p]
    d = EJ[EJ.activity.isin([a] + bs)]
    vals = [(c, max(roc_auc_score(d.activity == a, d[c]), 1 - roc_auc_score(d.activity == a, d[c]))) for c in E.columns]
    ref[p] = max(vals, key=lambda kv: kv[1])

pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
P = A[A.PASS].sort_values("auc_best", ascending=False)
print(f"\n  PASSED {len(P)} of {len(A)} features")
print(P[["feature", "best_pair", "auc_best", "MI", "eta2_act", "eta2_subj", "consistent", "max_rho_existing"]].round(3).to_string(index=False))
print("\n  Failed, with reasons (top 15 by AUC):")
print(A[~A.PASS].sort_values("auc_best", ascending=False).head(15)[["feature", "best_pair", "auc_best", "consistent",
                                                                     "eta2_act", "eta2_subj", "max_rho_existing", "fails"]].round(3).to_string(index=False))
print("\n  Per family passed/total:", A.groupby("family").PASS.agg(["sum", "size"]).apply(lambda r: f"{r['sum']}/{r['size']}", axis=1).to_dict())
print("\n  Best per hard pair: new (passing) vs existing")
for p in PAIRS:
    top = A[A.PASS].sort_values(f"auc_{p}", ascending=False).head(3)
    print(f"   {p:<11} existing {ref[p][0]} {ref[p][1]:.3f} | new: " +
          ", ".join(f"{r.feature} {r[f'auc_{p}']:.3f}" for _, r in top.iterrows()))
with open(os.path.join(OUT, "acceptance_summary.json"), "w") as fh:
    json.dump({"passed": P.feature.tolist(), "best_existing": {p: [c, float(v)] for p, (c, v) in ref.items()}}, fh, indent=2)
