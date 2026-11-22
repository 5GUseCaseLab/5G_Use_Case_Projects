"""
WiFeye: bottom-up feature EDA (no model training)
======================================================
Part 1 - audit of the current trial-level features (A2: fused-localiser event features + per-receiver
         window aggregates):
  * class separability per feature: eta^2 (activity), mutual information, Kruskal-Wallis
  * hard pairs: per-feature AUC for sit-vs-stand, hand-vs-empty, fall-vs-sit, fall-vs-ADL
  * subject dependence: eta^2 (subject) and cross-subject consistency of the fall-vs-ADL effect
  * redundancy: Spearman correlation clusters (|rho| > 0.85) and PCA effective dimension
Part 2 - raw-signal EDA on the localised event segment (to decide which new feature families to build):
  * band energy of the first principal component of the subcarrier amplitudes, from the ALIGNED
    (not yet low-passed) 100 Hz data: 0.3-2 / 2-5 / 5-10 / 10-20 / 20-35 Hz, by class
    -> is there class information above the 10 Hz low-pass?
  * PCA concentration: share of subcarrier variance in PC1 during the event, by class
  * Hilbert envelope of PC1: envelope peak and its asymmetry, by class
Outputs in data/models_exp/eda_v3/.  Usage: python eda_features.py
"""

import os
import glob
import json
import numpy as np
import pandas as pd
from scipy.stats import kruskal
from scipy.signal import welch, hilbert, butter, filtfilt
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import evaluate_event_model as em

OUT = "data/models_exp/eda_v3"
os.makedirs(OUT, exist_ok=True)
ADL = ["walk", "sit", "stand", "hand"]

# ------------------------------------------------------------------ Part 1
new = pd.read_csv("data/event_features_v3_fused.csv", dtype={"subject": str}).set_index("trial_id")
agg = em.window_aggregates("data/features_v3_rx1", "w1").join(em.window_aggregates("data/features_v3_rx2", "w2"))
T = new[["activity", "subject"] + [c for c in new.columns if c.startswith("ev_rx")]].join(agg).dropna()
feats = [c for c in T.columns if c not in ("activity", "subject")]
const = [c for c in feats if T[c].nunique() <= 1]           # e.g. spatial_cross_corr in single-receiver sets
feats = [c for c in feats if c not in const]
y = T.activity.values
H = T[T.activity != "empty_room"]


def eta2(x, g):
    x = np.asarray(x, float); m = x.mean(); sst = ((x - m) ** 2).sum() + 1e-12
    return sum((g == k).sum() * (x[g == k].mean() - m) ** 2 for k in np.unique(g)) / sst


def pair_auc(f, a, b, data=T):
    """Direction-free AUC of one feature for class a vs class b (0.5 = useless, 1 = perfect)."""
    d = data[data.activity.isin([a, b])]
    v = roc_auc_score(d.activity == a, d[f])
    return max(v, 1 - v)


mi = mutual_info_classif(np.nan_to_num(T[feats].values.astype(float)), y, random_state=0)
rows = []
for i, f in enumerate(feats):
    groups = [T.loc[T.activity == c, f].values for c in np.unique(y)]
    fall_sub = []
    for s in ["1", "2", "4", "5"]:
        d = H[H.subject == s]
        fall_sub.append(roc_auc_score(d.activity == "fall", d[f]))
    signs = np.sign(np.array(fall_sub) - 0.5)
    rows.append({"feature": f, "eta2_activity": eta2(T[f], y), "eta2_subject": eta2(H[f], H.subject.values),
                 "MI": mi[i], "kruskal_H": kruskal(*groups).statistic,
                 "auc_sit_stand": pair_auc(f, "sit", "stand"), "auc_hand_empty": pair_auc(f, "hand", "empty_room"),
                 "auc_fall_sit": pair_auc(f, "fall", "sit"),
                 "auc_fall_adl": max(roc_auc_score(H.activity == "fall", H[f]), 1 - roc_auc_score(H.activity == "fall", H[f])),
                 "fall_effect_consistent_subjects": int(max((signs > 0).sum(), (signs < 0).sum()))})
F = pd.DataFrame(rows).sort_values("MI", ascending=False)
F.to_csv(os.path.join(OUT, "feature_audit.csv"), index=False)

Zs = T[feats].rank().values                                   # Spearman via ranks
C = np.nan_to_num(np.corrcoef(Zs, rowvar=False))
cl = fcluster(linkage(1 - np.abs(C[np.triu_indices(len(feats), 1)]), method="average"), t=0.15, criterion="distance")
pca = PCA().fit(StandardScaler().fit_transform(T[feats].values.astype(float)))
ev90 = int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), 0.90) + 1)

part1 = {
    "n_features": len(feats), "constant_features_dropped": const,
    "median_eta2_activity": float(F.eta2_activity.median()), "median_eta2_subject": float(F.eta2_subject.median()),
    "share_features_MI_below_0.05": float((F.MI < 0.05).mean()),
    "best_auc_sit_stand": F.sort_values("auc_sit_stand", ascending=False).head(3)[["feature", "auc_sit_stand"]].round(3).values.tolist(),
    "best_auc_hand_empty": F.sort_values("auc_hand_empty", ascending=False).head(3)[["feature", "auc_hand_empty"]].round(3).values.tolist(),
    "best_auc_fall_sit": F.sort_values("auc_fall_sit", ascending=False).head(3)[["feature", "auc_fall_sit"]].round(3).values.tolist(),
    "best_auc_fall_adl": F.sort_values("auc_fall_adl", ascending=False).head(3)[["feature", "auc_fall_adl"]].round(3).values.tolist(),
    "top_MI": F.head(8)[["feature", "MI", "eta2_activity", "eta2_subject"]].round(3).values.tolist(),
    "share_fall_effect_consistent_all_4_subjects_among_auc>0.65": float(
        (F[F.auc_fall_adl > 0.65].fall_effect_consistent_subjects == 4).mean()),
    "redundancy_clusters_at_rho_0.85": int(len(np.unique(cl))),
    "pca_components_for_90pct_variance": ev90,
}

# ------------------------------------------------------------------ Part 2 (raw signal on the event segment)
bands = [(0.3, 2), (2, 5), (5, 10), (10, 20), (20, 35)]
b_lp, a_lp = butter(4, 10 / 50, btype="low")
r2 = []
cache = os.path.join(OUT, "raw_segment_eda.csv")
files = [] if (os.path.exists(cache) and not os.environ.get("EDA_FORCE")) else sorted(glob.glob("data/aligned_v3/*/*_aligned.csv"))
for f in files:
    tid = os.path.basename(f).replace("_aligned.csv", "")
    if tid not in new.index:
        continue
    on, off = new.loc[tid, "event_onset"], new.loc[tid, "event_offset"]
    df = pd.read_csv(f)
    t = df["time_rel_s"].values
    lo, hi = max(0, on - 0.5), min(t[-1], max(off + 0.5, on + 2.0))
    m = (t >= lo) & (t <= hi)
    row = {"trial_id": tid, "activity": new.loc[tid, "activity"]}
    for rx in ("rx1", "rx2"):
        X = df[[c for c in df.columns if c.startswith(f"{rx}_amp_")]].values
        X = X - np.median(X, axis=0)
        p = PCA(n_components=3).fit(X[m])
        pc1 = (X[m] - X[m].mean(0)) @ p.components_[0]
        fr, ps = welch(pc1, fs=100, nperseg=min(128, len(pc1)))
        tot = ps[(fr >= 0.3) & (fr <= 35)].sum() + 1e-12
        for lo_b, hi_b in bands:
            row[f"{rx}_band_{lo_b}_{hi_b}"] = ps[(fr >= lo_b) & (fr < hi_b)].sum() / tot
        row[f"{rx}_pc1_share"] = float(p.explained_variance_ratio_[0])
        env = np.abs(hilbert(filtfilt(b_lp, a_lp, pc1) if len(pc1) > 30 else pc1))
        ipk = int(np.argmax(env))
        row[f"{rx}_env_peak_over_median"] = float(env.max() / (np.median(env) + 1e-9))
        row[f"{rx}_env_peak_position"] = float(ipk / max(len(env) - 1, 1))   # 0 = start, 1 = end of segment
    r2.append(row)
R2 = pd.DataFrame(r2) if r2 else pd.read_csv(cache)
R2.to_csv(cache, index=False)
cols2 = [c for c in R2.columns if c.startswith("rx1_")]
med = R2.groupby("activity")[cols2].median().round(3)
hi_band = {}
for rx in ("rx1", "rx2"):
    for b in ("10_20", "20_35"):
        f_ = f"{rx}_band_{b}"
        H2 = R2[R2.activity != "empty_room"]
        a = roc_auc_score(H2.activity == "fall", H2[f_])
        hi_band[f"{f_}_auc_fall_vs_adl"] = float(max(a, 1 - a))
        a = roc_auc_score(R2[R2.activity.isin(["sit", "stand"])].activity == "sit",
                          R2[R2.activity.isin(["sit", "stand"])][f_])
        hi_band[f"{f_}_auc_sit_vs_stand"] = float(max(a, 1 - a))
extra = {}
for rx in ("rx1", "rx2"):
    for f_ in (f"{rx}_env_peak_position", f"{rx}_pc1_share", f"{rx}_band_0.3_2"):
        for a_, b_ in (("sit", "stand"), ("hand", "empty_room"), ("fall", "sit")):
            d = R2[R2.activity.isin([a_, b_])]
            v = roc_auc_score(d.activity == a_, d[f_])
            extra[f"{f_}:{a_}_vs_{b_}"] = float(max(v, 1 - v))
# is the most informative feature (absolute quiet level) an activity cue or a recording-session cue?
ln = T[["activity", "subject", "ev_rx1_log_noise"]].copy()
tn = ln.index.str.extract(r"_t(\d+)$")[0].astype(int).values
ln["block"] = np.where(ln.activity == "empty_room", "6_" + pd.Series(np.where(tn <= 10, "b", np.where(tn <= 20, "c", "m"))).values,
                       ln.subject + "_" + ln.activity)
ln["chunk"] = ln.groupby("block").cumcount() // 4
within_block_sd = ln.groupby("block")["ev_rx1_log_noise"].std().median()
between_block_sd = ln.groupby("block")["ev_rx1_log_noise"].mean().std()
# drift inside a block (first vs last 4 trials) indicates session drift rather than activity
drift = ln.groupby("block").apply(lambda g: g[g.chunk == g.chunk.max()]["ev_rx1_log_noise"].mean()
                                  - g[g.chunk == 0]["ev_rx1_log_noise"].mean()).abs().median()
extra["log_noise_by_class_median"] = ln.groupby("activity")["ev_rx1_log_noise"].median().round(3).to_dict()
extra["log_noise_within_block_sd"] = float(within_block_sd)
extra["log_noise_between_block_sd"] = float(between_block_sd)
extra["log_noise_median_abs_drift_first_vs_last_chunk"] = float(drift)
part2 = {"median_by_class_rx1": med.to_dict(), "above_10Hz_discriminability": hi_band, "extra": extra}
print("\nextra checks:", json.dumps(extra, indent=1, default=float))
res = {"part1_feature_audit": part1, "part2_raw_segment": part2}
with open(os.path.join(OUT, "eda_summary.json"), "w") as fh:
    json.dump(res, fh, indent=2, default=float)
print(json.dumps(part1, indent=2, default=float))
print("\nPart 2: medians by class (RX1, event segment of the aligned, not-yet-low-passed data)")
print(med.T.to_string())
print("\nabove-10 Hz band discriminability (AUC, 0.5 = none):", json.dumps({k: round(v, 3) for k, v in hi_band.items()}))
