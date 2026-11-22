"""
WiFeye: why is accuracy ~55-63 % even for seen people? Fast diagnostics (minutes)
======================================================================================
D1  Nearest-neighbour + clustering structure of the trial-level hybrid features:
    does a trial's nearest neighbour share its ACTIVITY, its SUBJECT, or its recording BLOCK?
    KMeans cluster agreement (ARI / NMI) with activity vs subject; silhouette by each label.
D2  Variance decomposition per feature (eta^2): share explained by activity vs by subject.
D3  RX1/RX2 synchronisation: lag between the two receivers' motion curves on strong events.
D4  Event-detection reliability: do RX1 and RX2 independently put the event at the same time?
D5  Per-subject feature normalisation (unsupervised, transductive) under LOSO, human classes
    only: a cheap upper estimate of what removing subject-specific feature offsets could gain
    (the goal of DANN / domain alignment).
Usage: python diagnostics_ceiling.py
"""

import os
import glob
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score, f1_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier

import train_classifier as tc
import evaluate_event_model as em
from event_features import motion_curve

OUT = "data/models_exp/diagnostics_v3"
os.makedirs(OUT, exist_ok=True)
res = {}

ev = pd.read_csv("data/event_features_v3.csv", dtype={"subject": str})
T = ev.set_index("trial_id").join(em.window_aggregates("data/features_v3_rx1", "w1")) \
      .join(em.window_aggregates("data/features_v3_rx2", "w2")).reset_index().dropna()
T = T.sort_values("trial_id").reset_index(drop=True)
col = lambda pre: [c for c in T.columns if c.startswith(pre)]
feats = col("ev_rx1_") + col("w1_") + col("ev_rx2_") + col("w2_")
tnum = T["trial_id"].str.extract(r"_t(\d+)$")[0].astype(int)
T["block"] = np.where(T.activity == "empty_room", "6_" + pd.Series(np.where(tnum <= 10, "bare", np.where(tnum <= 20, "chair", "mattress"))),
                      T.subject + "_" + T.activity)
T["chunk"] = T.groupby("block").cumcount() // 4
H = T[T.activity != "empty_room"].reset_index(drop=True)          # humans only for subject analyses
Z = StandardScaler().fit_transform(np.nan_to_num(H[feats].values.astype(float)))
Zp = PCA(n_components=15, random_state=0).fit_transform(Z)

# ---- D1 nearest neighbours / clustering
nn = NearestNeighbors(n_neighbors=2).fit(Zp)
idx = nn.kneighbors(Zp, return_distance=False)[:, 1]
same_act = (H.activity.values[idx] == H.activity.values)
same_sub = (H.subject.values[idx] == H.subject.values)
same_blk = (H.block.values[idx] == H.block.values)
res["D1_nn"] = {"same_activity": float(same_act.mean()), "same_subject": float(same_sub.mean()),
                "same_block": float(same_blk.mean()),
                "same_subject_different_activity": float((same_sub & ~same_act).mean()),
                "chance_same_activity": float(sum((H.activity.value_counts() / len(H)) ** 2)),
                "chance_same_subject": float(sum((H.subject.value_counts() / len(H)) ** 2))}
km = KMeans(n_clusters=5, n_init=10, random_state=0).fit(Zp)
res["D1_cluster"] = {"ARI_vs_activity": float(adjusted_rand_score(H.activity, km.labels_)),
                     "ARI_vs_subject": float(adjusted_rand_score(H.subject, km.labels_)),
                     "NMI_vs_activity": float(normalized_mutual_info_score(H.activity, km.labels_)),
                     "NMI_vs_subject": float(normalized_mutual_info_score(H.subject, km.labels_)),
                     "silhouette_by_activity": float(silhouette_score(Zp, H.activity)),
                     "silhouette_by_subject": float(silhouette_score(Zp, H.subject))}

# ---- D2 eta^2 per feature
def eta2(x, g):
    x = np.asarray(x, float); m = x.mean()
    ss_t = ((x - m) ** 2).sum() + 1e-12
    return sum(len(x[g == k]) * (x[g == k].mean() - m) ** 2 for k in np.unique(g)) / ss_t
e = pd.DataFrame({"feature": feats,
                  "eta2_activity": [eta2(H[f], H.activity.values) for f in feats],
                  "eta2_subject": [eta2(H[f], H.subject.values) for f in feats]})
e["subject_dominates"] = e.eta2_subject > e.eta2_activity
e.to_csv(os.path.join(OUT, "eta2_features.csv"), index=False)
res["D2"] = {"median_eta2_activity": float(e.eta2_activity.median()), "median_eta2_subject": float(e.eta2_subject.median()),
             "share_features_subject_dominates": float(e.subject_dominates.mean()),
             "top_activity_features": e.sort_values("eta2_activity", ascending=False).head(5)[["feature", "eta2_activity", "eta2_subject"]].round(3).values.tolist(),
             "top_subject_features": e.sort_values("eta2_subject", ascending=False).head(5)[["feature", "eta2_activity", "eta2_subject"]].round(3).values.tolist()}

# ---- D3 / D4 synchronisation and event agreement on strong-motion trials
lags, ev_dt = [], []
files = [f for a in ("fall", "walk", "sit", "stand") for f in sorted(glob.glob(f"data/processed_v3/{a}/*_aligned.csv"))]
for f in files:
    df = pd.read_csv(f)
    t = df["time_rel_s"].values
    c = {}
    for rx in ("rx1", "rx2"):
        cols = [k for k in df.columns if k.startswith(f"{rx}_amp_")]
        tc_, m = motion_curve(t, df[cols].values)
        c[rx] = (tc_, m / (np.percentile(m, 20) + 1e-9))
    (t1, m1), (t2, m2) = c["rx1"], c["rx2"]
    n = min(len(m1), len(m2))
    a, b = m1[:n] - m1[:n].mean(), m2[:n] - m2[:n].mean()
    L = range(-10, 11)                                       # +-1 s at 0.1 s hop
    cc = [np.dot(a[max(0, -k):n - max(0, k)], b[max(0, k):n - max(0, -k)]) for k in L]
    if max(m1[:n].max(), m2[:n].max()) > 3:                  # only clearly visible events
        lags.append(list(L)[int(np.argmax(cc))] * 0.1)
        ok = (t1[:n] >= 1) & (t1[:n] <= 8.5)
        ev_dt.append(abs(t1[:n][ok][np.argmax(m1[:n][ok])] - t2[:n][ok][np.argmax(m2[:n][ok])]))
lags, ev_dt = np.array(lags), np.array(ev_dt)
res["D3_sync"] = {"n_trials": int(len(lags)), "median_lag_s": float(np.median(lags)),
                  "share_abs_lag_le_0.2s": float(np.mean(np.abs(lags) <= 0.2)), "iqr_lag_s": [float(np.percentile(lags, 25)), float(np.percentile(lags, 75))]}
res["D4_event_agreement"] = {"median_abs_dt_s": float(np.median(ev_dt)), "share_within_0.5s": float(np.mean(ev_dt <= 0.5)),
                             "share_over_1s": float(np.mean(ev_dt > 1.0))}

# ---- D5 per-subject normalisation under LOSO (human classes only)
folds = [(f, ids) for f, ids in tc.loso_folds(T) ]
Hs = H.copy()
def loso(Xdf, label):
    y = Hs.activity.values; pred = np.empty(len(Hs), object); pf = np.zeros(len(Hs))
    for s in ["1", "2", "3", "4", "5"]:
        te = (Hs.subject == s).values
        p = []
        for cols in (col("ev_rx1_") + col("w1_"), col("ev_rx2_") + col("w2_")):
            clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced", n_jobs=-1,
                                         random_state=0).fit(Xdf.loc[~te, cols], y[~te])
            pr = clf.predict_proba(Xdf.loc[te, cols]); p.append(pd.DataFrame(pr, columns=clf.classes_))
        pm = (p[0] + p[1]) / 2
        pred[te] = pm.idxmax(axis=1).values; pf[te] = pm["fall"].values
    return {"macro_f1": float(f1_score(y, pred, average="macro")), "accuracy": float(np.mean(pred == y)),
            "fall_auc": float(roc_auc_score(y == "fall", pf))}
raw = Hs[feats].astype(float)
norm = raw.groupby(Hs.subject).transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
res["D5_per_subject_norm"] = {"raw": loso(raw, "raw"), "per_subject_z": loso(norm, "z")}

with open(os.path.join(OUT, "diagnostics.json"), "w") as fh:
    json.dump(res, fh, indent=2)
print(json.dumps(res, indent=2))
