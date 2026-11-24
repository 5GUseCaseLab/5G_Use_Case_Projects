"""
WiFeye: bottom-up EDA of CSI domain-specific features (no classifiers)
===========================================================================
Families (per receiver unless noted; computed on the fused-localiser event segment):
  F1  signed posture-change signature (FOLD-FIT, out-of-fold under LOSO): projection / cosine of the
      before->after static-level change vector (52 subcarriers) onto the training-subject sit-vs-stand
      and fall-vs-ADL directions
  F2  CARM-style speed spectrogram of the top-3 PCA components: band fractions 0.3-1.5 / 1.5-3 / 3-5 Hz at
      the event, peak "speed" frequency, centroid at peak and centroid slope (accelerating vs decelerating)
  F3  power burst curve (FallDeFi) and sharp decline (RT-Fall): burst ratio, rise rate, decline rate,
      post/pre power
  F4  DTW template distance (DeFall-style, FOLD-FIT): distance of the event motion profile to each class
      template built from training subjects only
  F5  autocorrelation: decorrelation lag (speed proxy), ACF at 0.5 s, periodicity peak height and period
  F6  Haar wavelet energies of PC1 (levels ~3-6 Hz, 1.6-3 Hz, 0.8-1.6 Hz, approximation <0.8 Hz)
  F7  subcarrier structure: PC1 share, eigenvalue entropy, components for 90 %, mean |inter-subcarrier r|
  F8  classic WiFall statistics on PC1: MAD, IQR, p90-p10, mean |derivative|, zero-crossing rate
  F9  Fresnel fringe rate: local extrema per second of single-subcarrier amplitude (median over subcarriers)
  F10 receiver geometry (trial-level): localiser weight of RX1, log peak ratio, burst time offset RX1-RX2
  F11 single-antenna phase and RSSI: temporal std of sanitised phase (event / pre), RSSI std, RSSI change
Acceptance test per feature:
  target-pair AUC >= 0.60 (direction-free for label-free features, signed for fold-fit ones) or MI >= 0.05,
  eta2(activity) > eta2(subject), effect direction consistent for >= 4 of 5 subjects (>= 3 of 4 for fall
  pairs), and not redundant with the current feature set (max |Spearman| < 0.85).
Usage: python eda_domain_features.py
"""

import os
import glob
import json
import numpy as np
import pandas as pd
from scipy.signal import stft
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score

import train_classifier as tc
import evaluate_event_model as em
from event_localizer import receiver_curve

OUT = "data/models_exp/eda_domain_v3"
os.makedirs(OUT, exist_ok=True)
FS = 100.0
ADL = ["walk", "sit", "stand", "hand"]
seg = pd.read_csv("data/event_features_v3_fused.csv", dtype={"subject": str}).set_index("trial_id")


def win(t, lo, hi):
    return (t >= lo) & (t <= hi)


def haar_energies(x, levels=6):
    a, e = np.asarray(x, float), []
    for _ in range(levels):
        if len(a) < 2:
            e.append(0.0); continue
        a = a[: len(a) // 2 * 2]
        d = (a[0::2] - a[1::2]) / np.sqrt(2); a = (a[0::2] + a[1::2]) / np.sqrt(2)
        e.append(float(np.sum(d ** 2)))
    e.append(float(np.sum(a ** 2)))
    tot = sum(e) + 1e-12
    return [v / tot for v in e]          # [25-50, 12.5-25, 6-12.5, 3-6, 1.6-3, 0.8-1.6 Hz, approx <0.8 Hz]


def rx_features(t, X, P, rssi, ev, rx):
    on, off, pk = ev["event_onset"], ev["event_offset"], ev["event_t"]
    lo, hi = max(0.0, on - 0.5), min(t[-1], max(off + 0.5, on + 2.0))
    m = win(t, lo, hi)
    pre, post = win(t, on - 2.2, on - 0.2), win(t, off + 0.2, off + 2.2)
    f = {}
    Xs = X[m] - X[m].mean(0)
    pca = PCA(n_components=min(10, Xs.shape[1])).fit(Xs)
    ev_ = pca.explained_variance_ratio_
    pcs = Xs @ pca.components_[:3].T
    pc1 = pcs[:, 0]
    # F7 subcarrier structure
    p = ev_ / ev_.sum()
    f["F7_pc1_share"] = float(ev_[0])
    f["F7_eig_entropy"] = float(-(p * np.log(p + 1e-12)).sum() / np.log(len(p)))
    f["F7_n90"] = float(np.searchsorted(np.cumsum(ev_), 0.9) + 1)
    C = np.corrcoef(X[m].T)
    f["F7_mean_abs_r"] = float(np.nanmean(np.abs(C[np.triu_indices_from(C, 1)])))
    # F2 speed spectrogram on the full trial (top-3 PCs projected), evaluated around the event
    full = (X - X[m].mean(0)) @ pca.components_[:3].T
    fr, tt, Z = stft(full.T, fs=FS, nperseg=128, noverlap=118, boundary=None)
    S = (np.abs(Z) ** 2).sum(axis=0)                          # (freq, time) summed over the 3 PCs
    tt = tt + t[0]
    band = lambda a, b: S[(fr >= a) & (fr < b)].sum(0)
    b1, b2, b3 = band(0.3, 1.5), band(1.5, 3.0), band(3.0, 5.0)
    tot = b1 + b2 + b3 + 1e-12
    ie = win(tt, on - 0.3, off + 0.3)
    if not ie.any():
        ie = np.abs(tt - pk) == np.abs(tt - pk).min()
    ipk = np.where(ie)[0][np.argmax(tot[ie])]
    f["F2_frac_0.3_1.5"] = float(b1[ipk] / tot[ipk]); f["F2_frac_1.5_3"] = float(b2[ipk] / tot[ipk])
    f["F2_frac_3_5"] = float(b3[ipk] / tot[ipk])
    lowm = (fr >= 0.3) & (fr < 5.0)
    cen = (S[lowm] * fr[lowm, None]).sum(0) / (S[lowm].sum(0) + 1e-12)
    f["F2_peak_freq"] = float(fr[lowm][np.argmax(S[lowm, ipk])])
    f["F2_centroid_at_peak"] = float(cen[ipk])
    i0, i1 = np.where(ie)[0][0], np.where(ie)[0][-1]
    f["F2_centroid_slope"] = float((cen[i1] - cen[i0]) / max(tt[i1] - tt[i0], 0.1))
    # F3 power burst curve and sharp decline
    base_pre = np.median(tot[win(tt, on - 2.2, on - 0.2)]) if win(tt, on - 2.2, on - 0.2).any() else np.median(tot)
    after = win(tt, tt[ipk] + 0.8, tt[ipk] + 1.2)
    before = win(tt, tt[ipk] - 1.2, tt[ipk] - 0.8)
    f["F3_burst_ratio"] = float(np.log((tot[ipk] + 1e-12) / (base_pre + 1e-12)))
    f["F3_decline_1s"] = float(1 - (tot[after].mean() if after.any() else tot[ipk]) / (tot[ipk] + 1e-12))
    f["F3_rise_1s"] = float(1 - (tot[before].mean() if before.any() else tot[ipk]) / (tot[ipk] + 1e-12))
    postw = win(tt, off + 0.2, off + 2.2)
    f["F3_post_over_pre"] = float(np.log(((tot[postw].mean() if postw.any() else base_pre) + 1e-12) / (base_pre + 1e-12)))
    # F5 autocorrelation of PC1 over the event window
    x = pc1 - pc1.mean()
    ac = np.correlate(x, x, "full")[len(x) - 1:]
    ac = ac / (ac[0] + 1e-12)
    zc = np.where(ac < 0)[0]
    f["F5_decorr_lag_s"] = float((zc[0] if len(zc) else len(ac)) / FS)
    f["F5_acf_0.5s"] = float(ac[50]) if len(ac) > 50 else 0.0
    seg_ac = ac[30:min(len(ac), 200)]
    if len(seg_ac) > 3:
        loc_max = [i for i in range(1, len(seg_ac) - 1) if seg_ac[i] > seg_ac[i - 1] and seg_ac[i] >= seg_ac[i + 1]]
        if loc_max:
            j = max(loc_max, key=lambda i: seg_ac[i])
            f["F5_period_peak"] = float(seg_ac[j]); f["F5_period_s"] = float((j + 30) / FS)
        else:
            f["F5_period_peak"], f["F5_period_s"] = 0.0, 0.0
    else:
        f["F5_period_peak"], f["F5_period_s"] = 0.0, 0.0
    # F6 Haar wavelet energies (relative)
    e = haar_energies(pc1, 6)
    for name, v in zip(["3_6Hz", "1.6_3Hz", "0.8_1.6Hz", "approx_lt0.8Hz"], e[3:]):
        f[f"F6_{name}"] = float(v)
    # F8 classic statistics on PC1
    f["F8_mad"] = float(np.median(np.abs(pc1 - np.median(pc1))))
    f["F8_iqr"] = float(np.subtract(*np.percentile(pc1, [75, 25])))
    f["F8_p90_p10"] = float(np.subtract(*np.percentile(pc1, [90, 10])))
    f["F8_mean_abs_deriv"] = float(np.mean(np.abs(np.diff(pc1))) * FS)
    f["F8_zcr"] = float(np.mean(np.diff(np.sign(x)) != 0) * FS)
    # F9 Fresnel fringe rate on single subcarriers
    ext = []
    for k in range(X.shape[1]):
        s = X[m, k] - X[m, k].mean()
        d = np.diff(s)
        ext.append(np.sum((d[1:] * d[:-1]) < 0) / max(m.sum() / FS, 0.1))
    f["F9_fringe_rate"] = float(np.median(ext))
    # F11 phase and RSSI
    if P is not None and pre.any():
        ph_ev = np.median(P[m].std(0)); ph_pre = np.median(P[pre].std(0))
        f["F11_phase_std_ratio"] = float(np.log((ph_ev + 1e-9) / (ph_pre + 1e-9)))
    else:
        f["F11_phase_std_ratio"] = 0.0
    f["F11_rssi_std_event"] = float(np.std(rssi[m])) if rssi is not None else 0.0
    f["F11_rssi_change"] = float(np.median(rssi[post]) - np.median(rssi[pre])) if (rssi is not None and pre.any() and post.any()) else 0.0
    # raw material for the fold-fit families
    if pre.any() and post.any():
        noise = np.sqrt((X[pre].var(0) + X[post].var(0)) / 2).mean() + 1e-9
        delta = (np.median(X[post], 0) - np.median(X[pre], 0)) / noise
    else:
        delta = np.zeros(X.shape[1])
    tc_, curve, _ = receiver_curve(t, X)
    cm = win(tc_, on - 1.0, off + 2.0)
    cv = curve[cm] if cm.sum() >= 4 else curve
    prof = np.interp(np.linspace(0, 1, 40), np.linspace(0, 1, len(cv)), cv)
    prof = (prof - prof.mean()) / (prof.std() + 1e-9)
    return {f"{k}_{rx}": v for k, v in f.items()}, delta, prof


rows, deltas, profs = [], {"rx1": {}, "rx2": {}}, {"rx1": {}, "rx2": {}}
files = sorted(glob.glob("data/processed_v3/*/*_aligned.csv"))
for i, fpath in enumerate(files):
    tid = os.path.basename(fpath).replace("_aligned.csv", "")
    if tid not in seg.index:
        continue
    df = pd.read_csv(fpath)
    t = df["time_rel_s"].values
    ev = seg.loc[tid]
    r = {"trial_id": tid, "activity": ev["activity"], "subject": ev["subject"]}
    for rx in ("rx1", "rx2"):
        X = df[[c for c in df.columns if c.startswith(f"{rx}_amp_")]].values
        pc = [c for c in df.columns if c.startswith(f"{rx}_phase_")]
        P = df[pc].values if pc else None
        rssi = df[f"{rx}_rssi"].values if f"{rx}_rssi" in df.columns else None
        fx, delta, prof = rx_features(t, X, P, rssi, ev, rx)
        r.update(fx); deltas[rx][tid] = delta; profs[rx][tid] = prof
    # F10 receiver geometry (trial level)
    r["F10_w_rx1"] = float(ev["loc_w_rx1"])
    r["F10_log_peak_ratio"] = float(np.log((ev.get("ev_rx1_peak", 1) + 1e-9) / (ev.get("ev_rx2_peak", 1) + 1e-9)))
    c1 = receiver_curve(t, df[[c for c in df.columns if c.startswith("rx1_amp_")]].values)
    c2 = receiver_curve(t, df[[c for c in df.columns if c.startswith("rx2_amp_")]].values)
    w = (c1[0] >= max(0, ev["event_onset"] - 1)) & (c1[0] <= ev["event_offset"] + 1)
    r["F10_burst_dt_s"] = float(c1[0][w][np.argmax(c1[1][w])] - c2[0][w][np.argmax(c2[1][:len(c1[1])][w])]) if w.any() else 0.0
    rows.append(r)
    if (i + 1) % 100 == 0:
        print(f"  {i + 1}/{len(files)}", flush=True)
D = pd.DataFrame(rows).set_index("trial_id")

# ---- fold-fit families (out-of-fold under LOSO; training subjects only)
trials = D.reset_index()[["trial_id", "activity", "subject"]]
for col_ in [f"F1_{a}_{rx}" for a in ("sitstand_proj", "sitstand_cos", "fall_proj") for rx in ("rx1", "rx2")] + \
            [f"F4_dtw_{c}_{rx}" for c in ("fall", "walk", "sit", "stand", "hand", "empty_room") for rx in ("rx1", "rx2")] + \
            [f"F4_dtw_fall_margin_{rx}" for rx in ("rx1", "rx2")]:
    D[col_] = np.nan


def dtw(a, b, band=8):
    n, mm = len(a), len(b)
    Dm = np.full((n + 1, mm + 1), np.inf); Dm[0, 0] = 0.0
    for ii in range(1, n + 1):
        for jj in range(max(1, ii - band), min(mm, ii + band) + 1):
            Dm[ii, jj] = (a[ii - 1] - b[jj - 1]) ** 2 + min(Dm[ii - 1, jj], Dm[ii, jj - 1], Dm[ii - 1, jj - 1])
    return float(np.sqrt(Dm[n, mm]))


for fold, test_ids in tc.loso_folds(trials):
    te = [t_ for t_ in test_ids if t_ in D.index]
    tr = [t_ for t_ in D.index if t_ not in test_ids]
    act = D.activity
    for rx in ("rx1", "rx2"):
        dl = deltas[rx]
        mean = lambda ids: np.mean([dl[t_] for t_ in ids], 0)
        d_ss = mean([t_ for t_ in tr if act[t_] == "sit"]) - mean([t_ for t_ in tr if act[t_] == "stand"])
        d_fa = mean([t_ for t_ in tr if act[t_] == "fall"]) - mean([t_ for t_ in tr if act[t_] in ADL])
        u_ss, u_fa = d_ss / (np.linalg.norm(d_ss) + 1e-9), d_fa / (np.linalg.norm(d_fa) + 1e-9)
        tmpl = {c: np.mean([profs[rx][t_] for t_ in tr if act[t_] == c], 0) for c in act.unique()}
        for t_ in te:
            v = dl[t_]
            D.loc[t_, f"F1_sitstand_proj_{rx}"] = float(v @ u_ss)
            D.loc[t_, f"F1_sitstand_cos_{rx}"] = float(v @ u_ss / (np.linalg.norm(v) + 1e-9))
            D.loc[t_, f"F1_fall_proj_{rx}"] = float(v @ u_fa)
            ds = {c: dtw(profs[rx][t_], tmpl[c]) for c in tmpl}
            for c, val in ds.items():
                D.loc[t_, f"F4_dtw_{c}_{rx}"] = val
            D.loc[t_, f"F4_dtw_fall_margin_{rx}"] = min(v_ for c, v_ in ds.items() if c != "fall") - ds["fall"]
    print(f"  fold-fit features done for {fold}", flush=True)
D.to_csv(os.path.join(OUT, "domain_features.csv"))

# ---- acceptance test
new = pd.read_csv("data/event_features_v3_fused.csv", dtype={"subject": str}).set_index("trial_id")
agg = em.window_aggregates("data/features_v3_rx1", "w1").join(em.window_aggregates("data/features_v3_rx2", "w2"))
E = new[[c for c in new.columns if c.startswith("ev_rx")]].join(agg)
E = E[[c for c in E.columns if E[c].nunique() > 1]].reindex(D.index)
feats = [c for c in D.columns if c.startswith("F")]
D = D.replace([np.inf, -np.inf], np.nan)
D[feats] = D[feats].fillna(D[feats].median())
y = D.activity.values
H = D[D.activity != "empty_room"]
mi = dict(zip(feats, mutual_info_classif(D[feats].values, y, random_state=0)))
PAIRS = {"sit_stand": ("sit", "stand"), "hand_empty": ("hand", "empty_room"), "fall_sit": ("fall", "sit"),
         "fall_adl": ("fall", ADL)}
SIGNED = {"F1_sitstand": ("sit_stand", "sit"), "F1_fall": ("fall_adl", "fall"), "F4_dtw_fall_margin": ("fall_adl", "fall")}
Er = E.rank(); Dr = D[feats].rank()


def eta2(x, g):
    x = np.asarray(x, float); m_ = x.mean(); sst = ((x - m_) ** 2).sum() + 1e-12
    return sum((g == k).sum() * (x[g == k].mean() - m_) ** 2 for k in np.unique(g)) / sst


def auc_pair(df, f, pair, signed_pos=None):
    a, b = PAIRS[pair]
    bs = b if isinstance(b, list) else [b]
    d = df[df.activity.isin([a] + bs)]
    if d.activity.nunique() < 2:
        return np.nan
    v = roc_auc_score(d.activity == a, d[f])
    return v if signed_pos else max(v, 1 - v)


rows = []
for f_ in feats:
    signed = next((v for k, v in SIGNED.items() if f_.startswith(k)), None)
    aucs = {p: auc_pair(D, f_, p, signed) for p in PAIRS}
    best_pair = signed[0] if signed else max(aucs, key=lambda p: aucs[p])
    subs = ["1", "2", "4", "5"] if best_pair.startswith("fall") else ["1", "2", "3", "4", "5"]
    raw_dirs = []
    for s in subs:
        a, b = PAIRS[best_pair]
        bs = b if isinstance(b, list) else [b]
        d = D[(D.subject == s) & D.activity.isin([a] + bs)]
        if d.activity.nunique() == 2:
            raw_dirs.append(roc_auc_score(d.activity == a, d[f_]) - 0.5)
    raw_dirs = np.array(raw_dirs)
    pooled_dir = np.sign(roc_auc_score(*(lambda d: (d.activity == PAIRS[best_pair][0], d[f_]))(
        D[D.activity.isin([PAIRS[best_pair][0]] + (PAIRS[best_pair][1] if isinstance(PAIRS[best_pair][1], list) else [PAIRS[best_pair][1]]))])) - 0.5)
    consistent = int(np.sum(np.sign(raw_dirs) == pooled_dir))
    need = 3 if best_pair.startswith("fall") else 4
    red = float(np.nanmax(np.abs(np.corrcoef(Dr[f_].values, Er.values.T)[0, 1:]))) if len(E.columns) else 0.0
    e_a, e_s = eta2(D[f_], y), eta2(H[f_], H.subject.values)
    passes = {"separates": (aucs[best_pair] >= 0.60) or (mi[f_] >= 0.05), "activity>subject": e_a > e_s,
              "consistent": consistent >= need, "not_redundant": red < 0.85}
    rows.append({"feature": f_, "family": f_.split("_")[0], "best_pair": best_pair, "auc_best": aucs[best_pair],
                 **{f"auc_{p}": aucs[p] for p in PAIRS}, "MI": mi[f_], "eta2_activity": e_a, "eta2_subject": e_s,
                 "consistent_subjects": f"{consistent}/{len(raw_dirs)}", "max_rho_existing": red,
                 "PASS": all(passes.values()), "fails": ",".join(k for k, v in passes.items() if not v)})
A = pd.DataFrame(rows).sort_values(["family", "auc_best"], ascending=[True, False])
A.to_csv(os.path.join(OUT, "acceptance_test.csv"), index=False)

# existing-feature reference for each hard pair
ref = {}
for p in PAIRS:
    vals = [(c, auc_pair(E.join(D[["activity", "subject"]]), c, p)) for c in E.columns]
    vals = [v for v in vals if not np.isnan(v[1])]
    ref[p] = max(vals, key=lambda kv: kv[1])
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
print("\n  Best EXISTING feature per pair:", {p: (c, round(v, 3)) for p, (c, v) in ref.items()})
print("\n  Acceptance test (per feature):")
print(A[["feature", "best_pair", "auc_best", "auc_sit_stand", "auc_hand_empty", "auc_fall_sit", "auc_fall_adl", "MI",
         "eta2_activity", "eta2_subject", "consistent_subjects", "max_rho_existing", "PASS", "fails"]].round(3).to_string(index=False))
summary = A.groupby("family").agg(n=("feature", "size"), passed=("PASS", "sum"), best_auc=("auc_best", "max")).to_dict("index")
print("\n  Per family:", json.dumps(summary, default=float))
for p in PAIRS:
    top = A.sort_values(f"auc_{p}", ascending=False).head(3)[["feature", f"auc_{p}", "PASS"]].values.tolist()
    print(f"  top new for {p}: {[(f, round(a, 3), ok) for f, a, ok in top]}  (best existing {ref[p][1]:.3f})")
with open(os.path.join(OUT, "summary.json"), "w") as fh:
    json.dump({"per_family": summary, "best_existing": {p: [c, float(v)] for p, (c, v) in ref.items()}}, fh, indent=2, default=float)
