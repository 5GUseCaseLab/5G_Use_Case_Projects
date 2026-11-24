"""
WiFeye: CSI domain-specific features, shared by training (train_final.py) and the live engine
==================================================================================================
Identical computations to eda_domain_features.py (which produced data/models_exp/eda_domain_v3/
domain_features.csv), packaged as importable functions:
  rx_domain_features(t, X, P, rssi, ev, rx) -> dict of label-free features for one receiver
      F2 speed spectrogram (CARM-style), F3 power burst / decline, F5 autocorrelation, F6 Haar wavelet
      energies, F7 subcarrier (PCA) structure, F8 classic statistics, F9 Fresnel fringe rate,
      F11 phase / RSSI
  trial_geometry(t, X1, X2, ev) -> F10 receiver-geometry features (trial level)
  posture_delta(t, X, ev), motion_profile(t, X, ev) -> raw material for the fold-fit features
  foldfit_params(fit_ids, act, deltas, profs) / foldfit_apply(delta, prof, params, rx)
      F1 signed posture signature and F4 DTW template distances (params learned from training data only)
"""

import numpy as np
from scipy.signal import stft
from sklearn.decomposition import PCA
from event_localizer import receiver_curve

FS = 100.0
ADL = ["walk", "sit", "stand", "hand"]


def _win(t, lo, hi):
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
    return [v / tot for v in e]


def rx_domain_features(t, X, P, rssi, ev, rx):
    on, off, pk = ev["event_onset"], ev["event_offset"], ev["event_t"]
    lo, hi = max(0.0, on - 0.5), min(t[-1], max(off + 0.5, on + 2.0))
    m = _win(t, lo, hi)
    pre, post = _win(t, on - 2.2, on - 0.2), _win(t, off + 0.2, off + 2.2)
    f = {}
    Xs = X[m] - X[m].mean(0)
    pca = PCA(n_components=min(10, Xs.shape[1])).fit(Xs)
    ev_ = pca.explained_variance_ratio_
    pcs = Xs @ pca.components_[:3].T
    pc1 = pcs[:, 0]
    p = ev_ / ev_.sum()
    f["F7_pc1_share"] = float(ev_[0])
    f["F7_eig_entropy"] = float(-(p * np.log(p + 1e-12)).sum() / np.log(len(p)))
    f["F7_n90"] = float(np.searchsorted(np.cumsum(ev_), 0.9) + 1)
    C = np.corrcoef(X[m].T)
    f["F7_mean_abs_r"] = float(np.nanmean(np.abs(C[np.triu_indices_from(C, 1)])))
    full = (X - X[m].mean(0)) @ pca.components_[:3].T
    fr, tt, Z = stft(full.T, fs=FS, nperseg=128, noverlap=118, boundary=None)
    S = (np.abs(Z) ** 2).sum(axis=0)
    tt = tt + t[0]
    band = lambda a, b: S[(fr >= a) & (fr < b)].sum(0)
    b1, b2, b3 = band(0.3, 1.5), band(1.5, 3.0), band(3.0, 5.0)
    tot = b1 + b2 + b3 + 1e-12
    ie = _win(tt, on - 0.3, off + 0.3)
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
    base_pre = np.median(tot[_win(tt, on - 2.2, on - 0.2)]) if _win(tt, on - 2.2, on - 0.2).any() else np.median(tot)
    after = _win(tt, tt[ipk] + 0.8, tt[ipk] + 1.2)
    before = _win(tt, tt[ipk] - 1.2, tt[ipk] - 0.8)
    f["F3_burst_ratio"] = float(np.log((tot[ipk] + 1e-12) / (base_pre + 1e-12)))
    f["F3_decline_1s"] = float(1 - (tot[after].mean() if after.any() else tot[ipk]) / (tot[ipk] + 1e-12))
    f["F3_rise_1s"] = float(1 - (tot[before].mean() if before.any() else tot[ipk]) / (tot[ipk] + 1e-12))
    postw = _win(tt, off + 0.2, off + 2.2)
    f["F3_post_over_pre"] = float(np.log(((tot[postw].mean() if postw.any() else base_pre) + 1e-12) / (base_pre + 1e-12)))
    x = pc1 - pc1.mean()
    ac = np.correlate(x, x, "full")[len(x) - 1:]
    ac = ac / (ac[0] + 1e-12)
    zc = np.where(ac < 0)[0]
    f["F5_decorr_lag_s"] = float((zc[0] if len(zc) else len(ac)) / FS)
    f["F5_acf_0.5s"] = float(ac[50]) if len(ac) > 50 else 0.0
    seg_ac = ac[30:min(len(ac), 200)]
    f["F5_period_peak"], f["F5_period_s"] = 0.0, 0.0
    if len(seg_ac) > 3:
        loc_max = [i for i in range(1, len(seg_ac) - 1) if seg_ac[i] > seg_ac[i - 1] and seg_ac[i] >= seg_ac[i + 1]]
        if loc_max:
            j = max(loc_max, key=lambda i: seg_ac[i])
            f["F5_period_peak"] = float(seg_ac[j]); f["F5_period_s"] = float((j + 30) / FS)
    e = haar_energies(pc1, 6)
    for name, v in zip(["3_6Hz", "1.6_3Hz", "0.8_1.6Hz", "approx_lt0.8Hz"], e[3:]):
        f[f"F6_{name}"] = float(v)
    f["F8_mad"] = float(np.median(np.abs(pc1 - np.median(pc1))))
    f["F8_iqr"] = float(np.subtract(*np.percentile(pc1, [75, 25])))
    f["F8_p90_p10"] = float(np.subtract(*np.percentile(pc1, [90, 10])))
    f["F8_mean_abs_deriv"] = float(np.mean(np.abs(np.diff(pc1))) * FS)
    f["F8_zcr"] = float(np.mean(np.diff(np.sign(x)) != 0) * FS)
    ext = []
    for k in range(X.shape[1]):
        s = X[m, k] - X[m, k].mean()
        d = np.diff(s)
        ext.append(np.sum((d[1:] * d[:-1]) < 0) / max(m.sum() / FS, 0.1))
    f["F9_fringe_rate"] = float(np.median(ext))
    if P is not None and pre.any():
        f["F11_phase_std_ratio"] = float(np.log((np.median(P[m].std(0)) + 1e-9) / (np.median(P[pre].std(0)) + 1e-9)))
    else:
        f["F11_phase_std_ratio"] = 0.0
    f["F11_rssi_std_event"] = float(np.std(rssi[m])) if rssi is not None else 0.0
    f["F11_rssi_change"] = float(np.median(rssi[post]) - np.median(rssi[pre])) if (rssi is not None and pre.any() and post.any()) else 0.0
    return {f"{k}_{rx}": v for k, v in f.items()}


def trial_geometry(t, X1, X2, ev):
    c1, c2 = receiver_curve(t, X1), receiver_curve(t, X2)
    w = (c1[0] >= max(0, ev["event_onset"] - 1)) & (c1[0] <= ev["event_offset"] + 1)
    dt = float(c1[0][w][np.argmax(c1[1][w])] - c2[0][w][np.argmax(c2[1][:len(c1[1])][w])]) if w.any() else 0.0
    return {"F10_w_rx1": float(ev.get("loc_w_rx1", 0.5)),
            "F10_log_peak_ratio": float(np.log((ev.get("ev_rx1_peak", 1) + 1e-9) / (ev.get("ev_rx2_peak", 1) + 1e-9))),
            "F10_burst_dt_s": dt}


def posture_delta(t, X, ev):
    on, off = ev["event_onset"], ev["event_offset"]
    pre, post = _win(t, on - 2.2, on - 0.2), _win(t, off + 0.2, off + 2.2)
    if pre.any() and post.any():
        noise = np.sqrt((X[pre].var(0) + X[post].var(0)) / 2).mean() + 1e-9
        return (np.median(X[post], 0) - np.median(X[pre], 0)) / noise
    return np.zeros(X.shape[1])


def motion_profile(t, X, ev):
    tc_, curve, _ = receiver_curve(t, X)
    cm = _win(tc_, ev["event_onset"] - 1.0, ev["event_offset"] + 2.0)
    cv = curve[cm] if cm.sum() >= 4 else curve
    p = np.interp(np.linspace(0, 1, 40), np.linspace(0, 1, len(cv)), cv)
    return (p - p.mean()) / (p.std() + 1e-9)


def dtw(a, b, band=8):
    n, m = len(a), len(b)
    D = np.full((n + 1, m + 1), np.inf); D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(max(1, i - band), min(m, i + band) + 1):
            D[i, j] = (a[i - 1] - b[j - 1]) ** 2 + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(np.sqrt(D[n, m]))


def foldfit_params(fit_ids, act, deltas, profs):
    """Learn F1 directions and F4 templates from training trials only (act: 6-class labels)."""
    out = {}
    for rx in ("rx1", "rx2"):
        mean = lambda ids: np.mean([deltas[rx][t] for t in ids], 0)
        d_ss = mean([t for t in fit_ids if act[t] == "sit"]) - mean([t for t in fit_ids if act[t] == "stand"])
        d_fa = mean([t for t in fit_ids if act[t] == "fall"]) - mean([t for t in fit_ids if act[t] in ADL])
        out[rx] = {"u_ss": d_ss / (np.linalg.norm(d_ss) + 1e-9), "u_fa": d_fa / (np.linalg.norm(d_fa) + 1e-9),
                   "templates": {c: np.mean([profs[rx][t] for t in fit_ids if act[t] == c], 0)
                                 for c in set(act[t] for t in fit_ids)}}
    return out


def foldfit_apply(delta, prof, params, rx):
    p = params[rx]
    f = {f"F1_sitstand_proj_{rx}": float(delta @ p["u_ss"]),
         f"F1_sitstand_cos_{rx}": float(delta @ p["u_ss"] / (np.linalg.norm(delta) + 1e-9)),
         f"F1_fall_proj_{rx}": float(delta @ p["u_fa"])}
    ds = {c: dtw(prof, tm) for c, tm in p["templates"].items()}
    for c, v in ds.items():
        f[f"F4_dtw_{c}_{rx}"] = v
    f[f"F4_dtw_fall_margin_{rx}"] = min(v for c, v in ds.items() if c != "fall") - ds.get("fall", 0.0)
    return f
