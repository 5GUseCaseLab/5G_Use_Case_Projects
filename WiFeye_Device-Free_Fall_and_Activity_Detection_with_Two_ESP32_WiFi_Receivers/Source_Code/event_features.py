"""
WiFeye: Event-level features (two-stage detection, stage 2)
================================================================
Stage 1 locates the dominant motion event in a ~10 s buffer; stage 2 describes the whole
event with physically motivated, mostly scale-free features, per receiver:

  motion curve m(t)   mean over subcarriers of the amplitude std in 0.2 s windows
                      (hop 0.1 s), divided by the buffer's quiet level (20th percentile)
  peak, log_peak      event strength
  pre, post, sway     median motion 1-3 s before / 1.5-3.5 s after the event and
                      log(post/pre): standing sways more than sitting, so sit-down < 0,
                      stand-up > 0, a fall ends in stillness (strongly < 0)
  still_after         share of post-event samples near the quiet level (fall confirmation)
  burst_dur, rise, decay, asym   envelope shape (walk = long; fall = short, abrupt stop)
  area                motion above the quiet level
  env_freq            dominant envelope frequency around the event (gait periodicity)
  posture_shift       |median level after - before| / within-segment noise
  log_noise           absolute quiet level (session-dependent; kept separate so its
                      influence can be ablated)
Cross-receiver (dual only): correlation of the two motion curves, peak-time difference.

The same function serves offline evaluation (whole 10 s trials from data/processed_v3)
and the live engine (a rolling buffer).

Usage (offline): python event_features.py --in-dir data/processed_v3 --out data/event_features_v3.csv
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd

WIN_S, HOP_S = 0.2, 0.1
PRE = (-3.0, -1.0)
POST = (1.5, 3.5)
SEARCH = (1.0, 8.5)


def motion_curve(t, x, fs=100.0):
    """t: (N,), x: (N, K) amplitudes on a uniform grid -> (centres, curve)."""
    w, h = int(round(WIN_S * fs)), int(round(HOP_S * fs))
    starts = np.arange(0, max(len(t) - w + 1, 1), h)
    c = np.array([x[s:s + w].std(axis=0).mean() for s in starts])
    centres = t[np.minimum(starts + w // 2, len(t) - 1)]
    return centres, c


def _seg(tc_, v, lo, hi):
    m = (tc_ >= lo) & (tc_ <= hi)
    return v[m] if m.any() else np.array([])


def rx_event_features(t, x, t_event, fs=100.0):
    tc_, c = motion_curve(t, x, fs)
    quiet = np.percentile(c, 20) + 1e-9
    m = c / quiet
    i_pk = int(np.argmin(np.abs(tc_ - t_event)))
    # local peak within +-0.5 s of the shared event time
    near = np.where(np.abs(tc_ - t_event) <= 0.5)[0]
    if len(near):
        i_pk = near[np.argmax(m[near])]
    peak = float(m[i_pk])
    pre = _seg(tc_, m, t_event + PRE[0], t_event + PRE[1])
    post = _seg(tc_, m, t_event + POST[0], t_event + POST[1])
    pre_l = float(np.median(pre)) if len(pre) else 1.0
    post_l = float(np.median(post)) if len(post) else 1.0
    half, low = 0.5 * peak, 0.2 * peak
    # contiguous burst around the peak
    l = i_pk
    while l > 0 and m[l - 1] > half:
        l -= 1
    r = i_pk
    while r < len(m) - 1 and m[r + 1] > half:
        r += 1
    burst = (r - l + 1) * HOP_S
    lr = i_pk
    while lr > 0 and m[lr - 1] > low:
        lr -= 1
    rr = i_pk
    while rr < len(m) - 1 and m[rr + 1] > low:
        rr += 1
    rise, decay = (i_pk - lr) * HOP_S + HOP_S / 2, (rr - i_pk) * HOP_S + HOP_S / 2
    ev = _seg(tc_, m - np.median(m), t_event - 1.5, t_event + 1.5)
    if len(ev) >= 8:
        spec = np.abs(np.fft.rfft(ev * np.hanning(len(ev))))
        freqs = np.fft.rfftfreq(len(ev), HOP_S)
        env_freq = float(freqs[1:][np.argmax(spec[1:])]) if len(spec) > 1 else 0.0
    else:
        env_freq = 0.0
    # posture shift on the level (median-removed amplitudes)
    pre_x = x[(t >= t_event + PRE[0]) & (t <= t_event + PRE[1])]
    post_x = x[(t >= t_event + POST[0]) & (t <= t_event + POST[1])]
    if len(pre_x) >= 10 and len(post_x) >= 10:
        noise = np.sqrt((pre_x.var(axis=0) + post_x.var(axis=0)) / 2).mean() + 1e-9
        shift = float(np.abs(np.median(post_x, axis=0) - np.median(pre_x, axis=0)).mean() / noise)
    else:
        shift = 0.0
    return {
        "peak": peak, "log_peak": float(np.log(peak + 1e-9)),
        "pre": pre_l, "post": post_l, "sway": float(np.log((post_l + 1e-9) / (pre_l + 1e-9))),
        "still_after": float(np.mean(post < 1.5)) if len(post) else 0.0,
        "burst_dur": burst, "rise": rise, "decay": decay, "asym": float(np.log(rise / decay)),
        "area": float(np.clip(m - 1.0, 0, None).sum() * HOP_S),
        "env_freq": env_freq, "posture_shift": shift, "log_noise": float(np.log(quiet)),
    }, (tc_, m)


def locate_event(t, mats, fs=100.0, search=SEARCH):
    """Shared event time: argmax of the summed normalised motion curves within `search`."""
    total, centres = None, None
    for x in mats.values():
        tc_, c = motion_curve(t, x, fs)
        mc = c / (np.percentile(c, 20) + 1e-9)
        total = mc if total is None else total[:len(mc)] + mc[:len(total)]
        centres = tc_[:len(total)]
    ok = (centres >= search[0]) & (centres <= search[1])
    idx = np.where(ok)[0] if ok.any() else np.arange(len(centres))
    sm = np.convolve(total, np.ones(3) / 3, mode="same")
    return float(centres[idx[np.argmax(sm[idx])]])


def event_features(t, mats, fs=100.0, search=SEARCH):
    """mats: {"rx1": (N,K) array, "rx2": (N,K) array} (any subset). Returns flat dict."""
    t_ev = locate_event(t, mats, fs, search)
    out, curves = {"event_t": t_ev}, {}
    for rx, x in mats.items():
        f, curves[rx] = rx_event_features(t, x, t_ev, fs)
        out.update({f"ev_{rx}_{k}": v for k, v in f.items()})
    if "rx1" in curves and "rx2" in curves:
        (t1, m1), (t2, m2) = curves["rx1"], curves["rx2"]
        n = min(len(m1), len(m2))
        out["ev_x_corr"] = float(np.corrcoef(m1[:n], m2[:n])[0, 1]) if n > 3 else 0.0
        out["ev_x_peak_dt"] = float(t1[np.argmax(m1)] - t2[np.argmax(m2)])
    return out


def rx_segment_features(t, x, loc, fs=100.0):
    """Event features for one receiver around a localised SEGMENT (onset -> peak -> offset) from
    event_localizer.localize(): 'before' = up to 2 s before onset, 'after' = up to 2 s after offset."""
    tc_, c = motion_curve(t, x, fs)
    quiet = np.percentile(c, 20) + 1e-9
    m = c / quiet
    on, off, pk = loc["onset"], loc["offset"], loc["t_peak"]
    in_seg = (tc_ >= on - 1e-6) & (tc_ <= off + 1e-6)
    seg = m[in_seg] if in_seg.any() else m[[int(np.argmin(np.abs(tc_ - pk)))]]
    pre = _seg(tc_, m, on - 2.2, on - 0.2)
    post = _seg(tc_, m, off + 0.2, off + 2.2)
    pre_l = float(np.median(pre)) if len(pre) else 1.0
    post_l = float(np.median(post)) if len(post) else 1.0
    peak = float(seg.max())
    rise, decay = max(pk - on, HOP_S / 2), max(off - pk, HOP_S / 2)
    if len(seg) >= 8:
        e = seg - seg.mean()
        spec = np.abs(np.fft.rfft(e * np.hanning(len(e))))
        env_freq = float(np.fft.rfftfreq(len(e), HOP_S)[1:][np.argmax(spec[1:])])
    else:
        env_freq = 0.0
    pre_x = x[(t >= on - 2.2) & (t <= on - 0.2)]
    post_x = x[(t >= off + 0.2) & (t <= off + 2.2)]
    if len(pre_x) >= 10 and len(post_x) >= 10:
        noise = np.sqrt((pre_x.var(axis=0) + post_x.var(axis=0)) / 2).mean() + 1e-9
        shift = float(np.abs(np.median(post_x, axis=0) - np.median(pre_x, axis=0)).mean() / noise)
    else:
        shift = 0.0
    return {"peak": peak, "log_peak": float(np.log(peak + 1e-9)), "seg_mean": float(seg.mean()),
            "pre": pre_l, "post": post_l, "sway": float(np.log((post_l + 1e-9) / (pre_l + 1e-9))),
            "still_after": float(np.mean(post < 1.5)) if len(post) else 0.0,
            "burst_dur": float(off - on), "rise": float(rise), "decay": float(decay),
            "asym": float(np.log(rise / decay)), "area": float(np.clip(seg - 1.0, 0, None).sum() * HOP_S),
            "env_freq": env_freq, "posture_shift": shift, "log_noise": float(np.log(quiet))}


def event_features_fused(t, mats, packet_times=None, ref=None, fs=100.0, prior=None):
    """Fused, reliability-weighted, prior-informed localisation (event_localizer) + segment features.
    prior: (lo, hi) seconds of the soft time prior; default = recording protocol (0.5, 7.5)."""
    from event_localizer import localize, PRIOR_LO, PRIOR_HI
    loc = localize(t, mats, packet_times, ref, fs, prior=prior or (PRIOR_LO, PRIOR_HI))
    out = {"event_t": loc["t_peak"], "event_onset": loc["onset"], "event_offset": loc["offset"],
           "loc_prominence": loc["prominence"], "loc_w_rx1": loc["w_rx1"], "loc_w_rx2": loc["w_rx2"],
           "loc_rel_rx1": loc["rel_rx1"], "loc_rel_rx2": loc["rel_rx2"]}
    for rx, x in mats.items():
        out.update({f"ev_{rx}_{k}": v for k, v in rx_segment_features(t, x, loc, fs).items()})
    return out


def trial_matrices(df):
    mats = {}
    for rx in ("rx1", "rx2"):
        cols = [c for c in df.columns if c.startswith(f"{rx}_amp_")]
        if cols:
            mats[rx] = df[cols].values
    return mats


def main():
    ap = argparse.ArgumentParser(description="Event-level features for all processed trials")
    ap.add_argument("--in-dir", default="data/processed_v3")
    ap.add_argument("--out", default="data/event_features_v3.csv")
    args = ap.parse_args()
    rows = []
    files = sorted(f for f in glob.glob(os.path.join(args.in_dir, "*", "*_aligned.csv")))
    for i, f in enumerate(files):
        df = pd.read_csv(f)
        tid = os.path.basename(f).replace("_aligned.csv", "")
        r = {"trial_id": tid, "activity": os.path.basename(os.path.dirname(f)), "subject": tid.split("_")[0]}
        r.update(event_features(df["time_rel_s"].values, trial_matrices(df)))
        rows.append(r)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(files)} trials")
    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(f"  Wrote {len(out)} trials x {out.shape[1] - 3} features to {args.out}")


if __name__ == "__main__":
    main()
