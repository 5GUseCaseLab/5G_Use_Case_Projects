"""
WiFeye: fused, reliability-weighted, prior-informed event localisation
===========================================================================
Replaces "argmax of the summed motion curves" (which put the two receivers' events >1 s apart in
62-75 % of non-walk trials). Design principles:
  * no requirement that both receivers see the event - a person moving along one receiver's
    line of sight may be invisible to it, so each receiver is weighted by how clearly IT sees a burst;
  * packet gaps cannot create fake bursts - windows with too few packets / long gaps are neutralised;
  * the recording protocol is used as a SOFT prior (action expected ~0.5-7.5 s after the GO beep,
    soft edges for human reaction-time error), never as hard labels;
  * the event is a segment (onset -> peak -> offset), not a fixed +-1 s around one point.
Label-free: uses no activity information, so the identical code runs at training, test and live time.

  localize(t, mats, packet_times, ref) -> dict(t_peak, onset, offset, w_rx1, w_rx2, prominence, ...)
"""

import numpy as np
from event_features import motion_curve

PRIOR_LO, PRIOR_HI, PRIOR_EDGE = 0.5, 7.5, 0.35       # seconds; logistic edges for human error
MIN_RATE_HZ, MAX_GAP_S, REL_HALF_S = 20.0, 0.12, 0.2   # packet-reliability rule (0.4 s window)
SMOOTH_PTS = 5                                         # 0.5 s moving average on the 0.1 s curve
ONSET_FRAC = 0.3                                       # onset/offset at 30 % of peak height above baseline


def soft_prior(tc, lo=PRIOR_LO, hi=PRIOR_HI, edge=PRIOR_EDGE):
    return 1.0 / (1.0 + np.exp(-(tc - lo) / edge)) / (1.0 + np.exp((tc - hi) / edge))


def packet_reliability(tc, pkt_t, half=REL_HALF_S, min_rate=MIN_RATE_HZ, max_gap=MAX_GAP_S):
    """True where the receiver delivered enough, evenly spread packets around each curve point."""
    if pkt_t is None or len(pkt_t) < 3:
        return np.ones(len(tc), bool)
    p = np.sort(np.asarray(pkt_t, float))
    rel = np.ones(len(tc), bool)
    for i, c in enumerate(tc):
        a, b = np.searchsorted(p, c - half), np.searchsorted(p, c + half)
        seg = np.concatenate([[c - half], p[a:b], [c + half]])
        if (b - a) < min_rate * 2 * half or np.max(np.diff(seg)) > max_gap:
            rel[i] = False
    return rel


def _smooth(x, k=SMOOTH_PTS):
    if len(x) < k:
        return x
    pad = np.pad(x, (k // 2, k - 1 - k // 2), mode="edge")
    return np.convolve(pad, np.ones(k) / k, mode="valid")


def receiver_curve(t, X, pkt_t=None, fs=100.0):
    """Normalised (by quiet level), gap-masked, smoothed log-motion curve for one receiver."""
    tc, c = motion_curve(t, X, fs)
    s = np.log(c / (np.percentile(c, 20) + 1e-9) + 1e-9)
    rel = packet_reliability(tc, pkt_t)
    if (~rel).any():
        s = np.where(rel, s, np.minimum(s, np.median(s)))     # an unreliable stretch cannot be a peak
    return tc, _smooth(s), float(rel.mean())


def localize(t, mats, packet_times=None, ref=None, fs=100.0, prior=(PRIOR_LO, PRIOR_HI)):
    """mats: {"rx1": (N,K), "rx2": (N,K)} on a shared grid t (seconds from trial/buffer start).
    packet_times: {"rx": array of packet times on the same time base} (optional, enables gap masking).
    ref: {"rx": log-peak level reached by NOISE alone (empty room)}; receivers are weighted by how far
    their strongest plausible burst exceeds it."""
    packet_times = packet_times or {}
    ref = ref or {}
    curves, weights, rel_share, peaks = {}, {}, {}, {}
    for rx, X in mats.items():
        tc, s, r = receiver_curve(t, X, packet_times.get(rx), fs)
        curves[rx], rel_share[rx] = (tc, s), r
    n = min(len(c[1]) for c in curves.values())
    tc = next(iter(curves.values()))[0][:n]
    prior = soft_prior(tc, prior[0], prior[1])
    for rx, (_, s) in curves.items():
        s = s[:n]
        base = np.median(s)
        peaks[rx] = float(np.max((s - base) * prior))
        weights[rx] = max(0.0, peaks[rx] - ref.get(rx, 0.0))
    if sum(weights.values()) <= 1e-9:                 # nobody clearly above noise: equal weights
        weights = {rx: 1.0 for rx in curves}
    wsum = sum(weights.values())
    F = sum(weights[rx] * (curves[rx][1][:n] - np.median(curves[rx][1][:n])) for rx in curves) / wsum
    score = F * prior
    ip = int(np.argmax(score))
    base = float(np.median(F))
    thr = base + ONSET_FRAC * (F[ip] - base)
    i0 = ip
    while i0 > 0 and F[i0 - 1] > thr:
        i0 -= 1
    i1 = ip
    while i1 < n - 1 and F[i1 + 1] > thr:
        i1 += 1
    return {"t_peak": float(tc[ip]), "onset": float(tc[i0]), "offset": float(tc[i1]),
            "prominence": float(F[ip] - base), "w_rx1": float(weights.get("rx1", 0.0) / wsum),
            "w_rx2": float(weights.get("rx2", 0.0) / wsum), "peak_rx1": peaks.get("rx1", np.nan),
            "peak_rx2": peaks.get("rx2", np.nan), "rel_rx1": rel_share.get("rx1", np.nan),
            "rel_rx2": rel_share.get("rx2", np.nan)}
