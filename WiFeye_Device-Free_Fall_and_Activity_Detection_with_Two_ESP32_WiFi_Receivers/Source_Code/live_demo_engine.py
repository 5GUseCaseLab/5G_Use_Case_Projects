"""
WiFeye: Live Demo Engine (2-ESP32, late fusion, 3-tier)
=============================================================
Roles
  master   Laptop A. Reads RX1 (COM3), receives RX2 packets relayed by the worker over
           UDP, runs the full pipeline and shows the live dashboard + fall alarm.
  worker   Laptop B. Reads RX2 (COM4), keeps only router->RX2 frames, stamps them in
           MASTER time (NTP-style sync from dataset_collector.NetworkSyncNode) and relays
           them to the master.
Network (same as data collection): both laptops join the router's Wi-Fi (SSID "CSI").
  The master finds the worker by UDP broadcast on port 5555 and the worker learns the
  master's address from the sync messages, so no IP addresses need to be typed. Relay
  packets go to UDP port 5600. Relay frames carry the laptops' MACs, which the
  router->ESP32 packet filter discards, so the relay does not contaminate the CSI.
Sources
  --port COMx            live ESP32 (also runs the 140 Hz UDP traffic pump that makes the
                         router transmit, exactly as during data collection)
  --replay TRIAL_ID      stream a recorded trial (both receivers) in real time
  --replay-demo          one recorded trial per class, back to back, with ground truth
  --loso-subject S       replay only subject S's trials with models retrained WITHOUT S
                         (an honest out-of-sample demo; normal replay uses the deployed
                         model, which saw every recorded trial)

Processing (identical to training, pipeline v3; see data/models/model_meta.json)
  packet filter   HT frames whose mac field is the receiver's own MAC (router -> ESP32),
                  subcarriers 0 and 27-37 dropped (52 kept)
  rolling buffer  last 10 s per receiver -> 100 Hz grid -> Hampel -> 10 Hz Butterworth
                  (zero phase) -> per-subcarrier median removal
  Tier 1          OPT-IN (--tier1, power saving for an edge device; off by default): dormant if no
                  receiver's SCALE-FREE window motion (window std / the buffer's own noise scale)
                  reached T1 in the last 6 s; T1 per receiver = 2.5 % quantile of the falls' peak
                  motion in training (LOSO: ~50 % of windows skipped, 4/79 falls lost; continuous
                  walking normalises itself away and is often skipped). An absolute empty-room
                  floor was tested and rejected: it would have passed only ~2.5 % of falls.
  Tier 2          a burst above T2 (95 % quantile of ADL bursts) shows "MOTION BURST - analysing";
                  it is not a fall warning (it also fires on ~1/3 of empty-room trials)
  Tier 3          adopted two-head classifier (bundle format two_head_v1), once >= 3.5 s have
                  passed after the event:
                    fused localiser   reliability-weighted over the receivers, packet-gap masked,
                                      soft time prior -> event segment (onset, peak, offset)
                    fall head         event-segment features + window aggregates, RF per receiver
                    activity head     the same + 42 CSI domain features (speed spectrogram, PCA
                                      structure, ACF, wavelet, fringe rate, RSSI, posture direction,
                                      DTW templates) that passed the acceptance test
                    late fusion       mean of the available receivers, per head
                    decision          FALL iff P(fall) >= tau (--fall-far 3: tau for ~3 % ADL false
                                      alarms under LOSO), else the most probable of walk / posture
                                      change (sit or stand) / hand gesture / no activity
                  (legacy single-head bundles are still accepted)
  fallback        if RX2 is silent > 1.5 s the engine runs on RX1 alone (and vice versa)

Warm-up: nothing is reported for the first --calibrate seconds (default 12) while the 10 s
buffer fills; the room does NOT need to be empty. Expected alert latency: 3.5-5 s after the fall.

Usage (both laptops on Wi-Fi "CSI"; --port auto picks the ESP32's USB-serial port)
  Laptop B:  python live_demo_engine.py --role worker --port auto
  Laptop A:  python live_demo_engine.py --role master --port auto --dual
  (optional: --master-ip / --worker-ip to skip discovery)
  Offline:   python live_demo_engine.py --replay-demo
             python live_demo_engine.py --loso-subject 4
             python live_demo_engine.py --selftest 30
"""

import os
import sys
import time
import json
import glob
import math
import socket
import struct
import argparse
import threading
from collections import deque
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
from scipy.signal import butter, filtfilt

from csi_preprocess import hampel_filter_2d
from feature_extraction import extract_window_features, extended_window_features, trial_noise_scale
from event_features import event_features, event_features_fused
from domain_features import rx_domain_features, trial_geometry, posture_delta, motion_profile, foldfit_apply

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

FS = 100.0
BUF_S = 10.0
POST_S = 3.5
MIN_COVER_S = 6.0
CLASSIFY_EVERY_S = 1.0
MIN_USE_HZ = 20.0        # receivers below this packet rate are not used (same rule as the localiser's gap mask)
KEEP_SC = [i for i in range(64) if i != 0 and not (27 <= i <= 37)]
CLASSES = ["fall", "walk", "sit", "stand", "hand", "empty_room"]
DISPLAY = {"fall": "FALL", "walk": "WALKING", "sit": "SITTING DOWN", "stand": "STANDING UP",
           "posture": "POSTURE CHANGE (SIT/STAND)", "hand": "HAND GESTURE", "empty_room": "NO ACTIVITY"}
RELAY_FMT = "<d53f"          # master-clock time, 52 amplitudes, RSSI
RELAY_SIZE = struct.calcsize(RELAY_FMT)


# ----------------------------------------------------------------------------------------
# Buffers and packet handling
# ----------------------------------------------------------------------------------------
class RxBuffer:
    """Thread-safe rolling buffer of (time, 52 amplitudes, RSSI) for one receiver."""

    def __init__(self, name, keep_s=BUF_S + 3.0):
        self.name, self.keep_s = name, keep_s
        self._t, self._a, self._r = deque(), deque(), deque()
        self._arr = deque(maxlen=600)       # wall-clock arrival times (packet-rate display)
        self._lag = deque(maxlen=100)       # arrival - timestamp (clock-sync / transport delay check)
        self._lock = threading.Lock()
        self.total = 0

    def push(self, t, amps, rssi=np.nan):
        now = time.time()
        with self._lock:
            self._t.append(t); self._a.append(amps); self._r.append(float(rssi)); self.total += 1
            self._arr.append(now); self._lag.append(now - t)
            while self._t and self._t[0] < t - self.keep_s:
                self._t.popleft(); self._a.popleft(); self._r.popleft()

    def arrival_rate(self, window=2.0):
        """Packets actually received per second (independent of timestamps / clock offset)."""
        with self._lock:
            cut = time.time() - window
            return sum(1 for x in self._arr if x >= cut) / window

    def lag(self):
        """Median arrival - timestamp delay (s): ~0 for the local ESP32, tens of ms for the relayed one;
        about a second or more means the worker's clock offset is wrong."""
        with self._lock:
            return float(np.median(self._lag)) if self._lag else float("nan")

    def snapshot(self, t0, t1):
        with self._lock:
            if not self._t:
                return np.empty(0), np.empty((0, len(KEEP_SC))), np.empty(0)
            t = np.fromiter(self._t, float); a = np.vstack(self._a); r = np.fromiter(self._r, float)
        m = (t >= t0) & (t <= t1)
        return t[m], a[m], r[m]

    def rate(self, now, window=2.0):
        with self._lock:
            return sum(1 for x in self._t if x >= now - window) / window

    def last_time(self):
        with self._lock:
            return self._t[-1] if self._t else -math.inf


class MacWatch:
    """Explains 'CSI arrives but nothing is kept': counts the address field of the HT frames seen on the
    serial port and, after 8 s with no frame matching this board's address, prints what it does see."""

    ADVICE = ("Usually the ESP32 is hearing other devices (e.g. a laptop's traffic pump flooding the channel) but "
              "not its own frames:\n  try --pump-hz 0 on this laptop (and/or the other one). Use --mac only if "
              "this really is a different ESP32 board,\n  never with a laptop's or router's address.")

    def __init__(self, role, own_mac):
        self.role, self.own = role, own_mac.upper()
        self.counts, self.kept, self.t0, self.warned = {}, 0, time.time(), False

    def see(self, pkt, kept):
        if pkt is not None and pkt.sig_mode == 1:
            m = pkt.mac.upper()
            self.counts[m] = self.counts.get(m, 0) + 1
        if kept:
            self.kept += 1

    def check(self):
        if self.warned or self.kept or not self.counts or time.time() - self.t0 < 8:
            return
        self.warned = True
        top = sorted(self.counts.items(), key=lambda kv: -kv[1])[:4]
        print(f"\n  WARNING {self.role}: CSI is arriving, but none of it carries this board's address {self.own}.\n"
              f"  Addresses seen: {', '.join(f'{m} ({n})' for m, n in top)}\n"
              f"  {self.ADVICE}\n")


def own_ht_amplitudes(pkt, own_mac):
    """Router->ESP32 HT frame filter (pipeline v3) and subcarrier selection."""
    if pkt is None or pkt.sig_mode != 1 or pkt.mac.upper() != own_mac.upper():
        return None
    a = np.asarray(pkt.amplitudes, dtype=float)
    return a[KEEP_SC] if len(a) >= 64 else None


# ----------------------------------------------------------------------------------------
# Processing: exactly the training pipeline on a rolling buffer
# ----------------------------------------------------------------------------------------
_B, _A = butter(4, 10.0 / (0.5 * FS), btype="low")


def preprocess(ts, A, t0, t1, R=None):
    """Irregular packets -> 100 Hz grid -> Hampel -> Butterworth -> median removal (amplitudes);
    RSSI is interpolated onto the same grid without filtering (as in merge_dual_rx.py)."""
    n = int(np.floor((t1 - t0) * FS))
    if len(ts) < 20 or n < 300:
        return None, None, None
    order = np.argsort(ts)
    ts, A = ts[order], A[order]
    R = None if R is None else np.asarray(R, float)[order]
    ts, uniq = np.unique(ts, return_index=True)
    A = A[uniq]
    # exactly merge_dual_rx.align_trial_pair's grid: at epoch magnitude (~1.7e9 s) float64 resolves
    # ~2e-7 s, and t0 + arange(n)/FS rounds differently, which flips samples that sit exactly on a
    # feature-window edge (e.g. onset - 2.2 s)
    grid = np.linspace(t0, t0 + (n - 1) / FS, n)
    X = np.column_stack([np.interp(grid, ts, A[:, k]) for k in range(A.shape[1])])
    X = hampel_filter_2d(X, window_size=7, n_sigmas=3.0)
    if len(X) > 12:
        X = filtfilt(_B, _A, X, axis=0)
    X = X - np.median(X, axis=0, keepdims=True)
    Rg = None
    if R is not None:
        R = R[uniq]
        ok = np.isfinite(R)
        Rg = np.interp(grid, ts[ok], R[ok]) if ok.sum() >= 2 else None
    return grid - t0, X, Rg


def window_table(trel, X):
    """Per-window features (2 s, hop 0.25 s) in the single-receiver 'rx1' slot naming."""
    cols = [f"rx1_amp_{k}" for k in range(X.shape[1])]
    df = pd.DataFrame(X, columns=cols)
    df["time_rel_s"] = trel
    noise = {"rx1": trial_noise_scale(df, cols)}
    rows = []
    for s in range(0, len(df) - 200 + 1, 25):
        sub = df.iloc[s:s + 200]
        f = extract_window_features(sub, cols, [], fs=FS)
        f.update(extended_window_features(sub, {"rx1": cols, "rx2": []}, noise, fs=FS))
        f["mid_s"] = (sub["time_rel_s"].iloc[0] + sub["time_rel_s"].iloc[-1]) / 2
        rows.append(f)
    return pd.DataFrame(rows)


def window_aggregates(W, prefix, half_width=1.0, search=(1.0, 8.5)):
    """Same event localisation + aggregation as train_classifier.apply_event_labels."""
    ok = (W["mid_s"] >= search[0]) & (W["mid_s"] <= search[1])
    cand = W[ok] if ok.any() else W
    centre = cand["mid_s"].values[np.argmax(cand["ext_motion_pk_rx1"].values)]
    region = W[(W["mid_s"] - centre).abs() <= half_width].drop(columns=["mid_s"])
    agg = {}
    for c in region.columns:
        agg[f"{prefix}_{c}_mean"] = float(region[c].mean())
        agg[f"{prefix}_{c}_max"] = float(region[c].max())
    return agg


def abs_motion(X, trel, seconds):
    """Absolute motion: mean over subcarriers of the std over the last `seconds`."""
    if X is None:
        return 0.0
    m = trel >= trel[-1] - seconds
    return float(X[m].std(axis=0).mean()) if m.sum() > 10 else 0.0


class EventClassifier:
    """Two-stage event classifier with per-receiver models and late fusion."""

    def __init__(self, bundle):
        self.models = bundle["event_models"]
        self.names = bundle["event_feature_names"]
        self.tau = float(bundle["fall_threshold"]["tau"])

    def classify(self, procs, search_ev=(1.0, 8.5)):
        """procs: {"rx1": (trel, X), "rx2": (trel, X)} (available receivers only)."""
        mats = {rx: X for rx, (trel, X) in procs.items()}
        trel = next(iter(procs.values()))[0]
        n = min(len(X) for X in mats.values())
        mats = {rx: X[:n] for rx, X in mats.items()}
        ev = event_features(trel[:n], mats, FS, search=search_ev)
        probs = []
        for rx, (tr, X) in procs.items():
            feats = dict(ev)
            feats.update(window_aggregates(window_table(tr, X), "w1" if rx == "rx1" else "w2"))
            vec = np.array([[feats.get(c, 0.0) for c in self.names[rx]]], dtype=float)
            clf = self.models[rx]
            p_raw = clf.predict_proba(np.nan_to_num(vec))[0]
            p = np.zeros(len(CLASSES))
            for j, c in enumerate(clf.classes_):
                p[CLASSES.index(c)] = p_raw[j]
            probs.append(p)
        p = np.mean(probs, axis=0)
        label = CLASSES[int(np.argmax(p))]
        p_fall = float(p[CLASSES.index("fall")])
        if p_fall >= self.tau:
            label = "fall"
        return {"event_t_rel": ev["event_t"], "label": label, "probs": p, "p_fall": p_fall,
                "receivers": sorted(procs)}


class TwoHeadClassifier:
    """Adopted system (bundle format two_head_v1, validated by validate_adopted.py):
      localisation   fused, reliability-weighted, gap-masked, soft protocol prior (event_localizer)
      fall head      event-segment features + window aggregates, one RF per receiver, late fusion
      activity head  the same + the CSI domain features that passed the acceptance test
                     (F1/F4 use the posture direction / DTW templates stored in the bundle)
      combination    P(fall) from the fall head; the activity head's other classes rescaled to 1-P(fall)
      decision       FALL iff P(fall) >= tau (the alarm operating point), else the most probable
                     other class. Classes: fall, walk, posture (sit + stand), hand, empty_room."""

    def __init__(self, bundle, far="3"):
        self.heads, self.feats = bundle["heads"], bundle["head_features"]
        for head in self.heads.values():                  # one sample per call: a thread pool costs more than
            for m in head.values():                       # it saves (2-receiver step 493 -> 324 ms) and steals
                m.n_jobs = 1                              # CPU from the serial reader
        self.classes = list(bundle["classes"])
        self.ffp, self.ref = bundle["foldfit_params"], bundle["localizer_ref"]
        th = bundle["fall_threshold"]
        self.tau = float(th.get(f"tau_far_{far}", th["tau"]))
        self.far = far
        self.fill = bundle.get("feature_fill", {})
        self.need_domain = any(c.startswith("F") for cs in self.feats["activity"].values() for c in cs)

    def features(self, procs, pkts=None, prior=None, wtabs=None):
        """procs: {rx: (trel, X, R)} on one shared grid; pkts: {rx: packet times on that grid};
        wtabs: precomputed window tables (the engine already builds them for the tier gates)."""
        trel = next(iter(procs.values()))[0]
        mats = {rx: p[1] for rx, p in procs.items()}
        ev = event_features_fused(trel, mats, pkts, self.ref, FS, prior=prior)
        feats = dict(ev)
        for rx, (tr, X, R) in procs.items():
            W = wtabs[rx] if wtabs and rx in wtabs else window_table(tr, X)
            feats.update(window_aggregates(W, "w1" if rx == "rx1" else "w2"))
            if self.need_domain:
                feats.update(rx_domain_features(tr, X, None, R, ev, rx))
                feats.update(foldfit_apply(posture_delta(tr, X, ev), motion_profile(tr, X, ev), self.ffp, rx))
        if self.need_domain and "rx1" in procs and "rx2" in procs:
            feats.update(trial_geometry(trel, procs["rx1"][1], procs["rx2"][1], ev))
        return feats

    def _head(self, head, rx, feats):
        clf = self.heads[head][rx]
        vals = []
        for c in self.feats[head][rx]:
            v = feats.get(c, np.nan)
            vals.append(v if np.isfinite(v) else self.fill.get(c, 0.0))   # as in training
        p_raw = clf.predict_proba(np.array([vals], dtype=float))[0]
        p = np.zeros(len(self.classes))
        for j, c in enumerate(clf.classes_):
            p[self.classes.index(c)] = p_raw[j]
        return p

    def decide(self, feats, rxs):
        pf = np.mean([self._head("fall", rx, feats) for rx in rxs], axis=0)
        pa = np.mean([self._head("activity", rx, feats) for rx in rxs], axis=0)
        fi = self.classes.index("fall")
        rest = [i for i in range(len(self.classes)) if i != fi]
        p = pa.copy()
        p[rest] = pa[rest] / (pa[rest].sum() + 1e-12) * (1.0 - pf[fi])
        p[fi] = pf[fi]
        p_fall = float(p[fi])
        label = "fall" if p_fall >= self.tau else self.classes[rest[int(np.argmax(p[rest]))]]
        return label, p, p_fall

    def classify(self, procs, pkts=None, prior=None, wtabs=None):
        feats = self.features(procs, pkts, prior, wtabs)
        label, p, p_fall = self.decide(feats, sorted(procs))
        return {"event_t_rel": feats["event_t"], "label": label, "probs": p, "p_fall": p_fall,
                "receivers": sorted(procs), "argmax": self.classes[int(np.argmax(p))]}


def make_classifier(bundle, far="3"):
    if bundle.get("format") == "two_head_v1":
        return TwoHeadClassifier(bundle, far)
    clf = EventClassifier(bundle)
    clf.classes = CLASSES
    return clf


# ----------------------------------------------------------------------------------------
# Engine
# ----------------------------------------------------------------------------------------
class LiveEngine:
    def __init__(self, bundle, use_rx=("rx1", "rx2"), calibrate_s=12.0, fast_factor=4.0,
                 log_dir="data/live_logs", ground_truth=None, quiet=False, far="3", tier1=False):
        self.use_tier1 = tier1
        self.clf = make_classifier(bundle, far)
        self.two_head = isinstance(self.clf, TwoHeadClassifier)
        self._pkts = {}
        self.buf = {rx: RxBuffer(rx) for rx in use_rx}
        self.calibrate_s, self.fast_factor = calibrate_s, fast_factor   # calibrate_s = warm-up only
        # scale-free tier gates per receiver (train_final.calibrate_tiers on the window features):
        # T1 = 2.5 % quantile of the falls' peak window motion, T2 = 95 % quantile of ADL bursts
        tiers = bundle.get("tiers", {})
        self.T1 = {rx: float(tiers[rx]["T1_dormant_motion"]) if rx in tiers else -math.inf for rx in use_rx}
        self.T2 = {rx: float(tiers[rx]["T2_fastpath_burst"]) if rx in tiers else math.inf for rx in use_rx}
        self.floor = {}   # legacy (absolute floor); kept so old scripts that set it still run
        self.start_t = None
        self.state = {"tier": 0, "status": "STARTING", "last_event": None, "alert_until": 0.0,
                      "pending_until": 0.0, "motion": {}, "rx_alive": {}}
        self.last_reported_t = -math.inf
        self.tier_counts = {1: 0, 2: 0, 3: 0}
        self.ground_truth = ground_truth
        self.quiet = quiet
        self._calib = {rx: [] for rx in use_rx}
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, f"session_{datetime.now():%Y%m%d_%H%M%S}.csv")
        with open(self.log_path, "w") as f:
            f.write("wall_time,event_time,label,p_fall,receivers,tier,ground_truth\n")
        self.events = []

    # -- receivers ------------------------------------------------------------------------
    def alive(self, now):
        """A receiver is used only if it delivered a packet in the last 1.5 s AND at least MIN_USE_HZ packets/s.
        A sparse stream (e.g. 8 pkt/s) gives unreliable features but would get an equal late-fusion vote."""
        return {rx: (now - b.last_time()) < 1.5 and b.arrival_rate() >= MIN_USE_HZ for rx, b in self.buf.items()}

    def processed(self, now, rxs, final=False):
        """Last BUF_S seconds of every usable receiver on ONE shared 100 Hz grid (as in
        merge_dual_rx.py: the grid covers the span where all receivers have data). Also keeps
        each receiver's packet times on that grid for the localiser's gap masking.
        final=True (end of a replayed trial): the whole buffered trial, exactly like training."""
        span = self.buf[rxs[0]].keep_s if (final and rxs) else BUF_S
        snaps = {}
        for rx in rxs:
            ts, A, R = self.buf[rx].snapshot(now - span, now)
            if len(ts) >= 20 and ts[-1] - ts[0] >= MIN_COVER_S:
                snaps[rx] = (ts, A, R)
        if not snaps:
            self._pkts = {}
            return {}
        t0 = max([now - span] + [s[0][0] for s in snaps.values()])
        t1 = min(s[0][-1] for s in snaps.values()) if final else now
        out, self._pkts, self._t0 = {}, {}, t0
        for rx, (ts, A, R) in snaps.items():
            trel, X, Rg = preprocess(ts, A, t0, t1, R)
            if X is not None:
                out[rx] = (trel, X, Rg)
                self._pkts[rx] = ts - t0
        return out

    # -- main step (called every CLASSIFY_EVERY_S) ------------------------------------------
    def step(self, now, final=False):
        if self.start_t is None:
            self.start_t = now
        alive = self.alive(now)
        rxs = [rx for rx, ok in alive.items() if ok] or ([*self.buf] if final else [])
        self.state["rx_alive"] = alive
        procs = self.processed(now, rxs, final)
        if not procs:
            self.state["status"] = "WAITING FOR DATA"
            return None
        # window features (2 s, hop 0.25 s) per receiver: used by the gates AND by the classifier
        wt = {rx: window_table(p[0], p[1]) for rx, p in procs.items()}
        end = {rx: p[0][-1] for rx, p in procs.items()}
        recent = lambda rx, s: wt[rx][wt[rx]["mid_s"] >= end[rx] - s]
        motion6 = {rx: float(recent(rx, 5.0)["ext_motion_rx1"].max()) for rx in procs}     # windows in the last 6 s
        burst1 = {rx: float(recent(rx, 1.25)["ext_motion_pk_rx1"].max()) for rx in procs}  # newest windows
        self.state["motion"] = {rx: float(recent(rx, 1.25)["ext_motion_rx1"].max()) for rx in procs}
        if now - self.start_t < self.calibrate_s and not final:
            self.state["status"] = f"WARMING UP - filling the {BUF_S:.0f} s buffer ({self.calibrate_s - (now - self.start_t):.0f}s left)"
            return None
        # Tier 1 (opt-in, --tier1: power saving for an edge device): dormant if no receiver's scale-free
        # motion reached T1 in the last 6 s. 6 s, not 2 s, so a person lying still after a fall still
        # reaches Tier 3. Costs: LOSO 4/79 falls lost, and CONTINUOUS activity (walking) normalises itself
        # away against the buffer's own noise scale and is often skipped - hence off by default.
        if self.use_tier1 and not final and all(motion6[rx] < self.T1[rx] for rx in procs) \
                and now > self.state["alert_until"]:
            self.state.update(tier=1, status="NO SIGNIFICANT MOTION (Tier 1 skip)")
            self.tier_counts[1] += 1
            return None
        # Tier 2: a burst above T2 only says 'something is happening' - in the recorded data it fires on
        # 41 % of falls but also 33 % of empty-room trials, so it is NOT shown as a fall warning
        if any(burst1[rx] >= self.T2[rx] for rx in procs):
            self.state["pending_until"] = now + POST_S + 1.0
            self.state["tier"] = 2
            self.tier_counts[2] += 1
        # Tier 3: event classification once >= POST_S of post-event data exists
        trel = next(iter(procs.values()))[0]
        hi = trel[-1] - POST_S if not final else 8.5
        if hi < 1.0:
            return None
        if self.two_head:
            # live: soft prior over [1 s, latest event with >= POST_S of after-data];
            # end of a replayed trial: the training prior (recording protocol, 0.5-7.5 s)
            res = self.clf.classify(procs, self._pkts, prior=None if final else (1.0, min(7.5, hi)), wtabs=wt)
        else:
            res = self.clf.classify({rx: p[:2] for rx, p in procs.items()}, search_ev=(1.0, min(8.5, hi)))
        self.tier_counts[3] += 1
        ev_abs = self._t0 + res["event_t_rel"]
        res["event_abs"] = ev_abs
        res["decided_at"] = now
        if final:   # replay end-of-trial summary: shown (and a fall alarm shown), but not logged as a new event
            self.state.update(tier=3, last_event=res, status=DISPLAY[res["label"]])
            if res["label"] == "fall" and not self.quiet and now >= self.state["alert_until"]:
                self.state["alert_until"] = now + 6.0
                threading.Thread(target=alarm, daemon=True).start()
            return res
        same_event = abs(ev_abs - self.last_reported_t) < POST_S
        alert_active = now < self.state["alert_until"] and res["label"] == "fall"
        if (same_event or alert_active) and not final:
            self.state["tier"] = 3
            return None
        if res["label"] == "empty_room" and not final:
            self.state.update(tier=3, status="PRESENCE / LOW ACTIVITY")
            return None
        self.last_reported_t = ev_abs
        self.state.update(tier=3, last_event=res, status=DISPLAY[res["label"]])
        if res["label"] == "fall":
            self.state["alert_until"] = now + 10.0
            threading.Thread(target=alarm, daemon=True).start()
        self.events.append(res)
        with open(self.log_path, "a") as f:
            f.write(f"{datetime.now().isoformat()},{ev_abs:.3f},{res['label']},{res['p_fall']:.4f},"
                    f"{'+'.join(res['receivers'])},3,{self.ground_truth or ''}\n")
        return res


def alarm():
    if HAS_WINSOUND:
        for _ in range(3):
            try:
                winsound.Beep(2200, 400); time.sleep(0.1)
            except Exception:
                return


def draw(engine, now, title):
    s = engine.state
    os.system("cls" if os.name == "nt" else "clear")
    print("=" * 72)
    print(f"  WiFeye LIVE  |  {title}")
    print("=" * 72)
    for rx, b in engine.buf.items():
        alive = s["rx_alive"].get(rx, False)
        t1 = engine.T1.get(rx, math.inf)
        mot = s["motion"].get(rx, 0.0)
        bar = "#" * int(min(30, 15 * mot / t1)) if np.isfinite(t1) and t1 > 0 else ""   # mid-bar = Tier-1 gate
        print(f"  {rx.upper()}: {'ONLINE ' if alive else 'OFFLINE'} {b.rate(now):6.1f} pkt/s  motion |{bar:<30}|")
    mode = "+".join(rx.upper() for rx, ok in s["rx_alive"].items() if ok) or "-"
    model = "two-head RF, fused localiser" if engine.two_head else "event RF"
    print(f"  Model: {model} | late fusion ({mode})   Tier: {s['tier']}   tiers so far {engine.tier_counts}")
    print("-" * 72)
    if now < s["alert_until"]:
        print("\n  " + "!" * 66)
        print("  !!!            >>> EMERGENCY ALERT: FALL DETECTED! <<<              !!!")
        print("  " + "!" * 66 + "\n")
    elif now < s["pending_until"]:
        print("\n  >>> MOTION BURST - analysing the event (Tier 2) ...\n")
    print(f"  STATUS: [{s['status']}]")
    ev = s["last_event"]
    if ev:
        top = sorted(zip(engine.clf.classes, ev["probs"]), key=lambda x: -x[1])[:3]
        print(f"  Last event: {DISPLAY[ev['label']]}  (P(fall)={ev['p_fall']:.2f}, tau={engine.clf.tau:.2f}, "
              f"receivers {'+'.join(ev['receivers'])})")
        print("  Top classes: " + ", ".join(f"{DISPLAY[c]} {p:.2f}" for c, p in top))
    if engine.ground_truth:
        gt = DISPLAY.get(engine.ground_truth, engine.ground_truth)
        if engine.two_head and engine.ground_truth in ("sit", "stand"):
            gt += "  -> class POSTURE CHANGE"
        print(f"  Ground truth (replay): {gt}")
    print("=" * 72)


# ----------------------------------------------------------------------------------------
# Sources
# ----------------------------------------------------------------------------------------
def trial_available(trial_id, raw_dir="data/raw"):
    act = "empty_room" if "_empty_room_" in trial_id else trial_id.split("_")[1]
    return all(os.path.exists(os.path.join(raw_dir, act, f"{trial_id}_{r}.csv")) for r in ("RX1", "RX2"))


def load_trial_packets(trial_id, raw_dir="data/raw"):
    """Recorded trial -> {rx: (epoch_times, amps52)} after the v3 packet filter."""
    act = "empty_room" if "_empty_room_" in trial_id else trial_id.split("_")[1]
    out = {}
    for rx, role, mac in (("rx1", "RX1", "C0:CD:D6:85:65:EC"), ("rx2", "RX2", "1C:C3:AB:B4:A2:48")):
        f = os.path.join(raw_dir, act, f"{trial_id}_{role}.csv")
        if not os.path.exists(f):
            continue
        df = pd.read_csv(f, usecols=["epoch_timestamp", "mac", "sig_mode", "rssi"] + [f"amp_{i}" for i in KEEP_SC])
        df = df[(df["mac"].str.upper() == mac) & (df["sig_mode"] == 1)]
        out[rx] = (df["epoch_timestamp"].values, df[[f"amp_{i}" for i in KEEP_SC]].values.astype(float),
                   df["rssi"].values.astype(float))
    return act, out


def replay(engine, trial_id, speed=1.0, show=True, relay_rx2=False, start_at=None, ui=None):
    """Stream one trial through the engine in (scaled) real time; final decision at the end.
    relay_rx2=True: RX2 is NOT read from file but arrives over UDP from a worker replay
    (integration test of the 2-laptop relay); start_at aligns both processes' clocks.
    ui: optional ui_dashboard.UiFeed, refreshed ~4x per second for the browser dashboard."""
    act, pk = load_trial_packets(trial_id)
    engine.ground_truth = act
    t0 = min(v[0][0] for v in pk.values())
    t_end = max(v[0][-1] for v in pk.values())
    if relay_rx2:
        pk = {rx: v for rx, v in pk.items() if rx != "rx2"}
        threading.Thread(target=_relay_listener, args=(engine, 5600), daemon=True).start()
    if start_at:
        time.sleep(max(0.0, start_at - time.time()))
    base = start_at or time.time()
    if speed <= 0:   # instant mode: push everything, one end-of-trial decision (batch evaluation)
        for rx, (ts, A, R) in pk.items():
            for t, a, r in zip(ts, A, R):
                engine.buf[rx].push(base + (t - t0), a, r)
        return act, engine.step(base + (t_end - t0), final=True)
    clock = lambda: (time.time() - base) * speed
    idx = {rx: 0 for rx in pk}
    next_cls, next_ui = CLASSIFY_EVERY_S, 0.0
    while True:
        now_rel = clock()
        for rx, (ts, A, R) in pk.items():
            while idx[rx] < len(ts) and ts[idx[rx]] - t0 <= now_rel:
                engine.buf[rx].push(base + (ts[idx[rx]] - t0), A[idx[rx]], R[idx[rx]])
                idx[rx] += 1
        if now_rel >= next_cls:
            engine.step(base + now_rel)
            next_cls += CLASSIFY_EVERY_S
            if show:
                draw(engine, base + now_rel, f"REPLAY {trial_id}")
        if ui is not None and now_rel >= next_ui:
            ui.update(base + now_rel)
            next_ui += ui.tick
        if now_rel > (t_end - t0) + 0.05:
            break
        time.sleep(0.01)
    res = engine.step(base + (t_end - t0), final=True)
    if ui is not None:
        ui.update(base + (t_end - t0))
    if show:
        draw(engine, base + (t_end - t0), f"REPLAY {trial_id} (end of trial)")
    return act, res


def run_serial(engine, port, own_mac, dual, worker_ip, relay_port, title, ui=None, show=True, pump_hz=140.0):
    from dataset_collector import NetworkSyncNode

    def waiting(msg):                                     # keep the dashboard alive while waiting for USB
        engine.state["status"] = msg
        if ui is not None:
            ui.update(time.time())

    probe, port = open_esp32(port, "RX1", waiting)        # wait until the port exists and is free ...
    probe.ser.close()                                     # ... then hand it to the reader process
    engine.state["status"] = "STARTING"
    # ESP32 reading + 140 Hz traffic pump in their own process: the classifier cannot starve them
    import multiprocessing as mp
    import serial_proc
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    stop_evt = ctx.Event()
    engine.serial_hz = ctx.Value("d", 0.0)
    kept = ctx.Value("i", 0)
    proc = ctx.Process(target=serial_proc.reader_main,
                       args=(port, own_mac, q, stop_evt, engine.serial_hz, kept, pump_hz), daemon=True)
    proc.start()
    print(f"  RX1 traffic pump: {'off' if pump_hz <= 0 else f'{pump_hz:.0f} Hz'}")

    def feeder():                                         # queue -> RX1 buffer (the queue absorbs any delay)
        while True:
            item = q.get()
            if item[0] == "pkt":
                engine.buf["rx1"].push(item[1], item[2].astype(float), item[3])
            elif item[0] == "ready":
                print(f"  RX1 reader process running on {item[1]}")
            elif item[0] == "macs":
                seen = ", ".join(f"{m} ({n})" for m, n in item[1])
                print(f"\n  WARNING RX1: CSI is arriving, but none of it carries this board's address {own_mac}.\n"
                      f"  Addresses seen: {seen}\n  {MacWatch.ADVICE}\n")
                engine.diag = (f"CSI is arriving but none carries this board's address {own_mac}. "
                               f"Addresses seen: {seen}")
            elif item[0] == "error":
                print(f"  RX1 reader error: {item[1]}")
                engine.diag = f"RX1 reader error: {item[1]}"
    threading.Thread(target=feeder, daemon=True).start()

    class _Watch:                                         # MAC diagnostics now happen in the reader process
        def check(self):
            pass
    watch = _Watch()

    class _Reader:
        def stop(self):
            stop_evt.set()
            proc.join(timeout=3)
    reader = _Reader()
    sync = None
    _check_wifi("Master")
    if dual:
        sync = NetworkSyncNode(is_master=True, target_ip=worker_ip or "<broadcast>")
        sync.start()
        threading.Thread(target=_relay_listener, args=(engine, relay_port), daemon=True).start()
        threading.Thread(target=_periodic_sync, args=(sync, worker_ip), daemon=True).start()
    try:
        next_step = time.time()
        while True:
            now = time.time()
            if now >= next_step:                          # classification once per second
                watch.check()
                engine.step(now)
                if show:
                    draw(engine, now, title)
                else:                                     # one updating status line (keeps earlier messages)
                    parts = [f"{rx.upper()} {b.arrival_rate():5.1f} pkt/s" for rx, b in engine.buf.items()]
                    if getattr(engine, "serial_hz", None) is not None:
                        parts.insert(1, f"RX1 serial {engine.serial_hz.value:5.1f}/s")
                    if sync is not None:
                        parts.append(f"worker {sync.worker_ip or 'not heard yet'}")
                    print(f"\r  {' | '.join(parts)} | {engine.state.get('status', '')[:40]:<40}", end="", flush=True)
                next_step = now + CLASSIFY_EVERY_S
            if ui is not None:                            # dashboard refresh ~4x per second
                ui.update(time.time())
                time.sleep(ui.tick)
            else:
                time.sleep(max(0.0, next_step - time.time()))
    except KeyboardInterrupt:
        pass
    finally:
        reader.stop()
        if sync:
            sync.stop()


def _check_wifi(role):
    """Warn when this laptop is not on the router's Wi-Fi 'CSI' (no discovery, no traffic pump)."""
    from dataset_collector import get_current_wifi_ssid
    ssid = get_current_wifi_ssid()
    if ssid and ssid.strip().lower() != "csi":
        print(f"\n  WARNING {role}: this laptop is on Wi-Fi '{ssid}', not 'CSI'. The two laptops cannot "
              f"find each other and the ESP32 gets few packets. Switch to 'CSI'.\n")


def _periodic_sync(sync, worker_ip):
    """Clock sync with the worker: retry every 2 s until it answers, then every 30 s. A worker that
    announces itself (WORKER_HELLO) is synced at once, by unicast."""
    known = None
    while True:
        try:
            sync.calibrate_clock(iterations=15, destination_ip=worker_ip)
        except Exception:
            pass
        if sync.worker_ip and sync.worker_ip != known:
            known = sync.worker_ip
            synced = sync.min_rtt_ms < 100
            print(f"\n  Worker heard at {known}" + (f" - clock synced (round trip {sync.min_rtt_ms:.1f} ms)"
                                                   if synced else " - syncing clock ..."))
        wait_until = time.time() + (30.0 if sync.min_rtt_ms < 100 else 2.0)
        while time.time() < wait_until and not (sync.worker_ip and sync.worker_ip != known):
            time.sleep(0.2)


def _relay_listener(engine, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 << 20)   # 4 MB: survives classifier bursts
    except OSError:
        pass
    s.bind(("0.0.0.0", port))
    while True:
        data, _ = s.recvfrom(4096)
        if len(data) == RELAY_SIZE:
            vals = struct.unpack(RELAY_FMT, data)
            engine.buf["rx2"].push(vals[0], np.array(vals[1:53], dtype=float), vals[53])


def worker_replay(trial_id, master_ip, relay_port, start_at, drop_after=None):
    """Relay test without hardware: send a recorded trial's RX2 packets as the worker would."""
    _, pk = load_trial_packets(trial_id)
    t0 = min(v[0][0] for v in pk.values())
    ts, A, R = pk["rx2"]
    out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    time.sleep(max(0.0, start_at - time.time()))
    sent = 0
    for t, a, r in zip(ts, A, R):
        if drop_after is not None and t - t0 > drop_after:
            break
        target = start_at + (t - t0)
        time.sleep(max(0.0, target - time.time()))
        out.sendto(struct.pack(RELAY_FMT, target, *a.astype(np.float32), float(r)), (master_ip, relay_port))
        sent += 1
    print(f"  worker replay: relayed {sent} RX2 packets of {trial_id}"
          + (f" (dropped after {drop_after}s)" if drop_after is not None else ""))


def resolve_port(port, wait=False):
    """'auto' -> first USB-serial port (same rule as dataset_collector.py).
    wait=True: return None instead of exiting when no port is present (the caller keeps waiting)."""
    if port and port.lower() != "auto":
        return port
    import serial.tools.list_ports
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if any(x in (p.description or "").lower() for x in ["cp210", "ch340", "uart", "usb", "serial", "silicon"]):
            print(f"  Auto-selected ESP32 port: {p.device} ({p.description})")
            return p.device
    if ports:
        return ports[0].device
    if wait:
        return None
    sys.exit("  No serial port found - is the ESP32 plugged in?")


def open_esp32(port_arg, role, on_wait=None):
    """Find and open the ESP32's serial port, WAITING (instead of quitting) until it is plugged in and free.
    on_wait(message) is called about once per second while waiting (e.g. to refresh the dashboard)."""
    from dataset_collector import SerialCSIReader
    last = None
    while True:
        p = resolve_port(port_arg, wait=True)
        if p is None:
            msg = f"PLUG IN ESP32 {role} (USB)"
            if msg != last:
                print(f"  No ESP32 found on USB. Waiting for {role} to be plugged in ... (Ctrl+C to stop)")
        else:
            try:
                reader = SerialCSIReader(p, 921600)
                reader.connect()
                try:
                    reader.ser.set_buffer_size(rx_size=1 << 20)   # Windows: 1 MB driver buffer if allowed
                except Exception:
                    pass
                print(f"  {role}: ESP32 opened on {p}")
                return reader, p
            except Exception as ex:
                if "FileNotFound" in str(ex) or "cannot find" in str(ex).lower():
                    msg = f"{p} NOT FOUND - PLUG IN ESP32 {role}"
                    hint = f"{p} does not exist: plug in the ESP32 and check its COM number in Device Manager"
                else:
                    msg = f"CANNOT OPEN {p} - CLOSE OTHER PROGRAMS USING IT"
                    hint = f"{p} is busy ({ex}): close any serial monitor / Arduino IDE using it"
                if msg != last:
                    print(f"  {hint}; retrying ...")
        last = msg
        if on_wait:
            on_wait(msg)
        time.sleep(1.0)


def run_worker(port, master_ip, relay_port, own_mac, pump_hz=140.0):
    """Relay RX2 to the master over the 'CSI' Wi-Fi. The master's IP is learned from its
    sync messages (UDP 5555) unless --master-ip is given."""
    from dataset_collector import SerialCSIReader, NetworkSyncNode
    sync = NetworkSyncNode(is_master=False)
    if master_ip:
        sync.master_ip = master_ip                        # --master-ip: announce straight to it
    sync.start()
    out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    reader, port = open_esp32(port, "RX2")                # waits until the ESP32 is plugged in and free
    sent = [0]
    target = lambda: master_ip or sync.master_ip
    watch = MacWatch("RX2", own_mac)

    def on_pkt(pkt):
        a = own_ht_amplitudes(pkt, own_mac)
        watch.see(pkt, a is not None)
        dest = target()
        if a is not None and dest:
            out.sendto(struct.pack(RELAY_FMT, pkt.sys_time + sync.clock_offset, *a.astype(np.float32),
                                   float(pkt.rssi)), (dest, relay_port))
            sent[0] += 1
    reader.pump_hz = pump_hz
    reader.start(on_pkt)
    print(f"  Worker: RX2 ({own_mac}) on {port}, traffic pump {'off' if pump_hz <= 0 else f'{pump_hz:.0f} Hz'}. "
          f"Waiting for the master on Wi-Fi 'CSI' ... Ctrl+C to stop.")
    _check_wifi("Worker")
    try:
        while True:
            time.sleep(1.0)
            sync.announce()                               # say hello every second (see NetworkSyncNode.announce)
            watch.check()
            dest = target() or (f"not found yet - {sync.hellos_sent} hellos sent, nothing heard back "
                                f"(is Laptop A on 'CSI' and running?)")
            own_rate = watch.kept - getattr(run_worker, "_last", 0); run_worker._last = watch.kept
            print(f"\r  master: {dest} | own frames {own_rate:3d}/s | relayed {sent[0]:7d} | "
                  f"serial {reader.live_rate_hz:5.1f} Hz | "
                  f"clock offset {sync.clock_offset * 1000:+.2f} ms   ", end="", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        reader.stop(); sync.stop()


# ----------------------------------------------------------------------------------------
# Honest replay (LOSO) and parity self-test
# ----------------------------------------------------------------------------------------
def demo_trials(version="v3", two_head=True):
    """One recorded trial per activity (sit and stand both shown) that the model classified
    correctly OUT OF SAMPLE (LOSO); a typical, median-confidence case of each."""
    fallback = ["2_fall_t05", "4_walk_t03", "5_sit_t02", "1_stand_t04", "3_hand_t06", "6_empty_room_t03"]
    path = "data/models_exp/adopted_v3/adopted_oof.csv" if two_head else f"data/models_exp/event_{version}/hyb_late_oof.csv"
    if not os.path.exists(path):
        return fallback
    oof = pd.read_csv(path)
    if two_head:
        alarm = oof["alarm_far3"].astype(bool)
        rest = oof[["p_walk", "p_posture", "p_hand", "p_empty_room"]].values.argmax(1)
        oof["pred"] = np.where(alarm, "fall", np.array(["walk", "posture", "hand", "empty_room"])[rest])
        ok = oof[oof["true5"] == oof["pred"]]
        groups = [(c, ok[ok["true6"] == c], "p_fall" if c == "fall" else f"p_{MERGE5.get(c, c)}") for c in CLASSES]
    else:
        ok = oof[oof["true"] == oof["pred"]]
        groups = [(c, ok[ok["true"] == c], f"p_{c}") for c in CLASSES]
    picks = []
    for c, rows, pc in groups:
        rows = rows.sort_values(pc, ascending=False)
        if len(rows):
            picks.append(rows["trial_id"].iloc[len(rows) // 2])
    return picks


MERGE5 = {"sit": "posture", "stand": "posture"}


def loso_bundle(bundle, subject, version="v3"):
    """Retrain without one subject (about a minute) for an honest replay. Two-head bundles are
    rebuilt by train_final.make_adopted_bundle (feature selection, posture direction and DTW
    templates refit without the subject) and use that LOSO fold's nested fall threshold.
    Caveat: the training subjects' F1/F4 inputs are the stored out-of-fold values, whose own
    folds included the held-out subject (3 of the 42 domain features; a minor, conservative-
    to-ignore leak, documented rather than hidden)."""
    if bundle.get("format") == "two_head_v1":
        import train_final
        b = train_final.make_adopted_bundle(exclude_subject=str(subject))
        rpath = "data/models_exp/adopted_v3/adopted_results.json"
        if os.path.exists(rpath):
            with open(rpath) as fh:
                r = json.load(fh)
            subjects = ["1", "2", "3", "4", "5"]          # tc.loso_folds order: fold k = subject k+1
            fold = str(subjects.index(str(subject))) if str(subject) in subjects else None
            tpf = r.get("tau_per_fold", {}).get(fold) if fold is not None else None
            if tpf:
                b["fall_threshold"] = dict(b["fall_threshold"], tau=tpf["0.05"], tau_far_3=tpf["0.03"],
                                           tau_far_5=tpf["0.05"], tau_far_10=tpf["0.1"],
                                           source=f"nested LOSO threshold, fold without subject {subject}")
        return b
    import evaluate_event_model as em
    from sklearn.ensemble import RandomForestClassifier
    ev = pd.read_csv(f"data/event_features_{version}.csv", dtype={"subject": str})
    T = ev.set_index("trial_id").join(em.window_aggregates(f"data/features_{version}_rx1", "w1")) \
          .join(em.window_aggregates(f"data/features_{version}_rx2", "w2")).reset_index().dropna()
    T = T[T["subject"] != str(subject)]
    b = dict(bundle)
    b["event_models"] = {rx: RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced",
                                                    n_jobs=-1, random_state=42).fit(
        T[bundle["event_feature_names"][rx]].values, T["activity"].values) for rx in ("rx1", "rx2")}
    return b


def selftest_two_head(bundle, n, far="3"):
    """Two-head parity: the live code path (raw packets -> buffer pipeline -> fused localiser ->
    features -> heads) against the offline training tables for the same recorded trials.
    Feature error is normalised by the feature's spread over all trials (IQR), so near-zero
    features do not inflate it. F1/F4 offline values are recomputed from the stored training
    material with the bundle's parameters (the training table holds out-of-fold versions)."""
    import evaluate_event_model as em
    seg = pd.read_csv("data/event_features_v3_fused.csv", dtype={"subject": str}).set_index("trial_id")
    agg = em.window_aggregates("data/features_v3_rx1", "w1").join(em.window_aggregates("data/features_v3_rx2", "w2"))
    dom = pd.read_csv("data/models_exp/eda_domain_v3/domain_features.csv", dtype={"subject": str}).set_index("trial_id")
    z = np.load("data/models_exp/eda_domain_v3/foldfit_material.npz", allow_pickle=True)
    deltas, profs = z["deltas"].item(), z["profs"].item()
    off_all = seg.join(agg).join(dom.drop(columns=[c for c in ("activity", "subject") if c in dom.columns]))
    clf = TwoHeadClassifier(bundle, far)
    names = sorted({c for h in clf.feats.values() for cs in h.values() for c in cs})
    names += [c for c in ("event_t", "event_onset", "event_offset") if c not in names]
    num = off_all[[c for c in names if c in off_all.columns]].apply(pd.to_numeric, errors="coerce")
    scale = (num.quantile(0.75) - num.quantile(0.25)).replace(0, np.nan).fillna(num.std()).fillna(1.0) + 1e-9
    available = [t for t in off_all.index if trial_available(t) and t in deltas["rx1"]]
    if not available:
        sys.exit("  No raw trials found under data/raw - copy the dataset or the sample data first.")
    ids = np.random.default_rng(0).choice(available, size=min(n, len(available)), replace=False)
    agree, agree_argmax, dpf, errs = [], [], [], {}
    for tid in ids:
        _, pk = load_trial_packets(tid)
        t0 = max(v[0][0] for v in pk.values()); t1 = min(v[0][-1] for v in pk.values())
        procs, pkts = {}, {}
        for rx, (ts, A, R) in pk.items():
            procs[rx] = preprocess(ts, A, t0, t1, R)
            pkts[rx] = ts - t0
        live = clf.features(procs, pkts, prior=None)
        off = {c: float(off_all.loc[tid, c]) for c in names if c in off_all.columns}
        for rx in ("rx1", "rx2"):
            off.update(foldfit_apply(deltas[rx][tid], profs[rx][tid], clf.ffp, rx))
        for c in names:
            if c in live and c in off and np.isfinite(off[c]) and np.isfinite(live[c]):
                s = float(scale[c]) if c in scale.index else max(abs(off[c]), 1.0)
                errs.setdefault(c, []).append(abs(live[c] - off[c]) / s)
        l_live, p_live, pf_live = clf.decide(live, ["rx1", "rx2"])
        l_off, p_off, pf_off = clf.decide(off, ["rx1", "rx2"])
        agree.append(l_live == l_off)
        agree_argmax.append(int(np.argmax(p_live)) == int(np.argmax(p_off)))
        dpf.append(abs(pf_live - pf_off))
    allerr = np.concatenate([np.asarray(v) for v in errs.values()])
    worst = sorted(((np.max(v), c) for c, v in errs.items()), reverse=True)[:5]
    fam = {}
    for c, v in errs.items():
        f = c.split("_")[0] if c.startswith("F") else ("event" if c.startswith(("ev_", "event")) else "window agg")
        fam.setdefault(f, []).extend(v)
    print(f"\n  Two-head self-test on {len(ids)} recorded trials (live code path vs offline training tables)")
    print(f"   decision agreement {np.mean(agree):.1%} | argmax agreement {np.mean(agree_argmax):.1%} | "
          f"max |dP(fall)| {np.max(dpf):.4f}")
    print(f"   feature error / IQR: median {np.median(allerr):.4f}, 90th pct {np.percentile(allerr, 90):.4f}, "
          f"99th pct {np.percentile(allerr, 99):.4f}  ({len(errs)} features)")
    for f, v in sorted(fam.items()):
        print(f"     {f:<11} median {np.median(v):.4f}  max {np.max(v):.4f}")
    print("   largest single deviations: " + ", ".join(f"{c} {e:.3f}" for e, c in worst))
    return float(np.mean(agree))


def selftest(bundle, n, version="v3"):
    """Live code path vs offline features/predictions on the same recorded trials."""
    if bundle.get("format") == "two_head_v1":
        return selftest_two_head(bundle, n)
    import evaluate_event_model as em
    ev = pd.read_csv(f"data/event_features_{version}.csv", dtype={"subject": str})
    T = ev.set_index("trial_id").join(em.window_aggregates(f"data/features_{version}_rx1", "w1")) \
          .join(em.window_aggregates(f"data/features_{version}_rx2", "w2"))
    rng = np.random.default_rng(0)
    available = [t for t in T.index.values if trial_available(t)]
    if not available:
        sys.exit("  No raw trials found under data/raw - copy the dataset or the sample data first.")
    ids = rng.choice(available, size=min(n, len(available)), replace=False)
    clf = EventClassifier(bundle)
    agree, rel_err = [], []
    for tid in ids:
        _, pk = load_trial_packets(tid)
        t0 = max(v[0][0] for v in pk.values()); t1 = min(v[0][-1] for v in pk.values())
        procs = {}
        for rx, (ts, A, _) in pk.items():
            trel, X, _ = preprocess(ts, A, t0, t1)
            procs[rx] = (trel, X)
        nmin = min(len(p[0]) for p in procs.values())
        procs = {rx: (tr[:nmin], X[:nmin]) for rx, (tr, X) in procs.items()}
        live = clf.classify(procs)
        off_vec = {rx: T.loc[tid, bundle["event_feature_names"][rx]].values.astype(float) for rx in ("rx1", "rx2")}
        off_p = np.mean([clf.models[rx].predict_proba(np.nan_to_num(off_vec[rx][None]))[0] for rx in ("rx1", "rx2")], axis=0)
        off_label = list(clf.models["rx1"].classes_)[int(np.argmax(off_p))]
        live_label = CLASSES[int(np.argmax(live["probs"]))]
        agree.append(live_label == off_label)
        # feature parity on the shared event features
        live_ev = event_features(procs["rx1"][0], {rx: X for rx, (_, X) in procs.items()}, FS)
        for k in ("ev_rx1_log_peak", "ev_rx1_posture_shift", "ev_rx1_burst_dur", "ev_rx2_log_peak"):
            off = float(T.loc[tid, k]); rel_err.append(abs(live_ev[k] - off) / (abs(off) + 1e-6))
    print(f"\n  Self-test on {len(ids)} trials: live vs offline label agreement {np.mean(agree):.1%}; "
          f"median relative feature error {np.median(rel_err):.2%} (90th pct {np.percentile(rel_err, 90):.2%})")
    return float(np.mean(agree))


# ----------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="WiFeye live demo engine")
    ap.add_argument("--role", choices=["master", "worker"], default="master")
    ap.add_argument("--port", help="ESP32 serial port, e.g. COM3, or 'auto'")
    ap.add_argument("--dual", action="store_true", help="master: also receive RX2 from the worker")
    ap.add_argument("--worker-ip", default=None)
    ap.add_argument("--master-ip", default=None)
    ap.add_argument("--relay-port", type=int, default=5600)
    ap.add_argument("--pump-hz", type=float, default=140.0,
                    help="UDP traffic pump on this laptop (140 = as in data collection; 0 = off). Lower it if the "
                         "ESP32s stop receiving their own frames because the channel is flooded")
    ap.add_argument("--mac", default=None,
                    help="this ESP32's own MAC if a different board is used (default: RX1 C0:CD:D6:85:65:EC on the "
                         "master, RX2 1C:C3:AB:B4:A2:48 on the worker)")
    ap.add_argument("--bundle", default="data/models/safesense_live_bundle.pkl")
    ap.add_argument("--calibrate", type=float, default=12.0,
                    help="warm-up seconds before reporting (buffer fill); the room need not be empty")
    ap.add_argument("--replay", default=None, help="trial id, e.g. 2_fall_t05")
    ap.add_argument("--replay-demo", action="store_true")
    ap.add_argument("--loso-subject", default=None)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--selftest", type=int, default=0)
    ap.add_argument("--relay-test", action="store_true",
                    help="replay: take RX2 from a worker replay over UDP instead of the file")
    ap.add_argument("--start-at", type=float, default=None, help="epoch start time shared by both test processes")
    ap.add_argument("--drop-after", type=float, default=None, help="worker replay: stop relaying after N s")
    ap.add_argument("--no-display", action="store_true", help="no console dashboard")
    ap.add_argument("--ui", action="store_true", help="open the browser dashboard (dashboard.html)")
    ap.add_argument("--ui-port", type=int, default=8050)
    ap.add_argument("--ui-exit", action="store_true", help="replay: exit when done instead of keeping the dashboard up")
    ap.add_argument("--no-browser", action="store_true", help="start the dashboard server without opening a browser")
    ap.add_argument("--tier1", action="store_true",
                    help="enable the Tier-1 dormant gate (saves compute; loses ~5%% of falls, skips steady walking)")
    ap.add_argument("--fall-far", default="3", choices=["3", "5", "10"],
                    help="fall-alarm operating point: target %% of ADL trials raising a false alarm "
                         "(LOSO: 3 %% and 5 %% catch the same 38 %% of falls, so 3 is the default)")
    args = ap.parse_args()

    if os.name == "nt":
        os.system("")  # enable ANSI on Windows consoles
    if args.role == "worker" and args.replay:
        worker_replay(args.replay, args.master_ip or "127.0.0.1", args.relay_port, args.start_at, args.drop_after)
        return
    if args.role == "worker":
        run_worker(args.port, args.master_ip, args.relay_port, args.mac or "1C:C3:AB:B4:A2:48", args.pump_hz)
        return

    bundle = joblib.load(args.bundle)
    two_head = bundle.get("format") == "two_head_v1"
    feed = None
    if args.ui and not args.selftest:
        import ui_dashboard
        feed = ui_dashboard.UiFeed("starting")
        ui_dashboard.start_server(feed, args.ui_port, open_browser=not args.no_browser)
    if args.selftest:
        if two_head:
            selftest_two_head(bundle, args.selftest, args.fall_far)
        else:
            selftest(bundle, args.selftest)
        return

    if args.replay or args.replay_demo or args.loso_subject:
        if args.loso_subject:
            print(f"  Retraining without subject {args.loso_subject} (honest replay) ...")
            bundle = loso_bundle(bundle, args.loso_subject)
            ev = pd.read_csv("data/event_features_v3_fused.csv" if two_head else "data/event_features_v3.csv",
                             dtype={"subject": str})
            trials = [t for t in ev.loc[ev["subject"] == str(args.loso_subject), "trial_id"] if trial_available(t)]
            if not trials:
                sys.exit(f"  No raw trials of subject {args.loso_subject} found under data/raw.")
        elif args.replay_demo:
            trials = [t for t in demo_trials(two_head=two_head) if trial_available(t)]
        else:
            trials = [args.replay]
        results = []
        for tid in trials:
            eng = LiveEngine(bundle, calibrate_s=0.0, quiet=bool(args.loso_subject) and feed is None,
                             far=args.fall_far)
            if feed is not None:
                kind = f"unseen subject {args.loso_subject}" if args.loso_subject else "recorded trial"
                feed.attach(eng, f"Replay · {kind} · {tid}")
            act, res = replay(eng, tid, speed=(0 if (args.loso_subject and feed is None) else args.speed),
                              show=not (args.loso_subject or args.no_display or feed is not None),
                              relay_rx2=args.relay_test, start_at=args.start_at, ui=feed)
            if args.relay_test:
                print(f"  relay test: RX2 packets received over UDP = {eng.buf['rx2'].total}, "
                      f"receivers used in final decision = {res['receivers'] if res else None}, "
                      f"events reported during stream = {[(e['label'], e['receivers']) for e in eng.events]}")
            truth = MERGE5.get(act, act) if two_head else act
            results.append((tid, truth, res["label"] if res else "none", res["p_fall"] if res else float("nan")))
            if not args.loso_subject or feed is not None:
                time.sleep(3.0 if feed is not None else 2.0)
        print(f"\n  Replay summary (trial, truth, decision, P(fall); FALL iff P(fall) >= tau = {eng.clf.tau:.3f}):")
        for r in results:
            print(f"   {r[0]:<20} {r[1]:<11} {r[2]:<11} {r[3]:.2f}")
        if args.loso_subject:
            acc = np.mean([r[1] == r[2] for r in results])
            print(f"  Honest accuracy for unseen subject {args.loso_subject}: {acc:.1%} over {len(results)} trials")
            if two_head:
                cls = sorted({r[1] for r in results})
                for c in cls:
                    rs = [r for r in results if r[1] == c]
                    print(f"     {c:<11} {np.mean([r[2] == c for r in rs]):.0%} of {len(rs)}")
        if not args.loso_subject:
            print("  NOTE: normal replay uses the deployed model, which was trained on these trials;"
                  " use --loso-subject for an out-of-sample demo.")
        if feed is not None and not args.ui_exit:
            print("  Replay finished; the dashboard stays up. Ctrl+C to quit.")
            try:
                while True:
                    time.sleep(1.0)
            except KeyboardInterrupt:
                pass
        return

    if not args.port:
        sys.exit("master needs --port COMx or --port auto (or --replay / --replay-demo / --loso-subject / --selftest)")
    use = ("rx1", "rx2") if args.dual else ("rx1",)
    eng = LiveEngine(bundle, use_rx=use, calibrate_s=args.calibrate, far=args.fall_far, tier1=args.tier1)
    title = "LIVE " + ("2-ESP32 (RX1 local + RX2 relayed over Wi-Fi 'CSI')" if args.dual else "1-ESP32 (RX1)")
    if feed is not None:
        feed.attach(eng, "Live · 2 ESP32 over Wi-Fi “CSI”" if args.dual else "Live · 1 ESP32 (RX1)")
    # with the browser dashboard the console is not redrawn, so startup messages and warnings stay readable
    run_serial(eng, args.port, args.mac or "C0:CD:D6:85:65:EC", args.dual, args.worker_ip, args.relay_port, title,
               ui=feed, show=not args.no_display and feed is None, pump_hz=args.pump_hz)


if __name__ == "__main__":
    main()
