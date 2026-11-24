"""
WiFeye: browser dashboard for the live engine
=============================================
A local web page (dashboard.html) that shows what the live engine detects, refreshed ~3x per second:
current activity, FALL alert, P(fall) vs the alarm threshold, class probabilities, packets/s and status per
receiver, a live CSI waterfall and motion trace per receiver, and the recent event timeline.

Uses only the Python standard library (http.server), so nothing extra to install and no internet needed.
The page is served on http://127.0.0.1:<port>/ and reads JSON from /state.

Usage (through the engine):
  python live_demo_engine.py --role master --port auto --dual --ui      live, 2 ESP32
  python live_demo_engine.py --replay-demo --ui                          recorded trials (no hardware)
"""

import os
import json
import time
import threading
import webbrowser
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PAGE = os.path.join(HERE, "dashboard.html")
DISPLAY = {"fall": "FALL", "walk": "Walking", "posture": "Posture change", "hand": "Hand gesture",
           "empty_room": "No activity", "sit": "Sitting down", "stand": "Standing up"}


class UiFeed:
    """Rolling, JSON-ready view of a LiveEngine. update(now) is called by the engine loop ~4x per second."""

    def __init__(self, mode, tick=0.25, hist_s=12.0):
        self.mode, self.tick = mode, tick
        self.n = int(round(hist_s / tick))
        self.lock = threading.Lock()
        self.snap = {"ready": False}
        self.history = []
        self._seen = set()
        self.t_start = time.time()
        self.e = None
        self._last = None                     # last shown event (kept across replayed trials)

    def attach(self, engine, label=None):
        """Follow a (new) engine; rolling plots restart, the event history is kept."""
        self.e = engine
        if label:
            self.mode = label
        self.cols = {rx: deque(maxlen=self.n) for rx in engine.buf}
        self.mot = {rx: deque(maxlen=self.n) for rx in engine.buf}
        self.rates = {rx: deque(maxlen=self.n) for rx in engine.buf}
        self.times = deque(maxlen=self.n)

    # -- rolling signals from the raw packet buffers (cheap, no model involved) ---------------------
    def _sample(self, now):
        self.times.append(now)
        for rx, b in self.e.buf.items():
            ts, A, _ = b.snapshot(now - 0.5, now)
            if len(ts) >= 3:
                recent = A[ts >= now - self.tick] if (ts >= now - self.tick).sum() >= 1 else A[-1:]
                self.cols[rx].append(recent.mean(0))
                self.mot[rx].append(float(A.std(0).mean()))
            else:
                self.cols[rx].append(self.cols[rx][-1] if self.cols[rx] else None)
                self.mot[rx].append(0.0)
            live = abs(time.time() - now) < 2.0           # live / real-time replay: count real arrivals
            self.rates[rx].append(float(b.arrival_rate() if live else b.rate(now)))

    def _waterfall(self, rx):
        cols = [c for c in self.cols[rx] if c is not None]
        if len(cols) < 4:
            return []
        M = np.vstack(cols)                                       # (time, 52)
        D = np.abs(M - np.median(M, axis=0, keepdims=True))
        D = D / (np.percentile(D, 97) + 1e-9)
        return np.round(np.clip(D, 0, 1), 2).tolist()

    def _motion(self, rx):
        m = np.asarray(self.mot[rx], float)
        if len(m) < 4 or not np.any(m > 0):
            return [0.0] * len(m)
        q = np.percentile(m[m > 0], 20) + 1e-9
        return np.round(m / q, 2).tolist()

    def update(self, now):
        e = self.e
        if e is None:
            return
        now = float(now)                      # replay times can be NumPy floats
        self._sample(now)
        st = e.state
        used = e.alive(now) if abs(time.time() - now) < 2.0 else {rx: True for rx in e.buf}
        alive = {rx: (now - b.last_time()) < 1.5 for rx, b in e.buf.items()}       # receiving at all
        reported = list(e.events)
        if st.get("last_event") is not None:                      # includes a replay's end-of-trial decision
            reported.append(st["last_event"])
        for res in reported:                                      # newly reported events -> timeline
            key = (round(res.get("event_abs", 0.0), 2), res["label"])
            if key not in self._seen:
                self._seen.add(key)
                self.history.append({"time": datetime.fromtimestamp(res.get("decided_at", now)).strftime("%H:%M:%S"),
                                     "label": res["label"], "display": DISPLAY.get(res["label"], res["label"]),
                                     "p_fall": round(float(res["p_fall"]), 3),
                                     "rx": "+".join(r.upper() for r in res["receivers"])})
        ev = st.get("last_event") or self._last
        if st.get("last_event") is not None:
            self._last = st["last_event"]
        event = None
        if ev is not None:
            classes = list(e.clf.classes)
            event = {"label": ev["label"], "display": DISPLAY.get(ev["label"], ev["label"]),
                     "p_fall": round(float(ev["p_fall"]), 3),
                     "probs": {c: round(float(p), 3) for c, p in zip(classes, ev["probs"])},
                     "receivers": [r.upper() for r in ev["receivers"]],
                     "age": round(max(0.0, now - ev.get("decided_at", now)), 1),
                     "latency": round(ev.get("decided_at", now) - ev.get("event_abs", now), 1)}
        warm = 0.0
        if e.start_t is not None:
            warm = max(0.0, e.calibrate_s - (now - e.start_t))
        snap = {
            "ready": True, "now": now, "clock": datetime.fromtimestamp(now).strftime("%H:%M:%S"),
            "uptime": round(time.time() - self.t_start), "mode": self.mode,
            "status": st.get("status", ""), "tier": st.get("tier", 0), "warmup_left": round(warm, 1),
            "alert": bool(now < float(st.get("alert_until", 0.0))),
            "burst": bool(now < float(st.get("pending_until", 0.0))),
            "event": event, "truth": DISPLAY.get(e.ground_truth, e.ground_truth) if e.ground_truth else None,
            "diag": getattr(e, "diag", None),
            "decisions": len(self.history), "history": self.history[-8:][::-1],
            "model": {"name": "two-head random forest" if getattr(e, "two_head", False) else "event random forest",
                      "tau": round(float(e.clf.tau), 3), "far": getattr(e.clf, "far", "5"),
                      "classes": list(e.clf.classes)},
            "rx": {rx: {"alive": bool(alive[rx]), "used": bool(used[rx] and alive[rx]),
                        "rate": round(self.rates[rx][-1], 1) if self.rates[rx] else 0.0,
                        "rate_hist": [round(v, 1) for v in self.rates[rx]],
                        "motion": self._motion(rx), "waterfall": self._waterfall(rx),
                        "total": int(e.buf[rx].total),
                        "lag_ms": (round(e.buf[rx].lag() * 1000) if np.isfinite(e.buf[rx].lag()) else None),
                        "serial_hz": (round(float(e.serial_hz.value), 1)
                                      if rx == "rx1" and getattr(e, "serial_hz", None) is not None else None)}
                   for rx in e.buf},
        }
        with self.lock:
            self.snap = snap

    def get(self):
        with self.lock:
            return self.snap


def start_server(feed, port=8050, open_browser=True):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/state"):
                body = json.dumps(feed.get()).encode()
                ctype = "application/json"
            elif self.path in ("/", "/index.html", "/dashboard.html"):
                with open(PAGE, "rb") as fh:
                    body = fh.read()
                ctype = "text/html; charset=utf-8"
            else:
                self.send_response(404); self.end_headers(); return
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):        # keep the console clean
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{port}/"
    print(f"  Dashboard: {url}  (press F11 in the browser for full screen)")
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    return srv
