import json, struct, threading, time
from websockets.sync.client import connect


def normalize_url(u):
    """'10.56.6.16' -> 'ws://10.56.6.16:8765'  (adds scheme and default port for ws://)."""
    u = (u or "").strip()
    if not u:
        return u
    if "://" not in u:
        u = "ws://" + u
    scheme, rest = u.split("://", 1)
    host, sep, path = rest.partition("/")
    if ":" not in host and scheme == "ws":
        host += ":8765"
    return f"{scheme}://{host}{sep}{path}"


MAGIC = b"VPU2"


class RemoteClient:
    """Persistent WebSocket to the laptop. A background thread connects, retries and pings,
    so the video pipeline never blocks on the network.
    Protocol v2 (used automatically when the server answers pings with "v": 2) adds the capture time and
    source generation to every frame and sends the camera calibration, so the laptop can run optical-flow
    ego-motion and looming. Replies then carry `ego` (own speed, yaw rate, dynamic horizon) and, per
    detection, an `ext` dict that is appended to the detection as a 7th element."""
    def __init__(self, url):
        self.url = normalize_url(url)
        self.v2 = False                         # server speaks protocol v2
        self.cal, self._cal_sent = None, None
        self.ego, self.ego_t = None, 0.0        # last ego-motion reply and when it arrived (monotonic)
        self.motion_ms = None
        self.ws, self.fail = None, 0
        self.rtt = self.infer = None            # ms (EWMA); rtt = round trip minus GPU time
        self.state, self.err, self.tries = "connecting", "", 0
        self.n_sent = self.n_ok = self.n_timeout = self.n_busy = self.n_err = 0   # diagnostics shown in the web page
        self.ping_rtt = None                                                    # ms, pings only (pure network delay)
        self.last_total = None                                                  # ms of the last full frame round trip
        self.io = threading.Lock()               # one request on the socket at a time
        self.last_use = self.last_ping = 0.0
        self._n = 0
        self._wake = threading.Event()
        threading.Thread(target=self._worker, daemon=True).start()

    @property
    def ready(self):
        return self.ws is not None and self.fail < 2 and self.rtt is not None

    def set_url(self, url):
        url = normalize_url(url)
        if url:
            self.url, self.err, self.state = url, "", "connecting"
            self._drop(self.ws)
            self._wake.set()

    def _drop(self, ws):
        if ws is not None and ws is self.ws:
            self.ws, self.rtt = None, None
            self._cal_sent, self.v2 = None, False
            try: ws.close()
            except Exception: pass

    def _worker(self):
        while True:
            if self.ws is None:
                self.state, self.tries = "connecting", self.tries + 1
                try:
                    ws = connect(self.url, open_timeout=6, close_timeout=1,
                                 max_size=2**22, compression=None)
                    self.ws, self.fail = ws, 0
                    if not self.ping(timeout=4):
                        raise RuntimeError("socket opened but no reply to ping")
                    self.err = ""
                    print("[remote] connected", self.url, "rtt", round(self.rtt or 0), "ms")
                except Exception as e:
                    self.err = f"{type(e).__name__}: {e}"[:160]
                    self.state = "failed"
                    print("[remote] connect failed:", self.err)
                    self._drop(self.ws)
                    self._wake.wait(2.0); self._wake.clear()
            else:
                now = time.monotonic()
                if now - self.last_use > 1.0 and now - self.last_ping > 1.0:
                    self.ping(timeout=3)
                self._wake.wait(0.4); self._wake.clear()

    def _ew(self, name, v, a=0.3):
        old = getattr(self, name)
        setattr(self, name, v if old is None else (1-a)*old + a*v)

    def _wait(self, ws, pred, deadline):
        while True:
            rem = deadline - time.perf_counter()
            if rem <= 0:
                return None
            try:
                m = json.loads(ws.recv(timeout=rem))
            except TimeoutError:
                return None
            if pred(m):                          # stale replies from earlier timeouts are skipped
                return m

    def _failed(self, ws, limit=3):
        self.fail += 1
        if self.fail >= limit:
            self._drop(ws)

    def ping(self, timeout=3.0):
        with self.io:
            ws = self.ws
            if ws is None:
                return False
            self._n += 1; n = self._n
            t0 = time.perf_counter()
            self.last_ping = time.monotonic()
            try:
                ws.send(json.dumps({"ping": n}))
                m = self._wait(ws, lambda m: m.get("pong") == n, t0 + timeout)
                if m is not None:
                    self.v2 = m.get("v", 1) >= 2
            except Exception as e:
                self.err = f"{type(e).__name__}: {e}"[:160]
                self._drop(ws); return False
            if m is None:
                self.err = "no reply (timeout)"
                self._failed(ws); return False
            self.fail = 0
            self.state = "connected"
            self._ew("rtt", (time.perf_counter()-t0)*1000)
            self._ew("ping_rtt", (time.perf_counter()-t0)*1000)
            return True

    def set_cal(self, cal):
        """Calibration dict (hfov, cam_h, horizon, foe_x); sent with the next frame when it changed."""
        self.cal = dict(cal)

    def detect(self, jpeg, fid, timeout, wait=0.15, max_fail=3, ts=None, gen=0):
        if not self.io.acquire(timeout=wait):     # a ping is in flight
            self.n_busy += 1
            return None
        try:
            ws = self.ws
            if ws is None:
                return None
            self.last_use = time.monotonic()
            self.n_sent += 1
            t0 = time.perf_counter()
            try:
                if self.v2 and ts is not None:
                    if self.cal is not None and self.cal != self._cal_sent:
                        ws.send(json.dumps({"cal": self.cal})); self._cal_sent = dict(self.cal)
                    ws.send(MAGIC + struct.pack("!IdI", fid, ts, gen & 0xFFFFFFFF) + jpeg)
                else:
                    ws.send(struct.pack("!I", fid) + jpeg)
                m = self._wait(ws, lambda m: m.get("id") == fid, t0 + timeout)
            except Exception as e:
                self.n_err += 1
                self.err = f"{type(e).__name__}: {e}"[:160]
                self._drop(ws); return None
            if m is None:
                self.n_timeout += 1
                self.err = f"frame ({len(jpeg)//1024} KB) got no reply within {timeout:.1f} s"
                self._failed(ws, max_fail); return None
            total = (time.perf_counter()-t0)*1000
            self.last_total = total
            self.n_ok += 1
            self.err = ""
            self.fail = 0
            self._ew("infer", m["infer_ms"])
            self._ew("rtt", max(0.0, total - m["infer_ms"]))
            if m.get("motion_ms") is not None:
                self._ew("motion_ms", m["motion_ms"])
            if m.get("ego") is not None:
                self.ego, self.ego_t = m["ego"], time.monotonic()
            dets, ext = m["dets"], m.get("ext")
            if ext and len(ext) == len(dets):
                dets = [d + [e] if e else d for d, e in zip(dets, ext)]
            return dets
        finally:
            self.io.release()
