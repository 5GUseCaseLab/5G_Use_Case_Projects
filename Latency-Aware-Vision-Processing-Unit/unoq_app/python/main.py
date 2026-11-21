import os, re, time, math, base64, cv2
from arduino.app_utils import App, Bridge
from arduino.app_bricks.web_ui import WebUI
from camera import Camera
from detectors import LocalDetector
from remote import RemoteClient
from tracking import Tracker
from advisor import Advisor

# ---------------- config (override with env vars or from the web UI) ----------------
DEFAULT_URL = "ws://10.56.6.16:8765"                           # <- your laptop's ZeroTier IP (can be changed in the web page)
REMOTE_BIAS = float(os.getenv("REMOTE_BIAS", "4"))             # use laptop unless it is > BIAS x slower than local
MAX_REMOTE_MS = float(os.getenv("MAX_REMOTE_MS", "800"))       # never use laptop if estimated total latency is above this
HFOV_DEG   = float(os.getenv("HFOV_DEG", "70"))                # horizontal field of view of the video's camera
CAM_H      = float(os.getenv("CAM_H", "1.3"))                  # camera height above the road in metres (dashcam ~1.2-1.5)
HORIZON    = float(os.getenv("HORIZON", "0.45"))               # where the horizon sits in the picture, 0 = top, 1 = bottom
FOE_X      = float(os.getenv("FOE_X", "0.5"))                  # image column the car drives towards, 0 = left, 1 = right
EGO_MODE   = os.getenv("EGO_MODE", "auto")                     # auto: laptop optical flow, else 'my speed' box | manual | off
EGO_MAX_AGE = 0.6                                              # s: laptop ego-motion older than this is not used
SOURCE     = os.getenv("SOURCE", "none")                       # none | camera | file
VIDEO_PATH = os.getenv("VIDEO_PATH", "")                       # used when SOURCE=file
USE_MCU = os.getenv("USE_MCU", "0") == "1"                       # 1 = send alerts to the Arduino sketch
SEND_LEVELS = [(480, 60), (384, 55), (320, 50), (256, 50), (192, 45)]   # (width, jpeg quality) sent to the laptop; adapts to the link
HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models")
UPLOAD_DIR = os.path.join(HERE, "uploads")
URL_FILE = os.path.join(HERE, "remote_url.txt")           # last URL typed in the web page is remembered here


def load_url():
    if os.getenv("LAPTOP_URL"):
        return os.getenv("LAPTOP_URL")
    try:
        u = open(URL_FILE).read().strip()
        if u: return u
    except Exception:
        pass
    return DEFAULT_URL

cfg = {"mode": "auto", "thr": 3.0, "max_age": 0.5, "hfov": HFOV_DEG, "bias": REMOTE_BIAS,
       "cam_h": CAM_H, "horizon": HORIZON, "foe_x": FOE_X, "ego_kmh": 0.0, "ego_mode": EGO_MODE}   # mode: auto | local | remote


def ew(old, v, a=0.25):
    return v if old is None else (1-a)*old + a*v


class Gate:
    """Process a frame only if it differs from the LAST PROCESSED frame by > thr
    (mean abs gray diff on a 64x36 blurred thumbnail), or if max_age expired."""
    def __init__(self):
        self.ref, self.t_last, self.diff = None, 0.0, 0.0

    def check(self, frame, now):
        g = cv2.GaussianBlur(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 36),
                                        interpolation=cv2.INTER_AREA), (5, 5), 0)
        if self.ref is None:
            go = True
        else:
            self.diff = float(cv2.absdiff(g, self.ref).mean())
            go = self.diff > cfg["thr"] or now - self.t_last > cfg["max_age"]
        if go:
            self.ref, self.t_last = g, now
        return go


class Pipeline:
    def __init__(self, ui):
        self.ui = ui
        self.cam = Camera()
        self.local = LocalDetector(MODEL_DIR)
        self.local_ms = self.local.warmup() if self.local.available else 150.0   # measured at start-up
        self.remote = RemoteClient(load_url())
        self.gate, self.tracker = Gate(), None
        self.seq = self.fid = self.gen = 0
        self.cal_used = None                           # (hfov, cam_h, horizon, foe_x) the tracker was built with
        self.ego, self.ego_src = None, "none"          # ego-motion used for the current frame
        self.advisor, self.mcu_level = Advisor(), 0
        self.mode = "local"
        self.enc_ms, self.e2e_ms = 8.0, None
        self.t_ui = self.t_stats = self.t_ping = self.t_probe = 0.0
        self.win = [0, 0, time.monotonic()]            # processed, skipped, window start
        self.alert = 0
        self.lvl, self.ok_streak, self.slow_streak, self.kb = 2, 0, 0, 0.0          # start small; step up when the link is fast
        self.up_f, self.up_path = None, ""
        for name, fn in (("config", self.on_config), ("upload_start", self.on_up_start),
                         ("upload_chunk", self.on_up_chunk), ("upload_end", self.on_up_end)):
            ui.on_message(name, fn)
        if SOURCE == "camera": self.cam.open_camera(0)
        elif SOURCE == "file" and VIDEO_PATH: self.cam.open_file(VIDEO_PATH)

    # ----- messages from the web page -----
    def on_config(self, *args):
        d = args[-1] or {}
        if d.get("mode") in ("auto", "local", "remote"): cfg["mode"] = d["mode"]
        if "thr" in d: cfg["thr"] = float(d["thr"])
        if "max_age" in d: cfg["max_age"] = max(0.1, min(5.0, float(d["max_age"])))
        if "hfov" in d: cfg["hfov"] = max(20.0, min(150.0, float(d["hfov"])))
        if "cam_h" in d: cfg["cam_h"] = max(0.3, min(4.0, float(d["cam_h"])))
        if "horizon" in d: cfg["horizon"] = max(0.15, min(0.85, float(d["horizon"])))
        if "ego_kmh" in d: cfg["ego_kmh"] = max(0.0, min(250.0, float(d["ego_kmh"])))
        if "foe_x" in d: cfg["foe_x"] = max(0.2, min(0.8, float(d["foe_x"])))
        if d.get("ego_mode") in ("auto", "manual", "off"): cfg["ego_mode"] = d["ego_mode"]
        if d.get("url"):
            self.remote.set_url(d["url"])
            try: open(URL_FILE, "w").write(self.remote.url)
            except Exception as e: print("[url] cannot save:", e)
        if "bias" in d: cfg["bias"] = max(0.5, min(10.0, float(d["bias"])))
        src = d.get("source")
        if src == "camera":
            self.cam.open_camera(int(d.get("index", 0)))
        elif src == "file" and d.get("path"):
            self.cam.open_file(d["path"].strip())
        elif src == "restart" and self.cam.kind == "file":
            self.cam.open_file(self.cam.info["name"] if os.path.isabs(self.cam.info["name"])
                               else os.path.join(UPLOAD_DIR, self.cam.info["name"]))

    def on_up_start(self, *args):
        d = args[-1] or {}
        name = re.sub(r"[^A-Za-z0-9._-]", "_", os.path.basename(d.get("name", "video.mp4"))) or "video.mp4"
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        self.up_path = os.path.join(UPLOAD_DIR, name)
        self.up_f = open(self.up_path, "wb")
        self.ui.send_message("upload_ack", {"n": 0})

    def on_up_chunk(self, *args):
        d = args[-1] or {}
        if self.up_f:
            self.up_f.write(base64.b64decode(d["b64"]))
            self.ui.send_message("upload_ack", {"n": 1})

    def on_up_end(self, *args):
        if self.up_f:
            self.up_f.close(); self.up_f = None
            print("[upload] saved", self.up_path)
            self.cam.open_file(self.up_path)
            self.ui.send_message("upload_done", {"name": os.path.basename(self.up_path)})

    # ----- the latency-based decision -----
    def remote_est(self):
        if not self.remote.ready:
            return None
        infer = self.remote.infer if self.remote.infer is not None else 40.0   # guess until the first real reply
        return self.remote.rtt + infer + self.enc_ms

    def choose(self):
        if cfg["mode"] == "remote":                   # FORCED: always the laptop, no matter what
            self.mode = "remote"
            return self.mode
        r = self.remote_est()
        if cfg["mode"] == "local" or r is None:
            self.mode = "local"                       # forced local, or link down
        elif not self.local.available:
            self.mode = "remote"
        else:                                         # auto: PREFER the laptop, with hysteresis
            lim = min(MAX_REMOTE_MS, cfg["bias"]*self.local_ms)
            if self.mode == "local" and r < lim: self.mode = "remote"
            elif self.mode == "remote" and r > 1.3*lim: self.mode = "local"
        return self.mode

    def current_ego(self, now):
        """Own motion for the tracker: laptop optical flow (fresh + good) > 'my speed' box > unknown."""
        r = self.remote.ego
        if cfg["ego_mode"] == "auto" and r and r.get("ok") and now - self.remote.ego_t < EGO_MAX_AGE:
            self.ego_src = "laptop flow"
            return r
        if cfg["ego_mode"] in ("auto", "manual") and cfg["ego_kmh"] > 0:
            self.ego_src = "manual"
            return {"v": cfg["ego_kmh"] / 3.6, "v_sd": 1.0, "yaw": 0.0}
        self.ego_src = "none"
        return None

    def run_local(self, frame):
        if not self.local.available:
            return []
        t0 = time.perf_counter()
        d = self.local.detect(frame)
        self.local_ms = ew(self.local_ms, (time.perf_counter()-t0)*1000)
        return d

    def step(self):
        now = time.monotonic()
        got = self.cam.latest(self.seq)
        if got is None:
            self.emit_stats(now)
            time.sleep(0.003); return
        frame, ts, self.seq, gen = got
        cal = (cfg["hfov"], cfg["cam_h"], cfg["horizon"], cfg["foe_x"])
        self.remote.set_cal(dict(zip(("hfov", "cam_h", "horizon", "foe_x"), cal)))
        if self.tracker is None or gen != self.gen or cal != self.cal_used:
            h, w = frame.shape[:2]                    # new video / loop / new calibration: reset state
            self.tracker = Tracker(w, h, *cal)
            self.gate.ref = None
            self.gen, self.cal_used = gen, cal

        if self.gate.check(frame, ts):
            dets = None
            forced = cfg["mode"] == "remote"
            if self.choose() == "remote":
                h, w = frame.shape[:2]
                sw, sq = SEND_LEVELS[self.lvl]
                t0 = time.perf_counter()
                ok, buf = cv2.imencode(".jpg", cv2.resize(frame, (sw, int(h*sw/w))),
                                       [cv2.IMWRITE_JPEG_QUALITY, sq])
                self.kb = ew(self.kb or None, len(buf)/1024)
                self.enc_ms = ew(self.enc_ms, (time.perf_counter()-t0)*1000)
                self.fid += 1
                est = self.remote_est() or 300
                tmo = max(1.5, min(4.0, 4*est/1000)) if forced else max(0.6, min(2.0, 3*est/1000))
                dets = self.remote.detect(buf.tobytes(), self.fid, tmo,
                                          wait=1.0 if forced else 0.15, max_fail=6 if forced else 3, ts=ts, gen=gen)
                last = len(SEND_LEVELS) - 1
                if dets is None:                      # timeout / lost: send smaller frames next time
                    self.lvl, self.ok_streak, self.slow_streak = min(self.lvl + 1, last), 0, 0
                else:                                 # adapt frame size to how long a frame really takes vs a bare ping
                    lt, base = self.remote.last_total or 0, self.remote.ping_rtt or 300
                    if lt > 2.0*base + 400:           # transfer dominates -> smaller
                        self.slow_streak += 1; self.ok_streak = 0
                        if self.slow_streak >= 4: self.lvl, self.slow_streak = min(self.lvl + 1, last), 0
                    elif lt < 1.5*base + 120:         # transfer is cheap -> bigger (better detection)
                        self.ok_streak += 1; self.slow_streak = 0
                        if self.ok_streak >= 15 and self.lvl > 0: self.lvl, self.ok_streak = self.lvl - 1, 0
                    else:
                        self.ok_streak = self.slow_streak = 0
                if dets is None and not forced:       # auto mode: timeout / link lost -> local fallback
                    self.mode = "local"
                elif dets is not None and not forced and now - self.t_probe > 20:
                    self.t_probe = now                # refresh local timing now and then
                    self.run_local(frame)
            if dets is None and forced:               # forced remote: never touch the local model
                self.win[1] += 1                      # frame not processed (laptop unreachable / slow)
                self.emit(frame); self.emit_stats(time.monotonic())
                return
            if dets is None:
                dets = self.run_local(frame)
            self.ego = self.current_ego(time.monotonic())
            self.tracker.update(dets, ts, self.ego)
            self.win[0] += 1
            self.e2e_ms = ew(self.e2e_ms, (time.monotonic()-ts)*1000)
        else:
            self.win[1] += 1
        self.emit(frame)
        self.emit_stats(time.monotonic())

    def emit(self, frame):
        now = time.monotonic()
        ego = (self.ego or {}).get("v") or 0.0
        tracks = self.tracker.snapshot(now)            # held + lightly extrapolated, so boxes never blink out
        adv = self.advisor.update(tracks, now, ego)    # CLEAR / CAUTION / BRAKE / BRAKE HARD
        self.alert = adv["level"]
        mcu = min(adv["level"], 2)
        if mcu != self.mcu_level:
            self.mcu_level = mcu
            if USE_MCU:                               # only if the sketch is installed and running
                try: Bridge.call("set_alert", mcu)
                except Exception as e: print("bridge:", e)
        if now - self.t_ui > 0.12:
            self.t_ui = now
            h, w = frame.shape[:2]
            ok, buf = cv2.imencode(".jpg", cv2.resize(frame, (320, int(h*320/w))),
                                   [cv2.IMWRITE_JPEG_QUALITY, 50])
            self.ui.send_message("frame", {"img": base64.b64encode(buf).decode(), "tracks": tracks, "advice": adv,
                                           "horizon": round(self.tracker.hy_now / self.tracker.h, 4)})

    def emit_stats(self, now):
        if now - self.t_stats < 0.5:
            return
        dt = now - self.win[2]
        fps = self.win[0]/dt if dt > 0 else 0
        tot = self.win[0] + self.win[1]
        skip = 100*self.win[1]/tot if tot else 0
        self.win = [0, 0, now]; self.t_stats = now
        rd = lambda v: None if v is None else round(v, 1)
        i = self.cam.info
        self.ui.send_message("stats", dict(
            mode=self.mode, cfg_mode=cfg["mode"], link=self.remote.ready,
            rtt=rd(self.remote.rtt), infer=rd(self.remote.infer), remote_est=rd(self.remote_est()),
            local_ms=rd(self.local_ms), e2e=rd(self.e2e_ms), fps=round(fps, 1),
            skip=round(skip), diff=round(self.gate.diff, 1), alert=self.alert,
            local_ok=self.local.available, url=self.remote.url, thr=cfg["thr"], max_age=cfg["max_age"], hfov=cfg["hfov"],
            cam_h=cfg["cam_h"], horizon=cfg["horizon"], ego_kmh=cfg["ego_kmh"], foe_x=cfg["foe_x"], ego_mode=cfg["ego_mode"],
            ego_src=self.ego_src, ego_v=rd((self.ego or {}).get("v") * 3.6) if self.ego else None,
            ego_yaw=rd(math.degrees((self.ego or {}).get("yaw") or 0.0)) if self.ego else None,
            hz_fit=(self.remote.ego or {}).get("horizon_fit"), foe_fit=(self.remote.ego or {}).get("foe_x"),
            motion_ms=rd(self.remote.motion_ms), v2=self.remote.v2,
            r_sent=self.remote.n_sent, r_ok=self.remote.n_ok, r_to=self.remote.n_timeout,
            r_busy=self.remote.n_busy, r_err=self.remote.n_err, send_w=SEND_LEVELS[self.lvl][0], kb=round(self.kb or 0, 1),
            link_state=self.remote.state, link_err=self.remote.err, tries=self.remote.tries, bias=cfg["bias"],
            source=f'{i["kind"]}: {i["name"]}' + (f' @ {i["fps"]} fps' if i["fps"] else '') +
                   (f' ({i["error"]})' if i["error"] else '')))


ui = WebUI()
pipe = Pipeline(ui)
App.run(user_loop=pipe.step)
