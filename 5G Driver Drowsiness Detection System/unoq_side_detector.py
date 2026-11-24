#!/usr/bin/env python3
"""
unoq_side_detector.py - Arduino UNO Q stand-alone drowsiness watcher (MediaPipe, Haar fallback)
----------------------------------------------------------------------------------------------
Runs on the UNO Q's Linux side, purely for show. It never sends anything to the laptop and
never affects the laptop app or the car. Its verdicts appear only in this terminal.

Detection is the SAME as laptop_app.py (code copied from it):
  MediaPipe Face Landmarker -> Eye Aspect Ratio with a 5 s personal calibration ->
  three triggers: eyes closed 1.5 s, PERCLOS 30 % of 60 s, 3 yawns in 60 s.
  If MediaPipe can't be used, it falls back to OpenCV Haar cascades automatically.

Video source (pick one):
  --stream http://<laptop-IP>:5000   watch the laptop dashboard's live video (/video.mjpg), read-only
  --camera 0                         or a USB webcam plugged into the UNO Q

Setup on the UNO Q (once):
  sudo apt install -y python3-opencv opencv-data python3-venv
  python3 -m venv --system-site-packages ~/mpenv
  ~/mpenv/bin/pip install mediapipe
Check: ~/mpenv/bin/python unoq_side_detector.py --check-mediapipe --stream http://<laptop-IP>:5000
Run:   ~/mpenv/bin/python unoq_side_detector.py --stream http://<laptop-IP>:5000
       (type r + Enter while it runs to recalibrate;  --haar forces the Haar method;  Ctrl+C stops)
"""

import argparse
import collections
import json
import os
import statistics
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
import zipfile

try:
    import cv2
    import numpy as np
except ImportError as e:
    sys.exit(f"ERROR: {e}.\nInstall OpenCV on the UNO Q with:\n"
             "    sudo apt update && sudo apt install -y python3-opencv opencv-data")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CAPTURE_W, CAPTURE_H = 640, 480
STREAM_TIMEOUT_S = 5.0           # reconnect if the video stream is silent this long

# ============ detection settings: identical to laptop_app.py ============
# --- MediaPipe model ---
MODEL_URLS = [   # official model bundle; 'latest' is the address in MediaPipe's current docs
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
]
MODEL_PATH = os.path.join(SCRIPT_DIR, "face_landmarker.task")

# --- eye closure ---
CALIBRATION_S = 5.0              # at start: look at the camera normally, eyes open, for 5 s
EAR_THRESHOLD_DEFAULT = 0.21     # used until calibration finishes
EAR_THRESHOLD_FACTOR = 0.72      # closed = EAR below 72 % of your calibrated open-eye EAR
EAR_THRESHOLD_MIN, EAR_THRESHOLD_MAX = 0.14, 0.30
BLINK_SCORE_CLOSED = 0.60        # MediaPipe eyeBlink score above this also counts as closed
EYES_CLOSED_ALERT_S = 1.5        # TRIGGER 1: eyes closed continuously this long (normal blink ~0.3 s)
OPEN_GRACE_S = 0.15              # a flicker of "open" shorter than this does not reset the timer

# --- PERCLOS: % of time eyes closed over a window ---
PERCLOS_WINDOW_S = 60.0
PERCLOS_MIN_DATA_S = 20.0        # need this much face time before PERCLOS can trigger
PERCLOS_ALERT = 0.30             # TRIGGER 2: eyes closed >= 30 % of the last 60 s (0 = off)

# --- yawns ---
YAWN_JAW_OPEN = 0.55             # MediaPipe jawOpen score that counts as a yawn
YAWN_MIN_S = 1.0                 # mouth must stay open this long
YAWN_WINDOW_S = 60.0
YAWN_ALERT_COUNT = 3             # TRIGGER 3: this many yawns within 60 s (0 = off)

NO_FACE_IS_DROWSY = False        # True: face gone for NO_FACE_DROWSY_S also triggers (head dropped)
NO_FACE_DROWSY_S = 3.0
# =========================================================================

# MediaPipe face-mesh landmark indices: p1..p6 around each eye
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]


def log(msg):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


def reason(e):
    if isinstance(e, urllib.error.HTTPError):
        return f"HTTP {e.code}"
    return str(getattr(e, "reason", e))


# ====================== analyzers (copied from laptop_app.py) ======================
def _dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def eye_aspect_ratio(p):
    """p = 6 points (p1..p6). EAR = (|p2-p6| + |p3-p5|) / (2 |p1-p4|). Open ~0.25-0.35, closed < ~0.15."""
    horiz = _dist(p[0], p[3])
    if horiz < 1e-6:
        return 0.0
    return (_dist(p[1], p[5]) + _dist(p[2], p[4])) / (2.0 * horiz)



def model_is_valid(path):
    """A real face_landmarker.task is a ~3.7 MB zip bundle. Anything else (an error page, half a download) is not."""
    try:
        return os.path.getsize(path) > 1_000_000 and zipfile.is_zipfile(path)
    except OSError:
        return False


def _download(url, dest):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (drowsiness unoq_side_detector)"})
    try:
        resp = urllib.request.urlopen(req, timeout=60)
    except urllib.error.URLError as e:
        if "CERTIFICATE_VERIFY_FAILED" not in str(e):
            raise
        import ssl                                   # this Python lacks root certificates: try certifi's set
        try:
            import certifi
        except ImportError:
            raise RuntimeError("SSL certificate check failed. Fix: pip install certifi   (then run again)") from e
        resp = urllib.request.urlopen(req, timeout=60, context=ssl.create_default_context(cafile=certifi.where()))
    with resp, open(dest, "wb") as f:
        while True:
            chunk = resp.read(65536)
            if not chunk:
                break
            f.write(chunk)


def ensure_model():
    """Returns the path of a valid face_landmarker.task, downloading it if needed."""
    if model_is_valid(MODEL_PATH):
        return MODEL_PATH
    if os.path.exists(MODEL_PATH):                  # an old broken download would make MediaPipe fail forever
        bad = MODEL_PATH + ".bad"
        os.replace(MODEL_PATH, bad)
        print(f"[model] {MODEL_PATH} was not a valid model ({os.path.getsize(bad)} bytes), renamed to .bad", flush=True)
    tmp = MODEL_PATH + ".part"
    problems = []
    for url in MODEL_URLS:
        print(f"[model] downloading the MediaPipe face model from {url} ...", flush=True)
        try:
            _download(url, tmp)
            if model_is_valid(tmp):
                os.replace(tmp, MODEL_PATH)
                print(f"[model] saved {MODEL_PATH} ({os.path.getsize(MODEL_PATH) // 1024} KB)", flush=True)
                return MODEL_PATH
            problems.append(f"{url} -> not a model file ({os.path.getsize(tmp)} bytes; a login page or proxy?)")
        except Exception as e:                       # no internet, proxy, firewall, SSL ...
            problems.append(f"{url} -> {e.__class__.__name__}: {e}")
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
    raise RuntimeError("could not download the face model.\n    " + "\n    ".join(problems) +
                       f"\n  Fix: open {MODEL_URLS[0]} in your browser and save the file as\n       {MODEL_PATH}")


def _mediapipe_api(mp):
    """(BaseOptions, vision module) using the official public names (mp.tasks...), which work on
    MediaPipe 0.10.x and 1.x; the old internal path is kept only as a last resort."""
    import importlib
    errors = []
    try:
        return mp.tasks.BaseOptions, mp.tasks.vision
    except AttributeError as e:
        errors.append(str(e))
    try:
        return (importlib.import_module("mediapipe.tasks.python").BaseOptions,
                importlib.import_module("mediapipe.tasks.python.vision"))
    except (ImportError, AttributeError) as e:
        errors.append(str(e))
    raise ImportError(f"mediapipe {getattr(mp, '__version__', '?')} has no Tasks API ({'; '.join(errors)})")



class MediaPipeAnalyzer:
    name = "MediaPipe + EAR"
    kind = "mp"

    def __init__(self):
        import mediapipe as mp
        BaseOptions, vision = _mediapipe_api(mp)
        path = ensure_model()
        with open(path, "rb") as f:                 # load from memory: avoids MediaPipe's trouble with
            model_bytes = f.read()                   # some Windows folder names (OneDrive, non-English)
        try:
            base = BaseOptions(model_asset_buffer=model_bytes)
        except TypeError:
            base = BaseOptions(model_asset_path=path)
        modes = vision.RunningMode
        video = getattr(modes, "VIDEO", None) or next((m for m in modes if "VIDEO" in m.name.upper()), None)
        if video is None:
            raise RuntimeError(f"this mediapipe has no VIDEO running mode ({[m.name for m in modes]})")
        opts = vision.FaceLandmarkerOptions(
            base_options=base,
            running_mode=video,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=True)
        self.landmarker = vision.FaceLandmarker.create_from_options(opts)
        self.mp = mp
        self.version = getattr(mp, "__version__", "?")
        # one test run on a blank frame now, so any problem shows up at start-up (and falls back to Haar)
        self.landmarker.detect_for_video(self._image(np.zeros((CAPTURE_H, CAPTURE_W, 3), np.uint8)), 1)
        self.last_ts = 1

    def _image(self, rgb):
        return self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))

    def process(self, frame):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ts = int(time.monotonic() * 1000)
        if ts <= self.last_ts:
            ts = self.last_ts + 1
        self.last_ts = ts
        res = self.landmarker.detect_for_video(self._image(rgb), ts)
        if not res.face_landmarks:
            return {"face": False}
        lms = res.face_landmarks[0]

        def pt(i):
            return (lms[i].x * w, lms[i].y * h)

        right = [pt(i) for i in RIGHT_EYE]
        left = [pt(i) for i in LEFT_EYE]
        ear = (eye_aspect_ratio(right) + eye_aspect_ratio(left)) / 2.0
        scores = {}
        if res.face_blendshapes:
            scores = {c.category_name: c.score for c in res.face_blendshapes[0]}
        blink = (scores.get("eyeBlinkLeft", 0.0) + scores.get("eyeBlinkRight", 0.0)) / 2.0
        xs = [l.x * w for l in lms]
        ys = [l.y * h for l in lms]
        box = (int(min(xs)), int(min(ys)), int(max(xs) - min(xs)), int(max(ys) - min(ys)))
        return {"face": True, "ear": ear, "blink": blink, "jaw": scores.get("jawOpen", 0.0),
                "box": box, "pts": [(int(x), int(y)) for (x, y) in right + left]}


class HaarAnalyzer:
    """Fallback: OpenCV Haar cascades. 'closed' = no open eye found."""
    name = "OpenCV Haar (fallback)"
    kind = "haar"

    def __init__(self):
        self.face = cv2.CascadeClassifier(self._find("haarcascade_frontalface_default.xml"))
        self.eye = cv2.CascadeClassifier(self._find("haarcascade_eye_tree_eyeglasses.xml"))
        if self.face.empty() or self.eye.empty():
            sys.exit("ERROR: could not load the Haar cascade files.")

    @staticmethod
    def _find(name):
        paths = []
        if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
            paths.append(os.path.join(cv2.data.haarcascades, name))
        paths += [f"/usr/share/opencv4/haarcascades/{name}", os.path.join(SCRIPT_DIR, name)]
        for p in paths:
            if os.path.isfile(p):
                return p
        sys.exit(f"ERROR: {name} not found. Install it with:  sudo apt install -y opencv-data")

    def process(self, frame):
        h0, w0 = frame.shape[:2]
        k = 320.0 / w0
        gray = cv2.equalizeHist(cv2.cvtColor(cv2.resize(frame, (320, int(h0 * k))), cv2.COLOR_BGR2GRAY))
        faces = self.face.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(60, 60))
        if not len(faces):
            return {"face": False}
        x, y, w, h = [int(v) for v in max(faces, key=lambda f: f[2] * f[3])]
        ey0, ey1 = y + int(0.18 * h), y + int(0.58 * h)
        eyes = self.eye.detectMultiScale(gray[ey0:ey1, x:x + w], scaleFactor=1.1, minNeighbors=4,
                                         minSize=(max(12, w // 9), max(12, w // 9)), maxSize=(w // 2, w // 2))
        s = 1.0 / k
        box = (int(x * s), int(y * s), int(w * s), int(h * s))
        eye_boxes = [(int((x + a) * s), int((ey0 + b) * s), int(c * s), int(d * s)) for (a, b, c, d) in eyes]
        return {"face": True, "eyes_found": len(eye_boxes), "box": box, "eye_boxes": eye_boxes}



# ====================== drowsiness rules (same as laptop_app.py) ======================
class Engine:
    """The per-frame rules of laptop_app.py's Detector.loop, without the server parts."""

    def __init__(self, kind):
        self.start(kind)

    def start(self, kind):
        self.kind = kind
        self.grace = OPEN_GRACE_S if kind == "mp" else 0.3
        self.closed_since = self.open_since = None
        self.perclos_hist, self.perclos_sum = collections.deque(), 0
        self.yawns, self.yawn_start, self.yawn_counted = collections.deque(), None, False
        self.last_face = time.time()
        self.recalibrate()

    def recalibrate(self):
        self.calibrated = self.kind != "mp"           # Haar has nothing to calibrate
        self.calib_start = None
        self.calib_samples = []
        self.ear_threshold = EAR_THRESHOLD_DEFAULT

    def _calibrate(self, obs, now):
        if self.calibrated:
            return
        if self.calib_start is None:
            self.calib_start = now
        if obs["blink"] < 0.4 and obs["ear"] > 0.12:          # only frames with clearly open eyes
            self.calib_samples.append(obs["ear"])
        if now - self.calib_start >= CALIBRATION_S and len(self.calib_samples) >= 15:
            base = statistics.median(self.calib_samples)
            thr = min(EAR_THRESHOLD_MAX, max(EAR_THRESHOLD_MIN, base * EAR_THRESHOLD_FACTOR))
            self.ear_threshold, self.calibrated = thr, True
            log(f"Calibrated: your open-eye EAR = {base:.3f}, closed below {thr:.3f}")

    def update(self, obs, now):
        face = obs["face"]
        closed = yawning = False
        if face:
            self.last_face = now
            if self.kind == "mp":
                self._calibrate(obs, now)
                closed = obs["ear"] < self.ear_threshold or obs["blink"] >= BLINK_SCORE_CLOSED
                yawning = obs["jaw"] >= YAWN_JAW_OPEN
            else:
                closed = obs["eyes_found"] == 0

        # ---- trigger 1: continuous eye closure (with a short grace for flicker) ----
        if face and closed:
            if self.closed_since is None:
                self.closed_since = now
            self.open_since = None
        elif face:
            if self.open_since is None:
                self.open_since = now
            if now - self.open_since >= self.grace:
                self.closed_since = None
        else:
            self.closed_since = self.open_since = None
        closed_for = (now - self.closed_since) if self.closed_since else 0.0

        # ---- trigger 2: PERCLOS ----
        if face:
            self.perclos_hist.append((now, closed))
            self.perclos_sum += closed
        while self.perclos_hist and now - self.perclos_hist[0][0] > PERCLOS_WINDOW_S:
            self.perclos_sum -= self.perclos_hist.popleft()[1]
        perclos = self.perclos_sum / len(self.perclos_hist) if self.perclos_hist else 0.0
        perclos_span = (now - self.perclos_hist[0][0]) if self.perclos_hist else 0.0

        # ---- trigger 3: yawns ----
        if yawning:
            if self.yawn_start is None:
                self.yawn_start = now
            elif not self.yawn_counted and now - self.yawn_start >= YAWN_MIN_S:
                self.yawns.append(now)
                self.yawn_counted = True
                log(f"yawn ({len(self.yawns)} in last {YAWN_WINDOW_S:.0f} s)")
        else:
            self.yawn_start, self.yawn_counted = None, False
        while self.yawns and now - self.yawns[0] > YAWN_WINDOW_S:
            self.yawns.popleft()

        # ---- decision ----
        why, kind = "", None
        if closed_for >= EYES_CLOSED_ALERT_S:
            why, kind = f"eyes closed for {closed_for:.1f} s", "eyes"
        elif PERCLOS_ALERT > 0 and perclos_span >= PERCLOS_MIN_DATA_S and perclos >= PERCLOS_ALERT:
            why, kind = f"eyes closed {perclos:.0%} of the last {PERCLOS_WINDOW_S:.0f} s", "perclos"
        elif YAWN_ALERT_COUNT > 0 and len(self.yawns) >= YAWN_ALERT_COUNT:
            why, kind = f"{len(self.yawns)} yawns in {YAWN_WINDOW_S:.0f} s", "yawns"
        elif NO_FACE_IS_DROWSY and not face and now - self.last_face >= NO_FACE_DROWSY_S:
            why, kind = f"face gone for {now - self.last_face:.1f} s", "noface"
        drowsy = kind is not None
        if kind in ("perclos", "yawns"):            # one-shot, as on the laptop: count again from zero
            self.perclos_hist.clear()
            self.perclos_sum = 0
            self.yawns.clear()

        if not face:
            label = "DROWSY" if drowsy else "NO_FACE"
        elif drowsy:
            label = "DROWSY"
        elif closed:
            label = "EYES_CLOSED"
        elif yawning:
            label = "YAWNING"
        elif not self.calibrated:
            label = "CALIBRATING"
        else:
            label = "AWAKE"
        return {"label": label, "drowsy": drowsy, "why": why, "closed_for": closed_for, "perclos": perclos,
                "yawns": len(self.yawns), "ear": obs.get("ear"), "face": face}


# ====================== video sources ======================
class StreamReader(threading.Thread):
    """Reads an MJPEG stream (multipart/x-mixed-replace) and keeps only the newest JPEG,
    so a slower UNO Q never falls behind the live video."""

    def __init__(self, url):
        super().__init__(daemon=True)
        self.url = url
        self.cond = threading.Condition()
        self.jpeg, self.seq = None, 0

    def latest(self, after, timeout=1.0):
        with self.cond:
            self.cond.wait_for(lambda: self.seq > after, timeout=timeout)
            return self.jpeg, self.seq

    def _publish(self, jpeg):
        with self.cond:
            self.jpeg = jpeg
            self.seq += 1
            self.cond.notify_all()

    def run(self):
        connected = None
        while True:
            try:
                with urllib.request.urlopen(self.url, timeout=STREAM_TIMEOUT_S) as r:
                    ctype = r.headers.get("Content-Type", "")
                    if "multipart" not in ctype:
                        log(f"{self.url} is not a video stream (it sent '{ctype or 'nothing'}').")
                        log("  -> use the laptop address, e.g. --stream http://10.37.177.20:5000")
                        time.sleep(5)
                        continue
                    if connected is not True:
                        log(f"watching the laptop's live video: {self.url}")
                        connected = True
                    self._read_parts(r)
                raise EOFError("the laptop closed the video stream")
            except (urllib.error.URLError, OSError, EOFError, ValueError) as e:
                if connected is not False:
                    log(f"no video from {self.url} ({reason(e)})")
                    log("  -> " + self._diagnose())
                    log("     retrying every 2 s ...")
                connected = False
                time.sleep(2)

    def _diagnose(self):
        """Read the laptop app's status page (read-only, like the dashboard) to say WHY there is no video."""
        base = self.url.split("/video.mjpg")[0].rstrip("/")
        try:
            with urllib.request.urlopen(base + "/api/status", timeout=4) as r:
                det = json.loads(r.read().decode("utf-8", "replace")).get("detector", {})
        except urllib.error.HTTPError as e:
            return f"the laptop answered HTTP {e.code}: is laptop_app.py the program on port 5000?"
        except (urllib.error.URLError, OSError, ValueError) as e:
            return (f"cannot reach the laptop at all ({reason(e)}). Check: laptop_app.py running? same Wi-Fi? "
                    "IP as printed by laptop_app.py? Windows Firewall allows Python on this network?")
        if det.get("error"):
            return f"laptop app is running, but its camera has a problem: {det['error']}"
        if not det.get("online"):
            return ("laptop app is running, but its camera is not sending frames "
                    "(started with --no-camera? webcam busy? see the laptop's terminal for an error)")
        return "laptop app and camera are fine; the video stream was interrupted"

    def _read_parts(self, r):
        while True:
            line = r.readline()
            if not line:
                return                                   # end of stream
            if not line.strip().startswith(b"--"):
                continue                                 # skip blank lines between parts
            length = None
            while True:                                  # part headers
                h = r.readline()
                if not h:
                    return
                h = h.strip()
                if not h:
                    break
                key, _, value = h.partition(b":")
                if key.strip().lower() == b"content-length":
                    length = int(value.strip())
            if length is not None:
                data = r.read(length)
                if len(data) < length:
                    return
            else:                                        # no length given: read until the JPEG end marker
                data = b""
                while not data.endswith(b"\xff\xd9"):
                    chunk = r.read(1)
                    if not chunk:
                        return
                    data += chunk
            if data[:2] == b"\xff\xd8":
                self._publish(data)


def open_camera(spec):
    src = int(spec) if spec.isdigit() else spec
    if isinstance(src, int) and sys.platform.startswith("linux"):
        cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
    else:
        cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def stream_url(text):
    url = text.strip()
    if "://" not in url:
        url = "http://" + url
    after_host = url.split("://", 1)[1]
    if "/" not in after_host.rstrip("/"):                # only an address was given
        url = url.rstrip("/") + "/video.mjpg"
    return url


# ====================== choosing the method ======================
SIGILL_HELP = ("this MediaPipe build needs CPU instructions (ARMv8.1 'LSE' atomics) that the UNO Q's "
               "Cortex-A53 processor does not have, so it crashes with 'Illegal instruction'")


def probe_mediapipe():
    """Start MediaPipe in a separate process first: if it crashes the whole process (e.g. 'Illegal
    instruction'), only the probe dies and this script can still fall back to Haar."""
    import subprocess
    try:
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--probe-mediapipe"],
                           capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return False, "MediaPipe took more than 10 minutes to start"
    out = (r.stdout + r.stderr).strip()
    if r.returncode == 0:
        return True, ""
    if r.returncode < 0 or "Illegal instruction" in out or "sigill" in out.lower():
        return False, SIGILL_HELP
    last = [l for l in out.splitlines() if l.strip()]
    return False, last[-1] if last else f"probe exited with code {r.returncode}"


def probe_main():
    try:
        a = MediaPipeAnalyzer()
        a.process(np.zeros((CAPTURE_H, CAPTURE_W, 3), np.uint8))
        print("MEDIAPIPE_OK", a.version, flush=True)
        return 0
    except Exception as e:
        print(f"{e.__class__.__name__}: {str(e).splitlines()[0] if str(e) else ''}", flush=True)
        return 2


def pick_analyzer(force_haar):
    if not force_haar:
        log("testing MediaPipe and the face model (this can take a little while on the UNO Q) ...")
        ok, why = probe_mediapipe()
        if ok:
            try:
                a = MediaPipeAnalyzer()
                log(f"MediaPipe {a.version} loaded")
                return a
            except Exception as e:
                why = f"{e.__class__.__name__}: {str(e).splitlines()[0]}"
        log("=" * 60)
        log(f"MediaPipe NOT used: {why}")
        log(f"  details: {sys.executable} {os.path.basename(__file__)} --check-mediapipe")
        log("  -> using the Haar method instead")
        log("=" * 60)
    return HaarAnalyzer()


def stdin_watch(flag):
    """Type r + Enter in this terminal to recalibrate."""
    try:
        for line in sys.stdin:
            if line.strip().lower() in ("r", "recalibrate"):
                flag.set()
    except Exception:
        pass


# ====================== self-test ======================
def check_mediapipe(stream):
    import platform
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    venv_fix = ("python3 -m venv --system-site-packages ~/mpenv  &&  ~/mpenv/bin/pip install mediapipe\n"
                "        then run this script with  ~/mpenv/bin/python  instead of  python3")
    pip_fix = f'"{sys.executable}" -m pip install mediapipe' if in_venv else venv_fix
    print("=" * 66)
    print(" MediaPipe check (Arduino UNO Q)")
    print(f" Python  : {sys.version.split()[0]} ({platform.system()} {platform.machine()})")
    print(f" Program : {sys.executable}  ({'separate environment' if in_venv else 'system Python'})")
    print("=" * 66)
    print(f"1. OpenCV ... OK, version {cv2.__version__}")

    print("2. start MediaPipe in a test process ...", end=" ", flush=True)
    ok, why = probe_mediapipe()
    if ok:
        print("OK")
    elif why == SIGILL_HELP:
        print("CRASHED (Illegal instruction)")
        print(f"   {why}.")
        print("   MediaPipe cannot run on this board. Use the Haar method (the script picks it automatically):")
        print(f"        {sys.executable} {os.path.basename(__file__)} --stream http://<laptop-IP>:5000")
        return 1
    else:
        print(f"failed ({why}), details below")

    print("3. import mediapipe ...", end=" ", flush=True)
    try:
        import mediapipe as mp
        print(f"OK, version {getattr(mp, '__version__', '?')}")
    except Exception as e:
        print(f"FAILED\n   {e.__class__.__name__}: {e}\n   Fix: {pip_fix}")
        return 1

    print("4. Tasks API (FaceLandmarker) ...", end=" ", flush=True)
    try:
        _b, vision = _mediapipe_api(mp)
        print(f"OK ({vision.FaceLandmarker.__name__})")
    except Exception as e:
        print(f"FAILED\n   {e.__class__.__name__}: {e}\n   Fix: {pip_fix} --upgrade")
        return 1

    print("5. face model file ...", end=" ", flush=True)
    try:
        path = ensure_model()
        print(f"OK, {path} ({os.path.getsize(path) // 1024} KB)")
    except Exception as e:
        print(f"FAILED\n   {e}")
        print("   Or copy it from the laptop (it sits next to laptop_app.py), in PowerShell on the laptop:")
        print(f"        scp face_landmarker.task arduino@<UNO-Q-IP>:{SCRIPT_DIR}/")
        return 1

    print("6. start the face landmarker + test run ...", end=" ", flush=True)
    try:
        analyzer = MediaPipeAnalyzer()
        print("OK")
    except Exception as e:
        print(f"FAILED\n   {e.__class__.__name__}: {e}")
        print("   Fix: delete face_landmarker.task next to this script and run again")
        return 1

    if stream:
        print(f"7. 5 s of the laptop's live video ({stream_url(stream)}) ...", flush=True)
        reader = StreamReader(stream_url(stream))
        reader.start()
        seq, frames, faces, ears, t_end, t0 = 0, 0, 0, [], time.time() + 5, time.time()
        while time.time() < t_end:
            jpeg, s = reader.latest(seq)
            if s == seq or jpeg is None:
                continue
            seq = s
            img = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            obs = analyzer.process(img)
            frames += 1
            if obs["face"]:
                faces += 1
                ears.append(obs["ear"])
        took = time.time() - t0
        if not frames:
            print("   no frames received (see the message above)")
        elif faces:
            print(f"   OK: {frames} frames ({frames / took:.1f} fps on the UNO Q), face in {faces}, "
                  f"eye openness (EAR) about {sum(ears) / len(ears):.2f}")
        else:
            print(f"   {frames} frames analysed but no face found: sit in front of the laptop camera")
    print("=" * 66)
    print(" RESULT: MediaPipe works on the UNO Q.")
    print(f" Run:  {sys.executable} {os.path.basename(__file__)} --stream http://<laptop-IP>:5000")
    print("=" * 66)
    return 0


# ====================== main loop ======================
def main():
    ap = argparse.ArgumentParser(description="UNO Q stand-alone drowsiness watcher (MediaPipe, Haar fallback)")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--stream", help="laptop address, e.g. http://172.20.10.2:5000 (reads /video.mjpg)")
    src.add_argument("--camera", help="USB webcam on the UNO Q: 0, 1 ... or /dev/video0")
    ap.add_argument("--haar", action="store_true", help="use the Haar method instead of MediaPipe")
    ap.add_argument("--check-mediapipe", action="store_true", help="test MediaPipe step by step and exit")
    ap.add_argument("--debug", action="store_true", help="print the result of every frame")
    ap.add_argument("--probe-mediapipe", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.probe_mediapipe:
        sys.exit(probe_main())
    if args.check_mediapipe:
        sys.exit(check_mediapipe(args.stream))
    if not (args.stream or args.camera):
        ap.error("give a video source: --stream http://<laptop-IP>:5000  or  --camera 0")

    log(f"UNO Q drowsiness watcher | OpenCV {cv2.__version__} | Python {sys.version.split()[0]}")
    analyzer = pick_analyzer(args.haar)
    engine = Engine(analyzer.kind)
    log(f"method: {analyzer.name}")
    if analyzer.kind == "mp":
        log(f"calibrating: look at the camera normally, eyes open, for {CALIBRATION_S:.0f} s "
            "(type r + Enter any time to recalibrate)")
    log("verdicts are shown here only; nothing is sent anywhere. Ctrl+C to stop.")

    recal = threading.Event()
    threading.Thread(target=stdin_watch, args=(recal,), daemon=True).start()

    reader, cap, last_seq = None, None, 0
    if args.stream:
        reader = StreamReader(stream_url(args.stream))
        reader.start()
    else:
        cap = open_camera(args.camera)
        if cap is None:
            sys.exit(f"ERROR: cannot open camera {args.camera}. Check it with:  ls /dev/video*")
        log(f"reading camera {args.camera}")

    frames, drowsy_events, fails = 0, 0, 0
    fps, fps_n, fps_t0 = 0.0, 0, time.time()
    t_start, last_print, prev_drowsy = time.time(), 0.0, False
    try:
        while True:
            # ---- newest frame ----
            if reader:
                jpeg, seq = reader.latest(last_seq)
                if seq == last_seq or jpeg is None:
                    continue
                last_seq = seq
                img = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
            else:
                ok, img = cap.read()
                if not ok:
                    fails += 1
                    if fails > 50:
                        sys.exit("ERROR: the camera stopped delivering frames.")
                    time.sleep(0.05)
                    continue
                fails = 0
            if img is None:
                continue

            if recal.is_set():
                recal.clear()
                engine.recalibrate()
                log("recalibrating: look at the camera with eyes open")

            # ---- detection on the UNO Q ----
            now = time.time()
            try:
                obs = analyzer.process(img)
            except Exception as e:
                if analyzer.kind == "mp":
                    traceback.print_exc()
                    log(f"MediaPipe failed while running ({e.__class__.__name__}: {e}) -> switching to Haar")
                    analyzer = HaarAnalyzer()
                    engine.start(analyzer.kind)
                    continue
                obs = {"face": False}
            r = engine.update(obs, now)
            frames += 1

            fps_n += 1
            if now - fps_t0 >= 1.0:
                fps, fps_n, fps_t0 = fps_n / (now - fps_t0), 0, now

            # ---- show it in this terminal ----
            if r["drowsy"] and not prev_drowsy:
                drowsy_events += 1
                log("=" * 58)
                log(f"  DROWSINESS DETECTED by the UNO Q  (#{drowsy_events})")
                log(f"  {r['why']}")
                log("=" * 58)
            elif prev_drowsy and not r["drowsy"]:
                log("driver awake again")
            prev_drowsy = r["drowsy"]

            ear = (f"EAR {r['ear']:.2f} (closed < {engine.ear_threshold:.2f}) | "
                   if analyzer.kind == "mp" and r["ear"] is not None else "")
            line = (f"{fps:4.1f} fps | {r['label']:<11} | {ear}closed {r['closed_for']:.1f} s | "
                    f"PERCLOS {r['perclos']:.0%}" + (f" | yawns {r['yawns']}" if analyzer.kind == "mp" else ""))
            if args.debug:
                log(line)
            elif now - last_print >= 1.0:
                last_print = now
                log(line)
    except KeyboardInterrupt:
        mins = (time.time() - t_start) / 60
        print(f"\nstopped. {frames} frames analysed in {mins:.1f} min with {analyzer.name}, "
              f"{drowsy_events} drowsiness detection(s).")
    finally:
        if cap is not None:
            cap.release()


if __name__ == "__main__":
    main()