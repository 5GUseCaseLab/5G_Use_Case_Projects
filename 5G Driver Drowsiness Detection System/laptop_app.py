#!/usr/bin/env python3
"""
laptop_app.py - Drowsiness-alert system, laptop-only version (v5.2: light / dark dashboard)
---------------------------------------------------------------------------------
Flow: car runs forward at full speed -> drowsiness detected -> car slows + buzzer
      -> click "Reset" on the dashboard (or press R) -> full speed again -> repeat.
The laptop decides the speed; the NodeMCU just follows. The push button is not used.

One program does everything on the laptop:
  * reads the webcam and detects drowsiness with MediaPipe Face Landmarker:
      - Eye Aspect Ratio (EAR) from the eyelid landmarks, auto-calibrated to YOUR eyes
      - MediaPipe "eyeBlink" scores as a second opinion
      - three triggers: eyes closed >= 1.5 s, PERCLOS (eyes-closed % over 60 s), repeated yawns
  * runs the server the ESP32/NodeMCU car talks to (protocol unchanged)
  * serves the dashboard at http://localhost:5000

Install once:   pip install flask opencv-python mediapipe
Run:            python laptop_app.py                (Windows: py laptop_app.py)
Check MediaPipe: python laptop_app.py --check-mediapipe   (tests every step, prints the fix)
Options:        --camera 1      use another webcam (default 0)
                --haar          use the old OpenCV Haar method instead of MediaPipe
                --no-camera     server + dashboard only
                --port 5000
First run downloads the MediaPipe model (face_landmarker.task, about 4 MB) next to this file.
"""

import argparse
import collections
import logging
import os
import socket
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
import zipfile

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================== settings ==============================
PORT = 5000
CAMERA_INDEX = 0
CAPTURE_W, CAPTURE_H = 640, 480

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
EYES_CLOSED_ALERT_S = 2.0        # TRIGGER 1: eyes closed continuously this long (normal blink ~0.3 s)
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

# --- car behaviour (sent to the NodeMCU every 150 ms) ---
DEFAULT_DRIVE = "forward"        # the motor runs as soon as the car connects ("stop" to start stopped)
NORMAL_SPEED = 255               # full speed, PWM 0-255 (lower to ~180 if the motor gets hot)
SLOW_SPEED = 90                  # speed while drowsiness alert is active

# --- server ---
ALERT_COOLDOWN_S = 3.0           # after Reset, ignore drowsiness this long (time to open your eyes)
CAR_TIMEOUT_S = 3.0
DETECTOR_TIMEOUT_S = 3.0
VALID_DRIVE = {"stop", "forward", "backward", "left", "right"}
# ======================================================================

# MediaPipe face-mesh landmark indices: p1..p6 around each eye
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]

app = Flask(__name__)
logging.getLogger("werkzeug").setLevel(logging.ERROR)
lock = threading.Lock()

state = {
    "detector": {"last_seen": 0.0, "data": {}, "error": None, "recalibrate": False},
    "car": {"last_seen": 0.0, "ip": None, "data": {}},
    "alert": {"active": False, "id": 0, "source": None, "raised_at": None},
    "drive": DEFAULT_DRIVE,
    "cooldown_until": 0.0,
    "alerts_total": 0,
}
events = collections.deque(maxlen=60)
next_alert_id = int(time.time()) % 1_000_000_000

frame_cond = threading.Condition()
latest_jpeg = {"data": None, "seq": 0}


def log_event(text):
    """Call with lock held."""
    stamp = time.strftime("%H:%M:%S")
    events.appendleft({"t": stamp, "text": text})
    print(f"[{stamp}] {text}", flush=True)


def raise_alert(source, reason=""):
    """Call with lock held. Returns True if a new alert was raised."""
    global next_alert_id
    now = time.time()
    if state["alert"]["active"]:
        return False
    if source == "camera" and now < state["cooldown_until"]:
        return False
    next_alert_id += 1
    state["alert"] = {"active": True, "id": next_alert_id, "source": source, "raised_at": now}
    state["alerts_total"] += 1
    why = f"webcam: {reason}" if source == "camera" else "test alert from dashboard"
    log_event(f"ALERT #{next_alert_id} raised ({why}) - car slowed, click Reset on the dashboard")
    return True


# ============================== analyzers ==============================
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
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (drowsiness laptop_app)"})
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
        sys.exit(f"ERROR: {name} not found. Install OpenCV with: pip install opencv-python")

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


# ============================== camera ==============================
def open_camera(index):
    if sys.platform.startswith("win"):
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    elif sys.platform == "darwin":
        backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
    else:
        backends = [cv2.CAP_V4L2, cv2.CAP_ANY]
    for be in backends:
        cap = cv2.VideoCapture(index, be)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_H)
            for _ in range(20):
                ok, frame = cap.read()
                if ok and frame is not None and frame.size > 0:
                    return cap
                time.sleep(0.05)
        cap.release()
    return None


# ============================== detector ==============================
class Detector(threading.Thread):
    def __init__(self, camera_index, force_haar=False):
        super().__init__(daemon=True)
        self.camera_index = camera_index
        self.analyzer = None
        self.mp_problem = None
        if not force_haar:
            try:
                self.analyzer = MediaPipeAnalyzer()
                print(f"[detector] MediaPipe {self.analyzer.version} loaded", flush=True)
            except Exception as e:  # missing package, no model, API change, unsupported Python ...
                self.mp_problem = f"{e.__class__.__name__}: {e}"
                print("[detector] " + "=" * 60, flush=True)
                print(f"[detector] MediaPipe NOT used: {self.mp_problem}", flush=True)
                if isinstance(e, ImportError):
                    print(f'[detector] Fix: "{sys.executable}" -m pip install --upgrade mediapipe', flush=True)
                print("[detector] Details: python laptop_app.py --check-mediapipe", flush=True)
                print("[detector] -> using the less accurate Haar method for now.", flush=True)
                print("[detector] " + "=" * 60, flush=True)
        if self.analyzer is None:
            self.analyzer = HaarAnalyzer()
        print(f"[detector] method: {self.analyzer.name}", flush=True)
        self.reset_calibration()

    def reset_calibration(self):
        self.calibrated = self.analyzer.kind != "mp"   # Haar has nothing to calibrate
        self.calib_start = None
        self.calib_samples = []
        self.ear_threshold = EAR_THRESHOLD_DEFAULT

    def set_error(self, msg):
        with lock:
            if state["detector"]["error"] != msg:
                log_event(f"Camera: {msg}")
            state["detector"]["error"] = msg

    def run(self):
        while True:
            cap = open_camera(self.camera_index)
            if cap is None:
                self.set_error(f"cannot open webcam {self.camera_index} "
                               f"(in use by another app, or no camera permission) - retrying")
                time.sleep(3)
                continue
            with lock:
                state["detector"]["error"] = None
                log_event(f"Webcam {self.camera_index} opened - look at the camera with eyes open "
                          f"for {CALIBRATION_S:.0f} s to calibrate")
            self.loop(cap)
            cap.release()
            time.sleep(1)

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
            with lock:
                log_event(f"Calibrated: your open-eye EAR = {base:.3f}, closed below {thr:.3f}")

    def loop(self, cap):
        grace = OPEN_GRACE_S if self.analyzer.kind == "mp" else 0.3
        closed_since = open_since = None
        perclos_hist, perclos_sum = collections.deque(), 0
        yawns, yawn_start, yawn_counted = collections.deque(), None, False
        last_face = time.time()
        drowsy_prev = False
        fps, fps_n, fps_t0 = 0.0, 0, time.time()
        last_print, fails = 0.0, 0

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                fails += 1
                if fails > 30:
                    self.set_error("webcam stopped delivering frames - reopening")
                    return
                time.sleep(0.02)
                continue
            fails = 0
            now = time.time()
            frame = cv2.flip(frame, 1)

            with lock:
                if state["detector"]["recalibrate"]:
                    state["detector"]["recalibrate"] = False
                    self.reset_calibration()
                    log_event("Recalibrating - look at the camera with eyes open")

            obs = self.analyzer.process(frame)
            face = obs["face"]
            closed = yawning = False
            if face:
                last_face = now
                if self.analyzer.kind == "mp":
                    self._calibrate(obs, now)
                    closed = obs["ear"] < self.ear_threshold or obs["blink"] >= BLINK_SCORE_CLOSED
                    yawning = obs["jaw"] >= YAWN_JAW_OPEN
                else:
                    closed = obs["eyes_found"] == 0

            # ---- trigger 1: continuous eye closure (with a short grace for flicker) ----
            if face and closed:
                if closed_since is None:
                    closed_since = now
                open_since = None
            elif face:
                if open_since is None:
                    open_since = now
                if now - open_since >= grace:
                    closed_since = None
            else:
                closed_since = open_since = None
            closed_for = (now - closed_since) if closed_since else 0.0

            # ---- trigger 2: PERCLOS ----
            if face:
                perclos_hist.append((now, closed))
                perclos_sum += closed
            while perclos_hist and now - perclos_hist[0][0] > PERCLOS_WINDOW_S:
                perclos_sum -= perclos_hist.popleft()[1]
            perclos = perclos_sum / len(perclos_hist) if perclos_hist else 0.0
            perclos_span = (now - perclos_hist[0][0]) if perclos_hist else 0.0

            # ---- trigger 3: yawns ----
            if yawning:
                if yawn_start is None:
                    yawn_start = now
                elif not yawn_counted and now - yawn_start >= YAWN_MIN_S:
                    yawns.append(now)
                    yawn_counted = True
                    print(f"[detector] yawn ({len(yawns)} in last {YAWN_WINDOW_S:.0f} s)", flush=True)
            else:
                yawn_start, yawn_counted = None, False
            while yawns and now - yawns[0] > YAWN_WINDOW_S:
                yawns.popleft()

            # ---- decision ----
            reason, kind = "", None
            if closed_for >= EYES_CLOSED_ALERT_S:
                reason, kind = f"eyes closed for {closed_for:.1f} s", "eyes"
            elif PERCLOS_ALERT > 0 and perclos_span >= PERCLOS_MIN_DATA_S and perclos >= PERCLOS_ALERT:
                reason, kind = f"eyes closed {perclos:.0%} of the last {PERCLOS_WINDOW_S:.0f} s", "perclos"
            elif YAWN_ALERT_COUNT > 0 and len(yawns) >= YAWN_ALERT_COUNT:
                reason, kind = f"{len(yawns)} yawns in {YAWN_WINDOW_S:.0f} s", "yawns"
            elif NO_FACE_IS_DROWSY and not face and now - last_face >= NO_FACE_DROWSY_S:
                reason, kind = f"face gone for {now - last_face:.1f} s", "noface"
            drowsy = kind is not None

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

            fps_n += 1
            if now - fps_t0 >= 1.0:
                fps, fps_n, fps_t0 = fps_n / (now - fps_t0), 0, now

            ear = obs.get("ear")
            with lock:
                state["detector"]["last_seen"] = now
                state["detector"]["data"] = {
                    "method": self.analyzer.name, "state": label, "drowsy": drowsy, "reason": reason,
                    "face": face, "calibrated": self.calibrated,
                    "ear": round(ear, 3) if ear is not None else None,
                    "ear_threshold": round(self.ear_threshold, 3) if self.analyzer.kind == "mp" else None,
                    "closed_for": round(closed_for, 1), "closed_alert_s": EYES_CLOSED_ALERT_S,
                    "perclos": round(perclos, 3), "perclos_alert": PERCLOS_ALERT,
                    "yawns": len(yawns), "yawn_alert_count": YAWN_ALERT_COUNT,
                    "fps": round(fps, 1)}
                fired = False
                if drowsy:
                    fired = raise_alert("camera", reason) or state["alert"]["active"]
            # PERCLOS / yawn triggers are one-shot: start counting again after they fire
            if fired and kind in ("perclos", "yawns"):
                perclos_hist.clear()
                perclos_sum = 0
                yawns.clear()

            self.draw(frame, obs, label, reason, closed_for, perclos, len(yawns), now)

            if drowsy and not drowsy_prev:
                print(f"[detector] >>> DROWSY: {reason}", flush=True)
            elif drowsy_prev and not drowsy:
                print("[detector] driver awake again", flush=True)
            drowsy_prev = drowsy
            if now - last_print >= 2.0:
                last_print = now
                e = f"EAR {ear:.3f}/{self.ear_threshold:.3f} | " if ear is not None else ""
                print(f"[detector] {fps:4.1f} fps | {label:<11} | {e}closed {closed_for:.1f}s | "
                      f"PERCLOS {perclos:.0%} | yawns {len(yawns)}", flush=True)

    def draw(self, frame, obs, label, reason, closed_for, perclos, n_yawns, now):
        red, green, orange, white = (0, 0, 255), (0, 200, 0), (0, 160, 255), (255, 255, 255)
        color = red if label in ("DROWSY", "EYES_CLOSED") else green
        if obs["face"]:
            x, y, w, h = obs["box"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            for p in obs.get("pts", []):
                cv2.circle(frame, p, 2, orange, -1)
            for (ex, ey, ew, eh) in obs.get("eye_boxes", []):
                cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), orange, 2)
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 64), (0, 0, 0), -1)
        cv2.putText(frame, label + (f": {reason}" if reason else ""), (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        if self.analyzer.kind == "mp" and obs.get("ear") is not None:
            line = (f"EAR {obs['ear']:.2f} (closed < {self.ear_threshold:.2f})  closed {closed_for:.1f}s  "
                    f"PERCLOS {perclos:.0%}  yawns {n_yawns}")
        else:
            line = f"closed {closed_for:.1f}s  PERCLOS {perclos:.0%}"
        if not self.calibrated and self.calib_start is not None:
            left = max(0.0, CALIBRATION_S - (now - self.calib_start))
            line = f"CALIBRATING: look at the camera, eyes open ({left:.0f}s)"
        cv2.putText(frame, line, (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, white, 1, cv2.LINE_AA)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            with frame_cond:
                latest_jpeg["data"] = buf.tobytes()
                latest_jpeg["seq"] += 1
                frame_cond.notify_all()


# ============================== ESP32 / NodeMCU car ==============================
@app.post("/api/car/sync")
def car_sync():
    data = request.get_json(silent=True) or {}
    now = time.time()
    with lock:
        car = state["car"]
        if now - car["last_seen"] >= CAR_TIMEOUT_S:
            log_event(f"Car online ({request.remote_addr})")
        car.update(last_seen=now, ip=request.remote_addr, data=data)
        # Acknowledgements from the car are ignored: only the dashboard Reset clears an alert.
        alert = state["alert"]
        return jsonify({"alert_active": alert["active"], "alert_id": alert["id"],
                        "drive": state["drive"],
                        "speed_normal": NORMAL_SPEED, "speed_slow": SLOW_SPEED,
                        "server_time": int(now)})


# ============================== dashboard API ==============================
@app.get("/api/status")
def api_status():
    now = time.time()
    with lock:
        det, car, alert = state["detector"], state["car"], state["alert"]
        return jsonify({
            "detector": {"online": now - det["last_seen"] < DETECTOR_TIMEOUT_S,
                         "error": det["error"], **det["data"]},
            "car": {"online": now - car["last_seen"] < CAR_TIMEOUT_S, "ip": car["ip"], **car["data"]},
            "alert": {"active": alert["active"], "id": alert["id"],
                      "seconds": round(now - alert["raised_at"], 1) if alert["active"] else 0},
            "cooldown_s": max(0.0, round(state["cooldown_until"] - now, 1)),
            "drive": state["drive"],
            "alerts_total": state["alerts_total"],
            "events": list(events),
        })


@app.get("/video.mjpg")
def video():
    def stream():
        seq = -1
        while True:
            with frame_cond:
                frame_cond.wait_for(lambda: latest_jpeg["seq"] != seq, timeout=2.0)
                if latest_jpeg["seq"] == seq or latest_jpeg["data"] is None:
                    continue
                seq, jpg = latest_jpeg["seq"], latest_jpeg["data"]
            yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                   + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
    return Response(stream(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.post("/api/drive")
def api_drive():
    cmd = str((request.get_json(silent=True) or {}).get("cmd", "")).lower()
    if cmd not in VALID_DRIVE:
        return jsonify({"ok": False, "error": f"cmd must be one of {sorted(VALID_DRIVE)}"}), 400
    with lock:
        if state["drive"] != cmd:
            state["drive"] = cmd
            log_event(f"Drive command: {cmd.upper()}")
    return jsonify({"ok": True, "drive": cmd})


@app.post("/api/test/drowsy")
def api_test():
    with lock:
        raised = raise_alert("test")
    return jsonify({"ok": True, "raised": raised})


@app.post("/api/reset")
def api_reset():
    now = time.time()
    with lock:
        alert = state["alert"]
        if not alert["active"]:
            return jsonify({"ok": True, "cleared": False})
        alert["active"] = False
        state["cooldown_until"] = now + ALERT_COOLDOWN_S
        log_event(f"ALERT #{alert['id']} reset from dashboard after "
                  f"{now - alert['raised_at']:.1f} s - back to full speed")
    return jsonify({"ok": True, "cleared": True})


@app.post("/api/recalibrate")
def api_recalibrate():
    with lock:
        state["detector"]["recalibrate"] = True
    return jsonify({"ok": True})


@app.get("/")
def dashboard():
    return Response(DASHBOARD_HTML, mimetype="text/html")


DASHBOARD_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Drowsiness Alert – Dashboard</title>
<script>
  (function () {   // saved choice, otherwise the computer's light/dark setting
    var t = null;
    try { t = localStorage.getItem("theme"); } catch (e) {}
    if (t !== "light" && t !== "dark")
      t = (window.matchMedia && matchMedia("(prefers-color-scheme: light)").matches) ? "light" : "dark";
    document.documentElement.dataset.theme = t;
  })();
</script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600;700&display=swap" rel="stylesheet">
<style>
  :root {
    --bg:#070B17; --panel:#0E1528; --panel2:#131C33; --edge:rgba(255,255,255,.07); --edge2:rgba(255,255,255,.12);
    --text:#E8ECF6; --muted:#8792AB; --dim:#5A6582;
    --amber:#F5A83B; --teal:#22C3A6; --red:#FF5A5F; --blue:#6E8BFF;
    --tint:rgba(255,255,255,.03); --tint2:rgba(255,255,255,.06); --hover:rgba(255,255,255,.08);
    --track:rgba(255,255,255,.07); --gridline:rgba(255,255,255,.025);
    --glow1:rgba(110,139,255,.16); --glow2:rgba(245,168,59,.10); --shadow:0 10px 30px rgba(0,0,0,.25);
    --pad-bg:linear-gradient(180deg, #1A2443, #121A33); --pad-base:#080D1C;
    --stop-bg:linear-gradient(180deg, #3A1620, #2A0F18); --stop-text:#FFB3B5;
    --log-text:#C9D0E2; --log-alert:#FFD0D2; --bar-end:#9AB0FF; --trace-line:#9AB0FF;
    --alert-bg:linear-gradient(110deg, rgba(255,90,95,.38), rgba(255,90,95,.10) 65%), #2A0F18; --alert-sub:#FFD3D4;
    color-scheme:dark;
    --sans:"Space Grotesk", "Segoe UI", system-ui, -apple-system, Roboto, sans-serif;
    --mono:"JetBrains Mono", ui-monospace, Consolas, monospace;
  }
  :root[data-theme="light"] {
    --bg:#EEF2F8; --panel:#FFFFFF; --panel2:#F9FBFE; --edge:rgba(15,26,56,.10); --edge2:rgba(15,26,56,.17);
    --text:#14203A; --muted:#55617A; --dim:#7A859B;
    --amber:#C2740E; --teal:#0C8A76; --red:#D3363C; --blue:#3D5BD9;
    --tint:rgba(15,26,56,.035); --tint2:rgba(15,26,56,.065); --hover:rgba(15,26,56,.09);
    --track:rgba(15,26,56,.09); --gridline:rgba(15,26,56,.045);
    --glow1:rgba(61,91,217,.10); --glow2:rgba(245,168,59,.14); --shadow:0 6px 22px rgba(15,26,56,.08);
    --pad-bg:linear-gradient(180deg, #FFFFFF, #EDF1F8); --pad-base:#C7D0E1;
    --stop-bg:linear-gradient(180deg, #FFF3F3, #FCE0E1); --stop-text:#B4232A;
    --log-text:#2D3954; --log-alert:#B4232A; --bar-end:#7D95F2; --trace-line:#3D5BD9;
    --alert-bg:linear-gradient(110deg, #E5484D, #C22C32); --alert-sub:#FFE6E7;
    color-scheme:light;
  }
  * { box-sizing:border-box; }
  html, body { margin:0; }
  body { min-height:100vh; font-family:var(--sans); color:var(--text); background:var(--bg); font-size:15px;
         background-image:
           radial-gradient(900px 500px at 12% -10%, var(--glow1), transparent 60%),
           radial-gradient(700px 500px at 100% 0%, var(--glow2), transparent 60%),
           linear-gradient(var(--gridline) 1px, transparent 1px),
           linear-gradient(90deg, var(--gridline) 1px, transparent 1px);
         transition:background-color .25s, color .25s;
         background-size:auto, auto, 32px 32px, 32px 32px; }
  .wrap { max-width:1400px; margin:0 auto; padding:20px 24px 32px; }

  /* ---------- top bar ---------- */
  header { display:flex; align-items:center; gap:14px; margin-bottom:18px; flex-wrap:wrap; }
  .mark { width:44px; height:44px; border-radius:12px; display:grid; place-items:center;
          background:linear-gradient(135deg, #1B2550, #0F1A38); border:1px solid var(--edge2);
          box-shadow:0 0 0 4px rgba(110,139,255,.08); }
  header h1 { margin:0; font-size:21px; font-weight:700; letter-spacing:-.01em; }
  header .sub { color:var(--muted); font-size:13px; margin-top:2px; }
  .spacer { flex:1; }
  #clock { font-family:var(--mono); font-size:12.5px; color:var(--muted); padding:7px 12px; border-radius:99px;
           background:var(--panel); border:1px solid var(--edge); }
  #clock::before { content:""; display:inline-block; width:7px; height:7px; border-radius:50%; background:var(--teal);
                   margin-right:8px; vertical-align:1px; box-shadow:0 0 8px var(--teal); animation:blink 2s infinite; }
  @keyframes blink { 50% { opacity:.35; } }
  .namebox { display:flex; align-items:center; gap:8px; padding:5px 5px 5px 12px; border-radius:99px;
             background:var(--panel); border:1px solid var(--edge2); transition:border-color .2s, box-shadow .2s; }
  .namebox:focus-within { border-color:var(--amber); box-shadow:0 0 0 3px rgba(245,168,59,.15); }
  .namebox svg { flex:none; color:var(--amber); }
  .namebox label { font-size:11px; font-weight:600; letter-spacing:.1em; text-transform:uppercase; color:var(--dim); }
  #driverName { width:150px; min-width:0; background:transparent; border:0; outline:0; color:var(--text);
                font:600 14.5px var(--sans); padding:4px 2px; }
  #driverName::placeholder { color:var(--dim); font-weight:500; }
  .namebox button { font:600 12.5px var(--sans); color:var(--text); cursor:pointer; padding:7px 12px; border-radius:99px;
                    border:1px solid var(--edge2); background:var(--tint2); transition:background .2s, color .2s, border-color .2s; }
  .namebox button:hover { background:var(--hover); }
  #saveName { background:var(--amber); border-color:var(--amber); color:#1A1205; }
  #saveName:hover { background:#FFBA55; }
  #saveName.done { background:rgba(34,195,166,.15); border-color:rgba(34,195,166,.5); color:var(--teal); }
  #armHint { display:none; font-size:12px; color:var(--amber); padding:6px 11px; border-radius:99px;
             border:1px dashed rgba(245,168,59,.5); background:rgba(245,168,59,.06); }
  #armHint.show { display:inline-block; }
  @media (max-width:720px) { .namebox { width:100%; } #driverName { flex:1; width:auto; } }

  /* ---------- grid ---------- */
  main { display:grid; grid-template-columns:repeat(12, 1fr); gap:16px; }
  .card { background:linear-gradient(180deg, var(--panel2), var(--panel)); border:1px solid var(--edge);
          border-radius:18px; padding:18px; box-shadow:var(--shadow); min-width:0; }
  .card h2 { display:flex; align-items:center; gap:10px; margin:0 0 14px; font-size:12px; font-weight:600;
             letter-spacing:.12em; text-transform:uppercase; color:var(--muted); }
  .card h2 .grow { flex:1; }
  .s-banner { grid-column:1 / -1; }
  .s-cam { grid-column:span 7; } .s-det { grid-column:span 5; }
  .s-car { grid-column:span 4; } .s-drive { grid-column:span 4; } .s-log { grid-column:span 4; }
  @media (max-width:1100px) { .s-cam, .s-det { grid-column:1 / -1; } .s-car, .s-drive { grid-column:span 6; } .s-log { grid-column:1 / -1; } }
  @media (max-width:720px) { .wrap { padding:14px; } .s-car, .s-drive { grid-column:1 / -1; } .mini { grid-template-columns:1fr; } .hero { flex-direction:column; align-items:flex-start; } .ear { text-align:left; } }

  /* ---------- status pills ---------- */
  .pill { display:inline-flex; align-items:center; gap:7px; padding:4px 11px; border-radius:99px; font-size:11.5px;
          font-weight:600; letter-spacing:.06em; text-transform:uppercase; border:1px solid; }
  .pill::before { content:""; width:7px; height:7px; border-radius:50%; background:currentColor; }
  .pill.on { color:var(--teal); border-color:rgba(34,195,166,.35); background:rgba(34,195,166,.10); }
  .pill.on::before { box-shadow:0 0 8px currentColor; animation:blink 2s infinite; }
  .pill.off { color:var(--red); border-color:rgba(255,90,95,.35); background:rgba(255,90,95,.08); }

  /* ---------- banner ---------- */
  #banner { position:relative; overflow:hidden; display:flex; align-items:center; gap:18px; flex-wrap:wrap;
            padding:20px 24px; border-radius:20px; border:1px solid rgba(34,195,166,.35);
            background:linear-gradient(110deg, rgba(34,195,166,.20), rgba(34,195,166,.04) 60%), var(--panel);
            transition:background .3s, border-color .3s; }
  #banner .ico { flex:none; width:58px; height:58px; border-radius:16px; display:grid; place-items:center;
                 background:rgba(34,195,166,.15); color:var(--teal); transition:inherit; }
  #banner .txt { flex:1; min-width:240px; }
  #bannerText { font-size:28px; font-weight:700; letter-spacing:-.015em; color:var(--teal); line-height:1.15; }
  #bannerSub { display:block; margin-top:4px; color:var(--muted); font-size:14.5px; }
  #banner.cool { border-color:rgba(245,168,59,.4);
                 background:linear-gradient(110deg, rgba(245,168,59,.20), rgba(245,168,59,.04) 60%), var(--panel); }
  #banner.cool .ico { background:rgba(245,168,59,.15); color:var(--amber); }
  #banner.cool #bannerText { color:var(--amber); }
  #banner.alert { border-color:rgba(255,90,95,.7); animation:alarm 1.1s ease-in-out infinite;
                  background:var(--alert-bg); }
  #banner.alert .ico { background:var(--red); color:#fff; }
  #banner.alert #bannerText { color:#fff; }
  #banner.alert #bannerSub { color:var(--alert-sub); }
  @keyframes alarm { 0%,100% { box-shadow:0 0 0 0 rgba(255,90,95,.0), 0 0 40px rgba(255,90,95,.25); }
                     50% { box-shadow:0 0 0 6px rgba(255,90,95,.18), 0 0 70px rgba(255,90,95,.45); } }
  #resetBtn { display:none; font:inherit; font-size:18px; font-weight:700; padding:15px 26px; border-radius:14px; border:0;
              background:#fff; color:#C9262B; cursor:pointer; box-shadow:0 8px 24px rgba(0,0,0,.35); }
  #resetBtn:hover { transform:translateY(-1px); } #resetBtn:active { transform:none; }
  #resetBtn kbd { font-family:var(--mono); font-size:13px; padding:2px 7px; border-radius:6px; margin-left:8px;
                  background:#FDE3E4; color:#C9262B; }
  #banner.alert #resetBtn { display:inline-block; }

  /* ---------- camera ---------- */
  .viewer { position:relative; border-radius:14px; overflow:hidden; background:#000; aspect-ratio:4 / 3; }
  #cam { width:100%; height:100%; object-fit:contain; display:block; }
  .viewer::after { content:""; position:absolute; inset:12px; pointer-events:none;
    background:
      linear-gradient(var(--amber),var(--amber)) top left / 26px 2px no-repeat, linear-gradient(var(--amber),var(--amber)) top left / 2px 26px no-repeat,
      linear-gradient(var(--amber),var(--amber)) top right / 26px 2px no-repeat, linear-gradient(var(--amber),var(--amber)) top right / 2px 26px no-repeat,
      linear-gradient(var(--amber),var(--amber)) bottom left / 26px 2px no-repeat, linear-gradient(var(--amber),var(--amber)) bottom left / 2px 26px no-repeat,
      linear-gradient(var(--amber),var(--amber)) bottom right / 26px 2px no-repeat, linear-gradient(var(--amber),var(--amber)) bottom right / 2px 26px no-repeat;
    opacity:.85; }
  .live { position:absolute; right:22px; top:22px; display:flex; align-items:center; gap:7px; padding:5px 10px; border-radius:8px;
          background:rgba(7,11,23,.7); backdrop-filter:blur(4px); font:600 11.5px var(--mono); letter-spacing:.1em; color:#fff; }
  .live::before { content:""; width:8px; height:8px; border-radius:50%; background:var(--red); box-shadow:0 0 8px var(--red); animation:blink 1.2s infinite; }
  #camnote { margin-top:12px; color:var(--muted); font-size:13px; }
  .legend { display:flex; flex-wrap:wrap; gap:6px 16px; margin-top:12px; font-size:12.5px; color:var(--muted); }
  .legend i { display:inline-block; width:10px; height:10px; border-radius:3px; margin-right:6px; vertical-align:-1px; }

  /* ---------- detector ---------- */
  .hero { display:flex; align-items:flex-end; justify-content:space-between; gap:12px; padding:14px 16px; border-radius:14px;
          background:var(--tint); border:1px solid var(--edge); margin-bottom:14px; }
  .k { font-size:11.5px; letter-spacing:.1em; text-transform:uppercase; color:var(--dim); font-weight:600; }
  #detState { display:block; margin-top:4px; font-size:22px; font-weight:700; letter-spacing:-.01em; color:var(--text); }
  #detState[data-s="AWAKE"] { color:var(--teal); } #detState[data-s="EYES_CLOSED"], #detState[data-s="YAWNING"] { color:var(--amber); }
  #detState[data-s="DROWSY"] { color:var(--red); } #detState[data-s="CALIBRATING"] { color:var(--blue); }
  #detState[data-s="NO_FACE"] { color:var(--muted); }
  .ear { text-align:right; }
  #detEar { display:block; margin-top:4px; font:600 14px var(--mono); color:var(--text); }
  .meter { margin:0 0 14px; }
  .meter .top { display:flex; justify-content:space-between; gap:10px; font-size:13.5px; color:var(--muted); margin-bottom:7px; }
  .meter .top b { font:600 13px var(--mono); color:var(--text); white-space:nowrap; }
  .bar { position:relative; height:10px; border-radius:99px; background:var(--tint2); overflow:hidden; }
  .bar > div { height:100%; width:0%; border-radius:99px; transition:width .25s ease, background .25s;
               background:linear-gradient(90deg, var(--blue), var(--bar-end)); box-shadow:0 0 12px rgba(110,139,255,.45); }
  .bar > div.hot { background:linear-gradient(90deg, #FF7A5C, var(--red)); box-shadow:0 0 14px rgba(255,90,95,.6); }
  .bar::after { content:""; position:absolute; right:0; top:0; bottom:0; width:2px; background:rgba(255,90,95,.6); }
  .mini { display:grid; grid-template-columns:repeat(3, 1fr); gap:10px; margin-top:4px; }
  .mini > div { padding:10px 12px; border-radius:12px; background:var(--tint); border:1px solid var(--edge); min-width:0; }
  .mini b { display:block; margin-top:4px; font:600 14px var(--mono); color:var(--text); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  #detMethod { font-family:var(--sans); font-size:13px; }

  .s-det { display:flex; flex-direction:column; }
  .trace { flex:1; min-height:130px; display:flex; flex-direction:column; margin:2px 0 14px; padding:12px 14px 10px;
           border-radius:14px; background:var(--tint); border:1px solid var(--edge); }
  #earLine { stroke:var(--trace-line); } #earFill stop { stop-color:var(--blue); } #earThr { stroke:var(--red); }
  .trace .top { display:flex; justify-content:space-between; gap:10px; margin-bottom:8px; }
  .trace .top i { font-style:normal; font-size:12px; color:var(--muted); }
  .trace .top i b { display:inline-block; width:14px; height:0; border-top:2px dashed var(--red); vertical-align:3px; margin-right:6px; }
  #earTrace { flex:1; width:100%; min-height:90px; display:block; }
  /* ---------- car ---------- */
  .gauge { position:relative; width:100%; max-width:300px; margin:0 auto 6px; }
  .gauge svg { width:100%; height:auto; display:block; }
  .gauge .track { fill:none; stroke:var(--track); stroke-width:16; stroke-linecap:round; }
  #carSpeed { fill:none; stroke:url(#speedGrad); stroke-width:16; stroke-linecap:round; stroke-dasharray:100; stroke-dashoffset:100;
              transition:stroke-dashoffset .3s ease; filter:drop-shadow(0 0 6px rgba(245,168,59,.5)); }
  .gauge .read { position:absolute; left:0; right:0; bottom:6px; text-align:center; }
  #carSpeedTxt { display:block; font:700 26px var(--mono); color:var(--text); }
  .gauge .read span { font-size:11.5px; letter-spacing:.12em; text-transform:uppercase; color:var(--dim); font-weight:600; }
  .rows { margin-top:10px; }
  .row { display:flex; justify-content:space-between; gap:12px; padding:9px 0; border-top:1px solid var(--edge); font-size:14px; color:var(--muted); }
  .row b { font:600 13.5px var(--mono); color:var(--text); text-align:right; }
  #carState { padding:2px 9px; border-radius:7px; background:var(--tint2); }

  /* ---------- drive ---------- */
  .pad { display:grid; grid-template-columns:repeat(3, 1fr); gap:10px; max-width:290px; margin:4px auto 0; }
  .pad button, .tools button { font:inherit; cursor:pointer; color:var(--text); transition:transform .08s, background .2s, border-color .2s, box-shadow .2s; }
  .pad button { aspect-ratio:1.2; border-radius:16px; font-size:13.5px; font-weight:600; border:1px solid var(--edge2);
                background:var(--pad-bg); box-shadow:inset 0 1px 0 rgba(255,255,255,.06), 0 4px 0 var(--pad-base);
                display:flex; flex-direction:column; align-items:center; justify-content:center; gap:4px; }
  .pad button .a { font-size:20px; line-height:1; color:var(--muted); }
  .pad button:hover { border-color:rgba(245,168,59,.5); }
  .pad button:active { transform:translateY(3px); box-shadow:inset 0 1px 0 rgba(255,255,255,.06), 0 1px 0 var(--pad-base); }
  .pad button.active { border-color:var(--amber); background:linear-gradient(180deg, rgba(245,168,59,.30), rgba(245,168,59,.12));
                       box-shadow:0 0 0 3px rgba(245,168,59,.18), 0 0 24px rgba(245,168,59,.30), 0 4px 0 var(--pad-base); }
  .pad button.active .a { color:var(--amber); }
  .pad .stop { background:var(--stop-bg); border-color:rgba(255,90,95,.45); color:var(--stop-text); }
  .pad .stop .a { color:var(--red); }
  .pad .stop.active { border-color:var(--red); background:linear-gradient(180deg, rgba(255,90,95,.35), rgba(255,90,95,.15));
                      box-shadow:0 0 0 3px rgba(255,90,95,.2), 0 0 24px rgba(255,90,95,.35), 0 4px 0 var(--pad-base); }
  .hint { margin-top:14px; text-align:center; color:var(--muted); font-size:12.5px; line-height:1.9; }
  kbd { font-family:var(--mono); font-size:11.5px; padding:2px 7px; border-radius:6px; background:var(--tint2);
        border:1px solid var(--edge2); border-bottom-width:2px; color:var(--text); }
  .tools { display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-top:16px; }
  .tools button { padding:11px 12px; border-radius:12px; font-size:13.5px; font-weight:600; border:1px solid var(--edge2);
                  background:var(--tint); display:flex; align-items:center; justify-content:center; gap:8px; }
  .tools button:hover { background:var(--hover); }
  #testBtn { grid-column:1 / -1; border-color:rgba(255,90,95,.4); color:var(--stop-text); background:rgba(255,90,95,.08); }
  #testBtn:hover { background:rgba(255,90,95,.16); }
  #soundBtn[data-on="1"] { border-color:var(--amber); color:var(--amber); background:rgba(245,168,59,.1); }

  /* ---------- log ---------- */
  #log { font-family:var(--mono); font-size:12.5px; max-height:372px; overflow-y:auto; margin:0 -6px; padding:0 6px; }
  #log div { display:grid; grid-template-columns:auto 1fr; gap:10px; padding:8px 0 8px 12px; border-left:2px solid var(--edge2);
             margin-left:2px; line-height:1.45; color:var(--log-text); }
  #log div + div { border-top:1px solid var(--edge); }
  #log span { color:var(--dim); }
  #log div.ev-alert { border-left-color:var(--red); color:var(--log-alert); }
  #log div.ev-reset { border-left-color:var(--teal); }
  #log div.ev-car { border-left-color:var(--blue); }
  #log div.ev-cal { border-left-color:var(--amber); }
  #log::-webkit-scrollbar { width:8px; } #log::-webkit-scrollbar-thumb { background:var(--hover); border-radius:8px; }
  button:focus-visible { outline:2px solid var(--amber); outline-offset:2px; }
  /* ---------- light / dark toggle ---------- */
  #themeBtn { flex:none; width:40px; height:40px; border-radius:50%; display:grid; place-items:center; cursor:pointer;
              background:var(--panel); border:1px solid var(--edge2); color:var(--text); padding:0;
              transition:background .2s, border-color .2s, transform .15s; }
  #themeBtn:hover { border-color:var(--amber); transform:rotate(-12deg); }
  #themeBtn svg { display:block; }
  #themeBtn .sun { color:var(--amber); }
  #themeBtn .moon { color:var(--blue); }
  :root[data-theme="dark"] #themeBtn .moon, :root[data-theme="light"] #themeBtn .sun { display:none; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div class="mark" aria-hidden="true">
      <svg width="28" height="28" viewBox="0 0 40 40"><path d="M3 20 Q20 6 37 20 Q20 34 3 20Z" fill="none" stroke="#E8ECF6" stroke-width="2.4" stroke-linejoin="round"/>
        <circle cx="20" cy="20" r="6.5" fill="#22C3A6"/><circle cx="20" cy="20" r="2.6" fill="#0F1A38"/>
        <g fill="#F5A83B"><circle cx="3" cy="20" r="2"/><circle cx="37" cy="20" r="2"/><circle cx="13" cy="13" r="2"/><circle cx="27" cy="13" r="2"/><circle cx="13" cy="27" r="2"/><circle cx="27" cy="27" r="2"/></g></svg>
    </div>
    <div>
      <h1>Drowsiness Alert System</h1>
      <div class="sub">Edge-AI driver monitoring · wireless vehicle slow-down</div>
    </div>
    <div class="spacer"></div>
    <div class="namebox">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true">
        <circle cx="12" cy="8" r="4"/><path d="M4 21c1.5-4 4.5-6 8-6s6.5 2 8 6"/></svg>
      <label for="driverName">Driver</label>
      <input id="driverName" maxlength="30" placeholder="Enter your name" autocomplete="off" spellcheck="false">
      <button id="saveName" type="button">Save</button>
      <button id="testVoice" type="button" title="Play the wake-up alert once">🔊 Test</button>
    </div>
    <span id="armHint">Click anywhere once to turn on the voice alert</span>
    <span id="clock">connecting…</span>
    <button id="themeBtn" type="button" title="Switch to light mode" aria-label="Switch to light mode">
      <svg class="sun" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true">
        <circle cx="12" cy="12" r="4.5"/><path d="M12 2v2.5M12 19.5V22M2 12h2.5M19.5 12H22M4.9 4.9l1.8 1.8M17.3 17.3l1.8 1.8M4.9 19.1l1.8-1.8M17.3 6.7l1.8-1.8"/></svg>
      <svg class="moon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <path d="M20.5 14.5A8.5 8.5 0 0 1 9.5 3.5a8.5 8.5 0 1 0 11 11z"/></svg>
    </button>
  </header>

  <main>
    <div class="s-banner">
      <div id="banner">
        <div class="ico" aria-hidden="true">
          <svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/></svg>
        </div>
        <div class="txt"><div id="bannerText">Waiting for data…</div><small id="bannerSub"></small></div>
        <button id="resetBtn">Reset – back to full speed<kbd>R</kbd></button>
      </div>
    </div>

    <section class="card s-cam">
      <h2>Driver camera<span class="grow"></span></h2>
      <div class="viewer">
        <img id="cam" src="/video.mjpg" alt="Live webcam view with face box and eye landmarks">
        <div class="live">LIVE</div>
      </div>
      <div class="legend">
        <span><i style="background:#22C35A"></i>Green box: awake</span>
        <span><i style="background:#FF4040"></i>Red box: eyes closed or drowsy</span>
        <span><i style="background:#FFA000"></i>Orange dots: eyelid landmarks</span>
      </div>
      <div id="camnote"></div>
    </section>

    <section class="card s-det">
      <h2>Detector<span class="grow"></span><span id="detOnline" class="pill off">offline</span></h2>
      <div class="hero">
        <div><span class="k">Driver state</span><b id="detState">–</b></div>
        <div class="ear"><span class="k">Eye openness (EAR)</span><b id="detEar">–</b></div>
      </div>
      <div class="meter">
        <div class="top"><span>Eyes closed for</span><b id="detClosedTxt">–</b></div>
        <div class="bar"><div id="detClosed"></div></div>
      </div>
      <div class="meter">
        <div class="top"><span>Eyes closed, last 60 s (PERCLOS)</span><b id="detPerclosTxt">–</b></div>
        <div class="bar"><div id="detPerclos"></div></div>
      </div>
      <div class="trace">
        <div class="top"><span class="k">Eye openness · last 60 s</span><i><b></b>closed threshold</i></div>
        <svg id="earTrace" viewBox="0 0 300 100" preserveAspectRatio="none" aria-label="Eye openness over the last 60 seconds">
          <defs><linearGradient id="earFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#6E8BFF" stop-opacity=".35"/><stop offset="1" stop-color="#6E8BFF" stop-opacity="0"/></linearGradient></defs>
          <path id="earArea" fill="url(#earFill)" d=""/>
          <polyline id="earLine" fill="none" stroke="#9AB0FF" stroke-width="2" vector-effect="non-scaling-stroke" stroke-linejoin="round" points=""/>
          <line id="earThr" x1="0" x2="300" y1="50" y2="50" stroke="#FF5A5F" stroke-width="1.5" stroke-dasharray="5 4" vector-effect="non-scaling-stroke"/>
        </svg>
      </div>
      <div class="mini">
        <div><span class="k">Yawns · 60 s</span><b id="detYawns">–</b></div>
        <div><span class="k">Frame rate</span><b id="detFps">–</b></div>
        <div><span class="k">Method</span><b id="detMethod">–</b></div>
      </div>
    </section>

    <section class="card s-car">
      <h2>Car<span class="grow"></span><span id="carOnline" class="pill off">offline</span></h2>
      <div class="gauge">
        <svg viewBox="0 0 220 128" aria-hidden="true">
          <defs><linearGradient id="speedGrad" x1="0" x2="1"><stop offset="0" stop-color="#22C3A6"/><stop offset="1" stop-color="#F5A83B"/></linearGradient></defs>
          <path class="track" d="M20 112 A90 90 0 0 1 200 112" pathLength="100"/>
          <path id="carSpeed" d="M20 112 A90 90 0 0 1 200 112" pathLength="100"/>
        </svg>
        <div class="read"><b id="carSpeedTxt">–</b><span>Motor PWM / 255</span></div>
      </div>
      <div class="rows">
        <div class="row"><span>Car state</span><b id="carState">–</b></div>
        <div class="row"><span>Drive command</span><b id="carDrive">–</b></div>
        <div class="row"><span>Wi‑Fi signal</span><b id="carRssi">–</b></div>
        <div class="row"><span>IP address</span><b id="carIp">–</b></div>
        <div class="row"><span>Alerts so far</span><b id="alertsTotal">0</b></div>
      </div>
    </section>

    <section class="card s-drive">
      <h2>Drive the car</h2>
      <div class="pad">
        <span></span><button data-cmd="forward"><span class="a">▲</span>Forward</button><span></span>
        <button data-cmd="left"><span class="a">◀</span>Left</button><button data-cmd="stop" class="stop"><span class="a">■</span>Stop</button><button data-cmd="right"><span class="a">▶</span>Right</button>
        <span></span><button data-cmd="backward"><span class="a">▼</span>Back</button><span></span>
      </div>
      <div class="hint"><kbd>↑</kbd> <kbd>↓</kbd> <kbd>←</kbd> <kbd>→</kbd> drive · <kbd>Space</kbd> stop · <kbd>R</kbd> reset alert</div>
      <div class="tools">
        <button id="testBtn">⚠ Simulate drowsiness</button>
        <button id="calBtn">◎ Recalibrate eyes</button>
        <button id="soundBtn">🔔 Enable alert sound</button>
      </div>
    </section>

    <section class="card s-log"><h2>Event log</h2><div id="log"></div></section>
  </main>
</div>
<script>
const $ = id => document.getElementById(id);
let soundOn = false, audioCtx = null;
let driverName = "", alertOn = false, voiceLooping = false, activated = false, voice = null, liveTones = [];
const sleep = ms => new Promise(r => setTimeout(r, ms));
const store = {
  get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch (e) {} },
};
function ensureAudio() {
  if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  if (audioCtx.state === "suspended") audioCtx.resume();
}
function tone(t0) {                       // one 880 Hz beep starting at audio time t0
  const o = audioCtx.createOscillator(), g = audioCtx.createGain();
  o.frequency.value = 880;
  g.gain.setValueAtTime(0.0001, t0);
  g.gain.exponentialRampToValueAtTime(0.22, t0 + 0.02);
  g.gain.setValueAtTime(0.22, t0 + 0.2);
  g.gain.exponentialRampToValueAtTime(0.0001, t0 + 0.25);
  o.connect(g); g.connect(audioCtx.destination);
  o.start(t0); o.stop(t0 + 0.26);
  liveTones.push(o); o.onended = () => { liveTones = liveTones.filter(x => x !== o); };
}
function beep() { if (!soundOn || !audioCtx) return; tone(audioCtx.currentTime); }
async function threeBeeps() {
  if (!audioCtx) return;
  const t = audioCtx.currentTime + 0.05;
  for (let i = 0; i < 3; i++) tone(t + i * 0.38);
  await sleep(3 * 380 + 120);
}
function pickVoice() {                    // prefer an Indian-English voice, then any English voice
  if (!("speechSynthesis" in window)) return;
  const vs = speechSynthesis.getVoices();
  voice = vs.find(v => /en[-_]IN/i.test(v.lang)) || vs.find(v => /^en/i.test(v.lang)) || null;
}
if ("speechSynthesis" in window) { pickVoice(); speechSynthesis.onvoiceschanged = pickVoice; }
function speak(text) {
  return new Promise(resolve => {
    if (!("speechSynthesis" in window)) return resolve();
    const u = new SpeechSynthesisUtterance(text);
    if (voice) { u.voice = voice; u.lang = voice.lang; } else { u.lang = "en-IN"; }
    u.rate = 0.95; u.pitch = 1; u.volume = 1;
    let done = false;
    const finish = () => { if (!done) { done = true; clearTimeout(timer); resolve(); } };
    const timer = setTimeout(finish, 6000);   // safety net if the browser never reports the end
    u.onend = finish; u.onerror = finish;
    speechSynthesis.speak(u);
  });
}
const wakeText = () => driverName ? `${driverName}, please wake up!` : "Please wake up!";
async function voiceLoop() {              // "<Name>, please wake up!" + 3 beeps, repeated until reset
  if (voiceLooping) return;
  voiceLooping = true;
  try {
    while (alertOn && soundOn) {
      await speak(wakeText());
      if (!(alertOn && soundOn)) break;
      await threeBeeps();
      if (!(alertOn && soundOn)) break;
      await sleep(600);
    }
  } finally { voiceLooping = false; }
}
function stopVoice() {
  if ("speechSynthesis" in window) speechSynthesis.cancel();
  liveTones.forEach(o => { try { o.stop(); } catch (e) {} });
  liveTones = [];
}
function setSound(on) {
  soundOn = on;
  store.set("alertSound", on ? "1" : "0");
  if (on) ensureAudio(); else stopVoice();
  $("soundBtn").textContent = on ? "🔕 Disable alert sound" : "🔔 Enable alert sound";
  $("soundBtn").dataset.on = on ? "1" : "0";
  updateArmHint();
  if (on && alertOn) voiceLoop();
}
function updateArmHint() {
  const active = activated || (navigator.userActivation && navigator.userActivation.hasBeenActive);
  $("armHint").classList.toggle("show", soundOn && !active);
}
function onFirstGesture() {               // browsers only allow sound after the user interacts with the page
  activated = true;
  if (soundOn) ensureAudio();
  updateArmHint();
  if (soundOn && alertOn) voiceLoop();
}
document.addEventListener("pointerdown", onFirstGesture, { once: true });
document.addEventListener("keydown", onFirstGesture, { once: true });
function saveName() {
  driverName = $("driverName").value.trim().slice(0, 30);
  $("driverName").value = driverName;
  store.set("driverName", driverName);
  if (!soundOn) setSound(true);            // saving a name switches the voice alert on
  const b = $("saveName");
  b.textContent = "Saved ✓"; b.classList.add("done");
  setTimeout(() => { b.textContent = "Save"; b.classList.remove("done"); }, 1500);
}
async function post(url, body) {
  try { await fetch(url, { method:"POST", headers:{ "Content-Type":"application/json" }, body:JSON.stringify(body || {}) }); }
  catch (e) { console.error(e); }
}
function pill(el, on) { el.textContent = on ? "online" : "offline"; el.className = "pill " + (on ? "on" : "off"); }
function bar(el, frac) {
  const p = Math.max(0, Math.min(1, frac || 0));
  if (el instanceof SVGElement) el.style.strokeDashoffset = 100 - p * 100;   // speed gauge arc
  else el.style.width = Math.round(p * 100) + "%";                           // linear meters
  el.classList.toggle("hot", frac >= 1);
}
function evClass(t) {
  if (/^ALERT .* raised/.test(t)) return "ev-alert";
  if (/reset|cleared/i.test(t)) return "ev-reset";
  if (/^Car |Drive command/.test(t)) return "ev-car";
  if (/Calibrat|Recalibrat|Webcam|Camera/.test(t)) return "ev-cal";
  return "";
}
const earHist = [];             // last 60 s of EAR samples (one per refresh, 400 ms)
function drawTrace(ear, thr) {
  earHist.push(ear); if (earHist.length > 150) earHist.shift();
  const Y = v => (100 - Math.min(1, Math.max(0, v / 0.45)) * 100).toFixed(1);
  const pts = earHist.map((v, i) => v == null ? null : [(i / 149 * 300).toFixed(1), Y(v)]);
  const ok = pts.filter(Boolean);
  $("earLine").setAttribute("points", ok.map(p => p.join(",")).join(" "));
  $("earArea").setAttribute("d", ok.length > 1 ? `M${ok[0][0]},100 L${ok.map(p => p.join(",")).join(" L")} L${ok[ok.length - 1][0]},100 Z` : "");
  if (thr != null) { const y = Y(thr); $("earThr").setAttribute("y1", y); $("earThr").setAttribute("y2", y); }
}
$("cam").addEventListener("error", () => setTimeout(() => { $("cam").src = "/video.mjpg?t=" + Date.now(); }, 2000));

async function refresh() {
  let s;
  try { s = await (await fetch("/api/status", { cache:"no-store" })).json(); }
  catch (e) { $("clock").textContent = "Not reachable – is laptop_app.py running?"; return; }
  $("clock").textContent = "Updated " + new Date().toLocaleTimeString();

  const wasAlert = alertOn;
  alertOn = !!s.alert.active;
  if (alertOn && soundOn) voiceLoop();
  if (wasAlert && !alertOn) stopVoice();

  const b = $("banner");
  if (s.alert.active) {
    b.className = "alert";
    $("bannerText").textContent = "DROWSINESS DETECTED – car slowed, buzzer on";
    $("bannerSub").textContent = `${driverName ? driverName + ", please wake up! · " : ""}${s.alert.seconds}s · click Reset (or press R) to go back to full speed`;
  } else if (s.cooldown_s > 0) {
    b.className = "cool"; $("bannerText").textContent = "Reset – back to full speed";
    $("bannerSub").textContent = `New alerts allowed in ${s.cooldown_s}s`;
  } else {
    b.className = ""; $("bannerText").textContent = "Driver alert – all normal";
    $("bannerSub").textContent = "Watching the driver’s eyes · car at full speed";
  }

  const d = s.detector, on = d.online;
  pill($("detOnline"), on);
  $("detMethod").textContent = d.method || "–";
  $("detState").textContent = on ? d.state + (d.reason ? " – " + d.reason : "") : (d.error ? "camera problem" : "–");
  $("detState").dataset.s = on ? d.state : "";
  $("detEar").textContent = on && d.ear != null
      ? `${d.ear.toFixed(2)} (closed below ${d.ear_threshold.toFixed(2)}${d.calibrated ? "" : ", calibrating"})` : "–";
  drawTrace(on && d.ear != null ? d.ear : null, d.ear_threshold);
  bar($("detClosed"), on ? d.closed_for / d.closed_alert_s : 0);
  $("detClosedTxt").textContent = on ? `${d.closed_for.toFixed(1)} s of ${d.closed_alert_s} s` : "–";
  bar($("detPerclos"), on && d.perclos_alert ? d.perclos / d.perclos_alert : 0);
  $("detPerclosTxt").textContent = on ? `${Math.round(d.perclos * 100)}% (alert at ${Math.round(d.perclos_alert * 100)}%)` : "–";
  $("detYawns").textContent = on ? `${d.yawns} (alert at ${d.yawn_alert_count})` : "–";
  $("detFps").textContent = on ? d.fps.toFixed(1) + " fps" : "–";
  $("camnote").textContent = d.error ? "Camera: " + d.error : "";

  const c = s.car;
  pill($("carOnline"), c.online);
  $("carState").textContent = c.online ? (c.state || "–") : "–";
  $("carState").style.color = c.online && c.state === "ALERT" ? "var(--red)" : (c.online ? "var(--teal)" : "");
  $("carDrive").textContent = s.drive.toUpperCase() + (c.online && c.link_ok === false ? " (link lost)" : "");
  const spd = c.online && c.speed !== undefined ? c.speed : 0;
  bar($("carSpeed"), spd / 255);
  $("carSpeed").classList.remove("hot");
  $("carSpeedTxt").textContent = c.online ? `${spd}` : "–";
  $("carRssi").textContent = c.online && c.rssi !== undefined ? c.rssi + " dBm" : "–";
  $("carIp").textContent = c.ip || "–";
  $("alertsTotal").textContent = s.alerts_total;
  document.querySelectorAll(".pad button").forEach(x => x.classList.toggle("active", x.dataset.cmd === s.drive));
  $("log").innerHTML = s.events.map(e => `<div class="${evClass(e.text)}"><span>${e.t}</span>${e.text.replace(/</g, "&lt;")}</div>`).join("") || "<div>No events yet.</div>";
}
document.querySelectorAll(".pad button").forEach(x => x.addEventListener("click", () => post("/api/drive", { cmd:x.dataset.cmd })));
function resetAlert() { alertOn = false; stopVoice(); post("/api/reset"); }
document.addEventListener("keydown", e => {
  if (e.target.closest && e.target.closest("input, textarea")) return;   // typing a name, not driving
  if (e.key === "r" || e.key === "R") { resetAlert(); return; }
  const m = { ArrowUp:"forward", ArrowDown:"backward", ArrowLeft:"left", ArrowRight:"right", " ":"stop" };
  if (m[e.key]) { e.preventDefault(); post("/api/drive", { cmd:m[e.key] }); }
});
$("testBtn").addEventListener("click", () => post("/api/test/drowsy"));
$("resetBtn").addEventListener("click", resetAlert);
$("calBtn").addEventListener("click", () => post("/api/recalibrate"));
$("soundBtn").addEventListener("click", () => { setSound(!soundOn); if (soundOn) beep(); });
function applyTheme(t, save) {
  document.documentElement.dataset.theme = t;
  const next = t === "dark" ? "light" : "dark";
  $("themeBtn").title = "Switch to " + next + " mode";
  $("themeBtn").setAttribute("aria-label", "Switch to " + next + " mode");
  if (save) store.set("theme", t);
}
applyTheme(document.documentElement.dataset.theme === "light" ? "light" : "dark", false);
$("themeBtn").addEventListener("click", e => {
  applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark", true);
  e.currentTarget.blur();          // so Space / R keep controlling the car, not this button
});
$("saveName").addEventListener("click", saveName);
$("driverName").addEventListener("keydown", e => { if (e.key === "Enter") { saveName(); e.target.blur(); } });
$("driverName").addEventListener("change", saveName);
$("testVoice").addEventListener("click", async () => {
  ensureAudio(); stopVoice();
  await speak(wakeText()); await threeBeeps();
});
driverName = (store.get("driverName") || "").slice(0, 30);
$("driverName").value = driverName;
if (store.get("alertSound") === "1") setSound(true);
refresh(); setInterval(refresh, 400);
</script>
</body>
</html>
"""


def local_ips():
    ips = set()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ips.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except OSError:
        pass
    return sorted(ip for ip in ips if not ip.startswith("127."))


def check_mediapipe(camera_index):
    """Step-by-step MediaPipe self-test. Prints exactly what fails and how to fix it."""
    import platform
    print("=" * 66)
    print(" MediaPipe check")
    print(f" Python  : {sys.version.split()[0]} ({platform.system()} {platform.machine()})")
    print(f" Program : {sys.executable}")
    print("=" * 66)
    pip_fix = f'"{sys.executable}" -m pip install --upgrade mediapipe'

    print("1. import mediapipe ...", end=" ", flush=True)
    try:
        import mediapipe as mp
        print(f"OK, version {getattr(mp, '__version__', '?')}")
    except Exception as e:
        print(f"FAILED\n   {e.__class__.__name__}: {e}")
        print(f"   Fix: install it into THIS Python (the one above):\n        {pip_fix}")
        return 1

    print("2. Tasks API (FaceLandmarker) ...", end=" ", flush=True)
    try:
        _BaseOptions, vision = _mediapipe_api(mp)
        print(f"OK ({vision.FaceLandmarker.__name__}, modes {[m.name for m in vision.RunningMode]})")
    except Exception as e:
        print(f"FAILED\n   {e.__class__.__name__}: {e}\n   Fix: {pip_fix}")
        return 1

    print("3. face model file ...", end=" ", flush=True)
    try:
        path = ensure_model()
        print(f"OK, {path} ({os.path.getsize(path) // 1024} KB)")
    except Exception as e:
        print(f"FAILED\n   {e}")
        return 1

    print("4. start the face landmarker + test run ...", end=" ", flush=True)
    try:
        analyzer = MediaPipeAnalyzer()
        print("OK")
    except Exception as e:
        print(f"FAILED\n   {e.__class__.__name__}: {e}")
        print("   Fix: delete face_landmarker.task next to laptop_app.py and run again; if it still fails,")
        print(f"        try the previous MediaPipe line:  \"{sys.executable}\" -m pip install \"mediapipe==0.10.21\"")
        return 1

    print(f"5. your webcam {camera_index} (look at it for 3 s) ...", end=" ", flush=True)
    cap = open_camera(camera_index)
    if cap is None:
        print("SKIPPED (webcam busy or not found: close laptop_app.py / other camera apps first)")
    else:
        faces, ears, t_end = 0, [], time.time() + 3
        while time.time() < t_end:
            ok, frame = cap.read()
            if not ok:
                continue
            obs = analyzer.process(cv2.flip(frame, 1))
            if obs["face"]:
                faces += 1
                ears.append(obs["ear"])
        cap.release()
        if faces:
            print(f"OK, face found in {faces} frames, eye openness (EAR) about {sum(ears) / len(ears):.2f}")
        else:
            print("no face found: check lighting and sit in front of the camera (MediaPipe itself works)")
    print("=" * 66)
    print(" RESULT: MediaPipe works. Start laptop_app.py normally; it will say 'MediaPipe ... loaded'.")
    print("=" * 66)
    return 0


def main():
    ap = argparse.ArgumentParser(description="Drowsiness alert: webcam detector + server + dashboard")
    ap.add_argument("--camera", type=int, default=CAMERA_INDEX, help="webcam index (0 = built-in)")
    ap.add_argument("--haar", action="store_true", help="use the old Haar method instead of MediaPipe")
    ap.add_argument("--no-camera", action="store_true", help="run without the webcam (test the car only)")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--check-mediapipe", action="store_true", help="test MediaPipe step by step and exit")
    args = ap.parse_args()
    if args.check_mediapipe:
        sys.exit(check_mediapipe(args.camera))

    print("=" * 66)
    print(" Drowsiness alert - laptop app (v5.2: MediaPipe + EAR, voice alert, light/dark dashboard)")
    print(f" Dashboard           : http://localhost:{args.port}")
    for ip in local_ips():
        print(f" SERVER_URL for ESP32: http://{ip}:{args.port}")
    print("=" * 66, flush=True)

    detector = None if args.no_camera else Detector(args.camera, force_haar=args.haar)
    with lock:
        log_event("Laptop app started")
        if detector and detector.mp_problem:
            log_event("MediaPipe not used, Haar fallback: " + detector.mp_problem.splitlines()[0][:160]
                      + " (run: python laptop_app.py --check-mediapipe)")
    if detector:
        detector.start()
    app.run(host="0.0.0.0", port=args.port, threaded=True, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()