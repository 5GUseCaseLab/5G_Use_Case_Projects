import os
import time
from pathlib import Path

import cv2
import joblib
import numpy as np

from pupil import crop_to_aspect_ratio, process_frame

try:  # only available inside Arduino App Lab on the UNO Q
    from arduino.app_utils import App, Bridge
except ImportError:
    App = Bridge = None


# ---------------- CONFIG ----------------
# must be set BEFORE importing unoq_service, which loads the model on import
print(str(Path(__file__).parent / "tier0.onnx"))
os.environ.setdefault("TIER0_MODEL", str(Path(__file__).parent / "tier0.onnx"))
os.environ.setdefault("TIER1_URL", "https://sim-landing-illinois-penny.trycloudflare.com")   # later
os.environ.setdefault("DASH_TOKEN", "5g-usecase-best")

from server import store, run_in_thread
os.environ.setdefault("EYE_CAM", "2")
os.environ.setdefault("FRONT_CAM", "0")
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATHS = (
    SCRIPT_DIR / "gaze_model.joblib",
    SCRIPT_DIR / "python" / "gaze_model.joblib",
)
WINDOW_SECONDS = 22
BLUR_SIGMA_PX = 25
MAX_DT = 0.2
CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 480
CAPTURE_FPS = 15
OUT_PREFIX = "gaze_heatmap"

SHOW_WINDOWS = os.getenv("SHOW_WINDOWS", "0") == "1"   # cv2 previews (off on headless UNO Q)
PUBLISH_EVERY = 2                                      # push every Nth frame to the dashboard
SERVER_PORT = int(os.getenv("DASH_PORT", "8000"))
TRIGGER_FN = "read_trigger"                            # must match Bridge.provide_safe name in sketch
# ----------------------------------------

CAM_PROFILES = {
    # eye: full size for pupil detection; MJPG is compressed so it is light on the bus
    "eye":   {"src": os.getenv("EYE_CAM", "2"),   "formats": ("MJPG", "YUYV", None),
              "size": (640, 480), "fps": 15},
    # front: YUYV only, so shrink it; it is upscaled to 640x480 by crop_to_aspect_ratio
    "front": {"src": os.getenv("FRONT_CAM", "0"), "formats": ("MJPG", "YUYV", None),
              "size": (160, 120), "fps": 15},
}


def open_camera(profile, name):
    """Open a V4L2 camera with per-camera format/size/fps and confirm a real frame."""
    source = profile["src"]
    source = int(source) if str(source).isdigit() else source
    w, h = profile["size"]

    for fourcc in profile["formats"]:
        cap = cv2.VideoCapture(source, cv2.CAP_V4L2)          # bypass GStreamer
        if not cap.isOpened():
            cap.release()
            time.sleep(0.3)
            continue
        if fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS, profile["fps"])             # must be set before the first read
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        ok, frame = cap.read()                                # first read starts streaming
        if ok and frame is not None and frame.size:
            print(f"{name} camera: {source!r}, format {fourcc or 'default'}, "
                  f"{frame.shape[1]}x{frame.shape[0]} @ {cap.get(cv2.CAP_PROP_FPS):.0f} fps")
            return cap
        cap.release()
        time.sleep(0.5)

    raise RuntimeError(
        f"Could not get frames from {name} camera at {source!r}. "
        "If dmesg shows 'Not enough bandwidth', lower that camera's size or fps in CAM_PROFILES."
    )

    
def read_frame(capture, name):
    ok, frame = capture.read()
    if not ok or frame is None or frame.size == 0:
        raise RuntimeError(
            f"{name} camera opened but did not return a frame. "
            "Check that another app is not using it and that both cameras "
            "can stream simultaneously."
        )
    return frame


def detect_pupil(eye_frame):
    """Return pupil center coordinates, or None when detection fails."""
    rotated_rect = process_frame(eye_frame)
    if rotated_rect is None or len(rotated_rect) == 0:
        return None

    center = rotated_rect[0]
    if center is None or len(center) < 2:
        return None
    center_x, center_y = map(int, center[:2])
    return center_x, center_y


def predict_gaze(model, pupil_xy):
    """Map pupil coordinates to frontal-camera pixel coordinates."""
    features = np.asarray(pupil_xy, dtype=np.float64).reshape(1, -1)
    fx, fy = model.predict(features)[0]
    return float(fx), float(fy)


def make_heatmap(dwell, frame_bgr, sigma):
    """Return the heatmap, its overlay, and the blurred dwell-time grid."""
    blurred = cv2.GaussianBlur(dwell, (0, 0), sigma)
    peak = blurred.max()
    norm = blurred / peak if peak > 0 else blurred
    heat_u8 = np.clip(norm * 255, 0, 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)

    alpha = (np.clip(norm * 1.5, 0, 1) * 0.7)[..., None]
    overlay = (frame_bgr * (1 - alpha) + heat_color * alpha).astype(np.uint8)
    return heat_color, overlay, blurred


def save_heatmap(dwell, frame):
    heat, overlay, blurred = make_heatmap(dwell, frame, BLUR_SIGMA_PX)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    paths = (
        f"{OUT_PREFIX}_{stamp}_heat.png",
        f"{OUT_PREFIX}_{stamp}_overlay.png",
        f"{OUT_PREFIX}_{stamp}_frame.png",
    )
    for path, image in zip(paths, (heat, overlay, frame)):
        if not cv2.imwrite(path, image):
            raise OSError(f"Could not save image: {path}")
    np.save(f"{OUT_PREFIX}_{stamp}_seconds.npy", blurred)
    print(f"Saved heatmap files with prefix {OUT_PREFIX}_{stamp}")
    return heat, overlay, blurred


class PinTrigger:
    """Polls the sketch's latched button over the Router Bridge."""

    MAX_FAILURES = 5

    def __init__(self):
        self.failures = 0

    @property
    def available(self):
        return Bridge is not None and self.failures < self.MAX_FAILURES

    def pressed(self):
        if not self.available:
            return False
        try:
            value = bool(int(Bridge.call(TRIGGER_FN)))
        except Exception as exc:
            self.failures += 1
            print(f"Bridge trigger read failed ({self.failures}/{self.MAX_FAILURES}): {exc}")
            if self.failures >= self.MAX_FAILURES:
                print("Pin trigger disabled; dashboard/keyboard triggers still work.")
            return False
        self.failures = 0
        return value


class GazeRecorder:
    def __init__(self):
        model_path = next((p for p in MODEL_PATHS if p.is_file()), None)
        if model_path is None:
            checked = ", ".join(str(p) for p in MODEL_PATHS)
            raise FileNotFoundError(f"Could not find gaze_model.joblib. Checked: {checked}")
        self.model = joblib.load(model_path)

        self.eye_cap = open_camera(CAM_PROFILES["eye"], "Eye")
        time.sleep(0.5)
        self.front_cap = open_camera(CAM_PROFILES["front"], "Frontal")

        # Crop before using the frame dimensions so the gaze grid matches the
        # 640x480 coordinates expected by the calibration model.
        front = crop_to_aspect_ratio(read_frame(self.front_cap, "Frontal"))
        self.height, self.width = front.shape[:2]

        self.dwell = np.zeros((self.height, self.width), dtype=np.float32)
        self.recording = False
        self.start_time = 0.0
        self.previous_time = 0.0
        self.valid_time = 0.0
        self.last_gaze = None
        self.last_pupil = None
        self.frame_count = 0
        self.pin = PinTrigger()

        run_in_thread(port=SERVER_PORT)
        print(f"Dashboard on port {SERVER_PORT}")
        print(f"Trigger: Arduino pin ({'ready' if self.pin.available else 'unavailable'}), "
              f"dashboard button{', keyboard g' if SHOW_WINDOWS else ''}.")

    # ---- recording control ----
    def start_recording(self):
        if self.recording:
            return
        self.dwell.fill(0)
        self.valid_time = 0.0
        self.start_time = self.previous_time = time.perf_counter()
        self.recording = True
        print("Recording started...")

    def finish_recording(self, front):
        self.recording = False
        heat, overlay, blurred = save_heatmap(self.dwell, front)
        store.publish("dwell", heat, quality=85)
        print(f"Valid gaze time: {self.valid_time:.1f}s of {WINDOW_SECONDS}s "
              f"(peak blurred dwell {blurred.max():.2f}s/px)")
        if SHOW_WINDOWS:
            cv2.imshow("heatmap", overlay)

    # ---- one iteration of the main loop ----
    def step(self):
        eye = crop_to_aspect_ratio(read_frame(self.eye_cap, "Eye"))
        front = crop_to_aspect_ratio(read_frame(self.front_cap, "Frontal"))
        now = time.perf_counter()
        store.raw_front = front

        # Triggers: Arduino pin OR dashboard button (keyboard handled below)
        pin_pressed = self.pin.pressed()
        dash_pressed = store.pop_trigger()
        if (pin_pressed or dash_pressed) and not self.recording:
            self.start_recording()

        display = front.copy()
        remaining = 0.0
        self.last_pupil = self.last_gaze = None

        if self.recording:
            elapsed = now - self.start_time
            dt = min(now - self.previous_time, MAX_DT)
            self.previous_time = now

            pupil = detect_pupil(eye)
            if pupil is not None:
                self.last_pupil = pupil
                gx, gy = predict_gaze(self.model, pupil)
                if np.isfinite(gx) and np.isfinite(gy):
                    xi, yi = int(round(gx)), int(round(gy))
                    if 0 <= xi < self.width and 0 <= yi < self.height:
                        self.dwell[yi, xi] += dt
                        self.valid_time += dt
                        self.last_gaze = (xi, yi)
                        cv2.circle(display, (xi, yi), 8, (0, 255, 0), 2)

            remaining = max(0.0, WINDOW_SECONDS - elapsed)
            cv2.putText(display, f"REC {remaining:4.1f}s", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            if elapsed >= WINDOW_SECONDS:
                self.finish_recording(front)

        # ---- publish to dashboard ----
        store.set_status(
            state="recording" if self.recording else "idle",
            remaining=remaining,
            window=WINDOW_SECONDS,
            valid_time=round(self.valid_time, 2),
            pupil=self.last_pupil,
            gaze=self.last_gaze,
            front_size=(self.width, self.height),
        )
        self.frame_count += 1
        if self.frame_count % PUBLISH_EVERY == 0 and store.has_viewers():
            store.publish("eye", eye)
            store.publish("front", front)     # raw POV; dashboard draws the gaze dot itself

        # ---- optional local previews / keyboard ----
        if SHOW_WINDOWS:
            cv2.imshow("frontal", display)
            cv2.imshow("eye", eye)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                return False
            if key == ord("g"):
                self.start_recording()
        return True

    def close(self):
        for cap in (self.eye_cap, self.front_cap):
            if cap is not None:
                cap.release()
        if SHOW_WINDOWS:
            cv2.destroyAllWindows()


def main():
    recorder = GazeRecorder()
    try:
        if App is not None:
            App.run(user_loop=recorder.step)       # UNO Q / App Lab: Bridge is active
        else:
            while recorder.step() is not False:    # desktop fallback (no pin trigger)
                pass
    finally:
        recorder.close()


if __name__ == "__main__":
    main()