from pathlib import Path
import time

import cv2
import joblib
import numpy as np

from detection.pupil import crop_to_aspect_ratio, process_frame


# ---------------- CONFIG ----------------
EYE_CAM_ID = 0
FRONT_CAM_ID = 2
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATHS = (
    SCRIPT_DIR / "gaze_model.joblib",
    SCRIPT_DIR / "python" / "gaze_model.joblib",
)
WINDOW_SECONDS = 25
BLUR_SIGMA_PX = 30
MAX_DT = 0.2
CAPTURE_WIDTH = 640
CAPTURE_HEIGHT = 480
CAPTURE_FPS = 30
OUT_PREFIX = "gaze_heatmap"
# ----------------------------------------


def open_camera(camera_id, name):
    """Open a camera using Windows DirectShow."""
    capture = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(
            f"Could not open {name} camera at index {camera_id}. "
            "Check the camera index, Windows camera permissions, and other "
            "apps that may be using the camera."
        )

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    capture.set(cv2.CAP_PROP_FPS, CAPTURE_FPS)
    backend = capture.getBackendName()
    print(f"{name} camera: index {camera_id}, backend {backend}")
    return capture


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
    images = (heat, overlay, frame)
    for path, image in zip(paths, images):
        if not cv2.imwrite(path, image):
            raise OSError(f"Could not save image: {path}")
    seconds_path = f"{OUT_PREFIX}_{stamp}_seconds.npy"
    np.save(seconds_path, blurred)
    print(f"Saved heatmap files with prefix {OUT_PREFIX}_{stamp}")
    return overlay, blurred


def main():
    model_path = next((path for path in MODEL_PATHS if path.is_file()), None)
    if model_path is None:
        checked_paths = ", ".join(str(path) for path in MODEL_PATHS)
        raise FileNotFoundError(
            f"Could not find gaze_model.joblib. Checked: {checked_paths}"
        )
    model = joblib.load(model_path)
    eye_cap = None
    front_cap = None

    try:
        front_cap = open_camera(FRONT_CAM_ID, "Frontal")
        eye_cap = open_camera(EYE_CAM_ID, "Eye")

        # Crop before using the frame dimensions so the gaze grid matches the
        # 640x480 coordinates expected by the calibration model.
        eye = crop_to_aspect_ratio(read_frame(eye_cap, "Eye"))
        front = crop_to_aspect_ratio(read_frame(front_cap, "Frontal"))
        height, width = front.shape[:2]

        dwell = np.zeros((height, width), dtype=np.float32)
        recording = False
        start_time = 0.0
        previous_time = 0.0
        valid_time = 0.0
        last_front = front

        print(
            f"Press 'g' to start a {WINDOW_SECONDS}s recording; "
            "press 'q' to quit."
        )
        print("Check the 'eye' and 'frontal' previews to confirm camera mapping.")

        while True:
            eye = crop_to_aspect_ratio(read_frame(eye_cap, "Eye"))
            front = crop_to_aspect_ratio(read_frame(front_cap, "Frontal"))
            last_front = front
            display = front.copy()
            now = time.perf_counter()

            if recording:
                elapsed = now - start_time
                dt = min(now - previous_time, MAX_DT)
                previous_time = now

                pupil = detect_pupil(eye)
                if pupil is not None:
                    gx, gy = predict_gaze(model, pupil)
                    if np.isfinite(gx) and np.isfinite(gy):
                        xi, yi = int(round(gx)), int(round(gy))
                        if 0 <= xi < width and 0 <= yi < height:
                            dwell[yi, xi] += dt
                            valid_time += dt
                            cv2.circle(display, (xi, yi), 8, (0, 255, 0), 2)

                remaining = max(0.0, WINDOW_SECONDS - elapsed)
                cv2.putText(
                    display,
                    f"REC {remaining:4.1f}s",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )

                if elapsed >= WINDOW_SECONDS:
                    recording = False
                    overlay, blurred = save_heatmap(dwell, last_front)
                    print(
                        f"Valid gaze time: {valid_time:.1f}s of "
                        f"{WINDOW_SECONDS}s "
                        f"(peak blurred dwell {blurred.max():.2f}s/px)"
                    )
                    cv2.imshow("heatmap", overlay)

            cv2.imshow("frontal", display)
            cv2.imshow("eye", eye)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord("g") and not recording:
                dwell.fill(0)
                valid_time = 0.0
                start_time = previous_time = time.perf_counter()
                recording = True
                print("Recording started...")
    finally:
        if eye_cap is not None:
            eye_cap.release()
        if front_cap is not None:
            front_cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
