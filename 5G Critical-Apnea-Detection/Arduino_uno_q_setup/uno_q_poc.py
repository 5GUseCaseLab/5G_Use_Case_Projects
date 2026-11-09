import os
import sys
import cv2
import time
import csv
import json
import socket
import msgpack
import requests
import torch
import threading
from collections import deque
import numpy as np
from scipy.signal import butter, filtfilt, welch
from flask import Flask, Response

# 1. MODEL DEFINITION
from neural_methods.model.PhysNet import PhysNet_padding_Encoder_Decoder_MAX

# ---------------- CONFIGURATION ----------------
VIDEO_SOURCE = "http://192.168.0.139:5000/video" 

# 5G MEC ThingsBoard Server Endpoint (Exp 4)
THINGSBOARD_HOST = "192.168.116.30"
THINGSBOARD_PORT = "9090"
THINGSBOARD_ACCESS_TOKEN = "xe2ryTbKLnBn9k5k09E0" 

THINGSBOARD_URL = f"http://{THINGSBOARD_HOST}:{THINGSBOARD_PORT}/api/v1/{THINGSBOARD_ACCESS_TOKEN}/telemetry"

FRAME_CHUNK = 64
CALIBRATION_SEC = 5.0  

# EXTREME SENSITIVITY APNEA THRESHOLDS
APNEA_ENTER_RATIO = 0.75  
APNEA_EXIT_RATIO  = 0.85  
HARD_APNEA_CEILING = 0.030 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PHYSNET_CHECKPOINT = os.path.join(BASE_DIR, "final_model_release", "UBFC-rPPG_PhysNet_DiffNormalized.pth")
CASCADE_PATH = os.path.join(BASE_DIR, "dataset", "haarcascade_frontalface_default.xml")
LOG_CSV_PATH = os.path.join(BASE_DIR, "apnea_uno_q_5g_log.csv")
# -----------------------------------------------

# --- BACKGROUND ROUTER SOCKET THREAD (ZERO VIDEO LAG) ---
ROUTER_SOCK = "/var/run/arduino-router.sock"
msg_id = 0

def bridge_call(method, *args):
    global msg_id
    msg_id += 1
    packet = msgpack.packb([0, msg_id, method, list(args)])
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(0.15)
            s.connect(ROUTER_SOCK)
            s.sendall(packet)
            try:
                s.recv(1024)
            except Exception:
                pass
    except Exception:
        pass

BUZZER = 4
LED_CHANNELS = [2, 3, 5, 6, 7, 8, 9, 13]
RED, GREEN, BLUE = 10, 11, 12

def set_buzzer(on):
    bridge_call("set_pin", BUZZER, 1 if on else 0)

def set_channel(pin, on):
    bridge_call("set_pin", pin, 0 if on else 1)

def set_colors(r, g, b):
    bridge_call("set_pin", RED, 0 if r else 1)
    bridge_call("set_pin", GREEN, 0 if g else 1)
    bridge_call("set_pin", BLUE, 0 if b else 1)

def all_off():
    set_buzzer(False)
    for ch in LED_CHANNELS:
        set_channel(ch, False)
    set_colors(False, False, False)

current_apnea_flag = 0
hardware_running = True

def hardware_alert_worker():
    alerts_active = False
    strobe_toggle = False
    all_off()
    while hardware_running:
        if current_apnea_flag == 2:
            if not alerts_active:
                set_buzzer(True)
                for ch in LED_CHANNELS:
                    set_channel(ch, True)
                alerts_active = True
            if strobe_toggle:
                set_colors(True, False, False)
            else:
                set_colors(False, False, True)
            strobe_toggle = not strobe_toggle
            time.sleep(0.25)
        else:
            if alerts_active:
                all_off()
                alerts_active = False
            time.sleep(0.05)

threading.Thread(target=hardware_alert_worker, daemon=True).start()
# --------------------------------------------------------

# --- ASYNCHRONOUS 5G THINGSBOARD PUBLISHER ---
telemetry_queue = deque(maxlen=10)

def thingsboard_publisher_thread():
    while True:
        if len(telemetry_queue) > 0:
            payload = telemetry_queue.popleft()
            try:
                headers = {'Content-Type': 'application/json'}
                requests.post(THINGSBOARD_URL, data=json.dumps(payload), headers=headers, timeout=1.0)
            except Exception:
                pass
        else:
            time.sleep(0.05)

threading.Thread(target=thingsboard_publisher_thread, daemon=True).start()

# --- ZERO-LAG CAMERA GRABBER ---
class FreshFrameGrabber(threading.Thread):
    def __init__(self, src):
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(src)
        self.ret = False
        self.frame = None
        self.running = True

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.ret = ret
                self.frame = frame

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.running = False
        self.cap.release()

camera_stream = FreshFrameGrabber(VIDEO_SOURCE)
camera_stream.start()
time.sleep(1.0)

# --- WEB DASHBOARD STREAMER (PORT 8080) ---
dashboard_app = Flask(__name__)
output_frame = None
frame_lock = threading.Lock()

@dashboard_app.route('/')
def index():
    return """
    <html>
        <head><title>5G Clinical Edge Monitor</title><meta name="viewport" content="width=device-width, initial-scale=1"></head>
        <body style="background:#0a0a0a; color:#fff; text-align:center; font-family:monospace; margin:0; padding:10px;">
            <h2 style="color:#00ff88; margin:5px;">ARDUINO UNO Q - 5G HEALTH MONITOR</h2>
            <p style="color:#888;">Telemetry Streaming to 5G MEC: 192.168.116.30:9090</p>
            <img src="/video_feed" style="border: 2px solid #333; border-radius: 8px; width: 95%; max-width: 900px;">
        </body>
    </html>
    """

def generate_stream():
    global output_frame, frame_lock
    while True:
        with frame_lock:
            if output_frame is None:
                time.sleep(0.02)
                continue
            ret, encoded_img = cv2.imencode('.jpg', output_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
            if not ret: continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + bytearray(encoded_img) + b'\r\n')
        time.sleep(0.03)

@dashboard_app.route('/video_feed')
def video_feed():
    return Response(generate_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

threading.Thread(target=lambda: dashboard_app.run(host='0.0.0.0', port=8080, debug=False, use_reloader=False, threaded=True), daemon=True).start()

device = torch.device("cpu")
print("\n" + "="*60)
print(f"[INFO] Initializing PhysNet on: {device}")

model = PhysNet_padding_Encoder_Decoder_MAX(frames=FRAME_CHUNK).to(device)
if os.path.exists(PHYSNET_CHECKPOINT):
    ckpt = torch.load(PHYSNET_CHECKPOINT, map_location=device, weights_only=False)
    raw = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    clean = {k.replace("module.", "").replace("model.", ""): v for k, v in raw.items()}
    model.load_state_dict(clean, strict=False)
    print(f"[INFO] Model Loaded: {os.path.basename(PHYSNET_CHECKPOINT)}")
else:
    print(f"[ERROR] Checkpoint not found: {PHYSNET_CHECKPOINT}")
    sys.exit(1)
model.eval()

# BACKGROUND AI WORKER
ai_input_queue = deque(maxlen=1)
pulse_stream = deque(maxlen=300)

def ai_worker():
    while True:
        if len(ai_input_queue) > 0:
            process_chunk = ai_input_queue.popleft()
            resized = [cv2.resize(f, (32, 32)) for f in process_chunk]
            rgb = np.array([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in resized], dtype=np.float32)
            denom = rgb[1:] + rgb[:-1] + 1e-6
            diff_norm = (rgb[1:] - rgb[:-1]) / denom
            tensor = torch.tensor(diff_norm, dtype=torch.float32).permute(3, 0, 1, 2).unsqueeze(0).to(device)

            with torch.no_grad():
                out = model(tensor)
                out_bvp = out[0] if isinstance(out, tuple) else out
                pulse_vals = out_bvp.squeeze().cpu().numpy()
                for val in pulse_vals[-3:]: pulse_stream.append(float(val))
        else:
            time.sleep(0.01)

threading.Thread(target=ai_worker, daemon=True).start()

csv_file = open(LOG_CSV_PATH, mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["time", "hr_bpm", "rr_brpm", "energy", "baseline", "apnea_status", "apnea_flag"])
csv_file.flush()

# SIGNAL PROCESSING
def butter_bandpass(data, lowcut, highcut, fs=30.0, order=2):
    if len(data) < 21: return data 
    nyq = 0.5 * fs
    low = max(0.005, min(0.99, lowcut / nyq))
    high = min(0.99, max(low + 0.01, highcut / nyq))
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def calculate_welch_freq(signal, low_hz, high_hz, fs=30.0, nfft=2048):
    if len(signal) == 0: return 0.0
    f, psd = welch(signal, fs=fs, nperseg=len(signal), nfft=nfft)
    mask = (f >= low_hz) & (f <= high_hz)
    f_valid, psd_valid = f[mask], psd[mask]
    if len(f_valid) == 0 or np.sum(psd_valid) == 0: return 0.0
    peak_idx = np.argmax(psd_valid)
    return round(float(f_valid[peak_idx] * 60.0), 1)

def physiological_hr_smooth(history, new_val):
    if new_val <= 45.0 or new_val >= 180.0: return history[-1] if history else 72.0
    if len(history) > 0 and abs(new_val - history[-1]) > 12.0:
        new_val = history[-1] + np.sign(new_val - history[-1]) * 12.0
    history.append(new_val)
    if len(history) > 7: history.pop(0)
    return float(np.median(history))

# OPENCV CASCADE
if hasattr(cv2, 'CascadeClassifier'):
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
else:
    import cv2.objdetect
    face_cascade = cv2.objdetect.CascadeClassifier(CASCADE_PATH)

face_crop_buffer = deque(maxlen=FRAME_CHUNK + 1)
facial_green_stream = deque(maxlen=300)
thoracic_motion_stream = deque(maxlen=300)
hr_history = deque(maxlen=7)

prev_chest_small = None
smoothed_box = None
face_miss_counter = 0

apnea_start_frame = None
in_apnea_state = False
recovery_frame_counter = 0
calibration_samples = []
baseline_energy = None
calibration_done = False

frame_counter = 0
last_log_frame = 0
last_hr = 72.0
start_time = time.time()
FPS = 30.0

print(f"[INFO] 5G Telemetry destination: {THINGSBOARD_URL}")

try:
    while True:
        ret, frame = camera_stream.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        frame_counter += 1
        video_time = time.time() - start_time

        h_img, w_img, _ = frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. ROBUST FACE DETECTION (Runs every 3rd frame to prevent CPU lag)
        if frame_counter % 3 == 0 or smoothed_box is None:
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(60, 60))

            if len(faces) > 0:
                face_miss_counter = 0
                primary_face = max(faces, key=lambda b: b[2] * b[3])
                fx, fy, fw, fh = primary_face

                if smoothed_box is None:
                    smoothed_box = np.array([fx, fy, fw, fh], dtype=np.float32)
                else:
                    smoothed_box = 0.15 * np.array([fx, fy, fw, fh], dtype=np.float32) + 0.85 * smoothed_box
            else:
                face_miss_counter += 1
                if face_miss_counter > 10:
                    smoothed_box = None

        # 2. DRAW TARGETING BOXES IF FACE IS LOCKED
        if smoothed_box is not None:
            sfx, sfy, sfw, sfh = smoothed_box.astype(int)
            sfx = max(0, min(w_img - 10, sfx))
            sfy = max(0, min(h_img - 10, sfy))
            sfw = min(w_img - sfx, sfw)
            sfh = min(h_img - sfy, sfh)

            face_crop = frame[sfy:sfy+sfh, sfx:sfx+sfw]
            if face_crop.shape[0] > 0 and face_crop.shape[1] > 0:
                face_crop_buffer.append(face_crop)
                facial_green_stream.append(np.mean(face_crop[:, :, 1]))

            if len(face_crop_buffer) == (FRAME_CHUNK + 1) and (frame_counter % 3 == 0):
                ai_input_queue.append(list(face_crop_buffer))

            # DRAW GREEN FACE RECTANGLE
            cv2.rectangle(frame, (sfx, sfy), (sfx+sfw, sfy+sfh), (0, 255, 120), 2)
            cv2.putText(frame, "PATIENT TARGET LOCKED", (sfx, sfy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 120), 2)

            # 3. DRAW CHEST / THORACIC MOTION BOX
            cx = max(0, sfx - int(0.20 * sfw))
            cw = min(w_img - cx, int(1.4 * sfw))
            cy = min(h_img - 10, sfy + sfh + int(0.10 * sfh))
            ch = min(h_img - cy, int(1.6 * sfh))

            if ch > 20 and cw > 20:
                chest_roi = gray[cy:cy+ch, cx:cx+cw]
                chest_small = cv2.resize(chest_roi, (60, 45))
                
                # DRAW YELLOW/CYAN CHEST RECTANGLE
                cv2.rectangle(frame, (cx, cy), (cx+cw, cy+ch), (255, 200, 0), 2)
                cv2.putText(frame, "THORAX MOTION ROI", (cx, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1)

                if prev_chest_small is not None:
                    flow = cv2.calcOpticalFlowFarneback(prev_chest_small, chest_small, None, 0.5, 1, 9, 2, 5, 1.1, 0)
                    v_y = float(np.mean(flow[..., 1]))
                    
                    # NOISE GATE (< 0.08 forced to 0.0000)
                    if abs(v_y) < 0.08:
                        v_y = 0.0
                    else:
                        v_y = np.clip(v_y, -2.5, 2.5)

                    thoracic_motion_stream.append(v_y)
                else:
                    thoracic_motion_stream.append(0.0)
                prev_chest_small = chest_small.copy()
            else:
                thoracic_motion_stream.append(0.0)
        else:
            facial_green_stream.append(facial_green_stream[-1] if len(facial_green_stream)>0 else 0.0)
            thoracic_motion_stream.append(0.0)
            cv2.putText(frame, "SEARCHING FOR PATIENT FACE...", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 4. VITAL SIGNS & APNEA STATE MACHINE (100% ORIGINAL LOGIC)
        current_hr = last_hr
        current_rr_motion = 0.0
        resp_energy = 0.0
        apnea_duration = 0.0
        apnea_status = "INITIALIZING..."
        apnea_flag = 0

        if len(thoracic_motion_stream) >= 20:
            if len(facial_green_stream) >= 45:
                green_arr = np.asarray(list(facial_green_stream)[-60:])
                green_detrended = green_arr - np.mean(green_arr)
                filt_pulse = butter_bandpass(green_detrended, 0.75, 2.5, fs=FPS)
                raw_hr = calculate_welch_freq(filt_pulse, 0.75, 2.5, fs=FPS)
                current_hr = physiological_hr_smooth(hr_history, raw_hr)
                last_hr = current_hr

            motion_slice = np.array(list(thoracic_motion_stream)[-20:])
            resp_energy = float(np.std(motion_slice))
            filt_motion = butter_bandpass(list(thoracic_motion_stream), 0.10, 0.50, fs=FPS)

            if not calibration_done:
                if video_time < CALIBRATION_SEC:
                    calibration_samples.append(resp_energy)
                    apnea_status = f"CALIBRATING... ({video_time:.1f}/{CALIBRATION_SEC:.0f}s)"
                else:
                    valid = [s for s in calibration_samples if s > 0]
                    baseline_energy = float(np.median(valid)) if valid else 0.08
                    baseline_energy = max(baseline_energy, 0.080)
                    calibration_done = True
            else:
                enter_threshold = baseline_energy * APNEA_ENTER_RATIO
                exit_threshold  = baseline_energy * APNEA_EXIT_RATIO

                is_breath_stopped = (resp_energy < enter_threshold) or (resp_energy < HARD_APNEA_CEILING)

                if not in_apnea_state:
                    if is_breath_stopped:
                        in_apnea_state = True
                        recovery_frame_counter = 0
                        apnea_start_frame = frame_counter
                else:
                    if resp_energy > exit_threshold and resp_energy > HARD_APNEA_CEILING:
                        recovery_frame_counter += 1
                        if recovery_frame_counter >= 10:
                            in_apnea_state = False
                            apnea_start_frame = None
                            recovery_frame_counter = 0
                    else:
                        recovery_frame_counter = 0

                if in_apnea_state:
                    apnea_duration = (frame_counter - apnea_start_frame) / float(FPS)
                    if apnea_duration >= 0.5:
                        apnea_status = f"CRITICAL APNEA ({apnea_duration:.1f}s)"
                        apnea_flag = 2
                    elif apnea_duration >= 0.2:
                        apnea_status = f"BREATH HOLD ({apnea_duration:.1f}s)"
                        apnea_flag = 1
                else:
                    apnea_status = "NORMAL RESPIRATION"
                    current_rr_motion = calculate_welch_freq(filt_motion[-90:], 0.10, 0.50, fs=FPS)

        # Pass flag to background hardware thread (non-blocking)
        current_apnea_flag = apnea_flag

        # 5. ALWAYS-ON CLINICAL HUD OVERLAY & WAVEFORMS
        hud_w, hud_h = 360, 240
        x0, y0 = frame.shape[1] - hud_w - 15, 15

        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + hud_w, y0 + hud_h), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        status_color = (0, 255, 0)
        if apnea_flag == 2: status_color = (0, 0, 255)
        elif apnea_flag == 1: status_color = (0, 165, 255)
        elif not calibration_done: status_color = (255, 255, 0)

        cv2.rectangle(frame, (x0, y0), (x0 + hud_w, y0 + 35), status_color, -1)
        cv2.putText(frame, apnea_status, (x0 + 10, y0 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(frame, f"5G PULSE: {current_hr:.1f} BPM", (x0 + 10, y0 + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        cv2.putText(frame, f"5G RESP:  {current_rr_motion:.1f} BrPM", (x0 + 10, y0 + 85), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 220, 0), 2)
        cv2.putText(frame, f"ENERGY:   {resp_energy:.5f}", (x0 + 10, y0 + 110), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1)

        # Real-time PPG Pulse Wave (Green)
        cv2.putText(frame, "PLETH / PPG", (x0 + 10, y0 + 130), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
        pulse_slice = list(facial_green_stream)[-50:]
        if len(pulse_slice) > 1 and np.ptp(pulse_slice) > 1e-5:
            norm_p = (pulse_slice - np.min(pulse_slice)) / (np.ptp(pulse_slice) + 1e-5)
            pts_p = [(x0 + 10 + int(i * (hud_w - 20) / len(norm_p)), int(y0 + 165 - norm_p[i] * 30)) for i in range(len(norm_p))]
            for i in range(len(pts_p) - 1): cv2.line(frame, pts_p[i], pts_p[i + 1], (0, 255, 0), 2)

        # Real-time Respiratory Wave (Yellow)
        cv2.putText(frame, "RESPIRATION", (x0 + 10, y0 + 185), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 220, 0), 1)
        resp_slice = list(thoracic_motion_stream)[-50:]
        if len(resp_slice) > 1 and np.ptp(resp_slice) > 1e-5:
            norm_r = (resp_slice - np.min(resp_slice)) / (np.ptp(resp_slice) + 1e-5)
            pts_r = [(x0 + 10 + int(i * (hud_w - 20) / len(norm_r)), int(y0 + 225 - norm_r[i] * 30)) for i in range(len(norm_r))]
            for i in range(len(pts_r) - 1): cv2.line(frame, pts_r[i], pts_r[i + 1], (255, 220, 0), 2)

        # 6. UPDATE VISUAL DASHBOARD FRAME
        with frame_lock:
            output_frame = frame.copy()

        # 7. TRANSMIT 5G TELEMETRY & LOG
        if (frame_counter - last_log_frame) >= 6:
            base_val = baseline_energy if baseline_energy else 0.0
            cutoff = max(base_val * APNEA_ENTER_RATIO, HARD_APNEA_CEILING)

            telemetry_packet = {
                "hr_bpm": round(current_hr, 1),
                "rr_brpm": round(current_rr_motion, 1),
                "resp_energy": round(resp_energy, 5),
                "apnea_status": apnea_status,
                "apnea_flag": apnea_flag
            }
            telemetry_queue.append(telemetry_packet)

            csv_writer.writerow([round(video_time, 2), round(current_hr, 1), round(current_rr_motion, 1),
                                 round(resp_energy, 5), round(base_val, 5), apnea_status, apnea_flag])
            csv_file.flush()

            status_icon = "🔴" if apnea_flag == 2 else ("🟠" if apnea_flag == 1 else "🟢")
            print(f"[{video_time:05.1f}s] {status_icon} {apnea_status:<22} | HR: {current_hr:05.1f} | RR: {current_rr_motion:04.1f} | 5G -> ThingsBoard")

            last_log_frame = frame_counter

except KeyboardInterrupt:
    print("\n[INFO] Stopped by user.")
finally:
    hardware_running = False
    all_off()
    camera_stream.stop()
    csv_file.close()
