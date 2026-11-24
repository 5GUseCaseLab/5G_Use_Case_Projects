# =====================================================================
#  UNO Q  -  Linux side (Python)
#  Gets raw pot values from the MCU, converts them to servo angles,
#  smooths + clamps them, and sends them to the ESP32 as UDP packets.
#
#  Packet format (plain text):  "base,shoulder,elbow"   e.g.  "92,45,130"
# =====================================================================
from arduino.app_utils import *
import socket
import threading
import time

# ---------------- settings you MUST edit ----------------
ESP32_IP   = "192.168.1.50"   # <-- IP the ESP32 prints on its Serial Monitor
ESP32_PORT = 5006
CALIBRATE  = True             # Step 1: True  -> only prints raw pot values
                              # Step 2: fill CAL below, then set False

# Calibration, one line per joint:
#   (raw at end A, raw at end B, servo angle at end A, servo angle at end B)
# Move each master joint to one end, note the raw value, then the other end.
# If a joint moves the wrong way, just swap the two angles.
CAL = {
    "base":     (400, 3700,   0, 180),
    "shoulder": (400, 3700,   0, 180),
    "elbow":    (400, 3700,   0, 180),
}

# Mechanical safety limits of the SLAVE arm (degrees). Tighten these if
# a joint hits the frame / cables / the table.
LIMITS = {
    "base":     (0, 180),
    "shoulder": (15, 165),
    "elbow":    (10, 170),
}

# ---------------- tuning ----------------
SEND_HZ  = 50     # packets per second
ALPHA    = 0.3    # smoothing: lower = smoother but laggier (0.2-0.5 is sensible)
MAX_STEP = 6      # max degrees a joint may move per packet (6 deg x 50 Hz = 300 deg/s)

JOINTS = ("base", "shoulder", "elbow")

_latest = None
_lock = threading.Lock()


def on_pots(base, shoulder, elbow):
    """Called by the Bridge every time the MCU sends a reading (50x per second)."""
    global _latest
    with _lock:
        _latest = (int(base), int(shoulder), int(elbow))


Bridge.provide("pots", on_pots)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
smoothed = None
sent = None
last_print = 0.0


def raw_to_angle(raw, cal):
    raw_a, raw_b, ang_a, ang_b = cal
    if raw_b == raw_a:
        return float(ang_a)
    t = (raw - raw_a) / (raw_b - raw_a)
    t = max(0.0, min(1.0, t))
    return ang_a + t * (ang_b - ang_a)


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def step_towards(current, target, max_step):
    delta = clamp(target - current, -max_step, max_step)
    return current + delta


def loop():
    global smoothed, sent, last_print
    time.sleep(1.0 / SEND_HZ)

    with _lock:
        raw = _latest
    if raw is None:                       # MCU hasn't sent anything yet
        if time.time() - last_print > 2:
            print("waiting for pot data from the MCU sketch...")
            last_print = time.time()
        return

    if CALIBRATE:
        if time.time() - last_print > 0.25:
            print("RAW   base=%4d   shoulder=%4d   elbow=%4d" % raw)
            last_print = time.time()
        return

    # 1) raw ADC -> angle, 2) clamp to safe range
    target = [clamp(raw_to_angle(r, CAL[j]), *LIMITS[j]) for r, j in zip(raw, JOINTS)]

    # 3) exponential moving average (low-pass filter against pot jitter)
    if smoothed is None:
        smoothed = target[:]
    else:
        smoothed = [ALPHA * t + (1 - ALPHA) * s for t, s in zip(target, smoothed)]

    # 4) rate limit: a loose wire can't make the slave arm jump
    if sent is None:
        sent = [round(s) for s in smoothed]
    else:
        sent = [round(step_towards(c, s, MAX_STEP)) for c, s in zip(sent, smoothed)]

    msg = "%d,%d,%d" % tuple(sent)
    try:
        sock.sendto(msg.encode(), (ESP32_IP, ESP32_PORT))
    except OSError as e:                  # e.g. Wi-Fi dropped for a moment
        print("send failed:", e)

    if time.time() - last_print > 0.5:
        print("raw=%s  ->  sent=%s" % (raw, msg))
        last_print = time.time()


App.run(user_loop=loop)
