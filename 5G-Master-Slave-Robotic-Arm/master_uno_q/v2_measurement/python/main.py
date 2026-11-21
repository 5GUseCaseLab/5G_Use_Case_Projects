# =====================================================================
#  UNO Q  -  Linux side (Python)   ***  MEASUREMENT VERSION  ***
#  Gets raw pot values from the MCU, converts them to servo angles,
#  smooths + clamps them, and sends them to the ESP32 as UDP packets.
#
#  On top of that it MEASURES, and prints every REPORT_EVERY_S seconds:
#    - latency  : round-trip time (RTT) of every packet, one-way = RTT/2
#    - jitter   : spread of the latency, spikes above SPIKE_MS
#    - delivery : % of packets that made it (both directions)
#    - pipeline : time spent in each stage (ADC, Bridge, wait, Python,
#                 Wi-Fi, ESP32) and the total pot -> servo-command delay
#    - data rate: bits per second on the link
#    - tracking : |master angle - angle the ESP32 actually applied|
#    - filter   : noise and lag for raw / alpha 0.3 / alpha 0.1 side by side
#  and logs every packet to arm_log.csv (for the angle-vs-time plot).
#
#  Packet  UNO Q -> ESP32 : "seq,t_us,base,shoulder,elbow"
#  Echo    ESP32 -> UNO Q : "A,seq,t_us,proc_us,base,shoulder,elbow,rx,lost"
#  t_us is the UNO Q's own clock. The ESP32 just sends it back, so the
#  round trip is measured on ONE clock - no clock sync needed.
# =====================================================================
from arduino.app_utils import *
from array import array
import math
import os
import socket
import threading
import time

# ---------------- settings you MUST edit ----------------
ESP32_IP   = "192.168.1.50"   # <-- IP the ESP32 prints on its Serial Monitor
ESP32_PORT = 5006
CALIBRATE  = False            # True -> only prints raw pot values

# Calibration, one line per joint  (copy YOUR numbers from your old main.py):
#   (raw at end A, raw at end B, servo angle at end A, servo angle at end B)
CAL = {
    "base":     (400, 3700,   0, 180),
    "shoulder": (400, 3700,   0, 180),
    "elbow":    (400, 3700,   0, 180),
}

# Mechanical safety limits of the SLAVE arm (degrees). Same as the ESP32 sketch.
LIMITS = {
    "base":     (0, 180),
    "shoulder": (15, 165),
    "elbow":    (10, 170),
}

# ---------------- tuning ----------------
SEND_HZ  = 50     # packets per second
ALPHA    = 0.3    # smoothing: lower = smoother but laggier
MAX_STEP = 6      # max degrees a joint may move per packet

# ---------------- measurement settings ----------------
REPORT_EVERY_S  = 5        # print a report this often
SPIKE_MS        = 100      # a round trip slower than this counts as a "spike"
ACK_TIMEOUT_S   = 1.0      # no echo after this long -> packet counted as lost
COMPARE_ALPHAS  = (1.0, 0.3, 0.1)   # filters compared side by side (1.0 = raw)
STILL_DEG       = 4.0      # arm counts as "still" if it moved less than this
MOVING_DEG      = 15.0     # arm counts as "moving" if it moved more than this
BRIDGE_PING     = True     # time Bridge.call("ping") once a second
LOG_CSV         = True
LOG_PATH        = os.path.join(os.path.dirname(os.path.abspath(__file__)), "arm_log.csv")

JOINTS = ("base", "shoulder", "elbow")
PERIOD = 1.0 / SEND_HZ
UDP_IP_HEADER_BYTES = 28   # 20 B IPv4 + 8 B UDP added to every packet


# =====================================================================
#  small statistics helpers
# =====================================================================
class Series:
    """Values for the current report window + the whole run."""
    def __init__(self):
        self.win = []
        self.all = array("d")

    def add(self, x):
        self.win.append(x)
        self.all.append(x)

    def roll(self):
        w, self.win = self.win, []
        return w


def summary(vals):
    if not vals:
        return None
    s = sorted(vals)
    n = len(s)
    mean = sum(s) / n
    var = sum((x - mean) ** 2 for x in s) / n
    pick = lambda q: s[min(n - 1, int(q * n))]
    return {"n": n, "mean": mean, "med": pick(0.5), "p95": pick(0.95),
            "p99": pick(0.99), "min": s[0], "max": s[-1], "std": math.sqrt(var)}


def mean(vals):
    return sum(vals) / len(vals) if vals else None


def fmt(x, nd=1, unit=""):
    return "n/a" if x is None else ("%.*f%s" % (nd, x, unit))


class Counters:
    def __init__(self):
        self.win = {}
        self.tot = {}

    def inc(self, name, n=1):
        self.win[name] = self.win.get(name, 0) + n
        self.tot[name] = self.tot.get(name, 0) + n

    def roll(self):
        w, self.win = self.win, {}
        return w


# =====================================================================
#  shared state (Bridge thread, echo thread and main loop all touch it)
# =====================================================================
_lock = threading.Lock()
_latest = None            # newest raw pot values
_latest_t = None          # when they arrived (perf_counter)
_last_arrival = None

M_rtt        = Series()   # round-trip time, ms
M_drtt       = Series()   # |RTT - previous RTT|, ms
M_esp_proc   = Series()   # ESP32 processing time, ms
M_adc        = Series()   # MCU ADC time, ms
M_bridge_int = Series()   # time between pot messages from the MCU, ms
M_bridge_rtt = Series()   # Bridge.call("ping") round trip, ms
M_age        = Series()   # how old the pot sample is when it's sent, ms
M_py         = Series()   # Python processing time, ms
M_err        = [Series() for _ in JOINTS]   # tracking error per joint, deg
C = Counters()
esp_counts = {"rx": None, "lost": None}
_last_rtt = None

pending = {}              # seq -> (t_sent, master_target, sent_angles)
t_start = time.perf_counter()

csv = None
if LOG_CSV and not CALIBRATE:
    try:
        csv = open(LOG_PATH, "w")
        csv.write("t_s,seq,rtt_ms,master_base,master_shoulder,master_elbow,"
                  "sent_base,sent_shoulder,sent_elbow,"
                  "slave_base,slave_shoulder,slave_elbow\n")
        print("logging every packet to", LOG_PATH)
    except OSError as e:
        print("CSV log disabled:", e)
        csv = None


def now_us():
    return int(time.perf_counter() * 1e6) & 0xFFFFFFFF


# =====================================================================
#  1) pot values from the MCU sketch
# =====================================================================
def on_pots(base, shoulder, elbow, adc_us=0):
    """Called by the Bridge every time the MCU sends a reading (50x per second)."""
    global _latest, _latest_t, _last_arrival
    t = time.perf_counter()
    with _lock:
        _latest = (int(base), int(shoulder), int(elbow))
        _latest_t = t
        if _last_arrival is not None:
            M_bridge_int.add((t - _last_arrival) * 1000)
        _last_arrival = t
        if adc_us:
            M_adc.add(int(adc_us) / 1000.0)


Bridge.provide("pots", on_pots)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", 0))           # fixed local port so the ESP32 can echo back


# =====================================================================
#  2) echoes coming back from the ESP32  (runs in its own thread)
# =====================================================================
def echo_worker():
    global _last_rtt
    while True:
        try:
            data, _ = sock.recvfrom(256)
        except OSError:
            time.sleep(0.05)
            continue
        t_now = time.perf_counter()
        parts = data.decode(errors="ignore").strip().split(",")
        if len(parts) < 7 or parts[0] != "A":
            continue
        try:
            v = [int(p) for p in parts[1:]]
        except ValueError:
            continue
        seq, t_us, proc_us = v[0], v[1], v[2]
        applied = v[3:6]
        rtt = (((int(t_now * 1e6) & 0xFFFFFFFF) - t_us) & 0xFFFFFFFF) / 1000.0

        with _lock:
            C.inc("echoes")
            C.inc("bytes_rx", len(data))
            M_rtt.add(rtt)
            if _last_rtt is not None:
                M_drtt.add(abs(rtt - _last_rtt))
            _last_rtt = rtt
            M_esp_proc.add(proc_us / 1000.0)
            if rtt > SPIKE_MS:
                C.inc("spikes")
            if len(v) >= 8:
                esp_counts["rx"], esp_counts["lost"] = v[6], v[7]

            entry = pending.pop(seq, None)
            if entry is None:
                C.inc("late")            # already counted as lost (came back > ACK_TIMEOUT_S)
                continue
            C.inc("delivered")
            _, target, sent = entry
            for j in range(3):
                M_err[j].add(abs(target[j] - applied[j]))
            if csv:
                csv.write("%.3f,%d,%.2f,%.1f,%.1f,%.1f,%d,%d,%d,%d,%d,%d\n" % (
                    t_now - t_start, seq, rtt, target[0], target[1], target[2],
                    sent[0], sent[1], sent[2], applied[0], applied[1], applied[2]))


# =====================================================================
#  3) Bridge latency: time a Bridge.call once a second
# =====================================================================
def bridge_ping_worker():
    fails = 0
    while fails < 3:
        time.sleep(1.0)
        t0 = time.perf_counter()
        try:
            Bridge.call("ping")
        except Exception as e:
            fails += 1
            print("Bridge ping failed (%s)" % e)
            continue
        fails = 0
        with _lock:
            M_bridge_rtt.add((time.perf_counter() - t0) * 1000)
    print("Bridge ping turned off - is the new sketch.ino uploaded?")


if not CALIBRATE:
    threading.Thread(target=echo_worker, daemon=True).start()
    if BRIDGE_PING:
        threading.Thread(target=bridge_ping_worker, daemon=True).start()


# =====================================================================
#  angle maths (same as before)
# =====================================================================
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
    return current + clamp(target - current, -max_step, max_step)


# =====================================================================
#  filter comparison (noise when still, lag when moving)
# =====================================================================
win_raw = [[] for _ in JOINTS]                       # clamped, unsmoothed angle
win_filt = {a: [[] for _ in JOINTS] for a in COMPARE_ALPHAS}
win_live = [[] for _ in JOINTS]                      # what was actually sent
shadow = None


def best_lag_ms(x, y, max_k=25):
    """How many ms y trails x: shift y back until it lines up best with x."""
    n = len(x)
    errs = []
    for k in range(max_k + 1):
        m = n - k
        if m < 20:
            break
        errs.append(sum(abs(y[i + k] - x[i]) for i in range(m)) / m)
    if not errs:
        return None
    k = min(range(len(errs)), key=errs.__getitem__)
    frac = 0.0
    if 0 < k < len(errs) - 1:                       # sub-sample refinement
        a, b, c = errs[k - 1], errs[k], errs[k + 1]
        d = a - 2 * b + c
        if d > 0:
            frac = 0.5 * (a - c) / d
    return (k + frac) * 1000.0 / SEND_HZ


def filter_report(raw, filt, live):
    if len(raw[0]) < 50:
        return ["Filter    not enough samples yet"]
    smooth = filt[min(COMPARE_ALPHAS)]
    span = max(max(s) - min(s) for s in smooth)
    state = "STILL" if span < STILL_DEG else ("MOVING" if span > MOVING_DEG else "slow/mixed")
    jm = max(range(3), key=lambda j: max(raw[j]) - min(raw[j]))   # joint that moved most

    rows = [("raw" if a == 1.0 else "a=%.2f" % a, filt[a]) for a in COMPARE_ALPHAS]
    rows.append(("LIVE", live))
    out = ["Filter    arm is %s in this window (%s)" % (state,
           "read NOISE now" if state == "STILL" else
           "read LAG now (from %s)" % JOINTS[jm] if state == "MOVING" else
           "hold still for noise, sweep a joint for lag"),
           "          %-6s  noise +-deg (base/shoulder/elbow)   std deg   lag ms" % ""]
    for name, series in rows:
        pm = ["%.2f" % ((max(s) - min(s)) / 2) for s in series]
        sd = mean([summary(s)["std"] for s in series])
        lag = best_lag_ms(raw[jm], series[jm]) if state == "MOVING" else None
        out.append("          %-6s  %5s / %5s / %5s               %5.2f     %s" % (
            name, pm[0], pm[1], pm[2], sd, fmt(lag, 0)))
    return out


# =====================================================================
#  the report
# =====================================================================
last_report = time.perf_counter()


def report():
    global last_report, win_raw, win_filt, win_live
    t = time.perf_counter()
    dt = t - last_report
    last_report = t

    with _lock:
        # packets with no echo for too long -> lost
        for s, (ts, _, _) in list(pending.items()):
            if t - ts > ACK_TIMEOUT_S:
                del pending[s]
                C.inc("lost")
        cw = C.roll()
        ct = dict(C.tot)
        w = {k: v.roll() for k, v in (("rtt", M_rtt), ("drtt", M_drtt), ("esp", M_esp_proc),
                                     ("adc", M_adc), ("bint", M_bridge_int),
                                     ("bping", M_bridge_rtt), ("age", M_age), ("py", M_py))}
        werr = [e.roll() for e in M_err]
        rtt_all = list(M_rtt.all)
        drtt_all = list(M_drtt.all)
        err_all = [list(e.all) for e in M_err]
        esp = dict(esp_counts)
    if csv:
        csv.flush()

    def deliv(c):
        d, l = c.get("delivered", 0), c.get("lost", 0)
        return 100.0 * d / (d + l) if d + l else None

    rw, ra = summary(w["rtt"]), summary(rtt_all)
    L = []
    L.append("=" * 78)
    L.append("REPORT  t=%.0f s        left: last %.0f s   |   right: whole run" % (t - t_start, dt))
    L.append("Packets   sent %d (%.1f/s), echoed %d, lost %d   |   sent %d, lost %d, late %d" % (
        cw.get("sent", 0), cw.get("sent", 0) / dt, cw.get("echoes", 0), cw.get("lost", 0),
        ct.get("sent", 0), ct.get("lost", 0), ct.get("late", 0)))
    fwd = None
    if esp["rx"] is not None and esp["rx"] + esp["lost"] > 0:
        fwd = 100.0 * esp["rx"] / (esp["rx"] + esp["lost"])
    L.append("Delivery  round trip %s   |   round trip %s,  UNO Q->ESP32 only %s" % (
        fmt(deliv(cw), 2, "%"), fmt(deliv(ct), 2, "%"), fmt(fwd, 2, "%")))
    if rw:
        L.append("Latency   RTT mean %.1f  median %.1f  p95 %.1f  p99 %.1f  max %.1f ms" % (
            rw["mean"], rw["med"], rw["p95"], rw["p99"], rw["max"]))
    if ra:
        L.append("          whole run: RTT mean %.1f  median %.1f  p95 %.1f  max %.1f ms"
                 "  ->  one-way (Wi-Fi) ~ %.1f ms" % (ra["mean"], ra["med"], ra["p95"], ra["max"],
                                                      ra["mean"] / 2))
    L.append("Jitter    RTT std %s ms, mean |change| %s ms, spikes >%d ms: %d   |   std %s, |change| %s, spikes %d" % (
        fmt(rw and rw["std"], 1), fmt(mean(w["drtt"]), 1), SPIKE_MS, cw.get("spikes", 0),
        fmt(ra and ra["std"], 1), fmt(mean(drtt_all), 1), ct.get("spikes", 0)))

    # ---- latency breakdown, pot -> servo command ----
    adc = mean(w["adc"])
    bridge = mean(w["bping"])
    bridge = bridge / 2 if bridge is not None else None
    age = mean(w["age"])
    py = mean(w["py"])
    wifi = rw["mean"] / 2 if rw else None
    espp = mean(w["esp"])
    parts = [adc, bridge, age, py, wifi, espp]
    total = sum(p for p in parts if p is not None)
    L.append("Pipeline  ADC %s | Bridge %s | wait for send %s | Python %s | Wi-Fi %s | ESP32 %s ms" % tuple(
        fmt(p, 2) for p in parts))
    L.append("          pot -> servo command ~ %.1f ms%s   (servo travel time not included)" % (
        total, "" if None not in parts else "  (n/a parts left out)"))
    bi = summary(w["bint"])
    if bi:
        L.append("          MCU->Python messages every %.1f ms (std %.2f ms)" % (bi["mean"], bi["std"]))

    # ---- data rate ----
    pay_tx, pay_rx = cw.get("bytes_tx", 0), cw.get("bytes_rx", 0)
    n_tx, n_rx = cw.get("sent", 0), cw.get("echoes", 0)
    L.append("Data rate UNO Q->ESP32 %.2f kbit/s (%.2f with UDP/IP headers), echo %.2f kbit/s (%.2f)" % (
        pay_tx * 8 / dt / 1000, (pay_tx + n_tx * UDP_IP_HEADER_BYTES) * 8 / dt / 1000,
        pay_rx * 8 / dt / 1000, (pay_rx + n_rx * UDP_IP_HEADER_BYTES) * 8 / dt / 1000))

    # ---- tracking error ----
    def err_line(errs):
        out = []
        for j, e in enumerate(errs):
            if not e:
                out.append("%s n/a" % JOINTS[j][0])
                continue
            out.append("%s %.1f/%.1f/%.0f" % (JOINTS[j][0], mean(e),
                       math.sqrt(sum(x * x for x in e) / len(e)), max(e)))
        allv = [x for e in errs for x in e]
        return "  ".join(out) + ("   all %.2f" % mean(allv) if allv else "")
    L.append("Tracking  |master - slave| deg, mean/RMS/max per joint")
    L.append("          last %.0f s : %s" % (dt, err_line(werr)))
    L.append("          whole run: %s" % err_line(err_all))

    L.extend(filter_report(win_raw, win_filt, win_live))
    win_raw = [[] for _ in JOINTS]
    win_filt = {a: [[] for _ in JOINTS] for a in COMPARE_ALPHAS}
    win_live = [[] for _ in JOINTS]
    print("\n".join(L))


# =====================================================================
#  main loop: 50 Hz
# =====================================================================
smoothed = None
sent = None
seq = 0
next_t = None
last_print = 0.0


def loop():
    global smoothed, sent, seq, next_t, last_print, shadow

    # steady 50 Hz (sleep only what's left of the period)
    if next_t is None:
        next_t = time.perf_counter()
    next_t += PERIOD
    delay = next_t - time.perf_counter()
    if delay > 0:
        time.sleep(delay)
    elif delay < -PERIOD:
        next_t = time.perf_counter()     # fell far behind -> resync

    with _lock:
        raw = _latest
        raw_t = _latest_t
    if raw is None:
        if time.time() - last_print > 2:
            print("waiting for pot data from the MCU sketch...")
            last_print = time.time()
        return

    if CALIBRATE:
        if time.time() - last_print > 0.25:
            print("RAW   base=%4d   shoulder=%4d   elbow=%4d" % raw)
            last_print = time.time()
        return

    t0 = time.perf_counter()
    # 1) raw ADC -> angle, 2) clamp to safe range
    target = [clamp(raw_to_angle(r, CAL[j]), *LIMITS[j]) for r, j in zip(raw, JOINTS)]

    # 3) exponential moving average
    if smoothed is None:
        smoothed = target[:]
    else:
        smoothed = [ALPHA * t + (1 - ALPHA) * s for t, s in zip(target, smoothed)]

    # 4) rate limit
    if sent is None:
        sent = [round(s) for s in smoothed]
    else:
        sent = [round(step_towards(c, s, MAX_STEP)) for c, s in zip(sent, smoothed)]

    seq += 1
    msg = ("%d,%d,%d,%d,%d" % (seq, now_us(), sent[0], sent[1], sent[2])).encode()
    py_ms = (time.perf_counter() - t0) * 1000
    t_send = time.perf_counter()
    try:
        sock.sendto(msg, (ESP32_IP, ESP32_PORT))
        ok = True
    except OSError as e:                  # e.g. Wi-Fi dropped for a moment
        print("send failed:", e)
        ok = False

    with _lock:
        if ok:
            pending[seq] = (t_send, tuple(target), tuple(sent))
            C.inc("sent")
            C.inc("bytes_tx", len(msg))
        M_py.add(py_ms)
        M_age.add((t_send - raw_t) * 1000)

    # side-by-side filters, for the comparison table only
    if shadow is None:
        shadow = {a: target[:] for a in COMPARE_ALPHAS}
    for a in COMPARE_ALPHAS:
        shadow[a] = [a * t + (1 - a) * s for t, s in zip(target, shadow[a])]
        for j in range(3):
            win_filt[a][j].append(shadow[a][j])
    for j in range(3):
        win_raw[j].append(target[j])
        win_live[j].append(sent[j])

    if time.perf_counter() - last_report >= REPORT_EVERY_S:
        report()


App.run(user_loop=loop)
