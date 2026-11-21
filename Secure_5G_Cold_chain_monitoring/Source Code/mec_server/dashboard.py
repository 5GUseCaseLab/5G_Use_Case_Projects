"""Read-only web dashboard served by the MEC server (stdlib only, no extra installs).

Open http://localhost:8080 on the laptop. Set DASHBOARD_HOST=0.0.0.0 to view it from a phone
on the same network (there is no login: keep it on your private hotspot).
"""
import json
import os
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import config

HOST = os.environ.get("DASHBOARD_HOST", "127.0.0.1")
PORT = int(os.environ.get("DASHBOARD_PORT", "8080"))
OFFLINE_AFTER_S = 10          # no verified telemetry for this long -> OFFLINE
ATTACK_HOLD_S = 10            # keep the red ATTACK state this long after a rejection
HTML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.html")


class LatencyStats:
    def __init__(self, keep=300):
        self.values = deque(maxlen=keep)
        self.total = 0

    def add(self, ms):
        self.values.append(float(ms))
        self.total += 1

    def summary(self):
        v = list(self.values)
        if not v:
            return {"n": 0, "last": None, "avg": None, "min": None, "max": None, "p95": None}
        s = sorted(v)
        return {"n": self.total, "last": round(v[-1], 1), "avg": round(sum(v) / len(v), 1),
                "min": round(s[0], 1), "max": round(s[-1], 1), "p95": round(s[int(0.95 * (len(s) - 1))], 1)}

    def recent(self, k=60):
        return [round(x, 1) for x in list(self.values)[-k:]]


class DashboardState:
    def __init__(self):
        self.lock = threading.Lock()
        self.history = deque(maxlen=200)        # (unix_time, temperature)
        self.events = deque(maxlen=60)          # (unix_time, level, text)
        self.latest = None
        self.last_telemetry_at = 0.0
        self.alarm = None
        self.sent_pwm = 0                       # what the server last asked for
        self.applied_pwm = None                 # what the board confirmed (signed event)
        self.attack_until = 0.0
        self.counts = {"accepted": 0, "rejected_telemetry": 0, "commands_sent": 0,
                       "commands_accepted": 0, "commands_rejected": 0}
        self.last_seq = None
        self.last_nonce = None
        self.broker_connected = False
        self.rtt = LatencyStats()          # command round-trip (server clock only)
        self.oneway = LatencyStats()       # telemetry one-way (needs synced clocks)

    def add_latency(self, kind, ms):
        with self.lock:
            (self.rtt if kind == "rtt" else self.oneway).add(ms)

    def add_event(self, level, text):
        with self.lock:
            self.events.appendleft((time.time(), level, text))

    def set_broker(self, ok):
        with self.lock:
            self.broker_connected = ok
        self.add_event("info" if ok else "warn", "Broker connected" if ok else "Broker disconnected")

    def telemetry_ok(self, payload, pwm, alarm):
        now = time.time()
        with self.lock:
            self.history.append((now, payload["temperature"]))
            self.latest = payload["temperature"]
            self.last_seq = payload["seq"]
            self.last_telemetry_at = now
            self.alarm = alarm
            self.counts["accepted"] += 1
            if alarm:
                self.events.appendleft((now, "warn", f"{alarm} at {payload['temperature']:.2f} C"))

    def rejected(self, kind, reason):
        now = time.time()
        with self.lock:
            self.attack_until = now + ATTACK_HOLD_S
            key = "rejected_telemetry" if kind == "telemetry" else "commands_rejected"
            self.counts[key] += 1
            self.events.appendleft((now, "bad", f"REJECTED {kind}: {reason}"))

    def command_sent(self, nonce, pwm):
        with self.lock:
            self.sent_pwm = pwm
            self.last_nonce = nonce
            self.counts["commands_sent"] += 1
            self.events.appendleft((time.time(), "info", f"Command sent: nonce={nonce} pwm={pwm}"))

    def board_event(self, payload):
        ev = payload.get("event")
        if ev == "command_accepted":
            with self.lock:
                self.applied_pwm = payload.get("pwm")
                self.counts["commands_accepted"] += 1
            self.add_event("good", f"Board accepted command nonce={payload.get('nonce')} pwm={payload.get('pwm')}")
        elif ev == "command_rejected":
            self.rejected("command (reported by board)", str(payload.get("reason")))
        elif ev in ("sensor_fault", "actuator_error"):
            self.add_event("bad", f"BOARD FAULT ({ev}): {payload.get('reason', payload.get('pwm'))}")
        else:
            self.add_event("info", f"Board event: {ev}")

    def snapshot(self):
        now = time.time()
        with self.lock:
            silent = now - self.last_telemetry_at if self.last_telemetry_at else None
            pwm = self.applied_pwm if self.applied_pwm is not None else self.sent_pwm
            if silent is None or silent > OFFLINE_AFTER_S:
                status = "OFFLINE"
            elif now < self.attack_until:
                status = "ATTACK"
            elif pwm and pwm > 0:
                status = "COOLING"
            else:
                status = "SECURE"
            return {
                "status": status, "temperature": self.latest, "alarm": self.alarm,
                "cooling_pwm": pwm, "seconds_since_telemetry": silent,
                "last_seq": self.last_seq, "last_nonce": self.last_nonce,
                "broker_connected": self.broker_connected, "counts": dict(self.counts),
                "thresholds": {"low": config.SAFE_LOW_C, "high": config.SAFE_HIGH_C,
                               "cool_on": config.COOL_ON_C, "cool_off": config.COOL_OFF_C},
                "history": [[round(t, 1), v] for t, v in self.history],
                "events": [[round(t, 1), lvl, txt] for t, lvl, txt in list(self.events)[:30]],
                "latency": {"rtt": self.rtt.summary(), "oneway": self.oneway.summary(),
                            "rtt_recent": self.rtt.recent(), "oneway_recent": self.oneway.recent()},
                "now": round(now, 1),
            }


def start(state: DashboardState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, ctype, body: bytes):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                with open(HTML_PATH, "rb") as f:
                    self._send(200, "text/html; charset=utf-8", f.read())
            elif self.path == "/api/state":
                self._send(200, "application/json", json.dumps(state.snapshot()).encode())
            else:
                self._send(404, "text/plain", b"not found")

    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv
