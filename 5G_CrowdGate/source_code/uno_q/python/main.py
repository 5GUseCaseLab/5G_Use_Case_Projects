"""CrowdGate gate controller - runs on the Arduino UNO Q's Linux side (Arduino App Lab -> Python).

Every second it asks the MEC for the crowd state (HTTP GET http://<MEC>:5000/status, JSON) and tells the MCU
sketch what to do through the Bridge. Only Python's standard library is used, so nothing needs installing and the
Wi-Fi does not need internet (the Kenstel 5G CPE network is enough).

Fail-safe: no valid reply for TIMEOUT seconds (MEC down, link lost, bad data) -> state 3 (red blink + buzzer + barrier closed).
State codes: 0 GREEN, 1 YELLOW, 2 RED, 3 FAIL-SAFE (also CAMERA_FAILURE or anything unknown).

MEC address: edit MEC_HOST below, or set the environment variable MEC_HOST, or pass it as the first argument:
    python3 main.py 192.168.116.30

Uses the App Lab Python API (Bridge.call, App.run) as in the Arduino App Lab examples.
"""
import json
import os
import sys
import time
import urllib.request

from arduino.app_utils import *

MEC_HOST = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("MEC_HOST", "192.168.116.30")
MEC_PORT = 5000
URL = f"http://{MEC_HOST}:{MEC_PORT}/status"
POLL_S = 1.0                     # ask once per second
HTTP_TIMEOUT = 1.5               # seconds to wait for one reply
TIMEOUT = 5.0                    # seconds without a valid reply => fail-safe
CODE = {"GREEN": 0, "YELLOW": 1, "RED": 2, "CAMERA_FAILURE": 3}   # anything unknown also -> 3

last_ok = 0.0                    # time of the last valid reply (0 = none yet -> start in fail-safe)
last_sent = None
last_poll = 0.0


def set_state(code):
    global last_sent
    if code != last_sent:                 # only call the MCU when something changed
        Bridge.call("set_state", code)
        last_sent = code
        print("state ->", code, flush=True)


def poll():
    global last_ok
    try:
        with urllib.request.urlopen(URL, timeout=HTTP_TIMEOUT) as r:
            data = json.loads(r.read().decode())
        code = CODE.get(data.get("state"), 3)
        last_ok = time.time()
        set_state(code)
    except Exception as e:                # unreachable, timeout, bad JSON ...
        print("poll failed:", e, flush=True)


def loop():
    global last_poll
    now = time.time()
    if now - last_poll >= POLL_S:
        last_poll = now
        poll()
    if time.time() - last_ok > TIMEOUT:
        set_state(3)                      # lost link to MEC -> safe mode
    time.sleep(0.1)


print("polling", URL, flush=True)
set_state(3)                              # start safe until the MEC answers
App.run(user_loop=loop)
