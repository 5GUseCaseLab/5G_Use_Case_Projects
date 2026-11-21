"""Send valid + malicious telemetry to the broker. Watch the SERVER log.

Run with the UNO Q app STOPPED (otherwise its real packets interleave):
  cd tools && python attack_telemetry.py [--host 192.168.x.x]
Expected: only test 1 is ACCEPTED; every other test is REJECTED.
"""
import argparse
import copy
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "mec_server"))
import config
import security
import paho.mqtt.client as mqtt

ap = argparse.ArgumentParser()
ap.add_argument("--host", default=config.BROKER_HOST)
a = ap.parse_args()

state_file = config.TELEMETRY_STATE_FILE
last = 0
if os.path.exists(state_file):
    last = json.load(open(state_file)).get(config.DEVICE_ID, 0)
seq = last + 1
topic = f"coldchain/{config.DEVICE_ID}/telemetry"
now = int(time.time())
good = security.make_envelope({"device_id": config.DEVICE_ID, "seq": seq, "timestamp": now, "temperature": 5.0})

tampered = copy.deepcopy(good); tampered["payload"]["temperature"] = 9.9
badsig = copy.deepcopy(good); badsig["payload"]["seq"] = seq + 1; badsig["sig"] = "0" * 64
nosig = {"payload": dict(good["payload"], seq=seq + 2)}
stale = security.make_envelope({"device_id": config.DEVICE_ID, "seq": seq + 3, "timestamp": now - 3600, "temperature": 5.0})
wrongkey = security.make_envelope({"device_id": config.DEVICE_ID, "seq": seq + 4, "timestamp": now, "temperature": 5.0}, key=b"attacker")

tests = [
    ("1 VALID packet                        -> expect ACCEPTED", json.dumps(good)),
    ("2 tampered temperature, old signature -> expect REJECTED (bad signature)", json.dumps(tampered)),
    ("3 replay of packet 1 (same seq)       -> expect REJECTED (replayed seq)", json.dumps(good)),
    ("4 forged signature                    -> expect REJECTED (bad signature)", json.dumps(badsig)),
    ("5 missing signature                   -> expect REJECTED (missing signature)", json.dumps(nosig)),
    ("6 stale timestamp (1h old)            -> expect REJECTED (stale)", json.dumps(stale)),
    ("7 signed with attacker's key          -> expect REJECTED (bad signature)", json.dumps(wrongkey)),
    ("8 garbage bytes                       -> expect REJECTED (malformed JSON)", "this is not json"),
]
c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="attacker")
c.connect(a.host, config.BROKER_PORT, 10)
c.loop_start()
for name, msg in tests:
    print("Sending:", name)
    c.publish(topic, msg, qos=1).wait_for_publish(5)
    time.sleep(1)
c.loop_stop()
