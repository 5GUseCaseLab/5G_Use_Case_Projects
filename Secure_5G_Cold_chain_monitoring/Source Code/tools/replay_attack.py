"""True replay attack: capture ONE real message from the broker, then re-send the exact bytes.
The attacker needs no key (plain MQTT can be read by anyone on the network), only a valid-looking
packet, and the system must still reject it.

  python replay_attack.py --kind telemetry            # replay a real temperature reading
  python replay_attack.py --kind command              # replay a real cooling command
  python replay_attack.py --kind command --delay 40   # replay late: rejected as stale (timestamp check)

Run while the server and the board app are running. Watch the dashboard / server log / board log.
"""
import argparse
import os
import threading
import time

import paho.mqtt.client as mqtt

ap = argparse.ArgumentParser()
ap.add_argument("--kind", choices=["telemetry", "command"], required=True)
ap.add_argument("--host", default=os.environ.get("MQTT_HOST", "localhost"))
ap.add_argument("--port", type=int, default=int(os.environ.get("MQTT_PORT", "1883")))
ap.add_argument("--device", default="uno_q_edge_01")
ap.add_argument("--times", type=int, default=2, help="how many times to replay it")
ap.add_argument("--delay", type=float, default=1.0, help="seconds to wait before replaying")
a = ap.parse_args()

topic = f"coldchain/{a.device}/" + ("telemetry" if a.kind == "telemetry" else "commands")
got = {}
ready = threading.Event()


def on_connect(c, u, f, rc, p):
    c.subscribe(topic, qos=1)


def on_message(c, u, msg):
    if not ready.is_set():
        got["payload"] = bytes(msg.payload)
        ready.set()


c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="attacker_replay")
c.on_connect, c.on_message = on_connect, on_message
c.connect(a.host, a.port, 30)
c.loop_start()
print(f"Listening on {topic} for one real message (up to 60 s)...")
if not ready.wait(60):
    c.loop_stop()
    raise SystemExit("Nothing captured: are the board app and server running?")
data = got["payload"]
print(f"Captured {len(data)} bytes: {data[:110].decode(errors='replace')}...")
print(f"Waiting {a.delay:.0f} s, then replaying it {a.times} time(s) (no key used)")
time.sleep(a.delay)
for i in range(a.times):
    c.publish(topic, data, qos=1).wait_for_publish(5)
    print(f"  replayed #{i + 1}")
    time.sleep(0.6)
c.loop_stop()
print("Done. Expected: REJECTED as replayed seq/nonce" + (" (or stale timestamp if delay > 30 s)" if a.kind == "command" else ""))
