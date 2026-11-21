"""Hand-craft commands to the UNO Q to test nonce/replay logic.

STOP mec_server first (so it does not send its own commands).
  python send_test_command.py --nonce 1000 --pwm 150          # valid
  python send_test_command.py --nonce 1000 --pwm 150          # again -> replay, REJECTED
  python send_test_command.py --nonce 1001 --pwm 0            # valid
  python send_test_command.py --nonce 2000 --pwm 255 --tamper # REJECTED (bad signature)
  python send_test_command.py --nonce 2000 --bad-sig | --no-sig | --wrong-device | --stale | --wrong-key
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
ap.add_argument("--nonce", type=int, required=True)
ap.add_argument("--pwm", type=int, default=150)
ap.add_argument("--tamper", action="store_true", help="change pwm after signing")
ap.add_argument("--bad-sig", action="store_true")
ap.add_argument("--no-sig", action="store_true", help="send the command without any signature")
ap.add_argument("--wrong-device", action="store_true")
ap.add_argument("--stale", action="store_true")
ap.add_argument("--wrong-key", action="store_true")
a = ap.parse_args()

dev = "some_other_device" if a.wrong_device else config.DEVICE_ID
now = time.time() - (3600 if a.stale else 0)
env = security.make_command(dev, a.nonce, a.pwm, now, key=b"attacker" if a.wrong_key else None)
if a.tamper:
    env = copy.deepcopy(env); env["payload"]["pwm"] = 255 if a.pwm != 255 else 0
if a.bad_sig:
    env["sig"] = "f" * 64
if a.no_sig:
    env.pop("sig", None)

c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="test_cmd_sender")
c.connect(a.host, config.BROKER_PORT, 10)
c.loop_start()
c.publish(config.COMMAND_TOPIC_FMT.format(device_id=config.DEVICE_ID), json.dumps(env), qos=1).wait_for_publish(5)
time.sleep(0.5)
c.loop_stop()
print("sent:", json.dumps(env["payload"]))
