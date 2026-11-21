"""Milestone test: can this machine reach the MQTT broker?
  python hello_publish.py <broker_ip>      (run on the UNO Q)
Watch on the laptop:  mosquitto_sub -h localhost -t "coldchain/#" -v
"""
import sys
import time
import paho.mqtt.client as mqtt

host = sys.argv[1] if len(sys.argv) > 1 else "localhost"
c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="hello_test")
c.connect(host, 1883, 10)
c.loop_start()
for i in range(3):
    info = c.publish("coldchain/hello", f"hello #{i} from {sys.platform}", qos=1)
    info.wait_for_publish(5)
    print("published", i)
    time.sleep(1)
c.loop_stop()
c.disconnect()
print("OK: broker reachable")
