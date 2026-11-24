"""UNO Q application: sensor -> signed MQTT telemetry; verified commands -> cooling output.

Run standalone:        python main.py
Inside Arduino App Lab: the same file works; it uses arduino.app_utils.App if present.
"""
import json
import logging
import os
import threading
import time

import paho.mqtt.client as mqtt

import config
import hardware
import security
import telemetry

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("uno_q")

store = security.NonceStore()
builder = telemetry.TelemetryBuilder()
lock = threading.Lock()                 # protects verify -> persist -> actuate as one unit
client = None
state = {"last_contact": time.time(), "next_sample": 0.0, "failsafe": False,
         "pwm": 0, "sensor_fault": False, "fault_reported": False}


# ------------------------------------------------------------ MQTT
def publish_event(event, **detail):
    if client and client.is_connected():
        client.publish(config.EVENTS_TOPIC, json.dumps(telemetry.build_event(event, **detail)), qos=1)


def on_connect(c, userdata, flags, reason_code, properties):
    if reason_code != 0:
        log.error("MQTT connect refused: %s", reason_code)
        return
    log.info("Connected to broker %s:%s", config.BROKER_HOST, config.BROKER_PORT)
    c.subscribe(config.COMMAND_TOPIC, qos=1)          # re-subscribe after every reconnect


def on_disconnect(c, userdata, flags, reason_code, properties):
    log.warning("Disconnected from broker (%s); will retry", reason_code)


def on_message(c, userdata, msg):
    now = time.time()
    with lock:
        try:
            cmd = security.verify_command(msg.payload, store, now)
            store.persist(cmd["nonce"])                 # persist BEFORE actuating
        except security.CommandRejected as e:
            log.warning("COMMAND REJECTED: %s", e)
            publish_event("command_rejected", reason=str(e))
            return
        except security.StateError as e:
            log.error("FAIL CLOSED (state): %s", e)
            publish_event("command_rejected", reason=f"state error: {e}")
            return
        if not apply_cooling(cmd["pwm"]):
            publish_event("actuator_error", pwm=cmd["pwm"])
            return
        state["last_contact"] = now
        if state["failsafe"]:
            log.info("MEC contact restored; leaving fail-safe")
            state["failsafe"] = False
        log.info("COMMAND ACCEPTED nonce=%d pwm=%d", cmd["nonce"], cmd["pwm"])
        publish_event("command_accepted", nonce=cmd["nonce"], pwm=cmd["pwm"])


# ------------------------------------------------------------ sampling
def apply_cooling(pwm):
    """Set the output and remember it. Returns False if the actuator could not be set.
    (If this fails the sketch's own watchdog still switches the output off.)"""
    try:
        hardware.set_cooling_power(pwm)
    except Exception:
        log.exception("Actuator error")
        return False
    state["pwm"] = pwm
    return True


def sample(now):
    try:
        temp = hardware.read_temperature()
    except hardware.SensorError as e:
        with lock:                                      # no trustworthy reading -> never keep cooling blind
            log.error("SENSOR FAULT: %s -> cooling OFF, no telemetry sent", e)
            apply_cooling(0)
            state["sensor_fault"] = True
            if not state["fault_reported"] and client.is_connected():
                publish_event("sensor_fault", reason=str(e))     # once per fault, when the broker is reachable
                state["fault_reported"] = True
        return
    with lock:
        if state["sensor_fault"]:
            log.info("Sensor recovered")
            state["sensor_fault"] = False
            state["fault_reported"] = False
        if now - state["last_contact"] > config.MEC_TIMEOUT_S:
            if not state["failsafe"]:
                log.error("No valid MEC command for %ds: LOCAL FAIL-SAFE active", config.MEC_TIMEOUT_S)
                state["failsafe"] = True
            state["pwm"] = config.FAILSAFE_PWM if temp >= config.FAILSAFE_TEMP_C else 0   # local decision
        apply_cooling(state["pwm"])                     # keep-alive: refreshes the sketch watchdog every sample

    if client.is_connected():
        env = builder.build(temp, now)
        client.publish(config.TELEMETRY_TOPIC, json.dumps(env), qos=1)
        log.info("TELEMETRY seq=%d temp=%.2fC pwm=%d", env["payload"]["seq"], temp, state["pwm"])
    else:
        log.warning("Broker not connected; telemetry skipped (temp=%.2fC)", temp)


def setup():
    global client
    if not store.ready:
        log.error("Nonce state is '%s' at %s -> ALL COMMANDS WILL BE REJECTED. "
                  "Run: python provision_state.py", store.status, store.path)
    log.info("Sensor: %s | Cooling: %s", "SIMULATED" if config.SENSOR_SIMULATION else "REAL (Bridge)",
             "SIMULATED" if config.COOLING_SIMULATION else "REAL (Bridge)")
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"{config.DEVICE_ID}")
    client.on_connect, client.on_disconnect, client.on_message = on_connect, on_disconnect, on_message
    client.reconnect_delay_set(1, 30)
    client.connect_async(config.BROKER_HOST, config.BROKER_PORT, keepalive=30)
    client.loop_start()                                 # network runs in a background thread


def loop():
    now = time.time()
    if now >= state["next_sample"]:
        state["next_sample"] = now + config.SAMPLE_PERIOD_S
        try:
            sample(now)
        except Exception:
            log.exception("sample() failed")
    time.sleep(0.1)


if __name__ == "__main__":
    setup()
    app = None
    if not os.environ.get("COLDCHAIN_STANDALONE"):
        try:
            from arduino.app_utils import App
            app = App
        except ImportError:
            app = None
    if app is not None:
        app.run(user_loop=loop)
    else:
        try:
            while True:
                loop()
        except KeyboardInterrupt:
            log.info("Stopped.")
