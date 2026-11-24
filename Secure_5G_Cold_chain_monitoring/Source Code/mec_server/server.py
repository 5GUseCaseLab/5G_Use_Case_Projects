"""MEC server: verify telemetry -> decide cooling -> publish signed commands."""
import json
import logging
import time

import paho.mqtt.client as mqtt

import config
import dashboard
import security
from controller import CoolingController
from logger import LatencyCsv, TelemetryCsv, setup_logging

log = logging.getLogger("mec")


class MecServer:
    def __init__(self):
        self.guard = security.TelemetryGuard()
        self.counter = security.CommandCounter()
        self.csv = TelemetryCsv()
        self.latency_csv = LatencyCsv()
        self.sent_at = {}            # nonce -> time the command was handed to MQTT
        self.dash = dashboard.DashboardState()
        self.controllers = {}
        self.last_sent = {}          # device -> (pwm, time)
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="mec_server")
        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect
        self.client.on_message = self.on_message
        self.client.reconnect_delay_set(1, 30)

    # ---- MQTT callbacks
    def on_connect(self, client, userdata, flags, reason_code, properties):
        if reason_code != 0:
            log.error("MQTT connect refused: %s", reason_code)
            return
        log.info("Connected to broker %s:%s", config.BROKER_HOST, config.BROKER_PORT)
        self.dash.set_broker(True)
        client.subscribe([(config.TELEMETRY_SUB, 1), (config.EVENTS_SUB, 1)])  # re-subscribe on every reconnect

    def on_disconnect(self, client, userdata, flags, reason_code, properties):
        log.warning("Disconnected from broker (%s); will retry", reason_code)
        self.dash.set_broker(False)

    def on_message(self, client, userdata, msg):
        try:
            parts = msg.topic.split("/")
            if len(parts) != 3 or parts[0] != "coldchain":
                return
            device, kind = parts[1], parts[2]
            if kind == "telemetry":
                self.handle_telemetry(device, msg.payload)
            elif kind == "events":
                self.handle_event(device, msg.payload)
        except Exception:
            log.exception("Unexpected error handling %s", msg.topic)   # never crash the loop

    # ---- handlers
    def handle_telemetry(self, topic_device, raw):
        recv = time.time()
        try:
            payload = security.verify_envelope(raw)
            if payload.get("device_id") != topic_device:
                raise security.VerificationError("topic/device_id mismatch")
            self.guard.check_and_accept(payload)
        except security.VerificationError as e:
            log.warning("REJECTED telemetry on %s: %s", topic_device, e)
            self.dash.rejected("telemetry", str(e))
            return
        dev = payload["device_id"]
        ctrl = self.controllers.setdefault(dev, CoolingController())
        pwm, alarm = ctrl.update(payload["temperature"])
        self.csv.write(payload, pwm, alarm)
        self.dash.telemetry_ok(payload, pwm, alarm)
        ts = payload["timestamp"]
        if isinstance(ts, float):                     # millisecond timestamp from the board
            delay_ms = (recv - ts) * 1000.0
            self.dash.add_latency("oneway", delay_ms)
            self.latency_csv.write("oneway", delay_ms, payload["seq"])
        log.info("ACCEPTED seq=%d temp=%.2fC -> pwm=%d%s", payload["seq"],
                 payload["temperature"], pwm, f"  ALARM={alarm}" if alarm else "")
        self.maybe_send_command(dev, pwm)

    def handle_event(self, topic_device, raw):
        recv = time.time()
        try:
            payload = security.verify_envelope(raw)
            security.check_event_fresh(payload)
        except security.VerificationError as e:
            log.warning("REJECTED event on %s: %s", topic_device, e)
            self.dash.rejected("event", str(e))
            return
        log.info("DEVICE EVENT: %s", {k: v for k, v in payload.items() if k != "device_id"})
        self.dash.board_event(payload)
        if payload.get("event") == "command_accepted":
            t0 = self.sent_at.pop(payload.get("nonce"), None)
            if t0 is not None:                        # round-trip: server publish -> board verify/persist/actuate -> event back
                rtt = (recv - t0) * 1000.0
                self.dash.add_latency("rtt", rtt)
                self.latency_csv.write("rtt", rtt, payload.get("nonce"))
                log.info("LATENCY command round-trip nonce=%s: %.0f ms", payload.get("nonce"), rtt)

    def maybe_send_command(self, dev, pwm):
        prev = self.last_sent.get(dev)
        due = prev is None or prev[0] != pwm or time.time() - prev[1] >= config.COMMAND_HEARTBEAT_S
        if not due:
            return
        try:
            nonce = self.counter.next()          # persisted before use
        except OSError:
            log.exception("Cannot persist nonce; NOT sending command (fail closed)")
            return
        env = security.make_command(dev, nonce, pwm)
        topic = config.COMMAND_TOPIC_FMT.format(device_id=dev)
        t_pub = time.time()
        self.client.publish(topic, json.dumps(env), qos=1)
        self.sent_at[nonce] = t_pub
        for k in [k for k, t in self.sent_at.items() if t_pub - t > 120]:
            del self.sent_at[k]                        # forget commands that were never acknowledged
        self.last_sent[dev] = (pwm, time.time())
        self.dash.command_sent(nonce, pwm)
        log.info("SENT command nonce=%d pwm=%d", nonce, pwm)

    def run(self):
        dashboard.start(self.dash)
        log.info("Dashboard: http://%s:%d", dashboard.HOST, dashboard.PORT)
        self.client.connect_async(config.BROKER_HOST, config.BROKER_PORT, keepalive=30)
        self.client.loop_forever(retry_first_connection=True)


if __name__ == "__main__":
    setup_logging()
    log.info("MEC server starting (broker %s:%s)", config.BROKER_HOST, config.BROKER_PORT)
    try:
        MecServer().run()
    except KeyboardInterrupt:
        log.info("Stopped.")
