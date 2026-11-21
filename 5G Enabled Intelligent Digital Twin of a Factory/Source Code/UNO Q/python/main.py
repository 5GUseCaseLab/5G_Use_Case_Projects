from arduino.app_utils import App, Bridge
import paho.mqtt.client as mqtt
import threading
import socket
import ssl
import json
import time

# ============================================================
# AWS IoT & UDP Configuration
# ============================================================
AWS_ENDPOINT = "a1s58c00pf7oul-ats.iot.eu-north-1.amazonaws.com"
AWS_PORT = 8883
CLIENT_ID = "UNO_Q_Linux_MPU"
PUBLISH_TOPIC = "esp32s3/pub"
SUBSCRIBE_TOPIC = "esp32s3/sub"

UDP_LISTEN_IP = "0.0.0.0"
UDP_LISTEN_PORT = 5005

UNO_Q_HOST_IP = "10.153.9.109"
UDP_DOWNLINK_PORT = 5006

CA_PATH = "/app/python/certs/AmazonRootCA1.pem"
CERT_PATH = "/app/python/certs/certificate.pem.crt"
KEY_PATH = "/app/python/certs/private.pem.key"

NANO_COMMANDS = {
    "FAN:ON", "FAN:OFF",
    "ML:FAN:ON", "ML:FAN:OFF",
    "MOTOR:ON", "MOTOR:OFF"
}

bridge_lock = threading.Lock()
udp_tx_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def to_signed_32(val):
    if val > 0x7FFFFFFF:
        val -= 0x100000000
    return val

# ============================================================
# UDP Listener Thread (Nano -> ESP32 -> UNO Q -> AWS)
# ============================================================
def udp_receiver_loop(mqtt_client):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_LISTEN_IP, UDP_LISTEN_PORT))
    print(f"[UDP] Listening for ESP32-S3 packets on port {UDP_LISTEN_PORT}...")

    while True:
        try:
            data, addr = sock.recvfrom(2048)
            payload_str = data.decode("utf-8").strip()
            if not payload_str:
                continue

            json.loads(payload_str)
            result = mqtt_client.publish(PUBLISH_TOPIC, payload_str, qos=1)
            status = "SENT" if result.rc == mqtt.MQTT_ERR_SUCCESS else f"ERR({result.rc})"
            print(f"[UDP -> AWS {status}] {payload_str}")

        except json.JSONDecodeError:
            print(f"[UDP] Ignored invalid JSON from {addr[0]}")
        except Exception as e:
            print(f"[UDP Error] {e}")

# ============================================================
# Command Router (AWS -> STM32 or AWS -> ESP32 -> Nano)
# ============================================================
def route_aws_command(payload_str):
    payload_str = payload_str.strip()
    if not payload_str:
        return

    # 1. Check if it is a Nano command (forward via UDP to host relay -> ESP32 -> Nano)
    if payload_str in NANO_COMMANDS:
        udp_tx_sock.sendto(payload_str.encode("utf-8"), (UNO_Q_HOST_IP, UDP_DOWNLINK_PORT))
        print(f"[AWS -> NANO] Forwarded: {payload_str}")
        return

    # 2. Check if it is a JSON command for UNO Q STM32 or Nano
    try:
        data = json.loads(payload_str)
        if isinstance(data, dict):
            if "nano_cmd" in data and str(data["nano_cmd"]).strip() in NANO_COMMANDS:
                cmd = str(data["nano_cmd"]).strip()
                udp_tx_sock.sendto(cmd.encode("utf-8"), (UNO_Q_HOST_IP, UDP_DOWNLINK_PORT))
                print(f"[AWS -> NANO] Forwarded: {cmd}")
                return

            if "pwm" in data:
                pwm_val = max(-255, min(255, int(data["pwm"])))
                with bridge_lock:
                    Bridge.call("set_motor_pwm", pwm_val & 0xFFFFFFFF)
                print(f"[AWS -> STM32] PWM set to {pwm_val}")
                return

            if "cmd" in data:
                cmd = str(data["cmd"]).strip().upper()
                if cmd == "ON":
                    with bridge_lock:
                        Bridge.call("set_motor_on", 1)
                    print("[AWS -> STM32] Motor ON")
                elif cmd in ("OFF", "STOP"):
                    with bridge_lock:
                        Bridge.call("set_motor_on", 0)
                    print("[AWS -> STM32] Motor OFF")
                return
    except json.JSONDecodeError:
        pass

    # 3. Fallback plain-text commands for UNO Q STM32
    cmd = payload_str.upper()
    if cmd == "ON":
        with bridge_lock:
            Bridge.call("set_motor_on", 1)
        print("[AWS -> STM32] Motor ON")
    elif cmd in ("OFF", "STOP"):
        with bridge_lock:
            Bridge.call("set_motor_on", 0)
        print("[AWS -> STM32] Motor OFF")
    elif cmd.startswith("PWM:"):
        try:
            pwm_val = max(-255, min(255, int(cmd[4:])))
            with bridge_lock:
                Bridge.call("set_motor_pwm", pwm_val & 0xFFFFFFFF)
            print(f"[AWS -> STM32] PWM set to {pwm_val}")
        except ValueError:
            print(f"[AWS CMD] Invalid PWM: {payload_str}")
    else:
        print(f"[AWS CMD] Unknown command: {payload_str}")

# ============================================================
# MQTT Callbacks
# ============================================================
def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print("[AWS] Connected to AWS IoT Core!")
        client.subscribe(SUBSCRIBE_TOPIC, qos=1)
        print(f"[AWS] Subscribed to {SUBSCRIBE_TOPIC}")
    else:
        print(f"[AWS] Connection failed with code: {rc}")

def on_message(client, userdata, msg):
    try:
        payload = msg.payload.decode("utf-8")
        print(f"[AWS RX] {payload}")
        route_aws_command(payload)
    except Exception as e:
        print(f"[AWS RX Error] {e}")

def create_mqtt_client():
    if hasattr(mqtt, "CallbackAPIVersion"):
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=CLIENT_ID)
    else:
        client = mqtt.Client(client_id=CLIENT_ID)

    client.tls_set(
        ca_certs=CA_PATH,
        certfile=CERT_PATH,
        keyfile=KEY_PATH,
        tls_version=ssl.PROTOCOL_TLSv1_2
    )
    client.on_connect = on_connect
    client.on_message = on_message
    return client

# ============================================================
# Main Telemetry Loop
# ============================================================
def telemetry_loop():
    print("Initializing AWS IoT MQTT Client...")
    client = create_mqtt_client()

    try:
        client.connect(AWS_ENDPOINT, AWS_PORT, keepalive=60)
        client.loop_start()
    except Exception as e:
        print(f"[ERROR] AWS connection failed: {e}")
        return

    threading.Thread(target=udp_receiver_loop, args=(client,), daemon=True).start()
    print("UNO Q Telemetry + Bidirectional UDP Bridge + AWS Control Active...")

    while True:
        try:
            with bridge_lock:
                raw_rpm = Bridge.call("get_rpm", 0)
                raw_enc = Bridge.call("get_enc", 0)
                raw_temp = Bridge.call("get_temp", 0)
                raw_hum = Bridge.call("get_hum", 0)
                raw_curr = Bridge.call("get_current", 0)

            rpm = to_signed_32(raw_rpm) / 100.0
            enc = to_signed_32(raw_enc)
            temp = raw_temp / 10.0
            hum = raw_hum / 10.0
            current_a = raw_curr / 1000.0

            payload = {
                "device_id": "UNO-Q-MPU",
                "motor_rpm": round(rpm, 2),
                "motor_current_a": round(current_a, 3),
                "encoder_pos": enc,
                "temperature_c": round(temp, 1),
                "humidity_pct": round(hum, 1),
                "timestamp_s": int(time.time())
            }
            json_payload = json.dumps(payload)

            result = client.publish(PUBLISH_TOPIC, json_payload, qos=1)
            status = "SENT" if result.rc == mqtt.MQTT_ERR_SUCCESS else f"ERR({result.rc})"
            print(f"[AWS {status}] {json_payload}")

        except Exception as e:
            print(f"Telemetry error: {e}")

        time.sleep(10)

App.run(user_loop=telemetry_loop)