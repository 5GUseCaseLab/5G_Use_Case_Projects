# =========================================================================
# 5G WEARABLE ASSISTIVE DEVICE - LINUX MPU CLIENT (ARDUINO UNO Q)
# Communicates with:
#   1. STM32 MCU via RouterBridge RPC / Serial (Live HC-SR04 Ultrasonic)
#   2. USB Camera (/dev/video*) with deep hardware diagnostic reporting
#   3. Edge Server on Laptop via Cloudflare Tunnel / UDP
# =========================================================================

import os
# Silence internal OpenCV / GStreamer C++ warnings when probing devices
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
os.environ["OPENCV_VIDEOIO_DEBUG"] = "0"

import socket
import json
import time
import cv2
import threading

if hasattr(cv2, "setLogLevel"):
    cv2.setLogLevel(0)

# --- 1. NETWORK CONFIGURATION ---
# Optional: Hardcode your Cloudflare URL if desired, OR leave empty to use
# 100% Autonomous Cloud Sync (Auto-discovers the active server with zero field edits).
CLOUDFLARE_URL   = "" 

def load_manual_url():
    """Reads manual Cloudflare URL override from CLOUDFLARE_URL or local tunnel_url.txt."""
    if CLOUDFLARE_URL and CLOUDFLARE_URL.strip():
        return CLOUDFLARE_URL.strip()
    try:
        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tunnel_url.txt")
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                u = f.read().strip()
                return u if u.startswith("https://") else ""
    except Exception:
        pass
    return "" 

# Shared Cloud Sync Topic (Matches edge_server/tunnel_manager.py for zero-touch discovery)
CLOUD_SYNC_TOPIC = "iitg_5g_wearable_nav_sync_9169957"

def fetch_active_cloud_url(timeout=3.0):
    """
    Auto-discovers the active Edge Server Cloudflare URL from the Cloud Sync topic.
    Eliminates the need to ever open Arduino App Lab or manually edit URLs in the field.
    """
    import urllib.request
    import urllib.error
    try:
        req = urllib.request.Request(
            f"https://ntfy.sh/{CLOUD_SYNC_TOPIC}/json?poll=1&since=45m",
            headers={"User-Agent": "Wearable-Arduino-Client"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8").strip()
            for line in reversed(body.split("\n")):
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    if data.get("event") == "message":
                        url = data.get("message", "").strip()
                        if url.startswith("http://") or url.startswith("https://"):
                            return url
                except Exception:
                    pass
    except urllib.error.HTTPError as e:
        if e.code == 429:
            print("⚠️ [CLOUD SYNC]: ntfy rate limit hit — slowing down discovery.")
        return None
    except Exception:
        pass
    return None

# Local Network Fallback IP (only used if Cloudflare / Cloud Sync is unavailable)
EDGE_SERVER_IP   = "192.168.0.108"
VIDEO_PORT       = 5005     # UDP video stream to Edge Server
TELEMETRY_PORT   = 5006     # UDP distance telemetry to Edge Server
DOWNLINK_PORT    = 5007     # UDP incoming obstacle alerts from Edge Server

# Global hardware state (Pure Real-World Hardware - Zero Synthetic Noise)
current_distance_cm  = 400.0
last_distance_update = 0.0
bridge_instance      = None
serial_device        = None
hardware_connected   = False

# -------------------------------------------------------------------------
# Zero-Dependency RFC 6455 WebSocket Client (Pure Python socket + ssl)
# Enables cross-network video & telemetry streaming over Cloudflare Tunnel
# -------------------------------------------------------------------------
import socket
import ssl
import base64

class RawWebSocketClient:
    """Zero-dependency WebSocket client for cross-network 5G uplink."""
    def __init__(self, url):
        self.url = url
        self.sock = None
        self.is_connected = False
        self._parse_url()

    def _parse_url(self):
        url = self.url.strip()
        if url.startswith("wss://"):
            self.use_ssl = True
            rem = url[6:]
        elif url.startswith("ws://"):
            self.use_ssl = False
            rem = url[5:]
        elif url.startswith("https://"):
            self.use_ssl = True
            rem = url[8:]
        elif url.startswith("http://"):
            self.use_ssl = False
            rem = url[7:]
        else:
            self.use_ssl = True
            rem = url

        parts = rem.split("/", 1)
        host_port = parts[0].split(":")
        self.host = host_port[0]
        self.port = int(host_port[1]) if len(host_port) > 1 else (443 if self.use_ssl else 80)
        self.path = "/" + parts[1] if len(parts) > 1 else "/ws"
        if not self.path.endswith("/ws"):
            self.path = self.path.rstrip("/") + "/ws"

    def connect(self):
        try:
            raw_sock = socket.create_connection((self.host, self.port), timeout=3.0)
            if self.use_ssl:
                ctx = ssl.create_default_context()
                self.sock = ctx.wrap_socket(raw_sock, server_hostname=self.host)
            else:
                self.sock = raw_sock

            key = base64.b64encode(os.urandom(16)).decode('utf-8')
            handshake = (
                f"GET {self.path} HTTP/1.1\r\n"
                f"Host: {self.host}\r\n"
                f"Upgrade: websocket\r\n"
                f"Connection: Upgrade\r\n"
                f"Sec-WebSocket-Key: {key}\r\n"
                f"Sec-WebSocket-Version: 13\r\n\r\n"
            )
            self.sock.sendall(handshake.encode('utf-8'))

            response = b""
            while b"\r\n\r\n" not in response:
                chunk = self.sock.recv(1024)
                if not chunk:
                    break
                response += chunk

            status_line = response.split(b"\r\n", 1)[0]
            if b" 101 " in status_line:
                self.is_connected = True
                try:
                    self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                except Exception:
                    pass
                try:
                    self.send_text(json.dumps({"type": "hello", "role": "wearable"}))
                except Exception:
                    pass
                return True
        except Exception:
            self.is_connected = False
            if self.sock:
                try: self.sock.close()
                except Exception: pass
                self.sock = None
        return False

    def send_binary(self, data_bytes):
        if not self.is_connected or not self.sock:
            return False
        return self._send_frame(0x02, data_bytes)

    def send_text(self, text_str):
        if not self.is_connected or not self.sock:
            return False
        return self._send_frame(0x01, text_str.encode('utf-8'))

    def _send_frame(self, opcode, payload):
        try:
            length = len(payload)
            header = bytearray()
            header.append(0x80 | (opcode & 0x0F))

            if length < 126:
                header.append(0x80 | length)
            elif length < 65536:
                header.append(0x80 | 126)
                header.extend(length.to_bytes(2, 'big'))
            else:
                header.append(0x80 | 127)
                header.extend(length.to_bytes(8, 'big'))

            mask = os.urandom(4)
            header.extend(mask)

            masked = bytearray(length)
            for i in range(length):
                masked[i] = payload[i] ^ mask[i % 4]

            self.sock.sendall(header + masked)
            return True
        except Exception:
            self.is_connected = False
            if self.sock:
                try: self.sock.close()
                except Exception: pass
                self.sock = None
            return False

    def close(self):
        self.is_connected = False
        if self.sock:
            try: self.sock.close()
            except Exception: pass
            self.sock = None

# -------------------------------------------------------------------------
# Connect to the STM32 MCU (RouterBridge Primary, Serial Fallback)
# -------------------------------------------------------------------------
def init_hardware_link():
    global bridge_instance, serial_device, hardware_connected

    # Method 1: Arduino UNO Q App Lab Bridge (arduino.app_utils)
    try:
        from arduino.app_utils import Bridge as AppBridge
        def bridge_worker():
            global current_distance_cm, last_distance_update
            errors = 0
            while True:
                try:
                    val = AppBridge.call("get_distance")
                    if val is not None:
                        current_distance_cm = float(val)
                        last_distance_update = time.time()
                        errors = 0
                except Exception as e:
                    errors += 1
                    if errors == 1 or errors % 100 == 0:
                        print(f"⚠️ [ULTRASONIC RPC ERROR x{errors}]: {e}")
                time.sleep(0.10)
        threading.Thread(target=bridge_worker, daemon=True).start()
        hardware_connected = True
        print("🔌 [HARDWARE LINK]: Ultrasonic via App Lab Bridge")
        return
    except ImportError:
        print("ℹ️ arduino.app_utils not available. Attempting serial fallback...")
    except Exception as e:
        print(f"ℹ️ App Lab Bridge note: {e}. Attempting serial fallback...")

    # Method 2: Serial Port Fallback (Reads JSON {"dist": XX.X} directly)
    import glob
    try:
        import serial
        potential_ports = (
            glob.glob("/dev/ttyACM*") + 
            glob.glob("/dev/ttyUSB*") + 
            glob.glob("/dev/ttyHS*") + 
            glob.glob("/dev/ttyMSM*") + 
            glob.glob("/dev/ttyS[0-9]*") + 
            [f"COM{i}" for i in range(1, 15)]
        )
        for port in potential_ports:
            try:
                ser = serial.Serial(port, 115200, timeout=0.1)
                t_end = time.time() + 2.0
                got = False
                while time.time() < t_end:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                    if line.startswith('{"dist"'):
                        got = True
                        break
                if not got:
                    ser.close()
                    continue

                print(f"🔌 [HARDWARE LINK]: Linked to STM32 MCU on Serial port {port}")
                serial_device = ser
                hardware_connected = True
                
                # Dedicated thread to keep serial buffer drained and distance fresh
                def serial_worker():
                    global current_distance_cm, last_distance_update
                    serial_errors = 0
                    while True:
                        try:
                            line = ser.readline().decode('utf-8', errors='ignore').strip()
                            if line.startswith("{") and line.endswith("}"):
                                data = json.loads(line)
                                if "dist" in data:
                                    current_distance_cm = float(data["dist"])
                                    last_distance_update = time.time()
                                    serial_errors = 0
                        except Exception as se:
                            serial_errors += 1
                            if serial_errors == 1 or serial_errors % 100 == 0:
                                print(f"⚠️ [SERIAL READ ERROR x{serial_errors}]: {se}")
                            time.sleep(0.02)
                
                threading.Thread(target=serial_worker, daemon=True).start()
                return
            except Exception:
                continue
    except Exception:
        pass

    if not hardware_connected:
        print("⚠️ [HARDWARE NOTICE]: Waiting for STM32 MCU heartbeat...")

# -------------------------------------------------------------------------
# Read Real Ultrasonic Distance from MCU (Non-blocking background read)
# -------------------------------------------------------------------------
def read_ultrasonic_distance():
    if time.time() - last_distance_update > 1.0:
        return None                       # sensor offline → server ignores it instead of "clear"
    return round(current_distance_cm, 1)

# -------------------------------------------------------------------------
# Comprehensive Hot-Pluggable Camera Detector with Raw Kernel Diagnostics
# -------------------------------------------------------------------------
def detect_camera():
    import glob, subprocess

    devices = sorted(glob.glob("/dev/video*"))
    
    # Periodic diagnostic print to inform the user exactly what the OS sees
    now = time.time()
    if now - getattr(detect_camera, "_last_diag_time", 0) >= 5.0:
        detect_camera._last_diag_time = now
        
        # 1. Enumerate video nodes and their driver names from sysfs
        video_info = []
        for dev in devices:
            node_name = os.path.basename(dev)
            name_file = f"/sys/class/video4linux/{node_name}/name"
            label = dev
            if os.path.exists(name_file):
                try:
                    with open(name_file, "r") as f:
                        label = f"{dev} ({f.read().strip()})"
                except Exception:
                    pass
            video_info.append(label)

        # 2. Enumerate USB devices from sysfs (zero external binaries required)
        usb_devices = []
        for prod_path in glob.glob("/sys/bus/usb/devices/*/product"):
            try:
                dev_dir = os.path.dirname(prod_path)
                with open(prod_path, "r") as f:
                    prod = f.read().strip()
                mfg = ""
                mfg_path = os.path.join(dev_dir, "manufacturer")
                if os.path.exists(mfg_path):
                    with open(mfg_path, "r") as f:
                        mfg = f.read().strip()
                usb_devices.append(f"{mfg} {prod}".strip())
            except Exception:
                pass

        # 3. Check and attempt to enforce USB Host Mode Role
        roles = []
        for role_path in glob.glob("/sys/class/usb_role/*/role"):
            try:
                with open(role_path, "r+") as f:
                    curr = f.read().strip()
                    roles.append(curr)
                    if curr != "host":
                        try:
                            f.seek(0)
                            f.write("host")
                            roles.append("-> switched to host")
                        except Exception as write_err:
                            roles.append(f"(write err: {write_err})")
            except Exception:
                pass

        print(f"🔍 [HARDWARE SCAN]: Video nodes: {video_info if video_info else 'None'}")
        print(f"   ↳ USB Bus Devices : {', '.join(usb_devices) if usb_devices else 'No USB devices enumerated by kernel'}")
        print(f"   ↳ USB Port Role   : {roles if roles else 'Unknown'}")

    # Exclude internal Qualcomm Venus encoder/decoder nodes (video0, video1)
    # External UVC webcams always enumerate on video2, video3, etc.
    camera_candidates = [d for d in devices if d not in ("/dev/video0", "/dev/video1")]
    if not camera_candidates:
        # Fallback to checking video2 directly
        camera_candidates = ["/dev/video2", "/dev/video3"]

    for dev in camera_candidates:
        suffix = dev.replace("/dev/video", "")
        dev_idx = int(suffix) if suffix.isdigit() else 2

        # Try opening via V4L2 backend and CAP_ANY
        open_attempts = [
            (dev_idx, getattr(cv2, "CAP_V4L2", cv2.CAP_ANY)),
            (dev_idx, cv2.CAP_ANY),
            (dev, getattr(cv2, "CAP_V4L2", cv2.CAP_ANY)),
            (dev, cv2.CAP_ANY)
        ]

        for target, backend in open_attempts:
            try:
                cap = cv2.VideoCapture(target, backend) if isinstance(target, int) else cv2.VideoCapture(target)
                if cap.isOpened():
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None:
                        print(f"📷 [CAMERA ONLINE]: Successfully streaming video from {dev} (Index {dev_idx})!")
                        return cap
                    cap.release()
            except Exception:
                continue

    return None

# -------------------------------------------------------------------------
# Main Execution Loop
# -------------------------------------------------------------------------
def main():
    print("==================================================")
    print("   5G WEARABLE ASSISTIVE DEVICE - ARDUINO UNO Q   ")
    print("==================================================")
    print(f"📡 Target Edge Server IP  : {EDGE_SERVER_IP}")
    print(f"🎥 Video Uplink Port      : {VIDEO_PORT} (UDP)")
    print(f"🧭 Telemetry Uplink Port  : {TELEMETRY_PORT} (UDP)")
    print(f"👂 Downlink Listen Port   : {DOWNLINK_PORT} (UDP)")
    print("==================================================")

    # 1. Establish hardware link with STM32 MCU
    init_hardware_link()

    # 2. Setup UDP transmission sockets
    uplink_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    downlink_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        downlink_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        downlink_sock.bind(("0.0.0.0", DOWNLINK_PORT))
        
        def downlink_worker():
            while True:
                try:
                    data, _ = downlink_sock.recvfrom(2048)
                    alert = json.loads(data.decode('utf-8'))
                    print(f"\n🚨 [SERVER ALERT]: {alert.get('text', 'Obstacle warning')}")
                except Exception:
                    time.sleep(0.01)
                    
        threading.Thread(target=downlink_worker, daemon=True).start()
    except Exception as e:
        print(f"ℹ️ Downlink UDP socket info: {e}. Continuing.")

    # 3. Detect onboard/USB camera
    cap = detect_camera()
    last_camera_retry_time = 0.0
    last_telemetry_time = 0.0
    last_ws_reconnect_time = 0.0
    last_frame_sent = 0.0
    skip_frames_until = 0.0

    # Video compression & rate limits matched to Edge Server YOLO 416
    VIDEO_FPS = 8                  # frames actually sent per second
    JPEG_QUALITY = 50
    FRAME_SIZE = (416, 277)        # matches the server's YOLO input size

    # Cross-Network Cloudflare Uplink Client with Autonomous Cloud Sync
    manual_override = load_manual_url()
    active_cloud_url = manual_override if manual_override else None
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cache_file = os.path.join(script_dir, "last_known_tunnel_url.txt")

    if not active_cloud_url:
        print("🔍 [CLOUD SYNC]: Searching for active 5G Edge Server...")
        discovered = fetch_active_cloud_url(timeout=3.0)
        if discovered:
            active_cloud_url = discovered
            print(f"✨ [CLOUD SYNC LOCKED]: Auto-discovered Edge Server: {active_cloud_url}")
            try:
                with open(cache_file, "w") as f:
                    f.write(active_cloud_url)
            except Exception:
                pass
        elif os.path.exists(cache_file):
            try:
                with open(cache_file, "r") as f:
                    cached = f.read().strip()
                    if cached.startswith("https://"):
                        active_cloud_url = cached
                        print(f"📂 [CLOUD SYNC CACHE]: Using last known URL: {active_cloud_url}")
            except Exception:
                pass
    else:
        print(f"🔒 [MANUAL URL LOCKED]: Using static Cloudflare URL: {active_cloud_url}")

    use_cloudflare = bool(active_cloud_url)
    ws_client = None
    if use_cloudflare:
        ws_client = RawWebSocketClient(active_cloud_url)
        print(f"🌐 [5G CLOUDFLARE MODE]: Connecting to {active_cloud_url}...")
        ws_client.connect()
    else:
        print(f"📡 [LOCAL UDP MODE]: Streaming to {EDGE_SERVER_IP}:{VIDEO_PORT}")

    print("\n🟢 5G Wearable Client running! Press Ctrl+C to terminate.\n")

    try:
        while True:
            now = time.time()
            dist_cm = read_ultrasonic_distance()

            # Auto-reconnect Cloudflare WebSocket if connection drops
            if not ws_client or not ws_client.is_connected:
                manual_url = load_manual_url()
                if manual_url:
                    if now - last_ws_reconnect_time >= 4.0:
                        last_ws_reconnect_time = now
                        if not ws_client or getattr(ws_client, "url", "") != manual_url:
                            if ws_client:
                                ws_client.close()
                            ws_client = RawWebSocketClient(manual_url)
                            use_cloudflare = True
                        ws_client.connect()
                else:
                    if now - last_ws_reconnect_time >= 10.0:
                        last_ws_reconnect_time = now
                        # Check if server announced a URL or changed URL
                        discovered = fetch_active_cloud_url(timeout=5.0)
                        if discovered:
                            current_url = getattr(ws_client, "url", "") if ws_client else ""
                            if discovered != current_url:
                                print(f"✨ [CLOUD SYNC]: Linked to Edge Server: {discovered}")
                                if ws_client:
                                    ws_client.close()
                                ws_client = RawWebSocketClient(discovered)
                                use_cloudflare = True
                                try:
                                    with open(cache_file, "w") as f:
                                        f.write(discovered)
                                except Exception:
                                    pass
                        if ws_client:
                            ws_client.connect()

            # --- 1. SENSOR TELEMETRY UPLINK (10 Hz / every 100ms) - ALWAYS SENT FIRST ---
            if now - last_telemetry_time >= 0.1:
                if use_cloudflare and ws_client and ws_client.is_connected:
                    telemetry_msg = json.dumps({
                        "type": "telemetry",
                        "ultrasonic_cm": dist_cm,
                        "timestamp": now
                    })
                    ws_client.send_text(telemetry_msg)
                elif not use_cloudflare:
                    telemetry = {
                        "ultrasonic_cm": dist_cm,
                        "timestamp": now
                    }
                    payload_json = json.dumps(telemetry).encode('utf-8')
                    try:
                        uplink_sock.sendto(payload_json, (EDGE_SERVER_IP, TELEMETRY_PORT))
                    except Exception:
                        pass

                last_telemetry_time = now
                if now - getattr(main, "_last_telemetry_print", 0) >= 2.0:
                    main._last_telemetry_print = now
                    dest = f"Cloudflare Tunnel ({'Connected' if (ws_client and ws_client.is_connected) else 'Connecting...'})" if use_cloudflare else f"UDP {EDGE_SERVER_IP}:{TELEMETRY_PORT}"
                    cam_status = "Online" if (cap is not None and cap.isOpened()) else "Standby (Detecting USB Camera...)"
                    dist_str = f"{dist_cm:.1f} cm" if dist_cm is not None else "-- (sensor offline)"
                    print(f"📡 [5G Telemetry ➔ {dest}] Distance: {dist_str} | Camera: {cam_status}", flush=True)

            # --- 2. CAMERA STREAMING WITH UPLINK PACING & BACKPRESSURE (Fix 2 & Fix 3) ---
            if cap is not None and cap.isOpened():
                if (now - last_frame_sent >= 1.0 / VIDEO_FPS) and (now >= skip_frames_until):
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        frame_resized = cv2.resize(frame, FRAME_SIZE)
                        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
                        result, encimg = cv2.imencode('.jpg', frame_resized, encode_param)
                        if result:
                            payload = encimg.tobytes()
                            if use_cloudflare and ws_client and ws_client.is_connected:
                                t0 = time.time()
                                ok_send = ws_client.send_binary(payload)
                                send_ms = (time.time() - t0) * 1000.0
                                last_frame_sent = now
                                if send_ms > 150.0:
                                    # Network link is congested; skip subsequent video frames to flush TCP queues
                                    skip_frames_until = now + 0.5
                            elif not use_cloudflare and len(payload) < 65000:
                                uplink_sock.sendto(payload, (EDGE_SERVER_IP, VIDEO_PORT))
                                last_frame_sent = now
                    else:
                        # Camera lost connection
                        print("⚠️ Camera frame drop. Re-detecting camera...")
                        cap.release()
                        cap = None
            else:
                # Hot-plug detection: periodically check for newly plugged USB camera
                if now - last_camera_retry_time >= 3.0:
                    last_camera_retry_time = now
                    cap = detect_camera()

            time.sleep(0.02) # ~50 Hz loop pace for tight, non-blocking telemetry & video timing

    except KeyboardInterrupt:
        print("\nStopping Wearable Client...")
    finally:
        if ws_client:
            ws_client.close()
        if cap is not None:
            cap.release()
        uplink_sock.close()
        downlink_sock.close()

if __name__ == "__main__":
    main()
