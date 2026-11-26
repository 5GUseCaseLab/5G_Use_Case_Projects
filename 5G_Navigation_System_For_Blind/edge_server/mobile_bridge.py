"""
5G Mobile Bridge Server
=======================
High-performance WebSocket & HTTP gateway for smartphone clients over 5G.
- Serves the static PWA web client (HTML/CSS/JS) to mobile browsers.
- Streams live camera video frames (JPEG) to the Edge AI pipeline.
- Receives GPS, speed, and motion sensor telemetry.
- Ingests microphone audio packets and transcribes voice destinations.
- Pushes turn-by-turn maneuvers, collision threat alerts, and voice prompts.
"""

import asyncio
import json
import time
import os
import sys
import math
import base64
import threading
import collections
import cv2
import numpy as np

# Ensure terminal uses UTF-8
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Ensure sibling packages can be resolved
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from navigation import decode_audio_to_np, transcribe_audio_np

STATIC_CLIENT_DIR = os.path.join(current_dir, "web_client")
INDEX_HTML_PATH = os.path.join(STATIC_CLIENT_DIR, "index.html")
STYLE_CSS_PATH = os.path.join(STATIC_CLIENT_DIR, "style.css")
APP_JS_PATH = os.path.join(STATIC_CLIENT_DIR, "app.js")
MAP_HTML_PATH = os.path.join(current_dir, "live_map.html")
NAV_STATE_PATH = os.path.join(current_dir, "navigation_state.json")

class MobileBridgeServer:
    """
    Dedicated 5G Smartphone Bridge Server.
    Provides non-blocking WebSocket & HTTP communication with the user's mobile device.
    """
    def __init__(self, host="0.0.0.0", port=8080):
        self.host = host
        self.port = port
        
        # Thread-safe buffers
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_frame_time = 0.0
        
        self.latest_telemetry = {
            "lat": None,
            "lon": None,
            "speed_kmh": 0.0,
            "accuracy_m": 999.0,
            "is_walking": False,
            "source": "none",
            "timestamp": 0.0
        }
        self.gps_history = collections.deque(maxlen=15)
        self.ptt_button_pressed = False
        self.is_voice_recording = False
        self.voice_recording_started = 0.0
        self.latest_voice_command = None
        self.latest_voice_command_is_exact = False
        self.pending_destination_confirmation = False
        self.pending_step_change = None
        self.latest_google_maps_url = ""
        self.last_ping_time = time.time()
        
        self.active_websockets = set()
        self.client_roles = {}
        self.loop = None
        self.thread = None
        self.is_running = False

    def start(self):
        """Launches the combined HTTP + WebSocket server in a background daemon thread."""
        self.is_running = True
        self.thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self.thread.start()
        print(f"🌐 5G Mobile Web & WebSocket Bridge started on port {self.port}")

    def _run_event_loop(self):
        """Asynchronous event loop running websockets and simple HTTP handler."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        try:
            import websockets
            from websockets.asyncio.server import serve
            from websockets.http11 import Response
            from websockets.datastructures import Headers
        except ImportError:
            print("⚠️ 'websockets' package missing. Please install: pip install websockets")
            return

        async def handler(websocket):
            # Register client connection
            with self.lock:
                self.active_websockets.add(websocket)
                self.client_roles[websocket] = "unknown"
            client_ip = websocket.remote_address[0] if websocket.remote_address else "5G_Client"
            print(f"\n📱 [5G CLIENT CONNECTED] from {client_ip}")
            
            try:
                async for message in websocket:
                    # Case 1: Binary data = JPEG camera frame (from wearable)
                    if isinstance(message, bytes):
                        with self.lock:
                            # Bug 4: Only set role if unknown. If phone already identified via hello, keep role as phone
                            if self.client_roles.get(websocket) in (None, "unknown"):
                                self.client_roles[websocket] = "wearable"
                        if len(message) > 0:
                            np_arr = np.frombuffer(message, dtype=np.uint8)
                            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                            if frame is not None and frame.size > 0:
                                with self.lock:
                                    first_frame = (self.latest_frame is None)
                                    self.latest_frame = frame
                                    self.latest_frame_time = time.time()
                                if first_frame:
                                    print(f"📷 [WEARABLE CAMERA ACTIVE]: Ingesting live video packets from {client_ip}!", flush=True)
                                
                    # Case 2: Text data = JSON telemetry, audio packet, or button
                    elif isinstance(message, str):
                        try:
                            data = json.loads(message)
                            msg_type = data.get("type")
                            
                            self.last_ping_time = time.time()
                            if msg_type == "hello":
                                role = str(data.get("role", "phone")).lower()
                                with self.lock:
                                    self.client_roles[websocket] = role
                                print(f"👋 [CLIENT IDENTIFIED]: Role = '{role}' from {client_ip}", flush=True)

                            elif msg_type == "ping":
                                # Problem 3: Independent WebSocket heartbeat ping (maintains connection, leaves GPS untouched)
                                continue

                            elif msg_type == "gps":
                                with self.lock:
                                    if self.client_roles.get(websocket) in (None, "unknown"):
                                        self.client_roles[websocket] = "phone"

                                # Bug 3: Treat heartbeat as connection-alive only; never refresh GPS coordinates, velocity, or age
                                if data.get("source") == "heartbeat":
                                    continue

                                lat_raw = data.get("lat")
                                lon_raw = data.get("lon")
                                if lat_raw is not None and lon_raw is not None:
                                    try:
                                        lat_val = float(lat_raw)
                                        lon_val = float(lon_raw)
                                        speed_val = float(data.get("speed_kmh") or 0.0)
                                        acc_val = float(data.get("accuracy")) if data.get("accuracy") is not None else 999.0
                                        source_val = str(data.get("source") or "live")
                                        client_moving = bool(data.get("moving") or data.get("is_moving", False))
                                        # Problem 1: Measure age relative to laptop clock only (prevents clock skew failures)
                                        fix_age = max(0.0, float(data.get("fix_age") or 0.0))
                                        reading_time = time.time() - fix_age
                                        
                                        with self.lock:
                                            first_fix = (self.latest_telemetry["lat"] is None)
                                            
                                            # Derive speed if client speed was 0 (common for pedestrian walking on iOS Safari)
                                            derived_speed = 0.0
                                            if self.gps_history:
                                                prev_lat, prev_lon, prev_t = self.gps_history[-1]
                                                dt = reading_time - prev_t
                                                if 0.5 <= dt <= 6.0:
                                                    R = 6371000.0
                                                    phi1, phi2 = math.radians(prev_lat), math.radians(lat_val)
                                                    dphi = math.radians(lat_val - prev_lat)
                                                    dlambda = math.radians(lon_val - prev_lon)
                                                    a = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2.0)**2
                                                    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
                                                    dist_m = R * c
                                                    # 3.0m minimum filters GPS jitter at ±14m accuracy while detecting real walking
                                                    if dist_m >= 3.0:
                                                        derived_speed = (dist_m / dt) * 3.6
                                            
                                            final_speed = speed_val if speed_val > 0.5 else derived_speed
                                            is_walking = client_moving or (final_speed >= 1.0)
                                            
                                            self.gps_history.append((lat_val, lon_val, reading_time))
                                            self.latest_telemetry["lat"] = lat_val
                                            self.latest_telemetry["lon"] = lon_val
                                            self.latest_telemetry["speed_kmh"] = max(0.0, final_speed)
                                            self.latest_telemetry["accuracy_m"] = max(0.0, acc_val)
                                            self.latest_telemetry["is_walking"] = is_walking
                                            self.latest_telemetry["source"] = source_val
                                            self.latest_telemetry["timestamp"] = reading_time
                                            if data.get("heading") is not None:
                                                try:
                                                    self.latest_telemetry["heading"] = float(data["heading"])
                                                except (ValueError, TypeError):
                                                    pass
                                            if data.get("heading_source") is not None:
                                                self.latest_telemetry["heading_source"] = str(data["heading_source"])
                                            if data.get("steps") is not None:
                                                try:
                                                    self.latest_telemetry["steps"] = int(data["steps"])
                                                except (ValueError, TypeError):
                                                    pass
                                            
                                        if first_fix:
                                            print(f"\n🛰️  [5G SATELLITE GPS LOCKED]: ({lat_val:.5f}, {lon_val:.5f}) ±{acc_val:.1f}m from iPhone!")
                                            
                                        # Ingest live ultrasonic proximity from phone ONLY if BLE is paired or hardware button pressed
                                        ultra_raw = data.get("ultrasonic_cm")
                                        if ultra_raw is not None and data.get("source") == "ble":
                                            try:
                                                dist_val = float(ultra_raw)
                                                with self.lock:
                                                    self.latest_telemetry["ultrasonic_cm"] = dist_val
                                                    self.latest_telemetry["ultrasonic_ts"] = time.time()
                                                now = time.time()
                                                if now - getattr(self, "_last_ble_log_time", 0) > 2.0:
                                                    self._last_ble_log_time = now
                                                    btn_val = data.get("hardware_button", 0)
                                                    print(f"📥 [BLE HOP 2]: Phone relayed BLE ➔ Dist: {dist_val:.1f} cm | Button: {btn_val}", flush=True)
                                            except (ValueError, TypeError):
                                                pass
                                        if data.get("hardware_button") == 1:
                                            self.ptt_button_pressed = True
                                    except (ValueError, TypeError):
                                        pass
                                    
                            elif msg_type in ("telemetry", "ultra", "distance"):
                                ultra_raw = data.get("ultrasonic_cm") or data.get("dist") or data.get("dist_cm")
                                if ultra_raw is not None:
                                    try:
                                        dist_val = float(ultra_raw)
                                        with self.lock:
                                            self.latest_telemetry["ultrasonic_cm"] = dist_val
                                            self.latest_telemetry["ultrasonic_ts"] = time.time()
                                        now = time.time()
                                        if now - getattr(self, "_last_telemetry_log_time", 0) > 2.0:
                                            self._last_telemetry_log_time = now
                                            print(f"📡 [5G WEARABLE TELEMETRY (CLOUDFLARE)]: Distance: {dist_val:.1f} cm", flush=True)
                                    except (ValueError, TypeError):
                                        pass

                            elif msg_type == "ptt_start":
                                with self.lock:
                                    self.is_voice_recording = True
                                    self.voice_recording_started = time.time()
                                print("\n🎙️  [5G IPHONE]: Push-to-Talk button tapped! Recording audio on phone...", flush=True)

                            elif msg_type == "voice_audio":
                                # Extract embedded GPS if phone sent it with audio
                                lat_raw = data.get("lat")
                                lon_raw = data.get("lon")
                                if lat_raw is not None and lon_raw is not None:
                                    try:
                                        lat_val = float(lat_raw)
                                        lon_val = float(lon_raw)
                                        # Problem 1: Measure age relative to laptop clock only
                                        fix_age = max(0.0, float(data.get("fix_age") or 0.0))
                                        reading_time = time.time() - fix_age
                                        with self.lock:
                                            current_ts = self.latest_telemetry.get("timestamp", 0.0)
                                            if reading_time >= current_ts:
                                                first_fix = (self.latest_telemetry["lat"] is None)
                                                self.latest_telemetry["lat"] = lat_val
                                                self.latest_telemetry["lon"] = lon_val
                                                if data.get("accuracy") is not None:
                                                    self.latest_telemetry["accuracy_m"] = float(data.get("accuracy"))
                                                self.latest_telemetry["speed_kmh"] = float(data.get("speed_kmh") or 0.0)
                                                self.latest_telemetry["source"] = "voice_audio"
                                                self.latest_telemetry["timestamp"] = reading_time
                                                if data.get("heading") is not None:
                                                    try:
                                                        self.latest_telemetry["heading"] = float(data["heading"])
                                                    except (ValueError, TypeError):
                                                        pass
                                                if first_fix:
                                                    print(f"\n🛰️  [5G SATELLITE GPS LOCKED FROM VOICE]: ({lat_val:.5f}, {lon_val:.5f}) ±{self.latest_telemetry['accuracy_m']:.1f}m from iPhone!")
                                    except (ValueError, TypeError):
                                        pass

                                b64_data = data.get("data", "")
                                sample_rate = int(data.get("sample_rate") or 16000)
                                fmt = data.get("format", "wav")
                                size_kb = len(b64_data) / 1024.0
                                print(f"\n📥 [5G IPHONE]: Voice audio packet received ({size_kb:.1f} KB, format: {fmt}, {sample_rate}Hz). Transcribing with Whisper AI...", flush=True)
                                
                                def process_voice_audio(b64_str, s_rate, a_fmt):
                                    try:
                                        raw_bytes = base64.b64decode(b64_str)
                                        print(f"📦 [AUDIO DECODER]: Processing {len(raw_bytes)} bytes of audio data (format: {a_fmt})...", flush=True)
                                        audio_np = decode_audio_to_np(raw_bytes, sample_rate=s_rate, fmt=a_fmt)
                                        
                                        if audio_np is None or len(audio_np) == 0:
                                            print(f"⚠️ [5G IPHONE AUDIO ERROR]: Could not decode audio stream ({len(raw_bytes)} bytes).", flush=True)
                                            self.send_speech("Audio decoding failed. Please speak again.")
                                            return
                                        
                                        destination = transcribe_audio_np(audio_np)
                                        if destination:
                                            print(f"\n🎯 [5G IPHONE DESTINATION IDENTIFIED]: '{destination}'", flush=True)
                                            with self.lock:
                                                self.latest_voice_command = destination
                                                self.latest_voice_command_is_exact = False
                                                self.is_voice_recording = False
                                        else:
                                            with self.lock:
                                                self.is_voice_recording = False
                                            print("\n❌ [5G IPHONE]: Whisper detected silence or unclear speech.", flush=True)
                                            self.send_speech("No destination heard. Please tap again and speak clearly.")
                                    except Exception as e:
                                        with self.lock:
                                            self.is_voice_recording = False
                                        print(f"⚠️ [5G IPHONE AUDIO ERROR]: {e}", flush=True)
                                        
                                threading.Thread(target=process_voice_audio, args=(b64_data, sample_rate, fmt), daemon=True).start()

                            elif msg_type == "voice_command":
                                lat_raw = data.get("lat")
                                lon_raw = data.get("lon")
                                if lat_raw is not None and lon_raw is not None:
                                    try:
                                        lat_val = float(lat_raw)
                                        lon_val = float(lon_raw)
                                        # Problem 1: Measure age relative to laptop clock only
                                        fix_age = max(0.0, float(data.get("fix_age") or 0.0))
                                        reading_time = time.time() - fix_age
                                        with self.lock:
                                            current_ts = self.latest_telemetry.get("timestamp", 0.0)
                                            if reading_time >= current_ts:
                                                first_fix = (self.latest_telemetry["lat"] is None)
                                                self.latest_telemetry["lat"] = lat_val
                                                self.latest_telemetry["lon"] = lon_val
                                                if data.get("accuracy") is not None:
                                                    self.latest_telemetry["accuracy_m"] = float(data.get("accuracy"))
                                                self.latest_telemetry["speed_kmh"] = float(data.get("speed_kmh") or 0.0)
                                                self.latest_telemetry["source"] = "voice_audio"
                                                self.latest_telemetry["timestamp"] = reading_time
                                                if data.get("heading") is not None:
                                                    try:
                                                        self.latest_telemetry["heading"] = float(data["heading"])
                                                    except (ValueError, TypeError):
                                                        pass
                                                if first_fix:
                                                    print(f"\n🛰️  [5G SATELLITE GPS LOCKED FROM VOICE]: ({lat_val:.5f}, {lon_val:.5f}) ±{self.latest_telemetry['accuracy_m']:.1f}m from iPhone!")
                                    except (ValueError, TypeError):
                                        pass

                                cmd_text = data.get("command") or data.get("text") or data.get("destination")
                                with self.lock:
                                    self.is_voice_recording = False
                                    if cmd_text:
                                        self.latest_voice_command = str(cmd_text).strip()
                                        self.latest_voice_command_is_exact = bool(data.get("is_exact") or data.get("exact", False))
                                if cmd_text:
                                    print(f"\n🎙️  [5G IPHONE VOICE COMMAND]: '{cmd_text}' (Exact: {self.latest_voice_command_is_exact})")

                            elif msg_type == "confirm_destination":
                                with self.lock:
                                    self.pending_destination_confirmation = bool(data.get("ok", True))
                                print("\n👆 [5G IPHONE]: Destination double-tap confirmation received!")
                                    
                            elif msg_type == "button_ptt":
                                with self.lock:
                                    self.ptt_button_pressed = True
                                print("\n🔘 [5G IPHONE PTT]: Hardware button triggered from mobile screen!")
                                
                            elif msg_type in ("step_change", "nav_step_change"):
                                req_step = data.get("step_index")
                                if req_step is not None:
                                    with self.lock:
                                        self.pending_step_change = int(req_step)
                                    print(f"📱 [MOBILE STEP CHANGE REQUEST]: Step index {req_step}")

                            elif msg_type == "nav_query":
                                q = str(data.get("query") or "").lower()
                                if q in ("where_am_i", "location"):
                                    self._handle_where_am_i(data)
                                
                        except json.JSONDecodeError:
                            pass
            except Exception as conn_err:
                from websockets.exceptions import ConnectionClosedOK
                if not isinstance(conn_err, ConnectionClosedOK):
                    print(f"ℹ️ WebSocket note from {client_ip}: {conn_err}")
            finally:
                with self.lock:
                    self.active_websockets.discard(websocket)
                    role = self.client_roles.pop(websocket, None)
                    if role != "wearable":
                        self.is_voice_recording = False
                print(f"📱 [5G CLIENT DISCONNECTED]: {client_ip}")

        def process_request(connection, request):
            """HTTP Request router serving clean static assets from disk."""
            raw_path = getattr(request, "path", "/")
            path = raw_path.split("?")[0]
            
            # 1. Main Mobile Client Web App
            if path in ["/", "/index.html", "/app", "/gps"]:
                try:
                    with open(INDEX_HTML_PATH, "r", encoding="utf-8") as f:
                        content = f.read().encode("utf-8")
                    headers = Headers([
                        ("Content-Type", "text/html; charset=utf-8"),
                        ("Cache-Control", "no-cache, no-store, must-revalidate"),
                        ("Permissions-Policy", "geolocation=*, microphone=*")
                    ])
                    return Response(200, "OK", headers, content)
                except Exception as e:
                    return Response(500, "Internal Error", Headers([("Content-Type", "text/plain")]), str(e).encode('utf-8'))
                    
            # 2. Client Stylesheet
            if path == "/style.css":
                try:
                    with open(STYLE_CSS_PATH, "r", encoding="utf-8") as f:
                        content = f.read().encode("utf-8")
                    headers = Headers([
                        ("Content-Type", "text/css; charset=utf-8"),
                        ("Cache-Control", "no-cache, no-store, must-revalidate")
                    ])
                    return Response(200, "OK", headers, content)
                except Exception:
                    return Response(404, "Not Found", Headers([("Content-Type", "text/plain")]), b"Not Found")

            # 3. Client JavaScript Application
            if path == "/app.js":
                try:
                    with open(APP_JS_PATH, "r", encoding="utf-8") as f:
                        content = f.read().encode("utf-8")
                    headers = Headers([
                        ("Content-Type", "application/javascript; charset=utf-8"),
                        ("Cache-Control", "no-cache, no-store, must-revalidate")
                    ])
                    return Response(200, "OK", headers, content)
                except Exception:
                    return Response(404, "Not Found", Headers([("Content-Type", "text/plain")]), b"Not Found")

            # 4. Redirect directly to Universal Google Maps Walking Navigation (No API keys, zero watermarks)
            if path in ["/map", "/live_map", "/map.html", "/google_map"]:
                with self.lock:
                    g_url = self.latest_google_maps_url
                if not g_url:
                    lat = self.latest_telemetry.get("lat") or 26.1885
                    lon = self.latest_telemetry.get("lon") or 91.6917
                    g_url = f"https://www.google.com/maps/@{lat:.5f},{lon:.5f},17z"
                headers = Headers([
                    ("Location", g_url),
                    ("Content-Type", "text/plain")
                ])
                return Response(302, "Found", headers, b"Redirecting to Google Maps...")

            # 5. Live Navigation State JSON API
            if path in ["/navigation_state.json", "/api/state", "/api/nav_state"]:
                if os.path.exists(NAV_STATE_PATH):
                    with open(NAV_STATE_PATH, "r", encoding="utf-8") as f:
                        state_json = f.read().encode("utf-8")
                else:
                    state_json = b"{}"
                headers = Headers([
                    ("Content-Type", "application/json; charset=utf-8"),
                    ("Cache-Control", "no-cache"),
                    ("Access-Control-Allow-Origin", "*")
                ])
                return Response(200, "OK", headers, state_json)

            # 5B. Ingest Wearable Telemetry via Cloudflare HTTP GET: /api/telemetry?dist=140.0&btn=0
            if path.startswith("/api/telemetry") or path.startswith("/telemetry") or raw_path.startswith("/api/telemetry") or raw_path.startswith("/telemetry"):
                try:
                    import urllib.parse
                    parsed = urllib.parse.urlparse(raw_path)
                    params = urllib.parse.parse_qs(parsed.query)
                    dist_val = None
                    if "dist" in params:
                        dist_val = float(params["dist"][0])
                    elif "dist_cm" in params:
                        dist_val = float(params["dist_cm"][0])
                    elif "ultra" in params:
                        dist_val = float(params["ultra"][0])
                    
                    btn_val = 0
                    if "btn" in params:
                        btn_val = 1 if params["btn"][0] in ("1", "true", "True") else 0
                    elif "button" in params:
                        btn_val = 1 if params["button"][0] in ("1", "true", "True") else 0

                    if dist_val is not None:
                        with self.lock:
                            self.latest_telemetry["ultrasonic_cm"] = dist_val
                            self.latest_telemetry["ultrasonic_ts"] = time.time()
                        now_ts = time.time()
                        if now_ts - getattr(self, "_last_cloud_telemetry_ts", 0) >= 1.0:
                            self._last_cloud_telemetry_ts = now_ts
                            print(f"📥 [CLOUDFLARE WEARABLE TELEMETRY]: Ultra: {dist_val:.1f} cm | Button: {btn_val}", flush=True)

                    if btn_val == 1:
                        self.ptt_button_pressed = True

                    headers = Headers([
                        ("Content-Type", "application/json; charset=utf-8"),
                        ("Cache-Control", "no-cache"),
                        ("Access-Control-Allow-Origin", "*")
                    ])
                    return Response(200, "OK", headers, b'{"status":"ok"}')
                except Exception as e:
                    return Response(400, "Bad Request", Headers([("Content-Type", "text/plain")]), str(e).encode('utf-8'))

            # 6. Favicon and touch icons
            if path in ["/favicon.ico", "/apple-touch-icon.png", "/apple-touch-icon-precomposed.png"]:
                return Response(204, "No Content", Headers([("Cache-Control", "max-age=86400")]), b"")

            if request.headers.get("Upgrade", "").lower() != "websocket":
                return Response(404, "Not Found", Headers([("Content-Type", "text/plain")]), b"Not Found")
                
            return None # Proceed to WebSocket upgrade

        async def main():
            async with serve(
                handler,
                self.host,
                self.port,
                process_request=process_request,
                max_size=10*1024*1024,
                ping_interval=None,
                ping_timeout=None
            ):
                while self.is_running:
                    await asyncio.sleep(0.5)

        self.loop.run_until_complete(main())

    # --- Server Accessors ---
    def get_latest_frame(self):
        """Returns the latest video frame from the iPhone 5G camera."""
        with self.lock:
            if self.latest_frame is not None:
                # Discard stale frames older than 2.0s
                if (time.time() - self.latest_frame_time) <= 2.0:
                    return self.latest_frame.copy(), self.latest_frame_time
                self.latest_frame = None
            return None, 0.0

    def get_latest_gps(self):
        """Returns current GPS coordinates from the iPhone."""
        with self.lock:
            lat = self.latest_telemetry.get("lat")
            lon = self.latest_telemetry.get("lon")
            if lat is not None and lon is not None:
                return (lat, lon)
            return None

    def get_latest_gps_with_metadata(self):
        """Returns ((lat, lon), source, age_seconds)."""
        with self.lock:
            lat = self.latest_telemetry.get("lat")
            lon = self.latest_telemetry.get("lon")
            ts = self.latest_telemetry.get("timestamp", 0.0)
            src = self.latest_telemetry.get("source", "iphone_5g")
            if lat is not None and lon is not None:
                age = time.time() - ts if ts > 0 else 999.0
                return (lat, lon), src, age
            return None, "none", 999.0

    def get_latest_speed(self):
        """Returns ground speed in km/h. Returns 0.0 if telemetry is stale (> 4.0s)."""
        with self.lock:
            ts = self.latest_telemetry.get("timestamp", 0.0)
            if ts <= 0 or (time.time() - ts) > 4.0:
                return 0.0
            return self.latest_telemetry.get("speed_kmh", 0.0)

    def get_last_fix_speed(self):
        """Speed reported with the most recent real GPS reading, regardless of its age."""
        with self.lock:
            return self.latest_telemetry.get("speed_kmh", 0.0)

    def is_walking(self):
        """Returns True if the user is actively walking based on fresh GPS velocity or motion sensor."""
        with self.lock:
            ts = self.latest_telemetry.get("timestamp", 0.0)
            now = time.time()
            # If telemetry is older than 4.0 seconds, phone cannot be actively walking
            if ts <= 0 or (now - ts) > 4.0:
                return False

            if self.latest_telemetry.get("speed_kmh", 0.0) >= 1.0:
                return True
            if self.latest_telemetry.get("is_walking", False):
                return True
            return False

    def get_latest_ultrasonic(self, max_age=1.5):
        """Returns the latest ultrasonic distance in cm from the wearable if fresh."""
        with self.lock:
            ts = self.latest_telemetry.get("ultrasonic_ts", 0)
            if time.time() - ts > max_age:
                return None
            return self.latest_telemetry.get("ultrasonic_cm")

    def get_latest_ultrasonic_ts(self):
        """Returns the timestamp of the latest ultrasonic reading."""
        with self.lock:
            return self.latest_telemetry.get("ultrasonic_ts", 0.0)

    def has_active_clients(self):
        """Returns True if at least one 5G mobile client is actively connected."""
        with self.lock:
            return len(self.active_websockets) > 0

    def has_phone_clients(self):
        """Returns True if at least one phone client is actively connected (excludes standalone wearable)."""
        with self.lock:
            if not self.active_websockets:
                return False
            # Check if any client has role "phone" or is an unlabelled client before first message
            return any(self.client_roles.get(ws) != "wearable" for ws in self.active_websockets)

    def _get_phone_sockets(self):
        """Returns only websockets connected to phones (never pushes audio/JSON to Arduino camera)."""
        with self.lock:
            return [ws for ws in self.active_websockets if self.client_roles.get(ws) != "wearable"]

    def get_latest_accuracy(self):
        """Returns the latest GPS horizontal accuracy in meters (Item 7 & Bug 10)."""
        with self.lock:
            acc = self.latest_telemetry.get("accuracy_m")
            if acc is None or acc <= 0.0:
                return 999.0
            return float(acc)

    def get_pending_step_change(self):
        """Returns requested step index from mobile client if any, then clears it."""
        with self.lock:
            step = self.pending_step_change
            self.pending_step_change = None
            return step

    def is_recording_voice(self):
        """Returns True if user is actively recording voice on the mobile client."""
        with self.lock:
            if getattr(self, "is_voice_recording", False) and (time.time() - getattr(self, "voice_recording_started", 0.0)) > 10.0:
                self.is_voice_recording = False  # Phone records max ~5 s; 10 s is a safe timeout limit
            return getattr(self, "is_voice_recording", False)

    def is_button_pressed(self):
        """Returns True if the user pressed Push-to-Talk on the phone."""
        with self.lock:
            pressed = self.ptt_button_pressed
            self.ptt_button_pressed = False
            return pressed

    def has_voice_command(self):
        """Returns True if a spoken destination command was received from the iPhone."""
        with self.lock:
            return self.latest_voice_command is not None

    def get_voice_command(self):
        """Retrieves and clears the latest spoken destination command from the iPhone."""
        with self.lock:
            cmd = self.latest_voice_command
            self.latest_voice_command = None
            return cmd

    async def _safe_ws_send(self, ws, payload):
        try:
            await ws.send(payload)
        except Exception:
            pass

    def send_threat_alert(self, text, obstacle="obstacle", distance_m=1.5, urgency="warning", haptic=None):
        """Pushes an immediate collision alert to the iPhone."""
        if not self.loop or not self.is_running:
            return
        payload = json.dumps({
            "type": "threat_alert",
            "obstacle": str(obstacle),
            "distance_m": float(distance_m),
            "urgency": str(urgency),
            "text": str(text),
            "haptic": haptic
        })
        sockets = self._get_phone_sockets()
        if not sockets:
            print("⚠️ [THREAT NOT DELIVERED]: no phone connected")
        else:
            print(f"📲 [THREAT → PHONE x{len(sockets)}]: \"{text}\"")
        for ws in sockets:
            asyncio.run_coroutine_threadsafe(self._safe_ws_send(ws, payload), self.loop)

    def send_proximity(self, level, distance_m=None, side=None):
        """Pushes real-time audio-spatial proximity feed to the phone (~4 Hz)."""
        if not self.loop or not self.is_running:
            return
        payload = json.dumps({
            "type": "proximity",
            "level": str(level),
            "distance_m": float(distance_m) if distance_m is not None else None,
            "side": side
        })
        for ws in self._get_phone_sockets():
            asyncio.run_coroutine_threadsafe(self._safe_ws_send(ws, payload), self.loop)

    def send_nav_step(self, step_text, speak=True):
        """Pushes a turn-by-turn walking maneuver to the iPhone."""
        if not self.loop or not self.is_running:
            return
        payload = json.dumps({
            "type": "nav_step",
            "step": str(step_text),
            "speak": bool(speak)
        })
        sockets = self._get_phone_sockets()
        for ws in sockets:
            asyncio.run_coroutine_threadsafe(self._safe_ws_send(ws, payload), self.loop)

    def get_latest_heading(self):
        """Returns the latest compass heading in degrees (0-360) from mobile device (Problem 9)."""
        with self.lock:
            return self.latest_telemetry.get("heading")

    def get_latest_heading_source(self):
        """Returns the source of the latest heading ('gps', 'compass', or 'none') (Problem 4c)."""
        with self.lock:
            return self.latest_telemetry.get("heading_source", "none")

    def pop_destination_confirmation(self):
        """Returns True if user double-tapped screen to confirm destination, clearing flag (Problem 11)."""
        with self.lock:
            val = getattr(self, "pending_destination_confirmation", False)
            self.pending_destination_confirmation = False
            return val

    def get_voice_command_metadata(self):
        """Returns (command_str, is_exact_bool)."""
        with self.lock:
            cmd = self.latest_voice_command
            exact = getattr(self, "latest_voice_command_is_exact", False)
            return cmd, exact

    def _handle_where_am_i(self, data):
        """Processes 'Where am I?' query from mobile client and responds with audio (Problem 11)."""
        with self.lock:
            lat = self.latest_telemetry.get("lat")
            lon = self.latest_telemetry.get("lon")
        if lat is None or lon is None:
            self.send_speech("Location unavailable. Please wait for GPS lock.", priority="high")
            return

        landmarks_path = os.path.join(current_dir, "navigation", "campus_landmarks.json")
        nearest_name = None
        min_dist_m = float('inf')
        if os.path.exists(landmarks_path):
            try:
                with open(landmarks_path, "r", encoding="utf-8") as f:
                    landmarks = json.load(f)
                import math
                R = 6371000.0
                for lm in landmarks:
                    coords = lm.get("coordinates")
                    if coords:
                        p_lat, p_lon = coords[0], coords[1]
                        dlat = math.radians(p_lat - lat)
                        dlon = math.radians(p_lon - lon)
                        a = math.sin(dlat/2)**2 + math.cos(math.radians(lat))*math.cos(math.radians(p_lat))*math.sin(dlon/2)**2
                        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
                        d_m = R * c
                        if d_m < min_dist_m:
                            min_dist_m = d_m
                            nearest_name = lm.get("canonical")
            except Exception:
                pass

        nav_info = ""
        if os.path.exists(NAV_STATE_PATH):
            try:
                with open(NAV_STATE_PATH, "r", encoding="utf-8") as f:
                    nstate = json.load(f)
                if nstate.get("is_active"):
                    cur_idx = nstate.get("step_index", 0)
                    steps = nstate.get("structured_steps", [])
                    if cur_idx + 1 < len(steps):
                        wp = steps[cur_idx + 1].get("waypoint")
                        if wp:
                            dlat = math.radians(wp[0] - lat)
                            dlon = math.radians(wp[1] - lon)
                            a = math.sin(dlat/2)**2 + math.cos(math.radians(lat))*math.cos(math.radians(wp[0]))*math.sin(dlon/2)**2
                            d_turn = int(round(6371000.0 * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))))
                            nav_info = f" Next turn in {d_turn} meters, about {int(round(d_turn / 0.7))} steps."
            except Exception:
                pass

        if nearest_name:
            query_reply = f"You are near {nearest_name}, about {int(round(min_dist_m))} meters away.{nav_info}"
        else:
            query_reply = f"Coordinates {lat:.4f}, {lon:.4f}.{nav_info}"
        print(f"\n📍 [WHERE AM I QUERY]: {query_reply}")
        self.send_speech(query_reply, priority="high")

    def send_speech(self, text, priority="normal", haptic=None):
        """Pushes a general spoken instruction to the mobile earphones with optional haptic pattern."""
        if not self.loop or not self.is_running:
            return
        payload = json.dumps({
            "type": "speech_prompt",
            "text": str(text),
            "priority": str(priority),
            "haptic": str(haptic) if haptic else None
        })
        sockets = self._get_phone_sockets()
        for ws in sockets:
            asyncio.run_coroutine_threadsafe(self._safe_ws_send(ws, payload), self.loop)

    def send_nav_route(self, destination, steps, total_distance_m, google_maps_url="", dest_coords=None, start_coords=None, route_points=None):
        """Pushes the complete structured turn-by-turn route to the mobile client."""
        if not self.loop or not self.is_running:
            return
        if google_maps_url:
            with self.lock:
                self.latest_google_maps_url = str(google_maps_url)
        payload = json.dumps({
            "type": "nav_route",
            "destination": str(destination),
            "dest_coords": list(dest_coords) if dest_coords else None,
            "start_coords": list(start_coords) if start_coords else None,
            "total_distance_m": int(total_distance_m),
            "total_steps": len(steps),
            "current_step_index": 0,
            "steps": steps,
            "route_points": list(route_points) if route_points else None,
            "google_maps_url": str(google_maps_url or "")
        })
        sockets = self._get_phone_sockets()
        for ws in sockets:
            asyncio.run_coroutine_threadsafe(self._safe_ws_send(ws, payload), self.loop)

    def send_nav_step_update(self, step_index, total_steps, step_text, speak=True, resync=False):
        """Pushes a sequential step advancement to the mobile client."""
        if not self.loop or not self.is_running:
            return
        payload = json.dumps({
            "type": "nav_step_update",
            "step_index": int(step_index),
            "total_steps": int(total_steps),
            "step_text": str(step_text),
            "speak": bool(speak),
            "resync": bool(resync)
        })
        sockets = self._get_phone_sockets()
        for ws in sockets:
            asyncio.run_coroutine_threadsafe(self._safe_ws_send(ws, payload), self.loop)

    def send_telemetry_update(self, ultra_cm=None, speed_kmh=None, gps_lat=None, gps_lon=None):
        """Pushes real-time sensor telemetry to connected mobile web clients."""
        if not self.loop or not self.is_running:
            return
        payload = json.dumps({
            "type": "telemetry_update",
            "ultra_cm": float(ultra_cm) if ultra_cm is not None else None,
            "speed_kmh": float(speed_kmh) if speed_kmh is not None else None,
            "gps": [float(gps_lat), float(gps_lon)] if (gps_lat is not None and gps_lon is not None) else None,
            "timestamp": time.time()
        })
        sockets = self._get_phone_sockets()
        for ws in sockets:
            asyncio.run_coroutine_threadsafe(self._safe_ws_send(ws, payload), self.loop)

    def stop(self):
        self.is_running = False
        print("🛑 5G Mobile Bridge stopped.")

if __name__ == "__main__":
    bridge = MobileBridgeServer(port=8080)
    bridge.start()
    print("Test running on http://localhost:8080")
    print("Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        bridge.stop()
