import socket
import threading
import json
import time
import sys
import cv2
import numpy as np

# Ensure UTF-8 output encoding across Windows terminals
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

class UDPNetworkHandler:
    """
    Dedicated 5G Edge Server UDP Network Handler.
    Manages non-blocking reception of live video streams and sensor telemetry
    from the wearable client (Arduino / ESP32 / Mock client) over 5G / Wi-Fi.
    """
    def __init__(self, host="0.0.0.0", video_port=5005, telemetry_port=5006, max_packet_size=65535):
        self.host = host
        self.video_port = video_port
        self.telemetry_port = telemetry_port
        self.max_packet_size = max_packet_size
        
        # State variables
        self.latest_frame = None
        self.latest_frame_time = 0
        self.latest_telemetry = {
            "lat": None,
            "lon": None,
            "speed_kmh": 0.0,
            "ultrasonic_cm": None,
            "button": False,
            "timestamp": 0
        }
        self.client_address = None  # Remember client IP/port for downlink
        
        # Thread locks
        self.frame_lock = threading.Lock()
        self.telemetry_lock = threading.Lock()
        
        # Sockets
        self.video_socket = None
        self.telemetry_socket = None
        
        # Running flags
        self.is_running = False
        self.video_thread = None
        self.telemetry_thread = None

    def start(self):
        """Initializes sockets and launches background listener threads."""
        self.is_running = True
        
        # 1. Setup Video Socket
        self.video_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Increase OS receive buffer size to prevent packet loss
        try:
            self.video_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
        except Exception:
            pass
        self.video_socket.bind((self.host, self.video_port))
        
        # 2. Setup Telemetry Socket
        self.telemetry_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.telemetry_socket.bind((self.host, self.telemetry_port))
        
        # 3. Start Listener Threads
        self.video_thread = threading.Thread(target=self._video_listener, daemon=True)
        self.telemetry_thread = threading.Thread(target=self._telemetry_listener, daemon=True)
        
        self.video_thread.start()
        self.telemetry_thread.start()
        
        local_ip = self.get_local_ip()
        print("==================================================")
        print("       5G EDGE SERVER - UDP NETWORK HANDLER       ")
        print("==================================================")
        print(f"📡 Server Local IP        : {local_ip}")
        print(f"🎥 Video Listening Port   : {self.video_port} (UDP)")
        print(f"🧭 Telemetry Listening Port: {self.telemetry_port} (UDP)")
        print(f"⚡ Status                 : LISTENING FOR WEARABLE CLIENT")
        print("==================================================")

    def _video_listener(self):
        """Continuously receives JPEG datagrams and decodes them into OpenCV frames."""
        while self.is_running:
            try:
                data, addr = self.video_socket.recvfrom(self.max_packet_size)
                if not data:
                    continue
                
                # Convert raw bytes directly to a NumPy buffer and decode JPEG
                np_arr = np.frombuffer(data, dtype=np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    with self.frame_lock:
                        self.latest_frame = frame
                        self.latest_frame_time = time.time()
            except Exception as e:
                if self.is_running:
                    print(f"⚠️ Video socket receive error: {e}")

    @staticmethod
    def _parse_nmea_coord(coord_str, direction):
        """
        Converts NMEA DDMM.MMMMM format (or decimal degrees) to signed decimal degrees.
        Example: '2611.31234', 'N' -> 26 + 11.31234/60 = 26.188539
        Example: '09141.50123', 'E' -> 91 + 41.50123/60 = 91.691687
        """
        if not coord_str or not direction:
            return None
        dot = coord_str.find('.')
        if dot == -1:
            return None
        if dot <= 2:
            val = float(coord_str)
        else:
            deg = float(coord_str[:dot-2])
            mins = float(coord_str[dot-2:])
            val = deg + (mins / 60.0)
        if direction.upper() in ['S', 'W']:
            val = -val
        return round(val, 6)

    @classmethod
    def parse_nmea_sentence(cls, line):
        """
        Parses common NMEA 0183 sentences ($GPRMC, $GPGGA, $GPGLL, $GNRMC, $GNGGA, etc.)
        Returns (lat, lon, speed_kmh) where speed_kmh is Doppler speed in km/h (or None).
        """
        line = line.strip()
        if not line.startswith('$'):
            return None
        if '*' in line:
            line = line.split('*')[0]
        parts = line.split(',')
        if len(parts) < 5:
            return None
        sentence_type = parts[0].upper()
        # $--RMC: Recommended Minimum Specific GNSS Data
        if sentence_type.endswith('RMC'):
            if len(parts) >= 7:
                # Accept coordinate values if present, even when indoors ('V' / receiver warning)
                if parts[3] and parts[5]:
                    lat = cls._parse_nmea_coord(parts[3], parts[4])
                    lon = cls._parse_nmea_coord(parts[5], parts[6])
                    if lat is not None and lon is not None:
                        speed_kmh = None
                        if len(parts) >= 8 and parts[7]:
                            try:
                                speed_knots = float(parts[7])
                                speed_kmh = round(speed_knots * 1.852, 2)
                            except (ValueError, TypeError):
                                speed_kmh = None
                        return lat, lon, speed_kmh
        elif sentence_type.endswith('GGA'):
            if len(parts) >= 6:
                if parts[2] and parts[4]:
                    lat = cls._parse_nmea_coord(parts[2], parts[3])
                    lon = cls._parse_nmea_coord(parts[4], parts[5])
                    if lat is not None and lon is not None:
                        return lat, lon, None
        elif sentence_type.endswith('GLL'):
            if len(parts) >= 5:
                if parts[1] and parts[3]:
                    lat = cls._parse_nmea_coord(parts[1], parts[2])
                    lon = cls._parse_nmea_coord(parts[3], parts[4])
                    if lat is not None and lon is not None:
                        return lat, lon, None
        return None

    def _telemetry_listener(self):
        """Continuously receives sensor telemetry packets (NMEA 0183, JSON, or CSV)."""
        has_locked = False
        last_reported_coords = None
        smoothed_lat = None
        smoothed_lon = None
        current_speed_kmh = 0.0
        consecutive_move_count = 0
        last_packet_time = 0.0

        while self.is_running:
            try:
                data, addr = self.telemetry_socket.recvfrom(4096)
                if not data:
                    continue
                
                # Cache client address for two-way communication
                self.client_address = addr
                message_str = data.decode('utf-8', errors='ignore').strip()
                
                parsed_gps = None
                parsed_speed = None
                gps_format = None
                
                # 1. Try NMEA 0183 sentence ($GPRMC, $GPGGA, $GPGLL, $GNRMC, etc.)
                if '$' in message_str:
                    for line in message_str.splitlines():
                        line = line.strip()
                        if line.startswith('$'):
                            res = self.parse_nmea_sentence(line)
                            if res:
                                parsed_gps = (res[0], res[1])
                                parsed_speed = res[2] if len(res) > 2 else None
                                gps_format = line.split(',')[0]
                                break
                                
                # 2. Try JSON format
                if not parsed_gps:
                    try:
                        telemetry_data = json.loads(message_str)
                        with self.telemetry_lock:
                            if "lat" in telemetry_data and "lon" in telemetry_data and telemetry_data["lat"] is not None:
                                parsed_gps = (float(telemetry_data["lat"]), float(telemetry_data["lon"]))
                                gps_format = "JSON"
                            if "speed_kmh" in telemetry_data and telemetry_data["speed_kmh"] is not None:
                                parsed_speed = float(telemetry_data["speed_kmh"])
                            elif "speed" in telemetry_data and telemetry_data["speed"] is not None:
                                parsed_speed = float(telemetry_data["speed"])
                            if "ultrasonic_cm" in telemetry_data and telemetry_data["ultrasonic_cm"] is not None:
                                with self.telemetry_lock:
                                    self.latest_telemetry["ultrasonic_cm"] = float(telemetry_data["ultrasonic_cm"])
                                    self.latest_telemetry["ultrasonic_ts"] = time.time()
                                now_ts = time.time()
                                if now_ts - getattr(self, "_last_udp_log_ts", 0) >= 1.0:
                                    self._last_udp_log_ts = now_ts
                                    btn_st = telemetry_data.get("button", False)
                                    u_val = self.latest_telemetry["ultrasonic_cm"]
                                    print(f"📥 [HOTSPOT UDP INGEST]: Received from Arduino UNO Q ➔ Ultra: {u_val:.1f} cm | Button: {btn_st}", flush=True)
                            if "button" in telemetry_data:
                                self.latest_telemetry["button"] = bool(telemetry_data["button"])
                    except json.JSONDecodeError:
                        pass
                        
                # 3. Try Raw CSV format: "LAT,LON" or "LAT,LON,ULTRASONIC" or "LAT,LON,ULTRASONIC,SPEED"
                if not parsed_gps and ',' in message_str and not message_str.startswith('$'):
                    try:
                        parts = message_str.split(',')
                        if len(parts) >= 2:
                            parsed_gps = (float(parts[0]), float(parts[1]))
                            gps_format = "CSV"
                            if len(parts) >= 3:
                                with self.telemetry_lock:
                                    self.latest_telemetry["ultrasonic_cm"] = float(parts[2])
                                    self.latest_telemetry["ultrasonic_ts"] = time.time()
                            if len(parts) >= 4:
                                parsed_speed = float(parts[3])
                    except Exception:
                        pass

                # If GPS was extracted, apply filtering & update telemetry
                if parsed_gps:
                    raw_lat, raw_lon = parsed_gps
                    now = time.time()
                    
                    # Update speed if reported in packet
                    if parsed_speed is not None:
                        current_speed_kmh = max(0.0, float(parsed_speed))
                    else:
                        # Fallback speed estimation if missing from current sentence (e.g. GGA)
                        if smoothed_lat is not None and last_packet_time > 0:
                            dt = max(0.1, now - last_packet_time)
                            dlat = (raw_lat - smoothed_lat) * 111139.0
                            dlon = (raw_lon - smoothed_lon) * 99736.0
                            inferred_dist_m = (dlat * dlat + dlon * dlon) ** 0.5
                            inferred_speed_kmh = (inferred_dist_m / dt) * 3.6
                            current_speed_kmh = min(current_speed_kmh, inferred_speed_kmh) if current_speed_kmh > 0 else inferred_speed_kmh
                    
                    last_packet_time = now
                    
                    # Adaptive EMA Coordinate Smoothing
                    # Rejects high-frequency indoor reflection jumps while retaining walking responsiveness
                    if smoothed_lat is None:
                        smoothed_lat = raw_lat
                        smoothed_lon = raw_lon
                    else:
                        if current_speed_kmh >= 1.5:
                            alpha = 0.55  # Walking pace: responsive tracking, minimal lag
                        elif current_speed_kmh >= 0.8:
                            alpha = 0.35  # Slow walking onset
                        else:
                            alpha = 0.20  # Physically stationary: strong noise rejection filter
                        
                        smoothed_lat = alpha * raw_lat + (1.0 - alpha) * smoothed_lat
                        smoothed_lon = alpha * raw_lon + (1.0 - alpha) * smoothed_lon
                    
                    final_lat = round(smoothed_lat, 6)
                    final_lon = round(smoothed_lon, 6)
                    
                    with self.telemetry_lock:
                        self.latest_telemetry["lat"] = final_lat
                        self.latest_telemetry["lon"] = final_lon
                        self.latest_telemetry["speed_kmh"] = round(current_speed_kmh, 1)
                        self.latest_telemetry["gps_source"] = "iphone"
                        self.latest_telemetry["gps_timestamp"] = now
                        self.latest_telemetry["timestamp"] = now
                    
                    # Event-Driven Console Logging with Doppler ZUPT & Temporal Debounce
                    if not has_locked:
                        has_locked = True
                        last_reported_coords = (final_lat, final_lon)
                        consecutive_move_count = 0
                        print(f"\n🛰️  [GPS LOCKED]: ({final_lat:.5f}, {final_lon:.5f}) from {addr[0]} via {gps_format}")
                    elif last_reported_coords:
                        # 1. Doppler Zero-Velocity Gating (ZUPT):
                        # Human walking speed is >= 1.5 - 3.5 km/h.
                        # If Doppler speed < 0.8 km/h, antenna is stationary; ignore indoor multipath jitter.
                        is_physically_moving = (current_speed_kmh >= 0.8)
                        
                        # Strict 15-meter threshold preserved
                        dlat = (final_lat - last_reported_coords[0]) * 111139.0
                        dlon = (final_lon - last_reported_coords[1]) * 99736.0
                        delta_m = (dlat * dlat + dlon * dlon) ** 0.5
                        
                        if is_physically_moving and delta_m >= 15.0:
                            consecutive_move_count += 1
                            # 2. Temporal Debounce: 2 consecutive packets required to confirm physical transit
                            if consecutive_move_count >= 2:
                                print(f"📍 [GPS MOVED]: ({final_lat:.5f}, {final_lon:.5f}) [Δ {delta_m:.0f}m @ {current_speed_kmh:.1f} km/h]")
                                last_reported_coords = (final_lat, final_lon)
                                consecutive_move_count = 0
                        else:
                            consecutive_move_count = 0
            except Exception as e:
                if self.is_running:
                    print(f"⚠️ Telemetry socket receive error: {e}")

    def get_latest_speed(self):
        """Thread-safe accessor for current ground speed in km/h."""
        with self.telemetry_lock:
            return self.latest_telemetry.get("speed_kmh", 0.0)

    def get_latest_frame(self):
        """Thread-safe accessor for the most recent video frame."""
        with self.frame_lock:
            if self.latest_frame is not None:
                # Discard stale frames older than 2.0s
                if (time.time() - self.latest_frame_time) <= 2.0:
                    return self.latest_frame.copy(), self.latest_frame_time
                self.latest_frame = None
            return None, 0

    def get_latest_gps(self):
        """Thread-safe accessor for current GPS coordinates."""
        with self.telemetry_lock:
            lat = self.latest_telemetry["lat"]
            lon = self.latest_telemetry["lon"]
            if lat is not None and lon is not None:
                return lat, lon
            return None

    def get_latest_gps_with_metadata(self):
        """
        Thread-safe accessor for current GPS coordinates with provenance metadata.
        Returns ((lat, lon), source, age_seconds).
        If no GPS is received yet, returns (None, 'none', 999.0).
        """
        with self.telemetry_lock:
            lat = self.latest_telemetry.get("lat")
            lon = self.latest_telemetry.get("lon")
            source = self.latest_telemetry.get("gps_source", "none")
            ts = self.latest_telemetry.get("gps_timestamp", 0)
            if lat is not None and lon is not None:
                age = time.time() - ts if ts > 0 else 999.0
                return (lat, lon), source, age
            return None, "none", 999.0

    def get_latest_ultrasonic(self, max_age=1.5):
        """Thread-safe accessor for latest ultrasonic distance in cm if fresh."""
        with self.telemetry_lock:
            ts = self.latest_telemetry.get("ultrasonic_ts", 0)
            if time.time() - ts > max_age:
                return None
            return self.latest_telemetry.get("ultrasonic_cm")

    def get_latest_ultrasonic_ts(self):
        """Thread-safe accessor for timestamp of latest ultrasonic distance."""
        with self.telemetry_lock:
            return self.latest_telemetry.get("ultrasonic_ts", 0.0)

    def is_button_pressed(self):
        """Checks if wearable client pressed the push-to-talk button."""
        with self.telemetry_lock:
            pressed = self.latest_telemetry["button"]
            # Reset after reading
            self.latest_telemetry["button"] = False
            return pressed

    def update_gps(self, lat, lon):
        """Thread-safe injector for live GPS (e.g. from iPhone Web GPS Bridge)."""
        with self.telemetry_lock:
            self.latest_telemetry["lat"] = float(lat)
            self.latest_telemetry["lon"] = float(lon)
            self.latest_telemetry["gps_source"] = "iphone"
            self.latest_telemetry["gps_timestamp"] = time.time()
            self.latest_telemetry["timestamp"] = time.time()

    def trigger_button(self):
        """Thread-safe trigger for Push-to-Talk button from web client."""
        with self.telemetry_lock:
            self.latest_telemetry["button"] = True

    def send_downlink_alert(self, message_dict, target_host=None, target_port=5007):
        """
        Sends an alert or audio control packet back to the client over UDP.
        """
        try:
            dest_host = target_host or (self.client_address[0] if self.client_address else "127.0.0.1")
            dest_port = target_port
            payload = json.dumps(message_dict, default=lambda o: float(o) if hasattr(o, '__float__') else str(o)).encode('utf-8')
            
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.sendto(payload, (dest_host, dest_port))
            sock.close()
            return True
        except Exception as e:
            print(f"❌ Failed to send downlink alert: {e}")
            return False

    def stop(self):
        """Cleanly terminates sockets and listener threads."""
        self.is_running = False
        if self.video_socket:
            self.video_socket.close()
        if self.telemetry_socket:
            self.telemetry_socket.close()
        print("🛑 UDP Network Handler stopped.")

    @staticmethod
    def get_local_ip():
        """Helper to find the machine's primary local IP on the Wi-Fi/5G hotspot."""
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # Doesn't need to be reachable, establishes routing table
            s.connect(('8.8.8.8', 80))
            ip = s.getsockname()[0]
        except Exception:
            ip = '127.0.0.1'
        finally:
            s.close()
        return ip


# ==============================================================
# STANDALONE TEST HARNESS
# ==============================================================
if __name__ == "__main__":
    handler = UDPNetworkHandler()
    handler.start()
    
    print("\n💡 Standalone Test Running...")
    print("Press 'q' in console or Ctrl+C to stop.")
    print("--------------------------------------------------")
    
    try:
        while True:
            # Check for incoming telemetry
            gps = handler.get_latest_gps()
            dist = handler.get_latest_ultrasonic()
            frame, frame_time = handler.get_latest_frame()
            
            status_line = []
            if gps:
                status_line.append(f"GPS: ({gps[0]:.5f}, {gps[1]:.5f})")
            else:
                status_line.append("GPS: Waiting...")
                
            if dist is not None:
                status_line.append(f"Ultrasonic: {dist:.1f} cm")
            else:
                status_line.append("Ultrasonic: Waiting...")
                
            if frame is not None:
                age_ms = (time.time() - frame_time) * 1000
                status_line.append(f"Frame: Received ({age_ms:.0f}ms ago)")
                cv2.imshow("5G Live Stream - Receiver Test", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                status_line.append("Frame: Waiting...")
                
            print(f"\r📡 [{time.strftime('%H:%M:%S')}] " + " | ".join(status_line), end="", flush=True)
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        handler.stop()
        cv2.destroyAllWindows()
