import socket
import json
import time
import cv2

def run_mock_wearable_client(server_ip="127.0.0.1", video_port=5005, telemetry_port=5006, downlink_port=5007):
    """
    Simulates the physical Arduino/ESP32 wearable hardware over UDP.
    Streams webcam video and periodic telemetry (GPS + Ultrasonic) to the Edge Server.
    """
    print("==================================================")
    print("   MOCK WEARABLE CLIENT (HARDWARE SIMULATOR)      ")
    print("==================================================")
    print(f"🎯 Target Server IP      : {server_ip}")
    print(f"🎥 Video Target Port     : {video_port} (UDP)")
    print(f"🧭 Telemetry Target Port : {telemetry_port} (UDP)")
    print(f"👂 Downlink Listen Port  : {downlink_port} (UDP)")
    print("==================================================")
    
    # Setup UDP sockets
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    # Downlink socket to receive warnings/audio back from Edge Server
    downlink_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    downlink_sock.bind(("0.0.0.0", downlink_port))
    downlink_sock.setblocking(False)
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("⚠️ Warning: Webcam not found, sending simulated color frames instead.")
        use_camera = False
    else:
        use_camera = True
        print("🎥 Webcam initialized.")

    print("\n🟢 Streaming simulated 5G hardware data! Press 'q' to stop.\n")
    
    last_telemetry_time = 0
    simulated_distance = 180.0
    sim_lat = 40.748817
    sim_lon = -73.985428
    
    try:
        while True:
            # 1. Capture and stream video frame
            if use_camera:
                ret, frame = cap.read()
                if not ret:
                    continue
                # Resize to standard wearable resolution (480x360) to stay within UDP payload limits
                frame_resized = cv2.resize(frame, (480, 360))
            else:
                # Synthetic dummy frame
                import numpy as np
                frame_resized = np.zeros((360, 480, 3), dtype=np.uint8)
                cv2.putText(frame_resized, f"MOCK HW: {time.time():.1f}", (40, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            
            # Compress to JPEG
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 65]
            result, encimg = cv2.imencode('.jpg', frame_resized, encode_param)
            
            if result:
                byte_data = encimg.tobytes()
                if len(byte_data) < 65000:  # Must fit in single UDP packet
                    sock.sendto(byte_data, (server_ip, video_port))
            
            # 2. Stream telemetry once every 500ms
            now = time.time()
            if now - last_telemetry_time > 0.5:
                # Slowly oscillate simulated distance for realistic testing
                simulated_distance = 100.0 + 50.0 * (time.time() % 10)
                
                telemetry_payload = {
                    "lat": sim_lat,
                    "lon": sim_lon,
                    "ultrasonic_cm": round(simulated_distance, 1),
                    "button": False,
                    "timestamp": now
                }
                
                sock.sendto(json.dumps(telemetry_payload).encode('utf-8'), (server_ip, telemetry_port))
                last_telemetry_time = now
                print(f"\r📡 [Client Streaming] Sent Frame (~{len(byte_data)/1024:.1f} KB) | Ultra: {telemetry_payload['ultrasonic_cm']}cm | GPS: ({sim_lat:.4f}, {sim_lon:.4f})", end="", flush=True)
            
            # 3. Check for Downlink alerts from the Server
            try:
                alert_data, addr = downlink_sock.recvfrom(2048)
                alert_msg = json.loads(alert_data.decode('utf-8'))
                print(f"\n🚨 [DOWNLINK RECEIVED FROM SERVER]: {alert_msg}")
            except BlockingIOError:
                pass
            except Exception:
                pass
                
            time.sleep(0.04) # ~25 FPS
            
    except KeyboardInterrupt:
        print("\nStopping Mock Client...")
    finally:
        if use_camera:
            cap.release()
        sock.close()
        downlink_sock.close()

if __name__ == "__main__":
    import sys
    ip = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
    run_mock_wearable_client(server_ip=ip)
