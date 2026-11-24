import socket
import subprocess
import threading
import time

CONTAINER_NAME = "5g-project-main-1"
UPLINK_PORT = 5005
DOWNLINK_PORT = 5006

container_ip = None
esp_ip = None

def watch_container_ip():
    global container_ip
    while True:
        try:
            ip = subprocess.check_output(
                ["docker", "inspect", "-f", "{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}", CONTAINER_NAME],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            if ip and ip != container_ip:
                container_ip = ip
                print(f"[Watcher] Found {CONTAINER_NAME} at {container_ip}", flush=True)
        except subprocess.CalledProcessError:
            if container_ip is not None:
                print(f"[Watcher] {CONTAINER_NAME} stopped. Waiting...", flush=True)
            container_ip = None
        time.sleep(5)

def uplink_loop():
    global esp_ip
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", UPLINK_PORT))
    print(f"[Relay UP] Listening on 0.0.0.0:{UPLINK_PORT}", flush=True)

    while True:
        data, addr = sock.recvfrom(2048)
        esp_ip = addr[0]
        if container_ip:
            sock.sendto(data, (container_ip, UPLINK_PORT))
            print(f"[UP] {len(data)}B from ESP ({esp_ip}) -> {container_ip}:{UPLINK_PORT}", flush=True)
        else:
            print(f"[UP] Waiting for {CONTAINER_NAME} to start...", flush=True)

def downlink_loop():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", DOWNLINK_PORT))
    print(f"[Relay DOWN] Listening on 0.0.0.0:{DOWNLINK_PORT}", flush=True)

    while True:
        data, addr = sock.recvfrom(2048)
        if esp_ip:
            sock.sendto(data, (esp_ip, DOWNLINK_PORT))
            print(f"[DOWN] {data.decode('utf-8', errors='ignore')} -> ESP ({esp_ip}:{DOWNLINK_PORT})", flush=True)
        else:
            print("[DOWN] Dropped: waiting for first packet from ESP32-S3.", flush=True)

threading.Thread(target=watch_container_ip, daemon=True).start()
threading.Thread(target=uplink_loop, daemon=True).start()
downlink_loop()