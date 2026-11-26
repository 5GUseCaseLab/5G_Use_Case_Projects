import os
import re
import sys
import time
import json
import threading
import subprocess
import urllib.request
import urllib.error
import cv2
import numpy as np

# Ensure terminal uses UTF-8 for ANSI block characters
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# =========================================================================
# 5G CLOUD SYNC CONFIGURATION (Zero-Touch Wearable Auto-Discovery)
# Enables Arduino UNO Q to automatically discover and connect to the active
# Cloudflare URL without requiring Arduino App Lab or manual edits in the field.
# =========================================================================
CLOUD_SYNC_TOPIC = "iitg_5g_wearable_nav_sync_9169957"

def publish_cloud_sync_url(url: str):
    """
    Publishes active Cloudflare URL to Cloud Sync bus for zero-touch Arduino discovery.
    Runs silently in background with a quick timeout.
    """
    if not url or not url.startswith("https://"):
        return False
    try:
        req = urllib.request.Request(
            f"https://ntfy.sh/{CLOUD_SYNC_TOPIC}",
            data=url.strip().encode('utf-8'),
            headers={'Title': 'Active5GTunnelURL', 'User-Agent': '5G-Edge-Server'},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=4.0) as resp:
            if resp.status == 200:
                print(f"📡 [CLOUD SYNC]: Published live tunnel URL for Wearable discovery: {url}")
                return True
    except urllib.error.HTTPError as e:
        reason = "daily ntfy quota reached" if e.code == 429 else e.reason
        print(f"❌ [CLOUD SYNC FAILED]: HTTP {e.code} — {reason}")
    except Exception as e:
        print(f"❌ [CLOUD SYNC FAILED]: {e}")
    return False

def generate_qr_image(url, size=(280, 280), border=3):
    """
    Generates a high-contrast 3-channel BGR QR code image using OpenCV's built-in encoder.
    Zero external pip dependencies required.
    """
    try:
        encoder = cv2.QRCodeEncoder_create()
        qr_matrix = encoder.encode(str(url))
        if qr_matrix is None or qr_matrix.size == 0:
            return None
            
        # Add white quiet-zone border (standard QR requires >= 3 modules)
        padded = cv2.copyMakeBorder(
            qr_matrix, border, border, border, border, 
            cv2.BORDER_CONSTANT, value=255
        )
        
        # In OpenCV QRCodeEncoder: 0 is black (module), 255 is white (background)
        # Convert to standard 3-channel RGB/BGR image
        qr_bgr = cv2.cvtColor(padded, cv2.COLOR_GRAY2BGR)
        
        # Scale with nearest-neighbor to preserve sharp module edges
        qr_scaled = cv2.resize(qr_bgr, size, interpolation=cv2.INTER_NEAREST)
        return qr_scaled
    except Exception as e:
        print(f"⚠️ QR Generation Error: {e}")
        return None

def print_terminal_qr(url):
    """
    Renders a scannable ASCII/Unicode block QR code directly in the terminal.
    Allows instant phone camera scanning from the terminal output.
    """
    try:
        encoder = cv2.QRCodeEncoder_create()
        m = encoder.encode(str(url))
        if m is None or m.size == 0:
            return
            
        border = 2
        h, w = m.shape
        padded = np.full((h + 2*border, w + 2*border), 255, dtype=np.uint8)
        padded[border:-border, border:-border] = m
        
        print("\n" + "=" * 60)
        print("📱 SCAN QR CODE WITH IPHONE CAMERA TO CONNECT 5G STREAM")
        print("=" * 60)
        
        # In dark terminals: '██' is bright (white background), '  ' is dark (black module)
        for row in padded:
            line = ''.join('  ' if val == 0 else '██' for val in row)
            print(line)
            
        print("=" * 60)
        print(f"🌐 5G Live Link: {url}")
        print("👉 Open the link on iPhone to stream 20 FPS Camera & Satellite GPS")
        print("=" * 60 + "\n")
    except Exception as e:
        print(f"⚠️ Terminal QR Error: {e}")

class CloudflareTunnelManager:
    """
    Automated Cloudflare Quick Tunnel Supervisor.
    - Automatically discovers and spawns cloudflared.exe
    - Captures the ephemeral trycloudflare.com URL in real time
    - Generates scannable QR codes for both the Terminal and OpenCV Monitor
    - Saves active link to active_tunnel_url.txt and qr_code.png
    """
    def __init__(self, port=8080, auto_start=True):
        self.port = port
        self.auto_start = auto_start
        self.tunnel_url = None
        self.qr_image = None
        self.process = None
        self.thread = None
        self.is_running = False
        self.lock = threading.Lock()
        
        # File paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.abspath(os.path.join(current_dir, ".."))
        self.url_file_path = os.path.join(self.project_root, "active_tunnel_url.txt")
        self.qr_file_path = os.path.join(current_dir, "qr_code.png")

    def _load_cached_url(self):
        try:
            if os.path.exists(self.url_file_path):
                mtime = os.path.getmtime(self.url_file_path)
                if (time.time() - mtime) < 900: # 15 minutes
                    with open(self.url_file_path, "r", encoding="utf-8") as f:
                        cached = f.read().strip()
                        if cached.startswith("https://") and "trycloudflare.com" in cached:
                            self.tunnel_url = cached
                            self.qr_image = generate_qr_image(cached, size=(280, 280))
                            threading.Thread(target=publish_cloud_sync_url, args=(cached,), daemon=True).start()
        except Exception:
            pass

    def _find_cloudflared_binary(self):
        candidates = [
            os.path.join(self.project_root, "cloudflared.exe"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "cloudflared.exe"),
            "cloudflared.exe",
            "cloudflared"
        ]
        for path in candidates:
            if os.path.exists(path) and os.path.isfile(path):
                return os.path.abspath(path)
                
        # Check PATH
        import shutil
        which_path = shutil.which("cloudflared")
        if which_path:
            return which_path
        return None

    def start(self):
        """Starts the tunnel manager in the background."""
        if not self.auto_start:
            return

        try:
            cfg = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "navigation", "nav_config.json"), encoding="utf-8"))
        except Exception:
            cfg = {}
        if cfg.get("tunnel_mode") == "fixed" and str(cfg.get("fixed_public_url", "")).startswith("https://"):
            self.tunnel_url = cfg["fixed_public_url"].rstrip("/")
            self.qr_image = generate_qr_image(self.tunnel_url, size=(280, 280))
            print_terminal_qr(self.tunnel_url)
            print(f"🔒 [FIXED TUNNEL MODE]: {self.tunnel_url} (cloudflared and ntfy disabled)")
            return

        binary_path = self._find_cloudflared_binary()
        if not binary_path:
            print("⚠️ cloudflared.exe not found. Manual tunnel can be started in a separate terminal.")
            return

        self.is_running = True
        self.thread = threading.Thread(target=self._run_tunnel_process, args=(binary_path,), daemon=True)
        self.thread.start()
        threading.Thread(target=self._heartbeat_sync_worker, daemon=True).start()

    def _heartbeat_sync_worker(self):
        """Refreshes the active tunnel URL in Cloud Sync only when it changes or every 30 minutes."""
        last_published = None
        last_publish_time = 0.0
        while self.is_running:
            sleep_duration = 10.0
            try:
                url = self.get_tunnel_url()
                now = time.time()
                # Only publish if URL changed, or to renew every 30 minutes (1800s)
                if url and (url != last_published or (now - last_publish_time) > 1800.0):
                    if publish_cloud_sync_url(url):
                        last_published = url
                        last_publish_time = now
                    else:
                        sleep_duration = 60.0
            except Exception:
                pass
            time.sleep(sleep_duration)

    def _run_tunnel_process(self, binary_path):
        cmd = [binary_path, "tunnel", "--url", f"http://localhost:{self.port}"]
        try:
            # Creationflags for Windows to detach/manage process cleanly
            creationflags = 0
            if sys.platform == "win32":
                creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
                
            self.process = subprocess.Popen(
                cmd,
                stderr=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                text=True,
                bufsize=1,
                creationflags=creationflags
            )
            
            # Read stderr lines until trycloudflare.com URL is announced
            while self.is_running and self.process and self.process.poll() is None:
                line = self.process.stderr.readline()
                if not line:
                    continue
                    
                # Match quick tunnel URL e.g. https://xxxx-xxxx.trycloudflare.com
                match = re.search(r'https://([a-zA-Z0-9\-]+)\.trycloudflare\.com', line)
                if match:
                    subdomain = match.group(1).lower()
                    # Filter out api or internal cloudflare endpoints
                    if subdomain not in ('api', 'www', 'pkg'):
                        full_url = match.group(0)
                        with self.lock:
                            if self.tunnel_url != full_url:
                                self.tunnel_url = full_url
                                self.qr_image = generate_qr_image(full_url, size=(280, 280))
                                
                                # Save URL and QR code image to disk
                                try:
                                    with open(self.url_file_path, "w", encoding="utf-8") as f:
                                        f.write(full_url + "\n")
                                    if self.qr_image is not None:
                                        cv2.imwrite(self.qr_file_path, self.qr_image)
                                except Exception:
                                    pass
                                    
                                # Print beautiful terminal QR Code
                                print_terminal_qr(full_url)
                                
        except Exception as e:
            if self.is_running:
                print(f"⚠️ Cloudflare tunnel process notice: {e}")

    def get_tunnel_url(self):
        """Returns the active tunnel URL if available."""
        with self.lock:
            return self.tunnel_url

    def get_qr_image(self):
        """Returns the rendered 280x280 BGR QR code image."""
        with self.lock:
            if self.qr_image is not None:
                return self.qr_image.copy()
            return None

    def render_standby_frame(self, target_shape=(480, 640)):
        """
        Renders a high-tech standby monitor frame displaying the QR code,
        connection status, and instructions for the iPhone.
        """
        h, w = target_shape[:2]
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:] = (16, 16, 22) # Modern deep slate background
        
        # Header Titles
        cv2.putText(frame, "5G WEARABLE ASSISTIVE SYSTEM", (int(w * 0.16), 36), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 229, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "[ AI EDGE SERVER STANDBY ]", (int(w * 0.28), 64), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 140), 1, cv2.LINE_AA)
        
        # QR Code Display
        qr = self.get_qr_image()
        url = self.get_tunnel_url()
        
        if qr is not None:
            qr_h, qr_w = qr.shape[:2]
            # Center QR code
            x_start = max(0, (w - qr_w) // 2)
            y_start = 82
            
            if y_start + qr_h < h:
                frame[y_start:y_start+qr_h, x_start:x_start+qr_w] = qr
                # Thin accent border around QR
                cv2.rectangle(frame, (x_start - 2, y_start - 2), (x_start + qr_w + 1, y_start + qr_h + 1), (0, 229, 255), 1)
                
            # Guidance text below QR
            text_y = y_start + qr_h + 30
            cv2.putText(frame, "SCAN WITH IPHONE CAMERA TO CONNECT 5G STREAM", (int(w * 0.08), text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
            
            if url:
                short_url = url.replace("https://", "")
                if len(short_url) > 52:
                    short_url = short_url[:50] + "..."
                cv2.putText(frame, f"URL: {short_url}", (int(w * 0.10), text_y + 24), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 229, 255), 1, cv2.LINE_AA)
                            
            cv2.putText(frame, "Awaiting iPhone video & satellite GPS packets...", (int(w * 0.14), min(h - 14, text_y + 48)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 160, 180), 1, cv2.LINE_AA)
        else:
            # Tunnel starting up
            cv2.putText(frame, "GENERATING 5G SECURE TUNNEL & QR CODE...", (int(w * 0.12), int(h * 0.45)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 229, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, "Starting cloudflared on port 8080...", (int(w * 0.22), int(h * 0.52)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)
            
        return frame

    def stop(self):
        """Cleanly terminates cloudflared subprocess."""
        self.is_running = False
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=2)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
            self.process = None
        print("🛑 Cloudflare Tunnel Manager stopped.")
