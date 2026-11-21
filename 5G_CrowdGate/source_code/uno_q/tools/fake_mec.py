"""Stand-in for the MEC's /status page, for testing the gate without the camera or the MEC.
Run on any computer on the same Wi-Fi as the UNO Q:  python fake_mec.py
Then point the gate at this computer's IP (MEC_HOST in uno_q/python/main.py) and type G / Y / R / C."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

state = {"state": "GREEN"}


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        b = json.dumps(state).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b)

    def log_message(self, *a):
        pass


threading.Thread(target=HTTPServer(("0.0.0.0", 5000), H).serve_forever, daemon=True).start()
while True:
    s = input("G / Y / R / C (camera failure): ").strip().upper()
    state["state"] = {"G": "GREEN", "Y": "YELLOW", "R": "RED", "C": "CAMERA_FAILURE"}.get(s, state["state"])
