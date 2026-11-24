# Setup (short version)

The complete, current setup guide is the [Setup section of the README](../README.md#setup). This page keeps only the quick Tailscale-based path. Other options (same Wi-Fi/5G hotspot, Cloudflare quick tunnel) are in the README.

## 1. Connect the two devices (any network): Tailscale
- Laptop (Windows/Linux): install Tailscale, sign in, note its IP:  `tailscale ip -4`  (100.x.y.z)
- UNO Q (SSH or `adb shell`):
  `curl -fsSL https://tailscale.com/install.sh | sh && sudo tailscale up`  (sign in with the same account)
- Check:  `tailscale ping <laptop-name>` -> "via <ip>:port" = direct (fast); "via DERP" = relayed (slower; the app will notice and use local more).
- Allow TCP 8765 on the laptop firewall (Tailscale interface).

## 2. Laptop
    cd laptop
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # CUDA build
    pip install -r requirements.txt
    python server.py            # add --model yolov8s.pt for more accuracy
    python server.py --flow raft                 # optical flow on the GPU (torchvision RAFT) instead of CPU DIS
    python server.py --motion off                # plain detector, like before

## 3. UNO Q
1. In Arduino App Lab create a new app (so it generates sketch.yaml etc.), add the **Web UI** brick.
2. Replace/add: `app.yaml`, `python/*`, `sketch/sketch.ino`, `assets/*` from unoq_app/.
3. Local model: on the laptop run `python ..\tools\export_onnx.py` (venv active) -> tools/yolov8n_320.onnx; copy it to python/models/ on the board.
4. Set the laptop address: edit `LAPTOP_URL` in main.py (or type it in the UI box), e.g. `ws://100.101.102.103:8765`.
5. Plug in a USB webcam, press Run, open the app's web UI.

## Offline / same-network setup (no internet)
Nothing in the runtime needs the internet. Put the laptop and the board on one router, hotspot or Ethernet switch:
1. Laptop: `ipconfig` (Windows) / `ip a` (Linux) -> note its address, e.g. `192.168.1.20`; allow TCP 8765 in the firewall.
2. Laptop: `python server.py` (models and packages must already be installed/downloaded once while online).
3. Board web page: connect to `ws://192.168.1.20:8765` instead of the Tailscale address.
4. Phone camera: the phone joins the same network; use `http://<phone-ip>:8080/video`.
Install the Python packages, `yolov8n.pt` and the exported ONNX model while you still have internet.
