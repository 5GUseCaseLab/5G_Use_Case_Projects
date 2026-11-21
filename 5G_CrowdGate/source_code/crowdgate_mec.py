"""MAIN SERVICE (runs on the MEC server).
Camera/video -> YOLO person detection -> keep people standing inside the zone -> density
-> state (GREEN/YELLOW/RED) -> JSON /status (polled by the Arduino UNO Q gate) + live web dashboard for the phone.

Run:  python crowdgate_mec.py --source rtsp://USER:PASSWORD@IP:554/stream1 --model best.pt --config config.json
Test with a video:  python crowdgate_mec.py --source mall.mp4 --area 20 --mqtt-host none
Dashboard: http://<MEC-IP>:5000   (JSON: /status)
"""
import argparse
import json
import os
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response, jsonify

import common

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay")

app = Flask(__name__)
latest = {"jpg": None, "status": {"state": "STARTING", "count": 0, "density": 0.0}}
lock = threading.Lock()

PAGE = """<html><head><meta name=viewport content='width=device-width,initial-scale=1'>
<style>body{font-family:sans-serif;background:#111;color:#fff;text-align:center}
#s{font-size:3em;padding:.3em;border-radius:12px;margin:10px}img{max-width:100%}</style></head><body>
<h2>CrowdGate</h2><div id=s>...</div><div id=d></div><img src=/video>
<script>setInterval(async()=>{let r=await (await fetch('/status')).json();
let c={GREEN:'#1a9850',YELLOW:'#e0b000',RED:'#d73027',CAMERA_FAILURE:'#7b1fa2'}[r.state]||'#555';
s.style.background=c;s.textContent=r.state==='CAMERA_FAILURE'?'CAMERA FAILURE - treat as RED':r.state;
d.textContent=r.state==='CAMERA_FAILURE'?'no fresh camera frame for '+r.stale_s+' s':r.count+' people | '+r.density.toFixed(2)+' per m2 | area '+r.area_m2+' m2'},1000)</script></body></html>"""


@app.route("/")
def index():
    return PAGE


@app.route("/status")
def status():
    with lock:
        return jsonify(latest["status"])


class FrameGrabber:
    """Reads the camera in a background thread and keeps ONLY the newest frame, so slow YOLO
    inference never makes the video lag behind (RTSP buffers old frames otherwise).
    Reconnects automatically; loops video files (paced to their fps) for offline demos."""

    def __init__(self, source):
        self.src = int(source) if str(source).isdigit() else source
        self.is_file = os.path.isfile(str(source))
        self.frame, self.idx, self.stop = None, 0, False
        self.cap = self._open()
        self.delay = 1.0 / (self.cap.get(cv2.CAP_PROP_FPS) or 25) if self.is_file else 0
        threading.Thread(target=self._run, daemon=True).start()

    def _open(self):
        cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
        for prop, ms in (("CAP_PROP_OPEN_TIMEOUT_MSEC", 5000), ("CAP_PROP_READ_TIMEOUT_MSEC", 5000)):
            if hasattr(cv2, prop):
                cap.set(getattr(cv2, prop), ms)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _run(self):
        while not self.stop:
            ok, f = self.cap.read()
            if not ok:
                if self.is_file:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                else:
                    print("Stream lost, reconnecting...")
                    self.cap.release()
                    time.sleep(1)
                    self.cap = self._open()
                continue
            self.frame, self.idx = f, self.idx + 1
            if self.delay:
                time.sleep(self.delay)

    def latest(self):
        return self.idx, self.frame


@app.route("/video")
@app.route("/people_det")          # same route name your lab's main.py used
def video():
    def gen():
        while True:
            with lock:
                jpg = latest["jpg"]
            if jpg:
                yield b"--f\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            time.sleep(0.04)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=f")


det = {"boxes": np.zeros((0, 4)), "state": "STARTING", "count": 0, "density": 0.0, "fps": 0.0, "zone": None}
COLORS = {"GREEN": (0, 200, 0), "YELLOW": (0, 200, 255), "RED": (0, 0, 255),
          "CAMERA_FAILURE": (200, 0, 200), "STARTING": (150, 150, 150)}


def render_loop(grabber, a):
    """Smooth dashboard video: shows the NEWEST camera frame at --view-fps with the LAST detections drawn on it.
    YOLO can be slow (a few fps) without the picture looking choppy; only the boxes update at YOLO speed.
    Frames are shrunk to --view-w before JPEG encoding (encoding 4K frames is slow)."""
    last_idx = -1
    while True:
        t0 = time.time()
        idx, frame = grabber.latest()
        if frame is not None and idx != last_idx:
            last_idx = idx
            h, w = frame.shape[:2]
            k = min(1.0, a.view_w / w)
            img = cv2.resize(frame, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA) if k < 1 else frame.copy()
            d = dict(det)
            color = COLORS.get(d["state"], (150, 150, 150))
            if d["zone"] is not None:
                cv2.polylines(img, [np.int32(np.array(d["zone"]) * k)], True, (255, 255, 0), 2)
            for x1, y1, x2, y2 in (d["boxes"] * k).astype(int):
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            txt = ("CAMERA FAILURE" if d["state"] == "CAMERA_FAILURE" else
                   f"{d['state']}  people={d['count']}  {d['density']:.2f}/m2  AI {d['fps']:.1f}fps")
            cv2.putText(img, txt, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()
            with lock:
                latest["jpg"] = jpg
            if a.show:
                cv2.imshow("CrowdGate", img)
                cv2.waitKey(1)
        time.sleep(max(0.0, 1.0 / a.view_fps - (time.time() - t0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=os.environ.get(
        "CAMERA_URL", "rtsp://USER:PASSWORD@192.168.128.10:554/avstream/channel=1/stream=0.sdp"),
        help="RTSP URL / video file / camera index (default: env CAMERA_URL, else the lab camera)")
    ap.add_argument("--model", default="yolo11s.pt", help="your trained best.pt (or a COCO model for testing)")
    ap.add_argument("--openvino", action="store_true",
                    help="export model to OpenVINO once and use it (faster on Intel CPUs, no GPU needed)")
    ap.add_argument("--config", default=None, help="config.json from calibrate_area.py")
    ap.add_argument("--area", type=float, default=None, help="manual area in m2 if no config")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--person-class", type=int, default=0, help="0 for COCO; use your class id for custom model")
    ap.add_argument("--yellow", type=float, default=2.0)
    ap.add_argument("--red", type=float, default=4.0)
    ap.add_argument("--mqtt-host", default="none", help="optional MQTT broker host; 'none' (default) disables MQTT")
    ap.add_argument("--mqtt-topic", default="crowdgate/state")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--show", action="store_true", help="open an OpenCV window")
    ap.add_argument("--log", default="log.csv")
    ap.add_argument("--view-w", type=int, default=960, help="width of the dashboard video (smaller = smoother)")
    ap.add_argument("--view-fps", type=float, default=15, help="dashboard video frame rate (independent of YOLO speed)")
    ap.add_argument("--stale-s", type=float, default=3.0,
                    help="no NEW camera frame for this many seconds => state CAMERA_FAILURE (gate goes to fail-safe)")
    ap.add_argument("--rescale-zone", action="store_true",
                    help="if the stream size differs from the calibration size, scale the zone instead of exiting "
                         "(only valid if the aspect ratio is the same AND the camera view/zoom is unchanged)")
    a = ap.parse_args()

    model = common.load_model(a.model, a.openvino, a.imgsz)

    zone, area, cal_size = None, a.area, None
    if a.config:
        cfg = common.load_config(a.config)
        zone, area = cfg["zone_px"], cfg["area_m2"]
        cal_size = cfg.get("source_size")
    if not area:
        raise SystemExit("Give --config or --area")
    sm = common.DensityState(area, a.yellow, a.red)

    client = None
    if a.mqtt_host.lower() != "none":
        import paho.mqtt.client as mqtt
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2) if hasattr(mqtt, "CallbackAPIVersion") else mqtt.Client()
        client.connect(a.mqtt_host, 1883, 60)
        client.loop_start()

    threading.Thread(target=lambda: app.run(host="0.0.0.0", port=a.port, threaded=True), daemon=True).start()

    grabber = FrameGrabber(a.source)
    det["zone"] = zone
    threading.Thread(target=render_loop, args=(grabber, a), daemon=True).start()
    last_pub, t_prev, last_idx = 0, time.time(), -1
    last_new, size_checked, failed = time.time(), False, False
    prev_small = None
    log = open(a.log, "w")
    log.write("time,raw_count,smooth_count,density,state,fps\n")

    while True:
        idx, frame = grabber.latest()
        if frame is None or idx == last_idx:          # nothing new yet
            now = time.time()
            if now - last_new > a.stale_s:            # camera dead/frozen -> unsafe, say so loudly
                if not failed:
                    print(f"CAMERA FAILURE: no new frame for {a.stale_s:.0f}s")
                    failed = True
                status = {"state": "CAMERA_FAILURE", "count": 0, "density": 0.0, "area_m2": area,
                          "ts": now, "stale_s": round(now - last_new, 1)}
                det.update(state="CAMERA_FAILURE", boxes=np.zeros((0, 4)))
                with lock:
                    latest["status"] = status
                if now - last_pub >= 1.0:
                    last_pub = now
                    if client:
                        client.publish(a.mqtt_topic, json.dumps(status))
            time.sleep(0.01)
            continue
        last_idx = idx
        small = cv2.cvtColor(cv2.resize(frame, (480, 270), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        if prev_small is not None and np.array_equal(small, prev_small):
            now = time.time()
            if now - last_new > a.stale_s:
                if not failed:
                    print(f"CAMERA FREEZE: same picture for {a.stale_s:.0f}s")
                    failed = True
                status = {"state": "CAMERA_FAILURE", "count": 0, "density": 0.0, "area_m2": area,
                          "ts": now, "stale_s": round(now - last_new, 1)}
                det.update(state="CAMERA_FAILURE", boxes=np.zeros((0, 4)))
                with lock:
                    latest["status"] = status
                if now - last_pub >= 1.0:
                    last_pub = now
                    if client:
                        client.publish(a.mqtt_topic, json.dumps(status))
            time.sleep(0.02)
            continue
        prev_small = small
        last_new = time.time()
        if failed:                                    # recovered: forget stale counts
            print("Camera recovered")
            sm.buf.clear()
            failed = False
        frame = frame.copy()

        if not size_checked:                          # calibration only valid for the same resolution
            size_checked = True
            h, w = frame.shape[:2]
            if zone is not None and cal_size and [w, h] != list(cal_size):
                cw, ch = cal_size
                if a.rescale_zone and abs(w / h - cw / ch) < 0.01:
                    zone = [[x * w / cw, y * h / ch] for x, y in zone]
                    det["zone"] = zone
                    print(f"WARNING: zone rescaled {cw}x{ch} -> {w}x{h}")
                else:
                    raise SystemExit(f"Stream is {w}x{h} but {a.config} was calibrated at {cw}x{ch}. "
                                     "Use the same stream (e.g. stream=0) or recalibrate.")

        r = model.predict(frame, classes=[a.person_class], conf=a.conf, imgsz=a.imgsz, verbose=False)[0]
        boxes = r.boxes.xyxy.cpu().numpy() if len(r.boxes) else np.zeros((0, 4))
        if zone is not None and len(boxes):
            keep = common.in_zone(common.foot_points(boxes), zone)
            boxes = boxes[np.array(keep, dtype=bool)]
        count = len(boxes)
        state, smooth, density = sm.update(count)

        now = time.time()
        fps, t_prev = 1 / max(now - t_prev, 1e-6), now
        det.update(boxes=boxes, state=state, count=count, density=density, fps=fps)

        status = {"state": state, "count": count, "density": round(density, 3), "area_m2": area, "ts": now}
        with lock:
            latest["status"] = status

        if now - last_pub >= 1.0:                       # publish once per second
            last_pub = now
            if client:
                client.publish(a.mqtt_topic, json.dumps(status))
            log.write(f"{now:.2f},{count},{smooth:.1f},{density:.3f},{state},{fps:.1f}\n")
            log.flush()


if __name__ == "__main__":
    main()
