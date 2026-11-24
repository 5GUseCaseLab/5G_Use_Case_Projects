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

PAGE = """<!doctype html><html><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>
<title>CrowdGate</title>
<style>
:root{--bg:#0b1220;--card:#131c2e;--line:#22304a;--txt:#e8eef9;--mut:#8ea0bd;--c:#3b4a63}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--txt);font-family:system-ui,Segoe UI,Roboto,Arial,sans-serif}
.wrap{max-width:1100px;margin:0 auto;padding:18px}
header{display:flex;justify-content:space-between;align-items:center;margin-bottom:16px}
h1{margin:0;font-size:1.5rem;letter-spacing:.5px}h1 span{color:var(--mut);font-weight:400;font-size:.95rem;margin-left:10px}
.live{font-size:.85rem;color:var(--mut);display:flex;align-items:center;gap:8px}
.dot{width:10px;height:10px;border-radius:50%;background:#e0b000}
.dot.on{background:#2ecc71;box-shadow:0 0 8px #2ecc71}.dot.off{background:#e74c3c}
.grid{display:grid;grid-template-columns:2fr 1fr;gap:16px}
@media(max-width:800px){.grid{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:16px;overflow:hidden}
.vid{position:relative;background:#000;min-height:200px}
.vid img{display:block;width:100%;height:auto}
.tag{position:absolute;top:10px;left:10px;background:rgba(0,0,0,.6);padding:4px 10px;border-radius:999px;font-size:.78rem}
.side{display:flex;flex-direction:column;gap:16px}
#st{padding:26px 16px;text-align:center;background:var(--c);transition:background .3s}
#sn{font-size:2.6rem;font-weight:800;letter-spacing:1px;line-height:1.1}
#sa{margin-top:8px;font-size:1rem;opacity:.92}
.tiles{display:grid;grid-template-columns:1fr 1fr;gap:12px;padding:14px}
.tile{background:#0f1728;border:1px solid var(--line);border-radius:12px;padding:12px}
.tile b{display:block;font-size:1.8rem}.tile small{color:var(--mut)}
.wide{grid-column:span 2}
.bar{height:10px;border-radius:999px;background:#0f1728;border:1px solid var(--line);margin-top:8px;overflow:hidden}
#bf{height:100%;width:0;background:var(--c);transition:width .4s,background .3s}
footer{margin-top:16px;color:var(--mut);font-size:.8rem;text-align:center}
</style></head><body><div class=wrap>
<header><h1>CrowdGate<span>Edge crowd-density monitor</span></h1>
<div class=live><span class=dot id=dt></span><span id=lt>connecting</span></div></header>
<div class=grid>
<div class='card vid'><span class=tag>LIVE &middot; 5G camera &rarr; MEC</span><img id=v alt=''></div>
<div class=side>
<div class=card><div id=st><div id=sn>...</div><div id=sa>waiting for data</div></div>
<div class=tiles>
<div class=tile><small>People in zone</small><b id=pc>-</b></div>
<div class=tile><small>Density (per m&sup2;)</small><b id=pd>-</b></div>
<div class='tile wide'><small>Zone area</small><b id=pa>-</b>
<div class=bar><div id=bf></div></div></div>
</div></div></div></div>
<footer>GREEN = open &nbsp;|&nbsp; YELLOW = caution &nbsp;|&nbsp; RED = gate closed &nbsp;|&nbsp; CAMERA FAILURE = fail-safe (treated as RED)</footer>
</div>
<script>
var COL={GREEN:'#1a9850',YELLOW:'#d9a400',RED:'#d73027',CAMERA_FAILURE:'#7b1fa2'};
var MSG={GREEN:'Safe - gate open',YELLOW:'Getting crowded - caution',RED:'Overcrowded - gate closed',CAMERA_FAILURE:'No camera frame - fail-safe, treat as RED',STARTING:'Starting up'};
function $(i){return document.getElementById(i)}
setInterval(async function(){
 try{var r=await (await fetch('/status',{cache:'no-store'})).json();
  var c=COL[r.state]||'#3b4a63';document.documentElement.style.setProperty('--c',c);
  $('sn').textContent=r.state==='CAMERA_FAILURE'?'CAMERA FAILURE':r.state;
  $('sa').textContent=MSG[r.state]||'';
  var fail=r.state==='CAMERA_FAILURE';
  $('pc').textContent=fail?'-':r.count;
  $('pd').textContent=fail?'-':Number(r.density).toFixed(2);
  $('pa').textContent=(r.area_m2!=null?r.area_m2:'-')+' m²';
  $('bf').style.width=Math.min(100,(r.density||0)/4*100)+'%';
  $('dt').className='dot on';$('lt').textContent='live '+new Date().toLocaleTimeString();
 }catch(e){$('dt').className='dot off';$('lt').textContent='dashboard offline'}
},1000);
function tick(){var n=new Image();n.onload=function(){$('v').src=n.src;setTimeout(tick,80)};
 n.onerror=function(){setTimeout(tick,500)};n.src='/frame.jpg?'+Date.now()}tick();
</script></body></html>
"""


@app.route("/")
def index():
    return PAGE


@app.route("/frame.jpg")
def frame_jpg():
    with lock:
        jpg = latest["jpg"]
    if not jpg:
        return Response(status=503)
    return Response(jpg, mimetype="image/jpeg", headers={"Cache-Control": "no-store"})


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
        last = None
        while True:
            with lock:
                jpg = latest["jpg"]
            if jpg and jpg is not last:
                last = jpg
                yield (b"--f\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpg)).encode()
                       + b"\r\n\r\n" + jpg + b"\r\n")
            time.sleep(0.02)
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
