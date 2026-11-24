"""EXPERIMENT 2 (server side): minimal inference server. Run the SAME file on (a) the MEC and (b) a public cloud VM,
then compare round-trip latency with latency_client.py.

  python infer_server.py --model best.pt --port 8000
Use --openvino on the MEC (same as crowdgate_mec.py) for a fair comparison with the real service.
NOTE: server_ms and the client's round trip measure upload + network + inference only, NOT camera-to-gate latency.
POST /infer  (body = JPEG bytes)  ->  {"count": N, "server_ms": T}
"""
import argparse
import time

import cv2
import numpy as np
from flask import Flask, jsonify, request

import common

app = Flask(__name__)
model = None
ARGS = None


@app.route("/infer", methods=["POST"])
def infer():
    t0 = time.perf_counter()
    img = cv2.imdecode(np.frombuffer(request.data, np.uint8), cv2.IMREAD_COLOR)
    r = model.predict(img, classes=[0], conf=ARGS.conf, imgsz=ARGS.imgsz, verbose=False)[0]
    return jsonify(count=len(r.boxes), server_ms=round((time.perf_counter() - t0) * 1000, 1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--openvino", action="store_true")
    ARGS = ap.parse_args()
    model = common.load_model(ARGS.model, ARGS.openvino, ARGS.imgsz)
    model.predict(np.zeros((480, 640, 3), np.uint8), verbose=False)   # warm-up
    app.run(host="0.0.0.0", port=ARGS.port, threaded=True)
