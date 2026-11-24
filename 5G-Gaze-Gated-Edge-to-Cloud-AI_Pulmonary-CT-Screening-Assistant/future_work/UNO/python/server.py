import asyncio, os, threading, time
import cv2
import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
import base64
import numpy as np
import unoq_service as tier0

CHANNELS = ("eye", "front", "dwell", "resnet", "resnet_heat")
TOKEN = os.getenv("DASH_TOKEN")          # optional; if set, POST endpoints require X-Token


class Store:
    def __init__(self):
        self.lock = threading.Lock()
        self.jpg = {c: None for c in CHANNELS}
        self.ts = {c: 0.0 for c in CHANNELS}
        self.status = {"state": "idle", "remaining": 0, "pupil": None, "gaze": None}
        self.viewers = 0
        self._trigger = False
        self.raw_front = None

    # ---- called from the gaze script ----
    def publish(self, name, frame_bgr, quality=70):
        ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            self.publish_jpeg(name, buf.tobytes())

    def publish_jpeg(self, name, data: bytes):
        with self.lock:
            self.jpg[name] = data
            self.ts[name] = time.time()

    def set_status(self, **kw):
        with self.lock:
            self.status.update(kw)

    def request_trigger(self):
        with self.lock:
            self._trigger = True

    def pop_trigger(self):
        with self.lock:
            t, self._trigger = self._trigger, False
            return t

    def has_viewers(self):
        return self.viewers > 0

    def get(self, name):
        with self.lock:
            return self.jpg[name], self.ts[name]


store = Store()
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def check(token):
    if TOKEN and token != TOKEN:
        raise HTTPException(401, "bad token")


@app.get("/")
def index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "dashboard.html"))


@app.get("/stream/{name}")
async def stream(name: str):
    if name not in CHANNELS:
        raise HTTPException(404)

    async def gen():
        store.viewers += 1
        last = 0.0
        try:
            while True:
                jpg, ts = store.get(name)
                if jpg is not None and ts != last:
                    last = ts
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                           + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                await asyncio.sleep(0.05)
        finally:
            store.viewers -= 1

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/image/{name}")
def image(name: str):
    if name not in CHANNELS:
        raise HTTPException(404)
    data, _ = store.get(name)
    if data is None:
        raise HTTPException(404, "no frame yet")
    mt = "image/png" if data[:4] == b"\x89PNG" else "image/jpeg"
    return Response(data, media_type=mt, headers={"Cache-Control": "no-store"})

@app.get("/api/status")
def status():
    with store.lock:
        return {**store.status, "ts": dict(store.ts), "server_time": time.time()}


@app.post("/api/trigger")
def trigger(x_token: str | None = Header(None)):
    check(x_token)
    store.request_trigger()
    return {"ok": True}


@app.post("/api/ingest/{name}")          # cloud ResNet posts its heatmap JPEG here
async def ingest(name: str, request: Request, x_token: str | None = Header(None)):
    check(x_token)
    if name not in CHANNELS:
        raise HTTPException(404)
    store.publish_jpeg(name, await request.body())
    return {"ok": True}


tier0.BUSY_CHECK = lambda: store.status.get("state") == "recording"
app.include_router(tier0.router)          # /infer, /expand, /health


def compose(frame, result):
    """Heat layer over the frame, with numbered boxes, for the dashboard's ResNet panel."""
    png = base64.b64decode(result["heatmap_png"])
    rgba = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)
    a = rgba[..., 3:4].astype(np.float32) / 255.0
    out = (frame * (1 - a) + rgba[..., :3] * a).astype(np.uint8)
    for b in result["boxes"]:
        cv2.rectangle(out, (b["x"], b["y"]), (b["x"] + b["w"], b["y"] + b["h"]), (0, 255, 0), 2)
        cv2.putText(out, f"#{b['id']}", (b["x"], max(b["y"] - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return out, png


@app.post("/api/analyze")
def analyze(x_token: str | None = Header(None)):
    check(x_token)
    frame = store.raw_front
    if frame is None:
        raise HTTPException(409, "no frontal frame yet")
    result = tier0.run_infer(frame.copy())
    comp, png = compose(frame, result)
    store.publish("resnet", comp, quality=85)
    store.publish_jpeg("resnet_heat", png)    # raw RGBA PNG for the overlay slider
    summary = {k: v for k, v in result.items() if k != "heatmap_png"}
    store.set_status(analysis=summary)
    return summary


def run_in_thread(host="0.0.0.0", port=8000):
    cfg = uvicorn.Config(app, host=host, port=port, log_level="warning")
    threading.Thread(target=uvicorn.Server(cfg).run, daemon=True).start()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)