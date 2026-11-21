import base64
import logging
import os
import time
import uuid
from collections import OrderedDict

import cv2
import numpy as np
import requests
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# 1. SERVE THE WEB UI
# This safely mounts the UI so it doesn't block the AI endpoints!
app.mount("/ui", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "www"), html=True), name="www")

STORE = OrderedDict()
MAX_IMAGES = 20
TIER1_URL = os.environ.get("TIER1_URL")

class StubModel:
    name = "STUB"
    def infer(self, gray):
        g = cv2.GaussianBlur(gray, (0, 0), 6).astype(np.float32)
        cam = np.clip((g - g.mean()) / (g.std() + 1e-6), 0, None)
        cam /= cam.max() + 1e-6
        return float(cam.max()), cam

class OnnxCamModel:
    MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 3, 1, 1)
    STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 3, 1, 1)

    def __init__(self, path):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        if os.environ.get("TIER0_THREADS"):
            opts.intra_op_num_threads = int(os.environ["TIER0_THREADS"])
        self.sess = ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])
        shape = self.sess.get_inputs()[0].shape
        self.size = int(shape[2])
        self.name = f"onnx:{os.path.basename(path)}"

    def infer(self, gray):
        x = cv2.resize(gray, (self.size, self.size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        x = (np.stack([x, x, x])[None] - self.MEAN) / self.STD
        logit, cam = self.sess.run(None, {"image": x})
        prob = float(1.0 / (1.0 + np.exp(-float(logit.reshape(-1)[0]))))
        cam = np.maximum(cam[0, 0], 0)
        cam = cam / (cam.max() + 1e-6)
        return prob, (cam * prob).astype(np.float32)

def load_model():
    path = os.environ.get("TIER0_MODEL")
    if not path:
        return StubModel()
    if not os.path.isfile(path):
        raise RuntimeError(f"TIER0_MODEL={path} does not exist")
    return OnnxCamModel(path)

MODEL = load_model()

# 2. HIGH SENSITIVITY TWEAKS
def extract_boxes(cam, min_peak=0.15, min_area=15, max_boxes=5):
    if cam.max() < min_peak: return []
    u8 = (cam * 255).astype(np.uint8)
    t, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (u8 > max(t, 0.25 * 255)).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if area < min_area: continue
        boxes.append({"x": x, "y": y, "w": w, "h": h, "score": float(cam[y:y + h, x:x + w].max())})
    boxes.sort(key=lambda b: -b["score"])
    boxes = boxes[:max_boxes]
    for i, b in enumerate(boxes): b["id"] = i
    return boxes

def heat_png(cam):
    u8 = (cam * 255).astype(np.uint8)
    bgr = cv2.applyColorMap(u8, cv2.COLORMAP_JET)
    alpha = np.where(u8 > 40, u8 * 0.7, 0).astype(np.uint8)
    ok, buf = cv2.imencode(".png", np.dstack([bgr, alpha]))
    return base64.b64encode(buf).decode()

@app.get("/health")
def health(): return {"ok": True, "model": MODEL.name}

# 3. THE MISSING ENDPOINT!
@app.post("/infer")
async def infer(file: UploadFile = File(...)):
    img = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None: raise HTTPException(400, "not a readable image")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    t0 = time.perf_counter()
    score, cam = MODEL.infer(gray)
    infer_ms = (time.perf_counter() - t0) * 1000
    cam = np.clip(cv2.resize(cam, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_CUBIC), 0, 1)
    image_id = uuid.uuid4().hex[:8]
    STORE[image_id] = img
    while len(STORE) > MAX_IMAGES: STORE.popitem(last=False)
    return {"image_id": image_id, "width": gray.shape[1], "height": gray.shape[0],
            "model": MODEL.name, "infer_ms": round(infer_ms, 1),
            "score": score, "boxes": extract_boxes(cam), "heatmap_png": heat_png(cam)}

class Expand(BaseModel):
    image_id: str
    x: int
    y: int
    w: int
    h: int
    margin: int = 16

# 4. MULTI-IMAGE CROP FOR TIER 1
@app.post("/expand")
def expand(r: Expand):
    img = STORE.get(r.image_id)
    if img is None: raise HTTPException(404, "unknown image_id")
    H, W = img.shape[:2]
    x0, y0 = max(r.x - r.margin, 0), max(r.y - r.margin, 0)
    x1, y1 = min(r.x + r.w + r.margin, W), min(r.y + r.h + r.margin, H)
    
    ok_crop, buf_crop = cv2.imencode(".png", img[y0:y1, x0:x1])  
    ok_full, buf_full = cv2.imencode(".png", img) 
    
    if not TIER1_URL:
        return {"stub": True, "crop_box": [x0, y0, x1, y1], "narrative": "Tier 1 not configured."}
    try:
        files = {"crop": ("crop.png", buf_crop.tobytes(), "image/png"), "full": ("full.png", buf_full.tobytes(), "image/png")}
        resp = requests.post(TIER1_URL.rstrip("/") + "/characterize", files=files, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise HTTPException(502, f"Tier 1 unreachable: {e}")
    return {**resp.json(), "crop_box": [x0, y0, x1, y1]}