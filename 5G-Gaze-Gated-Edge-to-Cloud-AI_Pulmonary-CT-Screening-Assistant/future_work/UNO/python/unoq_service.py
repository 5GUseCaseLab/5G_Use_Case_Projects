"""
UNO Q Tier 0 service.
Run:  uvicorn unoq_service:app --host 0.0.0.0 --port 8000
Env:  TIER0_MODEL=/path/to/tier0.onnx   real model (see export_onnx.py). Unset = STUB model, loudly labelled.
      TIER0_THREADS=4                   optional onnxruntime thread count
      TIER1_URL=http://<tier1-host>:9000   optional; without it /expand returns a stub
Deps: pip install fastapi uvicorn python-multipart opencv-python-headless numpy requests onnxruntime
"""
import base64
import logging
import os
import time
import uuid
from collections import OrderedDict

import cv2
import numpy as np
import requests
import threading
from fastapi import APIRouter, File, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel

router = APIRouter()          # was: app = FastAPI() + app.add_middleware(CORS...)
BUSY_CHECK = None             # server.py sets this; returns True while a gaze recording is running
INFER_LOCK = threading.Lock() # one inference at a time on the A53
STORE = OrderedDict()  # image_id -> original-resolution BGR image
MAX_IMAGES = 20
TIER1_URL = os.environ.get("TIER1_URL")


class StubModel:
    """Fake Tier 0 so the whole loop can be tested: bright compact regions light up.
    NOT a detector. Contract shared with the real model: infer(gray uint8) -> (score, cam in [0,1])."""
    name = "STUB"

    def infer(self, gray):
        g = cv2.GaussianBlur(gray, (0, 0), 6).astype(np.float32)
        cam = np.clip((g - g.mean()) / (g.std() + 1e-6), 0, None)
        cam /= cam.max() + 1e-6
        return float(cam.max()), cam


class OnnxCamModel:
    """Real Tier 0: the exported CamClassifier (export_onnx.py). One forward pass returns the
    logit AND the class-activation map, so no backprop is needed on the board.
    Preprocessing must match train.py exactly: grayscale -> 512x512 -> /255 -> replicate to 3 channels
    -> ImageNet mean/std. Inputs should be lung-windowed 8-bit slices like the ones data_prep.py writes."""
    MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 3, 1, 1)
    STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 3, 1, 1)

    def __init__(self, path):
        import onnxruntime as ort  # imported here so the stub path works without onnxruntime installed
        opts = ort.SessionOptions()
        if os.environ.get("TIER0_THREADS"):
            opts.intra_op_num_threads = int(os.environ["TIER0_THREADS"])
        self.sess = ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])
        shape = self.sess.get_inputs()[0].shape  # [1, 3, H, W]
        self.size = int(shape[2])
        outs = [o.name for o in self.sess.get_outputs()]
        if outs != ["logit", "cam"]:
            raise RuntimeError(f"unexpected model outputs {outs}; expected ['logit', 'cam'] from export_onnx.py")
        self.name = f"onnx:{os.path.basename(path)}"

    def infer(self, gray):
        x = cv2.resize(gray, (self.size, self.size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        x = (np.stack([x, x, x])[None] - self.MEAN) / self.STD
        logit, cam = self.sess.run(None, {"image": x})
        prob = float(1.0 / (1.0 + np.exp(-float(logit.reshape(-1)[0]))))
        cam = np.maximum(cam[0, 0], 0)
        cam = cam / (cam.max() + 1e-6)
        # scale by confidence so a low-probability slice shows a weak heatmap and yields no boxes
        return prob, (cam * prob).astype(np.float32)


def load_model():
    path = os.environ.get("TIER0_MODEL")
    if not path:
        logging.warning("TIER0_MODEL not set - running the STUB model (not a real detector)")
        return StubModel()
    if not os.path.isfile(path):
        # fail loudly: silently serving stub output when a real model was intended is the dangerous failure
        raise RuntimeError(f"TIER0_MODEL={path} does not exist; refusing to fall back to the stub")
    return OnnxCamModel(path)


MODEL = load_model()


def extract_boxes(cam, min_peak=0.15, min_area=15, max_boxes=5):
    """Option B: adaptive (Otsu) threshold + connected components. cam is float [0,1] at image size."""
    if cam.max() < min_peak:
        return []
    u8 = (cam * 255).astype(np.uint8)
    t, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (u8 > max(t, 0.25 * 255)).astype(np.uint8)  # floor so weak noise isn't boxed
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if area < min_area:
            continue
        boxes.append({"x": x, "y": y, "w": w, "h": h, "score": float(cam[y:y + h, x:x + w].max())})
    boxes.sort(key=lambda b: -b["score"])
    boxes = boxes[:max_boxes]
    for i, b in enumerate(boxes):
        b["id"] = i
    return boxes


def heat_png(cam):
    """Colorized RGBA heatmap; cold areas transparent so it overlays cleanly."""
    u8 = (cam * 255).astype(np.uint8)
    bgr = cv2.applyColorMap(u8, cv2.COLORMAP_JET)
    alpha = np.where(u8 > 40, u8 * 0.7, 0).astype(np.uint8)
    ok, buf = cv2.imencode(".png", np.dstack([bgr, alpha]))
    return base64.b64encode(buf).decode()


@router.get("/health")
def health():
    return {"ok": True, "model": MODEL.name}


def run_infer(img):
    """img: BGR uint8 ndarray at original resolution. Shared by /infer and /api/analyze."""
    if BUSY_CHECK and BUSY_CHECK():
        raise HTTPException(409, "gaze recording in progress; inference paused")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    with INFER_LOCK:
        t0 = time.perf_counter()
        score, cam = MODEL.infer(gray)
        infer_ms = (time.perf_counter() - t0) * 1000
    cam = np.clip(cv2.resize(cam, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_CUBIC), 0, 1)
    image_id = uuid.uuid4().hex[:8]
    STORE[image_id] = img
    while len(STORE) > MAX_IMAGES:
        STORE.popitem(last=False)
    return {"image_id": image_id, "width": gray.shape[1], "height": gray.shape[0],
            "model": MODEL.name, "infer_ms": round(infer_ms, 1),
            "score": score, "boxes": extract_boxes(cam), "heatmap_png": heat_png(cam)}


@router.post("/infer")
async def infer(file: UploadFile = File(...)):
    img = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "not a readable image")
    # run off the event loop, otherwise ONNX blocks the MJPEG streams and /api/status
    return await run_in_threadpool(run_infer, img)


# @router.post("/infer")
# async def infer(file: UploadFile = File(...)):
#     img = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_COLOR)
#     if img is None:
#         raise HTTPException(400, "not a readable image")
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     t0 = time.perf_counter()
#     score, cam = MODEL.infer(gray)
#     infer_ms = (time.perf_counter() - t0) * 1000
#     cam = np.clip(cv2.resize(cam, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_CUBIC), 0, 1)
#     image_id = uuid.uuid4().hex[:8]
#     STORE[image_id] = img
#     while len(STORE) > MAX_IMAGES:
#         STORE.popitem(last=False)
#     return {"image_id": image_id, "width": gray.shape[1], "height": gray.shape[0],
#             "model": MODEL.name, "infer_ms": round(infer_ms, 1),
#             "score": score, "boxes": extract_boxes(cam), "heatmap_png": heat_png(cam)}


class Expand(BaseModel):
    image_id: str
    x: int
    y: int
    w: int
    h: int
    margin: int = 16

@router.post("/expand")
def expand(r: Expand):
    img = STORE.get(r.image_id)
    if img is None:
        raise HTTPException(404, "unknown image_id - re-open the image")
    H, W = img.shape[:2]
    x0, y0 = max(r.x - r.margin, 0), max(r.y - r.margin, 0)
    x1, y1 = min(r.x + r.w + r.margin, W), min(r.y + r.h + r.margin, H)
    
    # 1. Encode the small crop
    ok_crop, buf_crop = cv2.imencode(".png", img[y0:y1, x0:x1])  
    # 2. Encode the full original image
    ok_full, buf_full = cv2.imencode(".png", img) 
    
    if not TIER1_URL:
        return {"stub": True, "crop_box": [x0, y0, x1, y1],
                "narrative": f"Tier 1 not configured. Prepared a {x1 - x0}x{y1 - y0}px crop."}
    try:
        # SEND BOTH FILES OVER 5G!
        files = {
            "crop": ("crop.png", buf_crop.tobytes(), "image/png"),
            "full": ("full.png", buf_full.tobytes(), "image/png")
        }
        resp = requests.post(TIER1_URL.rstrip("/") + "/characterize", files=files, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise HTTPException(502, f"Tier 1 unreachable: {e}")
    return {**resp.json(), "crop_box": [x0, y0, x1, y1]}