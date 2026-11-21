"""OPTIONAL / experimental: make a faster INT8 version using frames from YOUR video for calibration.
   Run on the laptop (venv):  pip install onnx onnxruntime
       python ..\\tools\\quantize_int8.py  path\\to\\your_video.mp4
   -> tools/yolov8n_320_int8.onnx. Copy it next to the FP32 file on the board; the app prefers it
   automatically. If boxes look worse or it is not faster, delete it (or set LOCAL_MODEL=yolov8n_320.onnx)."""
import sys
from pathlib import Path
import cv2, numpy as np, onnxruntime as ort
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType

here = Path(__file__).resolve().parent
src, dst = here / "yolov8n_320.onnx", here / "yolov8n_320_int8.onnx"
video = sys.argv[1]
S = 320
name = ort.InferenceSession(str(src), providers=["CPUExecutionProvider"]).get_inputs()[0].name


def prep(frame):
    h, w = frame.shape[:2]
    r = min(S / w, S / h); nw, nh = round(w * r), round(h * r)
    c = np.full((S, S, 3), 114, np.uint8)
    c[(S-nh)//2:(S-nh)//2+nh, (S-nw)//2:(S-nw)//2+nw] = cv2.resize(frame, (nw, nh))
    return cv2.dnn.blobFromImage(c, 1/255.0, (S, S), swapRB=True)


class Reader(CalibrationDataReader):
    def __init__(self, n=120):
        cap = cv2.VideoCapture(video); total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or n
        self.blobs = []
        for i in np.linspace(0, max(total - 1, 0), n).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i)); ok, f = cap.read()
            if ok: self.blobs.append(prep(f))
        print("calibration frames:", len(self.blobs)); self.it = iter(self.blobs)
    def get_next(self):
        b = next(self.it, None)
        return None if b is None else {name: b}


quantize_static(str(src), str(dst), Reader(), quant_format=QuantFormat.QDQ, per_channel=True,
                weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8)
print("DONE ->", dst, f"({dst.stat().st_size/1e6:.1f} MB)")
