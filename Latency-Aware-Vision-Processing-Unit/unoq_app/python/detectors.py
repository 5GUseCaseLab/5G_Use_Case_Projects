import os, cv2, numpy as np

# COCO ids kept (same set as the laptop server): person, bicycle, car, motorcycle, bus, truck
KEEP = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
KEEP_IDS = np.array(list(KEEP.keys()))
KEEP_NAMES = [KEEP[i] for i in KEEP_IDS]
CANDIDATES = ["yolov8n_320_int8.onnx", "yolov8n_320.onnx"]     # first one found is used


class LocalDetector:
    """YOLOv8n (ONNX, 320x320) on the UNO Q CPU. Uses onnxruntime if installed,
    otherwise OpenCV's built-in ONNX runner, so it works either way."""
    def __init__(self, model_dir, conf=0.40, iou=0.5, threads=None):
        self.conf, self.iou = conf, iou
        self.available, self.backend, self.size = False, "", 320
        name = os.getenv("LOCAL_MODEL")
        path = next((os.path.join(model_dir, n) for n in ([name] if name else CANDIDATES)
                     if os.path.exists(os.path.join(model_dir, n))), None)
        if not path:
            print("[local] no ONNX model in", model_dir, "- local fallback DISABLED")
            return
        threads = threads or int(os.getenv("LOCAL_THREADS", "3"))
        try:
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.intra_op_num_threads = threads
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
            inp = self.sess.get_inputs()[0]
            self.inp_name = inp.name
            if isinstance(inp.shape[2], int): self.size = inp.shape[2]
            self.backend = "onnxruntime"
        except Exception as e:
            print("[local] onnxruntime unavailable (", e, ") -> using cv2.dnn")
            self.net = cv2.dnn.readNetFromONNX(path)
            cv2.setNumThreads(threads)
            self.backend = "cv2.dnn"
        self.available = True
        print(f"[local] {os.path.basename(path)} via {self.backend}, input {self.size}x{self.size}")

    def warmup(self):
        """Run once on a blank frame; returns the typical inference time in ms."""
        import time
        img = np.full((360, 480, 3), 114, np.uint8)
        self.detect(img)
        t0 = time.perf_counter()
        for _ in range(3): self.detect(img)
        return (time.perf_counter() - t0) / 3 * 1000

    def _run(self, blob):
        if self.backend == "onnxruntime":
            return self.sess.run(None, {self.inp_name: blob})[0]
        self.net.setInput(blob)
        return self.net.forward()

    def detect(self, frame):
        h, w = frame.shape[:2]
        s = self.size
        r = min(s / w, s / h)                                     # letterbox (keep aspect ratio)
        nw, nh = int(round(w * r)), int(round(h * r))
        px, py = (s - nw) // 2, (s - nh) // 2
        canvas = np.full((s, s, 3), 114, np.uint8)
        canvas[py:py+nh, px:px+nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        blob = cv2.dnn.blobFromImage(canvas, 1/255.0, (s, s), swapRB=True)
        return self.decode(self._run(blob), w, h, r, px, py)

    def decode(self, out, w, h, r, px, py):
        """out: (1, 84, N) YOLOv8 head -> list of [cls, conf, x1, y1, x2, y2] normalised to the frame."""
        p = np.squeeze(out, 0)
        if p.shape[0] < p.shape[1]:
            p = p.T                                                # -> (N, 84)
        scores = p[:, 4:][:, KEEP_IDS]                              # only the classes we care about
        k = scores.argmax(1)
        conf = scores[np.arange(len(k)), k]
        m = conf >= self.conf
        if not m.any():
            return []
        p, k, conf = p[m], k[m], conf[m]
        cx, cy, bw, bh = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
        x1 = np.clip((cx - bw/2 - px) / r, 0, w); y1 = np.clip((cy - bh/2 - py) / r, 0, h)
        x2 = np.clip((cx + bw/2 - px) / r, 0, w); y2 = np.clip((cy + bh/2 - py) / r, 0, h)
        boxes = np.stack([x1, y1, x2 - x1, y2 - y1], 1)
        idx = cv2.dnn.NMSBoxesBatched(boxes.tolist(), conf.tolist(), k.tolist(), self.conf, self.iou)
        dets = []
        for i in np.array(idx).flatten():
            if x2[i] > x1[i] and y2[i] > y1[i]:
                dets.append([KEEP_NAMES[k[i]], round(float(conf[i]), 3),
                             float(x1[i]/w), float(y1[i]/h), float(x2[i]/w), float(y2[i]/h)])
        return dets
