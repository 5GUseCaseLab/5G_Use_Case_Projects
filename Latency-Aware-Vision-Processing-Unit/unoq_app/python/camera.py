import os, threading, time, cv2

STREAM_PREFIXES = ("http://", "https://", "rtsp://", "rtmp://", "udp://", "tcp://")
# RTSP over TCP: no garbled frames on wifi; low-latency flags for the FFmpeg backend
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay")


class Camera:
    """Video source: USB camera OR a video file. Keeps only the newest frame.
    Files are played back in REAL TIME (paced at the video's fps) and loop, so latency,
    frame-dropping and speed estimates behave as they would with a live camera."""
    def __init__(self, max_w=640):
        self.max_w = max_w
        self.frame, self.ts, self.seq, self.gen = None, 0.0, 0, 0
        self.lock = threading.Lock()
        self.cap, self.kind, self.fps, self.next_t = None, "none", 30.0, 0.0
        self.info = {"kind": "none", "name": "no source", "fps": 0, "error": ""}
        self._req = None
        threading.Thread(target=self._run, daemon=True).start()

    def open_camera(self, index=0):
        self._req = ("camera", index)

    def open_file(self, path):
        """A path to a video file, or a stream URL (http://, rtsp://, ...) from a phone IP-camera app."""
        path = str(path).strip()
        self._req = ("stream", path) if path.lower().startswith(STREAM_PREFIXES) else ("file", path)

    def _open(self, kind, arg):
        if self.cap is not None:
            self.cap.release(); self.cap = None
        cap = cv2.VideoCapture(arg)
        if not cap.isOpened():
            self.kind = "none"
            self.info = {"kind": "none", "name": str(arg), "fps": 0, "error": "cannot open"}
            print("[source] cannot open", arg)
            if kind == "stream":                             # phone not reachable yet / app not started: keep trying
                self.info["error"] = "cannot open, retrying"
                time.sleep(2.0)
                if self._req is None:
                    self._req = ("stream", arg)
            return
        if kind == "camera":
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if 1 <= fps <= 120 else 30.0
        self.cap, self.kind, self.next_t = cap, kind, time.monotonic()
        with self.lock:
            self.frame, self.gen = None, self.gen + 1      # gen change -> tracker/gate reset
        if kind == "stream":
            fps = 0.0                                        # unknown / variable: no pacing, newest frame wins
            self.fps = 30.0
        name = str(arg) if kind in ("camera", "stream") else str(arg).split("/")[-1]
        self.info = {"kind": kind, "name": name, "fps": round(self.fps, 1) if kind != "stream" else 0, "error": ""}
        self.url = arg if kind == "stream" else None
        print("[source] opened", kind, arg, "fps", self.fps)

    def _run(self):
        while True:
            req, self._req = self._req, None
            if req:
                self._open(*req)
            if self.cap is None:
                time.sleep(0.1); continue
            ok, f = self.cap.read()
            if not ok:
                if self.kind == "stream":                  # dropped connection: reconnect, keep trying
                    self.info["error"] = "stream lost, reconnecting"
                    time.sleep(1.0)
                    self._req = ("stream", self.url)
                    continue
                if self.kind == "file":                    # loop the video
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    with self.lock: self.gen += 1
                    time.sleep(0.02)
                else:
                    time.sleep(0.05)
                continue
            h, w = f.shape[:2]
            if w > self.max_w:
                f = cv2.resize(f, (self.max_w, int(h*self.max_w/w)))
            with self.lock:
                self.frame, self.ts, self.seq = f, time.monotonic(), self.seq + 1
            if self.kind == "file":                        # pace to real time
                self.next_t += 1.0 / self.fps
                d = self.next_t - time.monotonic()
                if d > 0: time.sleep(d)
                else: self.next_t = time.monotonic()

    def latest(self, after_seq):
        with self.lock:
            if self.frame is None or self.seq == after_seq:
                return None
            return self.frame, self.ts, self.seq, self.gen
