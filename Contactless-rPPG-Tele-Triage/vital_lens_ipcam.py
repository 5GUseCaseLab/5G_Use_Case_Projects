import os

# Guard BLAS/OMP threads before importing numerical or vision libraries.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import ctypes
import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from importlib.resources import as_file, files
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import vitallens.ssd
import vitallens_core as vc
from prpy.numpy.face import get_roi_from_det
from vitallens import VitalLens

# vitallens.client sets root logger to INFO at import time.
logging.getLogger().setLevel(logging.ERROR)

# Python 3.13 requires entering the Traversable context returned by files().
vitallens.ssd.files = lambda *args, **kwargs: as_file(files(*args, **kwargs))

CAMERA_URL = "http://192.168.137.35:8080"
WINDOW_SECONDS = 15.0
MIN_WINDOW_SECONDS = 8.0
EVAL_INTERVAL_SECONDS = 0.5
RESAMPLE_FPS = 30.0
FACE_DETECT_HZ = 3.0
FACE_DEADBAND_PX = 3.0
FACE_EMA_ALPHA = 0.25
FACE_LOST_RESET_SECONDS = 2.0

WINDOW_NAME = "VitalLens POS - Real-Time rPPG"
PANEL_W = 380
MIN_CANVAS_H = 600
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
STREAM_ENDPOINTS = ("/video", "/videofeed", "/mjpegfeed", "/stream")


class TriageLevel(Enum):
    CALIBRATING = "CALIBRATING"
    NO_FACE = "NO_FACE"
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"


TRIAGE_META: dict[TriageLevel, tuple[str, tuple[int, int, int]]] = {
    TriageLevel.RED: ("RED - Immediate Attention", (60, 60, 235)),
    TriageLevel.YELLOW: ("YELLOW - Abnormal / High", (0, 200, 255)),
    TriageLevel.GREEN: ("GREEN - Normal / Stable", (70, 210, 80)),
    TriageLevel.CALIBRATING: ("CALIBRATING", (230, 185, 50)),
    TriageLevel.NO_FACE: ("NO FACE DETECTED", (150, 150, 160)),
}


def classify_triage(bpm: float | None, has_face: bool) -> TriageLevel:
    if not has_face:
        return TriageLevel.NO_FACE
    if bpm is None:
        return TriageLevel.CALIBRATING
    if bpm > 120.0 or bpm < 40.0:
        return TriageLevel.RED
    if bpm > 100.0 or bpm < 60.0:
        return TriageLevel.YELLOW
    return TriageLevel.GREEN


@dataclass(frozen=True)
class StreamConfig:
    source: str = CAMERA_URL
    window_seconds: float = WINDOW_SECONDS
    min_window_seconds: float = MIN_WINDOW_SECONDS
    eval_interval_seconds: float = EVAL_INTERVAL_SECONDS
    resample_fps: float = RESAMPLE_FPS
    face_detect_hz: float = FACE_DETECT_HZ
    face_deadband_px: float = FACE_DEADBAND_PX
    face_ema_alpha: float = FACE_EMA_ALPHA
    face_lost_reset_seconds: float = FACE_LOST_RESET_SECONDS
    expected_bpm: float | None = None
    no_gui: bool = False
    fast_replay: bool = False
    save_hud_frame: str | None = None

    def __post_init__(self) -> None:
        floats = (
            self.window_seconds,
            self.min_window_seconds,
            self.eval_interval_seconds,
            self.resample_fps,
            self.face_detect_hz,
            self.face_deadband_px,
            self.face_ema_alpha,
            self.face_lost_reset_seconds,
        )
        if not all(math.isfinite(val) for val in floats):
            raise ValueError("Numeric configuration values must be finite.")
        if self.min_window_seconds <= 1.0:
            raise ValueError("--min-window-seconds must be > 1.0.")
        if self.window_seconds < self.min_window_seconds:
            raise ValueError("--window-seconds must be >= --min-window-seconds.")
        if self.resample_fps <= 5.0:
            raise ValueError("--target-fps must be > 5.0.")
        if self.eval_interval_seconds <= 0.0:
            raise ValueError("--eval-interval must be > 0.0.")
        if self.face_detect_hz <= 0.0:
            raise ValueError("face_detect_hz must be > 0.0.")
        if self.face_deadband_px < 0.0:
            raise ValueError("face_deadband_px must be >= 0.0.")
        if not (0.0 < self.face_ema_alpha <= 1.0):
            raise ValueError("face_ema_alpha must be in (0.0, 1.0].")
        if self.face_lost_reset_seconds < 0.0:
            raise ValueError("face_lost_reset_seconds must be >= 0.0.")
        if self.expected_bpm is not None and (not math.isfinite(self.expected_bpm) or self.expected_bpm <= 0.0):
            raise ValueError("--expected-bpm must be a positive finite number.")


class RgbSample(NamedTuple):
    timestamp: float
    r: float
    g: float
    b: float
    face_box: tuple[int, int, int, int]


@dataclass
class VitalsSnapshot:
    triage: TriageLevel
    bpm: float | None
    confidence: float
    buffer_seconds: float
    target_window_seconds: float
    measured_fps: float
    sample_count: int
    ppg_waveform: np.ndarray | None
    face_box: tuple[int, int, int, int] | None
    roi_box: tuple[int, int, int, int] | None
    face_conf: float
    is_warming_up: bool
    solve_ms: float
    eval_count: int
    stream_size: tuple[int, int]
    source_label: str
    expected_bpm: float | None


def enable_dpi_awareness() -> None:
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def get_screen_geometry() -> tuple[int, int]:
    try:
        user32 = ctypes.windll.user32
        sw = int(user32.GetSystemMetrics(0))
        sh = int(user32.GetSystemMetrics(1))
        if sw > 400 and sh > 400:
            return sw, sh
    except Exception:
        pass
    return 1920, 1080


def estimate_vitals_from_trace(
    vl: VitalLens,
    samples: list[RgbSample],
    resample_fps: float,
) -> tuple[float | None, float, np.ndarray | None, float, float]:
    if len(samples) < 30:
        return None, 0.0, None, 0.0, 0.0

    ts = np.array([s.timestamp for s in samples], dtype=np.float64)
    rgb = np.array([[s.r, s.g, s.b] for s in samples], dtype=np.float64)
    dur = float(ts[-1] - ts[0])
    if dur <= 1.0:
        measured = float((len(ts) - 1) / dur) if dur > 0.0 else 0.0
        return None, 0.0, None, measured, 0.0

    measured_fps = (len(ts) - 1) / dur
    n_u = max(2, int(round(dur * resample_fps)) + 1)
    t_u = np.linspace(ts[0], ts[-1], n_u)
    rgb_u = np.column_stack([np.interp(t_u, ts, rgb[:, c]) for c in range(3)])
    eff_fps = (n_u - 1) / dur
    use_ts = (t_u - t_u[0]).tolist()

    idx_map = np.clip(
        np.round(np.interp(t_u, ts, np.arange(len(samples)))).astype(int),
        0,
        len(samples) - 1,
    )
    use_boxes = [list(samples[int(i)].face_box) for i in idx_map]

    t0 = time.perf_counter()
    sig = vl.rppg.algorithm(rgb_u, eff_fps)
    session = vc.Session(vl.rppg.session_config)
    signals_in = {"ppg_waveform": vc.SignalInput(data=sig.tolist(), confidence=np.ones_like(sig).tolist())}
    face_in = vc.FaceInput(coordinates=use_boxes, confidence=np.ones(len(sig)).tolist())
    res = session.process(vc.SessionInput(face=face_in, signals=signals_in, timestamp=use_ts), "Global")
    solve_ms = (time.perf_counter() - t0) * 1000.0

    hr = res.vitals.get("heart_rate")
    ppg = res.waveforms.get("ppg_waveform")
    bpm_val = float(hr.value) if hr and hr.value is not None else None
    conf_val = float(hr.confidence) if hr else 0.0
    waveform = np.asarray(ppg.data, dtype=np.float64) if ppg else sig
    return bpm_val, conf_val, waveform, float(measured_fps), float(solve_ms)


def _probe_source(source: str) -> tuple[cv2.VideoCapture, str, bool, np.ndarray]:
    if os.path.exists(source):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {source}")
        ok, frame = cap.read()
        if not ok or frame is None or frame.size == 0:
            cap.release()
            raise RuntimeError(f"Failed to read initial frame from video file: {source}")
        return cap, os.path.basename(source) or source, False, frame

    if "://" not in source and Path(source).suffix.lower() in VIDEO_EXTENSIONS:
        raise FileNotFoundError(f"Video file not found: {source}")

    stripped = source.rstrip("/")
    candidates = list(dict.fromkeys([source, *(f"{stripped}{ep}" for ep in STREAM_ENDPOINTS)]))
    for url in candidates:
        cap = cv2.VideoCapture(url)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0:
                return cap, url, True, frame
        cap.release()
    raise ConnectionError(f"Could not connect to live video stream from: {source}")


def open_source(source: str) -> tuple[cv2.VideoCapture, str, bool]:
    cap, label, is_live, _ = _probe_source(source)
    if not is_live:
        # Rewind file capture so callers read starting from frame 0.
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if int(round(cap.get(cv2.CAP_PROP_POS_FRAMES))) != 0:
            cap.release()
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                raise RuntimeError(f"Failed to reopen video file: {source}")
    return cap, label, is_live


def detect_face_box(
    vl: VitalLens,
    frame_bgr: np.ndarray,
) -> tuple[tuple[int, int, int, int] | None, float]:
    h, w = frame_bgr.shape[:2]
    if max(w, h) > 320:
        scale = 320.0 / max(w, h)
        small = cv2.resize(
            frame_bgr,
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        small = frame_bgr

    small_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    boxes, info = vl.face_detector(inputs=small_rgb[np.newaxis], n_frames=1, fps=30.0)
    if len(boxes) == 0 or len(boxes[0]) == 0:
        return None, 0.0

    rx0, ry0, rx1, ry1 = boxes[0, 0]
    x0 = max(0, min(w, int(round(float(rx0) * w))))
    y0 = max(0, min(h, int(round(float(ry0) * h))))
    x1 = max(0, min(w, int(round(float(rx1) * w))))
    y1 = max(0, min(h, int(round(float(ry1) * h))))
    if x1 <= x0 or y1 <= y0:
        return None, 0.0

    conf = float(info[0, 0, 3])
    return (x0, y0, x1, y1), conf


def smooth_face_box(
    prev_box: tuple[int, int, int, int] | None,
    new_box: tuple[int, int, int, int],
    deadband_px: float,
    ema_alpha: float,
) -> tuple[int, int, int, int]:
    if prev_box is None:
        return new_box
    mean_shift = sum(abs(a - b) for a, b in zip(prev_box, new_box)) / 4.0
    if mean_shift < deadband_px:
        return prev_box
    blended = tuple(int(round(p * (1.0 - ema_alpha) + n * ema_alpha)) for p, n in zip(prev_box, new_box))
    if blended[2] <= blended[0] or blended[3] <= blended[1]:
        return new_box
    return (blended[0], blended[1], blended[2], blended[3])


def extract_roi_rgb(
    vl: VitalLens,
    frame_bgr: np.ndarray,
    face_box: tuple[int, int, int, int],
) -> tuple[tuple[float, float, float], tuple[int, int, int, int]] | None:
    h, w = frame_bgr.shape[:2]
    roi = get_roi_from_det(face_box, roi_method=vl.rppg.roi_method, clip_dims=(w, h))
    x0 = max(0, int(roi[0]))
    y0 = max(0, int(roi[1]))
    x1 = min(w, int(roi[2]))
    y1 = min(h, int(roi[3]))
    if x1 <= x0 or y1 <= y0:
        return None
    mean_bgr = cv2.mean(frame_bgr[y0:y1, x0:x1])[:3]
    return (float(mean_bgr[2]), float(mean_bgr[1]), float(mean_bgr[0])), (x0, y0, x1, y1)


class StreamState:
    def __init__(self, source_label: str, stream_size: tuple[int, int], first_frame: np.ndarray) -> None:
        self.lock = threading.Lock()
        self.samples: deque[RgbSample] = deque()
        self.latest_frame: np.ndarray | None = first_frame
        self.stream_size = stream_size
        self.source_label = source_label
        self.face_box: tuple[int, int, int, int] | None = None
        self.roi_box: tuple[int, int, int, int] | None = None
        self.face_conf: float = 0.0
        self.last_face_time: float | None = None
        self.bpm: float | None = None
        self.confidence: float = 0.0
        self.ppg_waveform: np.ndarray | None = None
        self.measured_fps: float = 0.0
        self.solve_ms: float = 0.0
        self.eval_count: int = 0
        self.buffer_seconds: float = 0.0
        self.is_warming_up: bool = False
        self.reset_generation: int = 0

    def _clear_vitals_and_samples_locked(self) -> None:
        self.samples.clear()
        self.bpm = None
        self.confidence = 0.0
        self.ppg_waveform = None
        self.buffer_seconds = 0.0
        self.is_warming_up = False
        self.solve_ms = 0.0
        self.reset_generation += 1

    def _reset_face_locked(self) -> None:
        self.face_box = None
        self.roi_box = None
        self.face_conf = 0.0
        self.last_face_time = None
        self._clear_vitals_and_samples_locked()

    def reset_buffer(self) -> None:
        with self.lock:
            self._clear_vitals_and_samples_locked()

    def update_face(
        self,
        box: tuple[int, int, int, int] | None,
        conf: float,
        timestamp: float,
        frame_bgr: np.ndarray,
        vl: VitalLens,
        config: StreamConfig,
    ) -> None:
        if box is None:
            with self.lock:
                self.face_conf = 0.0
            return

        with self.lock:
            smoothed = smooth_face_box(self.face_box, box, config.face_deadband_px, config.face_ema_alpha)
            self.face_box = smoothed
            self.face_conf = conf
            self.last_face_time = timestamp
            roi_res = extract_roi_rgb(vl, frame_bgr, smoothed)
            if roi_res is not None:
                self.roi_box = roi_res[1]

    def ingest_frame(
        self,
        frame_bgr: np.ndarray,
        timestamp: float,
        vl: VitalLens,
        config: StreamConfig,
    ) -> None:
        h, w = frame_bgr.shape[:2]
        with self.lock:
            self.latest_frame = frame_bgr
            self.stream_size = (w, h)
            if self.last_face_time is not None and (timestamp - self.last_face_time) > config.face_lost_reset_seconds:
                self._reset_face_locked()
                return
            active_box = self.face_box

        if active_box is None:
            return
        roi_res = extract_roi_rgb(vl, frame_bgr, active_box)
        if roi_res is None:
            return
        (r, g, b), roi_box = roi_res

        with self.lock:
            if self.face_box is None:
                return
            self.roi_box = roi_box
            self.samples.append(RgbSample(timestamp, r, g, b, active_box))
            cutoff = timestamp - (config.window_seconds + 0.5)
            while self.samples and self.samples[0].timestamp < cutoff:
                self.samples.popleft()
            if len(self.samples) >= 2:
                span = self.samples[-1].timestamp - self.samples[0].timestamp
                self.buffer_seconds = min(span, config.window_seconds)
                if self.bpm is None and span > 0.5:
                    self.measured_fps = (len(self.samples) - 1) / span
            else:
                self.buffer_seconds = 0.0

    def snapshot_window(self, window_seconds: float) -> tuple[list[RgbSample], bool, int]:
        with self.lock:
            if not self.samples or self.face_box is None:
                return [], False, self.reset_generation
            latest_t = self.samples[-1].timestamp
            win_start = latest_t - window_seconds
            window_samples = [s for s in self.samples if s.timestamp >= win_start]
            return window_samples, True, self.reset_generation

    def commit_vitals(
        self,
        bpm: float | None,
        conf: float,
        ppg: np.ndarray | None,
        measured_fps: float,
        solve_ms: float,
        buf_span: float,
        window_seconds: float,
        generation: int,
    ) -> bool:
        with self.lock:
            # Discard solve if a reset or face-loss happened while inference ran.
            if generation != self.reset_generation or self.face_box is None:
                return False
            self.bpm = bpm
            self.confidence = conf
            self.ppg_waveform = ppg
            self.measured_fps = measured_fps
            self.solve_ms = solve_ms
            self.eval_count += 1
            self.is_warming_up = buf_span < (window_seconds - 0.5)
            return True

    def latest_frame_and_snapshot(self, config: StreamConfig) -> tuple[np.ndarray | None, VitalsSnapshot]:
        with self.lock:
            frame = self.latest_frame
            has_face = self.face_box is not None
            win_count = 0
            if self.samples:
                win_start = self.samples[-1].timestamp - config.window_seconds
                win_count = sum(1 for s in self.samples if s.timestamp >= win_start)
            snapshot = VitalsSnapshot(
                triage=classify_triage(self.bpm, has_face),
                bpm=self.bpm,
                confidence=self.confidence,
                buffer_seconds=self.buffer_seconds,
                target_window_seconds=config.window_seconds,
                measured_fps=self.measured_fps,
                sample_count=win_count,
                ppg_waveform=self.ppg_waveform,
                face_box=self.face_box,
                roi_box=self.roi_box,
                face_conf=self.face_conf,
                is_warming_up=self.is_warming_up,
                solve_ms=self.solve_ms,
                eval_count=self.eval_count,
                stream_size=self.stream_size,
                source_label=self.source_label,
                expected_bpm=config.expected_bpm,
            )
            return frame, snapshot


def _format_eval_log(snapshot: VitalsSnapshot) -> str:
    hr_str = f"{snapshot.bpm:5.1f} BPM" if snapshot.bpm is not None else " --.- BPM"
    err_str = ""
    if snapshot.expected_bpm is not None and snapshot.bpm is not None:
        err_str = f" | Err: {abs(snapshot.bpm - snapshot.expected_bpm):.1f}"
    return (
        f"[Eval #{snapshot.eval_count:03d}] HR: {hr_str}{err_str} | "
        f"Conf: {snapshot.confidence:.2f} | Triage: {snapshot.triage.value} | "
        f"Buf: {snapshot.buffer_seconds:.1f}/{snapshot.target_window_seconds:.1f}s | "
        f"FPS: {snapshot.measured_fps:.1f} | Solve: {snapshot.solve_ms:.1f} ms"
    )


def _evaluate_window(vl: VitalLens, state: StreamState, config: StreamConfig) -> VitalsSnapshot | None:
    window_samples, has_face, gen = state.snapshot_window(config.window_seconds)
    if not has_face or len(window_samples) < 30:
        return None
    buf_span = window_samples[-1].timestamp - window_samples[0].timestamp
    if buf_span < config.min_window_seconds:
        return None

    bpm, conf, ppg, measured_fps, solve_ms = estimate_vitals_from_trace(vl, window_samples, config.resample_fps)
    if not state.commit_vitals(bpm, conf, ppg, measured_fps, solve_ms, buf_span, config.window_seconds, gen):
        return None

    _, snapshot = state.latest_frame_and_snapshot(config)
    print(_format_eval_log(snapshot))
    return snapshot


def draw_fitted_text(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    font_face: int,
    max_font_scale: float,
    color: tuple[int, int, int],
    thickness: int,
    max_width: int,
    min_font_scale: float = 0.35,
) -> tuple[int, int]:
    scale = max_font_scale
    (tw, th), _ = cv2.getTextSize(text, font_face, scale, thickness)
    if tw > max_width and max_width > 0:
        scale = max(min_font_scale, scale * (max_width / float(tw)))
        (tw, th), _ = cv2.getTextSize(text, font_face, scale, thickness)
        while tw > max_width and scale > min_font_scale:
            scale = max(min_font_scale, scale - 0.02)
            (tw, th), _ = cv2.getTextSize(text, font_face, scale, thickness)
            if scale == min_font_scale:
                break

    render_text = text
    if tw > max_width and max_width > 12 and len(render_text) > 4:
        while len(render_text) > 4:
            render_text = render_text[:-4] + "..."
            (tw, th), _ = cv2.getTextSize(render_text, font_face, scale, thickness)
            if tw <= max_width:
                break

    cv2.putText(img, render_text, org, font_face, scale, color, thickness, cv2.LINE_AA)
    return tw, th


def _draw_pill(
    img: np.ndarray,
    rect: tuple[int, int, int, int],
    bg_color: tuple[int, int, int],
    text: str,
    text_color: tuple[int, int, int],
    font_scale: float = 0.48,
    thickness: int = 1,
    alpha: float = 1.0,
) -> None:
    x0, y0, x1, y1 = rect
    if alpha < 1.0:
        overlay = img[y0:y1, x0:x1].copy()
        cv2.rectangle(overlay, (0, 0), (x1 - x0 - 1, y1 - y0 - 1), bg_color, -1)
        cv2.addWeighted(overlay, alpha, img[y0:y1, x0:x1], 1.0 - alpha, 0.0, img[y0:y1, x0:x1])
    else:
        cv2.rectangle(img, (x0, y0), (x1, y1), bg_color, -1)

    pad_x = 8
    max_w = max(10, (x1 - x0) - 2 * pad_x)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    tx = x0 + max(pad_x, ((x1 - x0) - min(tw, max_w)) // 2)
    ty = y0 + ((y1 - y0) + th) // 2
    draw_fitted_text(img, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, max_w)


def _draw_progress_bar(
    img: np.ndarray,
    x0: int,
    y_top: int,
    width: int,
    label: str,
    value_text: str,
    ratio: float,
    fill_color: tuple[int, int, int],
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    draw_fitted_text(img, label, (x0, y_top + 14), font, 0.48, (165, 172, 184), 1, width // 2)
    (vw, _), _ = cv2.getTextSize(value_text, font, 0.48, 1)
    vx = max(x0 + width // 2, x0 + width - vw)
    draw_fitted_text(img, value_text, (vx, y_top + 14), font, 0.48, (230, 235, 242), 1, width // 2)

    bar_y0, bar_y1 = y_top + 22, y_top + 32
    cv2.rectangle(img, (x0, bar_y0), (x0 + width, bar_y1), (34, 37, 46), -1)
    clamped = max(0.0, min(1.0, ratio))
    fill_w = int(round(width * clamped))
    if fill_w > 0:
        cv2.rectangle(img, (x0, bar_y0), (x0 + fill_w, bar_y1), fill_color, -1)


def _draw_waveform(
    img: np.ndarray,
    x0: int,
    y_top: int,
    width: int,
    box_h: int,
    waveform: np.ndarray | None,
    line_color: tuple[int, int, int],
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    draw_fitted_text(img, "PPG Waveform", (x0, y_top + 12), font, 0.48, (180, 186, 198), 1, width - 80)
    status = "5s trace" if waveform is not None and len(waveform) >= 2 else "waiting"
    (sw, _), _ = cv2.getTextSize(status, font, 0.42, 1)
    draw_fitted_text(img, status, (x0 + width - sw, y_top + 12), font, 0.42, (130, 136, 150), 1, 80)

    by0 = y_top + 18
    by1 = by0 + box_h
    cv2.rectangle(img, (x0, by0), (x0 + width, by1), (14, 15, 19), -1)
    cv2.rectangle(img, (x0, by0), (x0 + width, by1), (42, 46, 56), 1)
    mid_y = by0 + box_h // 2
    cv2.line(img, (x0 + 2, mid_y), (x0 + width - 2, mid_y), (34, 37, 46), 1)

    if waveform is None or len(waveform) < 2:
        return

    tail = np.nan_to_num(np.asarray(waveform[-150:], dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    vmin, vmax = float(np.min(tail)), float(np.max(tail))
    pad_x, pad_y = 6, 8
    xs = np.linspace(x0 + pad_x, x0 + width - pad_x, len(tail))
    if vmax - vmin < 1e-9:
        ys = np.full(len(tail), float(mid_y), dtype=np.float64)
    else:
        norm = (tail - vmin) / (vmax - vmin)
        ys = (by0 + pad_y) + (1.0 - norm) * (box_h - 2 * pad_y)

    pts = np.column_stack([np.round(xs), np.round(ys)]).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], isClosed=False, color=line_color, thickness=2, lineType=cv2.LINE_AA)


def _scale_box(
    box: tuple[int, int, int, int],
    scale: float,
    y_off: int,
    vid_w: int,
    vid_h: int,
) -> tuple[int, int, int, int]:
    x0 = max(0, min(vid_w - 1, int(round(box[0] * scale))))
    y0 = max(y_off, min(y_off + vid_h - 1, int(round(box[1] * scale)) + y_off))
    x1 = max(x0 + 1, min(vid_w, int(round(box[2] * scale))))
    y1 = max(y0 + 1, min(y_off + vid_h, int(round(box[3] * scale)) + y_off))
    return x0, y0, x1, y1


def _draw_video_overlays(
    canvas: np.ndarray,
    snapshot: VitalsSnapshot,
    scale: float,
    y_off: int,
    vid_w: int,
    vid_h: int,
) -> None:
    src_w, src_h = snapshot.stream_size
    if snapshot.face_box is not None:
        fx0, fy0, fx1, fy1 = _scale_box(snapshot.face_box, scale, y_off, vid_w, vid_h)
        cv2.rectangle(canvas, (fx0, fy0), (fx1, fy1), (175, 195, 210), 1)

    if snapshot.roi_box is not None:
        rx0, ry0, rx1, ry1 = _scale_box(snapshot.roi_box, scale, y_off, vid_w, vid_h)
        cv2.rectangle(canvas, (rx0, ry0), (rx1, ry1), (80, 220, 120), 2)
        label = "POS Face ROI"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        pill_w = min(vid_w - 8, lw + 16)
        pill_h = lh + 10
        px0 = max(4, min(rx0, vid_w - pill_w - 4))
        py0 = ry0 - pill_h - 4 if (ry0 - pill_h - 4) >= y_off + 4 else min(y_off + vid_h - pill_h - 4, ry0 + 4)
        _draw_pill(
            canvas,
            (px0, py0, px0 + pill_w, py0 + pill_h),
            (22, 48, 30),
            label,
            (130, 240, 160),
            font_scale=0.45,
            alpha=0.82,
        )

    info_text = f"{src_w}x{src_h} @ {snapshot.measured_fps:.1f} FPS"
    (iw, ih), _ = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
    pill_w = min(max(80, vid_w - 20), iw + 18)
    pill_h = ih + 12
    ix0, iy0 = 10, y_off + 10
    if ix0 + pill_w < vid_w and iy0 + pill_h < y_off + vid_h:
        _draw_pill(
            canvas,
            (ix0, iy0, ix0 + pill_w, iy0 + pill_h),
            (18, 20, 26),
            info_text,
            (225, 232, 240),
            font_scale=0.48,
            alpha=0.72,
        )


def _header_status(snapshot: VitalsSnapshot) -> tuple[str, tuple[int, int, int]]:
    if snapshot.triage == TriageLevel.NO_FACE:
        return "NO FACE", TRIAGE_META[TriageLevel.NO_FACE][1]
    if snapshot.bpm is None:
        return "CALIBRATING", TRIAGE_META[TriageLevel.CALIBRATING][1]
    if snapshot.is_warming_up:
        return "WARMING UP", TRIAGE_META[TriageLevel.YELLOW][1]
    return "LOCKED", TRIAGE_META[TriageLevel.GREEN][1]


def render_hud_frame(
    frame_bgr: np.ndarray,
    snapshot: VitalsSnapshot,
    screen_size: tuple[int, int] | None = None,
) -> np.ndarray:
    src_h, src_w = frame_bgr.shape[:2]
    screen_w, screen_h = screen_size or get_screen_geometry()

    max_vid_w = max(360, int(screen_w * 0.85) - PANEL_W)
    max_vid_h = max(360, int(screen_h * 0.82))
    scale = min(max_vid_w / float(src_w), max_vid_h / float(src_h), 1.0)
    vid_w = max(1, int(round(src_w * scale)))
    vid_h = max(1, int(round(src_h * scale)))

    canvas_h = max(vid_h, MIN_CANVAS_H)
    canvas_w = vid_w + PANEL_W
    canvas = np.full((canvas_h, canvas_w, 3), (18, 18, 22), dtype=np.uint8)

    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(frame_bgr, (vid_w, vid_h), interpolation=interp)
    y_off = (canvas_h - vid_h) // 2
    canvas[y_off : y_off + vid_h, 0:vid_w] = resized

    _draw_video_overlays(canvas, snapshot, scale, y_off, vid_w, vid_h)
    cv2.line(canvas, (vid_w, 0), (vid_w, canvas_h - 1), (45, 48, 56), 1)

    x0 = vid_w + 20
    inner_w = PANEL_W - 40
    x1 = x0 + inner_w
    font = cv2.FONT_HERSHEY_SIMPLEX

    status_text, status_color = _header_status(snapshot)
    draw_fitted_text(canvas, "VITALLENS POS", (x0, 38), font, 0.62, (238, 242, 248), 2, inner_w - 122)
    _draw_pill(canvas, (x1 - 112, 20, x1, 44), status_color, status_text, (16, 18, 22), font_scale=0.45, thickness=2)

    cv2.rectangle(canvas, (x0, 58), (x1, 154), (28, 30, 38), -1)
    cv2.rectangle(canvas, (x0, 58), (x1, 154), (48, 52, 62), 1)
    bpm_str = f"{snapshot.bpm:.1f}" if snapshot.bpm is not None else "--.-"
    bpm_w, _ = draw_fitted_text(canvas, bpm_str, (x0 + 16, 114), font, 1.65, (248, 250, 252), 3, inner_w - 92)
    draw_fitted_text(canvas, "BPM", (x0 + 26 + bpm_w, 114), font, 0.62, (165, 172, 184), 1, 60)

    if snapshot.expected_bpm is not None and snapshot.bpm is not None:
        sub_text = f"Target {snapshot.expected_bpm:.1f} BPM  (err {abs(snapshot.bpm - snapshot.expected_bpm):.1f})"
        sub_color = (175, 205, 230)
    elif snapshot.expected_bpm is not None:
        sub_text = f"Target {snapshot.expected_bpm:.1f} BPM"
        sub_color = (150, 158, 172)
    else:
        sub_text = f"Source: {snapshot.source_label}"
        sub_color = (150, 158, 172)
    draw_fitted_text(canvas, sub_text, (x0 + 16, 142), font, 0.45, sub_color, 1, inner_w - 32)

    triage_label, triage_color = TRIAGE_META[snapshot.triage]
    _draw_pill(canvas, (x0, 166, x1, 200), triage_color, triage_label, (16, 18, 22), font_scale=0.52, thickness=2)

    buf_ratio = snapshot.buffer_seconds / max(snapshot.target_window_seconds, 1e-6)
    buf_color = (70, 210, 80) if buf_ratio >= 0.96 else (70, 175, 240)
    _draw_progress_bar(
        canvas,
        x0,
        214,
        inner_w,
        "Buffer",
        f"{snapshot.buffer_seconds:.1f} / {snapshot.target_window_seconds:.1f} s",
        buf_ratio,
        buf_color,
    )

    conf_color = triage_color if snapshot.bpm is not None else (110, 118, 130)
    _draw_progress_bar(
        canvas,
        x0,
        258,
        inner_w,
        "Confidence",
        f"{snapshot.confidence:.2f}",
        snapshot.confidence,
        conf_color,
    )

    _draw_waveform(canvas, x0, 304, inner_w, 90, snapshot.ppg_waveform, triage_color)

    cv2.line(canvas, (x0, 422), (x1, 422), (40, 44, 52), 1)
    metric_rows = (
        ("Window param", f"{snapshot.target_window_seconds:.1f} s"),
        ("Measured FPS", f"{snapshot.measured_fps:.1f} fps"),
        ("Solve time", f"{snapshot.solve_ms:.1f} ms"),
        ("Samples", f"{snapshot.sample_count}"),
        ("Evals", f"{snapshot.eval_count}"),
    )
    for idx, (key, val) in enumerate(metric_rows):
        row_y = 444 + idx * 24
        draw_fitted_text(canvas, key, (x0, row_y), font, 0.48, (155, 162, 176), 1, inner_w // 2)
        (vw, _), _ = cv2.getTextSize(val, font, 0.48, 1)
        vx = max(x0 + inner_w // 2, x1 - vw)
        draw_fitted_text(canvas, val, (vx, row_y), font, 0.48, (230, 235, 242), 1, inner_w // 2)

    cv2.line(canvas, (x0, canvas_h - 38), (x1, canvas_h - 38), (40, 44, 52), 1)
    draw_fitted_text(
        canvas,
        "[R] Reset buffer   [Q / Esc] Quit",
        (x0, canvas_h - 16),
        font,
        0.44,
        (135, 142, 155),
        1,
        inner_w,
    )
    return canvas


def _save_hud_png(
    output_path: str,
    frame_bgr: np.ndarray,
    snapshot: VitalsSnapshot,
    screen_size: tuple[int, int],
) -> None:
    out_file = Path(output_path)
    if out_file.parent:
        out_file.parent.mkdir(parents=True, exist_ok=True)
    hud = render_hud_frame(frame_bgr, snapshot, screen_size=screen_size)
    if not cv2.imwrite(str(out_file), hud):
        raise RuntimeError(f"Failed to save HUD image to: {output_path}")
    print(f"[INFO] Saved HUD frame to: {output_path}")


def _poll_gui_key(window_name: str, wait_ms: int, state: StreamState) -> bool:
    key = cv2.waitKey(max(1, wait_ms)) & 0xFF
    if key in (ord("q"), ord("Q"), 27):
        return False
    if key in (ord("r"), ord("R")):
        state.reset_buffer()
        print("[INFO] Buffer reset by user.")
    try:
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            return False
    except cv2.error:
        return False
    return True


def _reconnect_stream(
    source_label: str,
    fallback_source: str,
    stop_event: threading.Event,
) -> tuple[cv2.VideoCapture, np.ndarray] | None:
    backoff = 0.2
    while not stop_event.is_set():
        for candidate in dict.fromkeys((source_label, fallback_source)):
            if stop_event.is_set():
                return None
            try:
                cap, _, is_live, frame = _probe_source(candidate)
                if is_live:
                    print(f"[INFO] Reconnected to live stream: {candidate}")
                    return cap, frame
                cap.release()
            except Exception:
                pass
        stop_event.wait(backoff)
        backoff = min(2.0, backoff * 2.0)
    return None


def _run_live(
    vl: VitalLens,
    cap: cv2.VideoCapture,
    first_frame: np.ndarray,
    source_label: str,
    config: StreamConfig,
) -> None:
    h, w = first_frame.shape[:2]
    state = StreamState(source_label, (w, h), first_frame)
    t0 = time.monotonic()
    box0, conf0 = detect_face_box(vl, first_frame)
    state.update_face(box0, conf0, t0, first_frame, vl, config)
    if box0 is not None:
        state.ingest_frame(first_frame, t0, vl, config)

    stop_event = threading.Event()
    screen_size = get_screen_geometry()
    if not config.no_gui:
        enable_dpi_awareness()
        screen_size = get_screen_geometry()
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    def capture_loop() -> None:
        active_cap = cap
        try:
            while not stop_event.is_set():
                ok, frame = active_cap.read()
                t_now = time.monotonic()
                if not ok or frame is None or frame.size == 0:
                    active_cap.release()
                    reconnected = _reconnect_stream(source_label, config.source, stop_event)
                    if reconnected is None:
                        break
                    active_cap, frame = reconnected
                    t_now = time.monotonic()
                state.ingest_frame(frame, t_now, vl, config)
        finally:
            active_cap.release()

    def face_loop() -> None:
        interval = 1.0 / config.face_detect_hz
        while not stop_event.is_set():
            cycle_start = time.monotonic()
            frame_copy: np.ndarray | None = None
            with state.lock:
                if state.latest_frame is not None:
                    frame_copy = state.latest_frame.copy()
            if frame_copy is not None:
                box, conf = detect_face_box(vl, frame_copy)
                state.update_face(box, conf, time.monotonic(), frame_copy, vl, config)
            elapsed = time.monotonic() - cycle_start
            stop_event.wait(max(0.01, interval - elapsed))

    def vitals_loop() -> None:
        while not stop_event.wait(config.eval_interval_seconds):
            _evaluate_window(vl, state, config)

    threads = [
        threading.Thread(target=capture_loop, name="CaptureThread", daemon=True),
        threading.Thread(target=face_loop, name="FaceDetectorThread", daemon=True),
        threading.Thread(target=vitals_loop, name="VitalsWorkerThread", daemon=True),
    ]
    for thread in threads:
        thread.start()

    hud_saved = False
    try:
        while not stop_event.is_set():
            frame, snapshot = state.latest_frame_and_snapshot(config)
            if config.save_hud_frame and not hud_saved and frame is not None and snapshot.bpm is not None:
                _save_hud_png(config.save_hud_frame, frame, snapshot, screen_size)
                hud_saved = True

            if config.no_gui:
                time.sleep(0.05)
                continue

            if frame is not None:
                hud = render_hud_frame(frame, snapshot, screen_size=screen_size)
                cv2.imshow(WINDOW_NAME, hud)
            if not _poll_gui_key(WINDOW_NAME, 15, state):
                break
    except KeyboardInterrupt:
        print("[INFO] Interrupted by user.")
    finally:
        stop_event.set()
        for thread in threads:
            thread.join(timeout=2.0)
        if config.save_hud_frame and not hud_saved:
            frame, snapshot = state.latest_frame_and_snapshot(config)
            if frame is not None:
                _save_hud_png(config.save_hud_frame, frame, snapshot, screen_size)
        if not config.no_gui:
            cv2.destroyAllWindows()


def _run_replay(
    vl: VitalLens,
    cap: cv2.VideoCapture,
    first_frame: np.ndarray,
    source_label: str,
    config: StreamConfig,
) -> None:
    raw_fps = float(cap.get(cv2.CAP_PROP_FPS))
    video_fps = raw_fps if raw_fps > 1.0 else config.resample_fps
    detect_every = max(1, int(round(video_fps / config.face_detect_hz)))

    h, w = first_frame.shape[:2]
    state = StreamState(source_label, (w, h), first_frame)
    screen_size = get_screen_geometry()
    if not config.no_gui:
        enable_dpi_awareness()
        screen_size = get_screen_geometry()
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    last_valid_hud: tuple[np.ndarray, VitalsSnapshot] | None = None
    last_frame_hud: tuple[np.ndarray, VitalsSnapshot] | None = None
    last_eval_idx = -1
    next_eval_ts = config.min_window_seconds
    replay_wall_start = time.monotonic()
    user_aborted = False

    frame_idx = 0
    current_frame: np.ndarray | None = first_frame
    last_processed_idx = 0

    try:
        while current_frame is not None:
            ts = frame_idx / video_fps
            if frame_idx == 0 or frame_idx % detect_every == 0:
                box, conf = detect_face_box(vl, current_frame)
                state.update_face(box, conf, ts, current_frame, vl, config)

            state.ingest_frame(current_frame, ts, vl, config)

            if ts + 1e-9 >= next_eval_ts:
                snap = _evaluate_window(vl, state, config)
                if snap is not None:
                    last_eval_idx = frame_idx
                    next_eval_ts = ts + config.eval_interval_seconds

            latest_frame, snapshot = state.latest_frame_and_snapshot(config)
            if latest_frame is not None:
                last_frame_hud = (latest_frame, snapshot)
                if snapshot.bpm is not None:
                    last_valid_hud = (latest_frame.copy(), snapshot)

            if not config.no_gui:
                hud = render_hud_frame(current_frame, snapshot, screen_size=screen_size)
                cv2.imshow(WINDOW_NAME, hud)
                wait_ms = 1
                if not config.fast_replay:
                    target_wall = replay_wall_start + ts
                    wait_ms = max(1, int(round((target_wall - time.monotonic()) * 1000.0)))
                if not _poll_gui_key(WINDOW_NAME, wait_ms, state):
                    user_aborted = True
                    break
            elif not config.fast_replay:
                delay = (replay_wall_start + ts) - time.monotonic()
                if delay > 0.0:
                    time.sleep(delay)

            last_processed_idx = frame_idx
            frame_idx += 1
            ok, next_frame = cap.read()
            current_frame = next_frame if ok and next_frame is not None and next_frame.size > 0 else None

        if not user_aborted and last_eval_idx != last_processed_idx:
            final_snap = _evaluate_window(vl, state, config)
            if final_snap is not None and last_frame_hud is not None:
                last_frame_hud = (last_frame_hud[0], final_snap)
                if final_snap.bpm is not None:
                    last_valid_hud = (last_frame_hud[0].copy(), final_snap)

        if config.save_hud_frame:
            chosen = last_valid_hud if last_valid_hud is not None else last_frame_hud
            if chosen is not None:
                _save_hud_png(config.save_hud_frame, chosen[0], chosen[1], screen_size)

        _, final_snapshot = state.latest_frame_and_snapshot(config)
        print(_format_summary(final_snapshot))
    except KeyboardInterrupt:
        print("[INFO] Interrupted by user.")
    finally:
        cap.release()
        if not config.no_gui:
            cv2.destroyAllWindows()


def _format_summary(snapshot: VitalsSnapshot) -> str:
    label = TRIAGE_META[snapshot.triage][0]
    if snapshot.bpm is None:
        return f"[SUMMARY] {snapshot.source_label}: No valid BPM produced (Evals: {snapshot.eval_count}, Status: {label})"
    err_part = ""
    if snapshot.expected_bpm is not None:
        err_part = f" | Expected: {snapshot.expected_bpm:.1f} BPM (Error: {abs(snapshot.bpm - snapshot.expected_bpm):.1f} BPM)"
    return (
        f"[SUMMARY] {snapshot.source_label}: {snapshot.bpm:.1f} BPM | "
        f"Confidence: {snapshot.confidence:.2f} | Triage: [{label}]{err_part} | "
        f"Evals: {snapshot.eval_count}"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Real-time IP camera and video rPPG estimator using VitalLens POS.")
    parser.add_argument("--source", type=str, default=CAMERA_URL, help="HTTP/RTSP stream URL or local video file path.")
    parser.add_argument("--window-seconds", type=float, default=WINDOW_SECONDS, help="Rolling estimation window in seconds.")
    parser.add_argument("--min-window-seconds", type=float, default=MIN_WINDOW_SECONDS, help="Minimum window before first estimate.")
    parser.add_argument("--eval-interval", type=float, default=EVAL_INTERVAL_SECONDS, help="Vitals evaluation period in seconds.")
    parser.add_argument("--target-fps", type=float, default=RESAMPLE_FPS, help="Uniform resampling framerate for POS.")
    parser.add_argument("--expected-bpm", type=float, default=None, help="Optional reference heart rate in BPM.")
    parser.add_argument("--no-gui", action="store_true", help="Run headless without opening an OpenCV window.")
    parser.add_argument("--fast-replay", action="store_true", help="Process local video without wall-clock pacing.")
    parser.add_argument("--save-hud-frame", type=str, default=None, help="Optional PNG path to save a rendered HUD frame.")
    return parser


def parse_args(argv: list[str] | None = None) -> StreamConfig:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        return StreamConfig(
            source=args.source,
            window_seconds=args.window_seconds,
            min_window_seconds=args.min_window_seconds,
            eval_interval_seconds=args.eval_interval,
            resample_fps=args.target_fps,
            expected_bpm=args.expected_bpm,
            no_gui=args.no_gui,
            fast_replay=args.fast_replay,
            save_hud_frame=args.save_hud_frame,
        )
    except ValueError as exc:
        parser.error(str(exc))


def main() -> None:
    config = parse_args()
    print("[INFO] Initializing VitalLens (POS, export_to_json=False)...")
    vl = VitalLens(method="pos", export_to_json=False)

    cap, source_label, is_live, first_frame = _probe_source(config.source)
    mode_name = "LIVE STREAM" if is_live else ("FAST REPLAY" if config.fast_replay else "PACED REPLAY")
    print(f"[INFO] Source: {source_label} | Mode: {mode_name} | Window: {config.window_seconds:.1f}s")

    if is_live:
        _run_live(vl, cap, first_frame, source_label, config)
    else:
        _run_replay(vl, cap, first_frame, source_label, config)


if __name__ == "__main__":
    main()
