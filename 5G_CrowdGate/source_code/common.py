"""Shared helpers for CrowdGate: config, geometry (homography / zone / area), state machine."""
import json
from collections import deque

import cv2
import numpy as np

# ---------------------------------------------------------------- config
def save_config(path, cfg):
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)


def load_config(path):
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------- geometry
# Homography = a 3x3 matrix that maps a pixel (x, y) on the floor to real floor metres.
def compute_homography(px_pts, m_pts):
    """px_pts: Nx2 pixels, m_pts: Nx2 metres (N >= 4). Returns 3x3 homography."""
    H, _ = cv2.findHomography(np.float32(px_pts), np.float32(m_pts), method=0)
    if H is None:
        raise ValueError("Could not compute homography (points degenerate?)")
    return H


def px_to_m(H, pts_px):
    pts = np.float32(pts_px).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, np.float32(H)).reshape(-1, 2)


def polygon_area_m2(H, poly_px, obstacles_px=()):
    """Floor area of the zone in m^2 (shoelace formula via cv2.contourArea),
    minus obstacle polygons (pillars, stage...)."""
    area = cv2.contourArea(px_to_m(H, poly_px))
    for ob in obstacles_px:
        area -= cv2.contourArea(px_to_m(H, ob))
    return float(area)


def foot_points(xyxy):
    """Bottom-centre of each box = where the person touches the floor."""
    xyxy = np.asarray(xyxy, dtype=float).reshape(-1, 4)
    return np.stack([(xyxy[:, 0] + xyxy[:, 2]) / 2, xyxy[:, 3]], axis=1)


def in_zone(points_px, poly_px):
    """Boolean list: is each point inside (or on the edge of) the zone polygon?"""
    poly = np.int32(poly_px).reshape(-1, 1, 2)
    return [cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0 for x, y in points_px]


def detect_aruco(frame, dict_name="DICT_4X4_50"):
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    det = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())
    corners, ids, _ = det.detectMarkers(frame)
    return corners, ids


def distance_m(H, p_px, q_px):
    """Real floor distance (metres) between two image pixels that lie ON the floor plane."""
    a, b = px_to_m(H, [p_px, q_px])
    return float(np.hypot(*(a - b)))


def aruco_homography(frame, ref_tl, marker_m, dict_name="DICT_4X4_50", min_markers=3):
    """Pixel->metre homography from ArUco markers whose floor positions are known.
    ref_tl: {marker_id: (x_m, y_m)} position of each marker's top-left corner (+x = marker's right, +y = marker's down;
    all markers laid with the same orientation). Needs >= min_markers visible: one small marker alone gives a
    badly conditioned homography (area errors of tens of percent), so we refuse it.
    Returns (H, corners_by_id) where corners_by_id[id] = 4x2 pixel corners (TL, TR, BR, BL)."""
    corners, ids = detect_aruco(frame, dict_name)
    if ids is None:
        raise ValueError("No ArUco markers found")
    ids = [int(i) for i in ids.flatten()]
    ref_tl = {int(k): v for k, v in ref_tl.items()}
    by_id = {mid: corners[i][0] for i, mid in enumerate(ids)}
    used = [m for m in by_id if m in ref_tl]
    if len(used) < min_markers:
        raise ValueError(f"Need at least {min_markers} reference markers visible, found {sorted(used)} "
                         f"(all detected: {sorted(by_id)}; --ref lists {sorted(ref_tl)})")
    src, dst = [], []
    for mid in used:
        x, y = ref_tl[mid]
        src += by_id[mid].tolist()
        dst += [(x, y), (x + marker_m, y), (x + marker_m, y + marker_m), (x, y + marker_m)]
    return compute_homography(src, dst), by_id


def aruco_zone(frame, ref_tl, marker_m, vertex_corner=0, dict_name="DICT_4X4_50", vertex_ids=None):
    """Zone = polygon through one corner of chosen markers (default: all ids in ref_tl, sorted by id).
    vertex_ids: ids that form the polygon; other ids in ref_tl are helper markers (only improve the homography).
    Returns (H, poly_px, area_m2)."""
    ref_tl = {int(k): v for k, v in ref_tl.items()}
    vertex_ids = sorted(int(v) for v in (vertex_ids if vertex_ids is not None else ref_tl))
    H, by_id = aruco_homography(frame, ref_tl, marker_m, dict_name, min_markers=1)
    missing = [v for v in vertex_ids if v not in by_id]
    if missing:
        raise ValueError(f"Zone-corner markers not detected: {missing} (seen: {sorted(by_id)})")
    poly_px = [by_id[v][vertex_corner].tolist() for v in vertex_ids]
    return H, poly_px, polygon_area_m2(H, poly_px)


# ---------------------------------------------------------------- state machine
GREEN, YELLOW, RED = "GREEN", "YELLOW", "RED"
STATE_CODE = {GREEN: 0, YELLOW: 1, RED: 2, "FAILSAFE": 3}


class DensityState:
    """Smooths raw counts over N frames and applies thresholds with hysteresis
    (hysteresis = different up/down trip points so the state doesn't flicker)."""

    def __init__(self, area_m2, yellow=2.0, red=4.0, margin=0.2, window=10):
        self.area, self.yellow, self.red, self.margin = area_m2, yellow, red, margin
        self.buf = deque(maxlen=window)
        self.state = GREEN

    def update(self, count):
        self.buf.append(count)
        smooth = float(np.median(self.buf))          # median ignores single-frame glitches
        d = smooth / self.area
        s, m = self.state, self.margin
        if s == GREEN and d >= self.yellow:
            s = YELLOW
        elif s == YELLOW:
            if d >= self.red:
                s = RED
            elif d < self.yellow - m:
                s = GREEN
        elif s == RED and d < self.red - m:
            s = YELLOW if d >= self.yellow else GREEN
        if s == GREEN and d >= self.red:             # big jump straight to red
            s = RED
        self.state = s
        return s, smooth, d


def load_model(path, openvino=False, imgsz=960):
    """Load a YOLO model; with openvino=True export once and use the faster Intel-CPU version.
    Shared by the MEC service, the accuracy experiment and the latency server so all three use the same inference."""
    import os
    from ultralytics import YOLO
    model = YOLO(path)
    if openvino:
        ov_dir = os.path.splitext(path)[0] + "_openvino_model"
        if not os.path.isdir(ov_dir):
            model.export(format="openvino", imgsz=imgsz)      # needs: pip install openvino
        model = YOLO(ov_dir, task="detect")
    return model


def count_people(model, frame, conf, imgsz, person_class=0, zone=None):
    """Number of people whose foot point is inside `zone` (all people if zone is None). Same rule as the MEC service."""
    r = model.predict(frame, classes=[person_class], conf=conf, imgsz=imgsz, verbose=False)[0]
    boxes = r.boxes.xyxy.cpu().numpy() if len(r.boxes) else np.zeros((0, 4))
    if zone is not None and len(boxes):
        boxes = boxes[np.array(in_zone(foot_points(boxes), zone), dtype=bool)]
    return len(boxes)
