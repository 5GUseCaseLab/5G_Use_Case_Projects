"""Offline tests for the area code (no camera needed):  python tests/test_area.py"""
import json
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
import common


def synthetic_floor(markers_tl, M=0.167, room=(4.5, 3.5), ppm=300):
    """Floor plane drawn in metres (marker ids 0.. at given top-left positions), then warped by a strong
    perspective into a 1920x1080 'camera' image. Returns image, projector world->pixel."""
    pad = 100
    cw, ch = int(room[0] * ppm) + 2 * pad, int(room[1] * ppm) + 2 * pad
    canvas = np.full((ch, cw), 215, np.uint8)
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    side = int(M * ppm)
    for mid, (x, y) in markers_tl.items():
        mk = cv2.aruco.generateImageMarker(d, mid, side)
        mk = cv2.copyMakeBorder(mk, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)   # white quiet zone
        px, py = int(x * ppm) + pad - 12, int(y * ppm) + pad - 12
        canvas[py:py + side + 24, px:px + side + 24] = mk
    src = np.float32([[0, 0], [cw, 0], [cw, ch], [0, ch]])
    dst = np.float32([[500, 380], [1450, 380], [1850, 1000], [90, 1000]])
    Mw = cv2.getPerspectiveTransform(src, dst)
    img = cv2.cvtColor(cv2.warpPerspective(canvas, Mw, (1920, 1080), borderValue=120), cv2.COLOR_GRAY2BGR)

    def world_to_px(x, y):
        return cv2.perspectiveTransform(np.float32([[[x * ppm + pad, y * ppm + pad]]]), Mw)[0][0]

    return img, world_to_px


REF = {0: (1.5, 1.2), 1: (3.0, 1.2), 2: (3.0, 2.2), 3: (1.5, 2.2)}      # 1.5 x 1.0 m rectangle inside a 4.5 x 3.5 room


def test_click_mode_room_area():
    img, w2p = synthetic_floor(REF)
    H, _ = common.aruco_homography(img, REF, 0.167)
    room_px = [w2p(0, 0), w2p(4.5, 0), w2p(4.5, 3.5), w2p(0, 3.5)]
    area = common.polygon_area_m2(H, room_px)
    err = abs(area - 15.75) / 15.75
    print(f"room area from 4 markers covering only 1.5x1.0 m: computed {area:.2f}, true 15.75, error {err:.2%}")
    assert err < 0.05


def test_single_marker_is_refused():
    img, _ = synthetic_floor({0: (1.5, 1.2)})
    try:
        common.aruco_homography(img, {0: (1.5, 1.2)}, 0.167)
        raise AssertionError("should refuse a single marker")
    except ValueError as e:
        print("single marker ->", str(e)[:60])


def test_cli_click_mode_headless():
    img, w2p = synthetic_floor(REF)
    with tempfile.TemporaryDirectory() as t:
        cv2.imwrite(os.path.join(t, "f.jpg"), img)
        zone = " ".join(f"{p[0]:.1f},{p[1]:.1f}" for p in [w2p(0, 0), w2p(4.5, 0), w2p(4.5, 3.5), w2p(0, 3.5)])
        ref = " ".join(f"{k}:{v[0]},{v[1]}" for k, v in REF.items())
        r = subprocess.run([sys.executable, os.path.join(ROOT, "calibrate_area.py"), "--source", os.path.join(t, "f.jpg"),
                            "--method", "aruco-click", "--marker-m", "0.167", "--ref", ref, "--zone-px", zone,
                            "--out", os.path.join(t, "c.json")], capture_output=True, text=True, cwd=t)
        print(r.stdout.strip().splitlines()[-2:] or r.stderr[-300:])
        cfg = json.load(open(os.path.join(t, "c.json")))
        assert abs(cfg["area_m2"] - 15.75) / 15.75 < 0.05, cfg["area_m2"]


def test_marker_rectangle_mode_and_helper_and_missing():
    ref = {0: (1.0, 1.0), 1: (3.5, 1.0), 2: (3.5, 2.7), 3: (1.0, 2.7), 4: (2.2, 1.8)}
    img, w2p = synthetic_floor(ref)
    H, poly, area = common.aruco_zone(img, ref, 0.167, vertex_ids=[0, 1, 2, 3])
    assert len(poly) == 4 and abs(area - 2.5 * 1.7) / 4.25 < 0.03, area
    d = common.distance_m(H, w2p(1.2, 0.5), w2p(3.8, 3.0))
    assert abs(d - np.hypot(2.6, 2.5)) / np.hypot(2.6, 2.5) < 0.03, d
    img2 = img.copy()
    x, y = np.int32(w2p(3.55, 1.05))
    cv2.rectangle(img2, (x - 60, y - 60), (x + 60, y + 60), (215, 215, 215), -1)
    try:
        common.aruco_zone(img2, ref, 0.167, vertex_ids=[0, 1, 2, 3])
        raise AssertionError("expected ValueError for hidden corner marker")
    except ValueError as e:
        print("hidden marker ->", str(e)[:50])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
