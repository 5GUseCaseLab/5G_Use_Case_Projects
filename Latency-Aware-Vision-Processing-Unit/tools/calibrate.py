"""Find HORIZON / FOE_X (and check CAM_H / HFOV) from a drive video or a folder of frames.

    python tools/calibrate.py videos/curvy_road_20_min_apart.mp4
    python tools/calibrate.py "DriveCache/Road Scenario 2/YOLO inference of frames 20 min apart" --hfov 68 --cam-h 1.55
    python tools/calibrate.py drive.mp4 --speed-kmh 32        # you know how fast you drove -> CAM_H is solved
    python tools/calibrate.py drive.mp4 --lane 900,410,1530 --lane-width 3.5   # a lane measured on the overlay

Two independent horizon estimates:
  1. VANISHING POINT of straight edges (lane lines, kerbs, fences). Every horizontal line in the world - also
     on a curve - vanishes ON the horizon line, so the median VP height over many frames is the horizon.
  2. OPTICAL FLOW (laptop/motion.py): the ego-motion fit locates the focus of expansion = where the car is
     heading (FOE_X) and the horizon, while moving. Works on night clips too, where edges are scarce.
If they agree to within ~2 % of the image height, use the VP value; if not, look at the overlay.

What each setting does (all as used by unoq_app/python/tracking.py):
  HORIZON  0..1 from the top. Ground distance Z = f*CAM_H/(y_bottom - y_horizon): 1 % of image height wrong
           at 30 m is ~15-20 % distance error, so this is the most important one.
  CAM_H    metres from road to lens. Scales every ground distance AND the ego speed linearly.
  HFOV     horizontal field of view of the frames as recorded (after any stabilisation crop). Sets f.
           Size-based distance scales with f; ground distance scales with f too.
  FOE_X    0..1 from the left: the image column the car drives towards. If the camera is mounted turned,
           'ahead / left / right' (lane zone) is wrong without it.

The overlay (calib_overlay.png) draws the horizon, the FOE and 3.5 m-wide lane bars at 5/10/20/30/50 m.
If the bars are wider than your lane, CAM_H is too small (or HFOV too large), and vice versa.
"""
import argparse, json, math, os, sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "laptop"))
from motion import MotionEstimator, FlowEngine                      # noqa: E402

EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
CAMERAS = {   # approximate horizontal FOV of 16:9 video (check your own camera; stabilisation crops a bit more)
    "phone": 68, "moto-edge-50-fusion": 68, "gopro-linear": 86, "gopro-wide": 118, "dashcam": 100, "webcam": 60,
}


def read_frames(src, every, max_frames, width):
    """-> list of (index, frame_at_width, full_res_frame_or_None) as consecutive PAIRS (i, i+1) every `every`."""
    p = Path(src)
    out = []
    if p.is_dir():
        files = sorted(f for f in p.iterdir() if f.suffix.lower() in EXTS)
        if files and files[0].stat().st_size < 1024:
            sys.exit(f"{src}: frames look like Git LFS pointers. Run `git lfs pull --include=\"<folder>/**\"` first.")
        idx = list(range(0, max(0, len(files) - 1), every))[:max_frames]
        for i in idx:
            for j in (i, i + 1):
                img = cv2.imread(str(files[j]))
                if img is not None:
                    out.append((j, img))
    else:
        cap = cv2.VideoCapture(str(p))
        if not cap.isOpened():
            sys.exit(f"cannot open {src}")
        n, k = 0, 0
        while k < max_frames * 2:
            ok, img = cap.read()
            if not ok: break
            if n % every in (0, 1):
                out.append((n, img)); k += 1
            n += 1
    if not out:
        sys.exit("no frames read")
    full = out[len(out) // 2][1].copy()
    if width:
        out = [(i, cv2.resize(im, (width, int(im.shape[0] * width / im.shape[1])), interpolation=cv2.INTER_AREA))
               for i, im in out]
    return out, full


# ------------------------------------------------------------------ vanishing point of straight edges
def segments(gray):
    H, W = gray.shape
    e = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 120)
    e[: int(0.18 * H)] = 0                                     # sky / canopy
    L = cv2.HoughLinesP(e, 1, np.pi / 180, 40, minLineLength=int(0.05 * W), maxLineGap=8)
    if L is None:
        return np.zeros((0, 4))
    L = L.reshape(-1, 4).astype(np.float64)
    ang = np.degrees(np.arctan2(L[:, 3] - L[:, 1], L[:, 2] - L[:, 0])) % 180
    a = np.minimum(ang, 180 - ang)                             # 0 = horizontal, 90 = vertical
    return L[(a > 12) & (a < 78)]                              # no box edges, no tree trunks, no far horizontals


def vanishing_point(L, W, H, rng, iters=400):
    if len(L) < 4:
        return None
    p1, p2 = L[:, :2], L[:, 2:]
    d = p2 - p1
    ln = np.hypot(d[:, 0], d[:, 1])
    hom = np.cross(np.c_[p1, np.ones(len(L))], np.c_[p2, np.ones(len(L))])
    slope_sign = np.sign(d[:, 0] * d[:, 1])
    best, best_s = None, 0.0
    for _ in range(iters):
        i, j = rng.choice(len(L), 2, replace=False, p=ln / ln.sum())
        if slope_sign[i] == slope_sign[j]:
            continue                                           # want one line from each side of the road
        v = np.cross(hom[i], hom[j])
        if abs(v[2]) < 1e-9:
            continue
        v = v[:2] / v[2]
        if not (-0.5 * W < v[0] < 1.5 * W and 0.05 * H < v[1] < 0.8 * H):
            continue
        mid = (p1 + p2) / 2
        to = v - mid
        cosang = np.abs((to * d).sum(1)) / (np.hypot(to[:, 0], to[:, 1]) * ln + 1e-9)
        inl = cosang > math.cos(math.radians(1.5))
        s = (ln * inl).sum()
        if s > best_s and len(set(slope_sign[inl])) == 2:
            best_s, best = s, (v, inl)
    if best is None:
        return None
    v, inl = best
    A = hom[inl, :2] / np.hypot(hom[inl, 0], hom[inl, 1])[:, None]   # least-squares refine
    b = -hom[inl, 2] / np.hypot(hom[inl, 0], hom[inl, 1])
    try:
        v = np.linalg.lstsq(A * ln[inl, None], b * ln[inl], rcond=None)[0]
    except np.linalg.LinAlgError:
        pass
    return v, int(inl.sum()), L[inl]


# ------------------------------------------------------------------ overlay
def overlay(img, horizon, foe_x, hfov, cam_h, vp_lines=None, scale=1.0):
    H, W = img.shape[:2]
    o = img.copy()
    f = (W / 2) / math.tan(math.radians(hfov) / 2)
    yh, xf = horizon * H, foe_x * W
    t = max(1, int(round(W / 640)))
    if vp_lines is not None:
        for x1, y1, x2, y2 in vp_lines * scale:
            cv2.line(o, (int(x1), int(y1)), (int(x2), int(y2)), (255, 200, 0), t)
    cv2.line(o, (0, int(yh)), (W, int(yh)), (0, 255, 255), 2 * t)
    cv2.putText(o, f"horizon {horizon:.3f}", (10 * t, int(yh) - 8 * t), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * t, (0, 255, 255), 2 * t)
    cv2.drawMarker(o, (int(xf), int(yh)), (0, 0, 255), cv2.MARKER_CROSS, 30 * t, 2 * t)
    for Z in (5, 10, 15, 20, 30, 50):
        y = yh + f * cam_h / Z
        if y >= H: continue
        half = f * 1.75 / Z
        cv2.line(o, (int(xf - half), int(y)), (int(xf + half), int(y)), (0, 255, 0), 2 * t)
        for x in (xf - half, xf + half):
            cv2.line(o, (int(x), int(y - 6 * t)), (int(x), int(y + 6 * t)), (0, 255, 0), 2 * t)
        cv2.putText(o, f"{Z} m", (int(xf + half + 8 * t), int(y + 5 * t)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * t, (0, 255, 0), t + 1)
    cv2.putText(o, f"green bars = 3.5 m lane at that distance  (CAM_H {cam_h} m, HFOV {hfov} deg)", (10 * t, H - 12 * t),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55 * t, (255, 255, 255), 2 * t)
    for x in range(0, W, 100 * t):                              # pixel ruler for --lane measurements
        cv2.line(o, (x, 0), (x, 8 * t), (255, 255, 255), t)
        cv2.putText(o, str(x), (x + 2, 20 * t), cv2.FONT_HERSHEY_SIMPLEX, 0.35 * t, (255, 255, 255), t)
    for y in range(0, H, 100 * t):
        cv2.line(o, (0, y), (8 * t, y), (255, 255, 255), t)
        cv2.putText(o, str(y), (10 * t, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35 * t, (255, 255, 255), t)
    return o


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("source", help="video file or folder of frames")
    ap.add_argument("--hfov", type=float, default=None, help="horizontal FOV in degrees (default: --camera preset)")
    ap.add_argument("--camera", default="phone", choices=sorted(CAMERAS), help="FOV preset if --hfov not given")
    ap.add_argument("--cam-h", type=float, default=1.55, help="camera height above the road, m")
    ap.add_argument("--horizon", type=float, default=0.40, help="starting guess for the flow fit")
    ap.add_argument("--speed-kmh", type=float, help="your true average speed over the clip -> solves CAM_H")
    ap.add_argument("--lane", help="Y,XL,XR: pixel row and the two lane edges on it (full-res overlay pixels)")
    ap.add_argument("--lane-width", type=float, default=3.5, help="real lane width for --lane, m")
    ap.add_argument("--every", type=int, default=3, help="use a frame pair every N frames")
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--out", default="calib_out", help="folder for overlay + json")
    a = ap.parse_args()
    hfov = a.hfov or CAMERAS[a.camera]

    pairs, full = read_frames(a.source, a.every, a.max_frames, 960)
    Hf, Wf = full.shape[:2]
    H, W = pairs[0][1].shape[:2]
    rng = np.random.default_rng(0)

    # 1) vanishing points
    vps, show_lines = [], None
    for k, (i, img) in enumerate(pairs[::2]):
        r = vanishing_point(segments(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)), W, H, rng)
        if r:
            vps.append(r[0])
            if show_lines is None or k == len(pairs) // 4:
                show_lines = r[2]
    vps = np.array(vps) if vps else np.zeros((0, 2))
    vp_h = float(np.median(vps[:, 1]) / H) if len(vps) >= 3 else None
    vp_iqr = float(np.subtract(*np.percentile(vps[:, 1], [75, 25])) / H) if len(vps) >= 3 else None
    vp_x = float(np.median(vps[:, 0]) / W) if len(vps) >= 3 else None

    # 2) optical flow ego-motion (consecutive pairs only)
    est = MotionEstimator(FlowEngine("dis"))
    est.set_cal(dict(hfov=hfov, cam_h=a.cam_h, horizon=vp_h or a.horizon, foe_x=0.5))
    fps = 30.0
    speeds, hz = [], []
    for k in range(0, len(pairs) - 1, 2):
        (i0, f0), (i1, f1) = pairs[k], pairs[k + 1]
        if i1 != i0 + 1: continue
        est.prev = None
        est.flow_step(f0, i0 / fps, 0)
        ego, _ = est.analyze(est.flow_step(f1, i1 / fps, 0), [])
        if ego and ego.get("ok"):
            speeds.append(ego["v_raw"]); hz.append(ego["horizon"])
    moving = [s for s in speeds if s > 2.0]
    fl_h = float(np.median(hz)) if len(hz) >= 5 else None
    fl_learned = (est.cal["horizon"] * est.prev[0].shape[0] + est.lt_dy) / est.prev[0].shape[0] if est.prev else None
    foe = est.cal["foe_x"] + est.lt_dx / 320 if est.n_lt >= 5 else None
    v_med = float(np.median(moving)) if moving else None

    # recommendations
    horizon = vp_h if vp_h is not None else (fl_learned if est.n_lt >= 5 else a.horizon)
    foe_x = foe if foe is not None else (vp_x if vp_x is not None and abs(vp_x - 0.5) < 0.1 else 0.5)
    cam_h, cam_h_why = a.cam_h, "your --cam-h (not measured)"
    if a.speed_kmh and v_med:
        cam_h = round(a.cam_h * (a.speed_kmh / 3.6) / v_med, 2)
        cam_h_why = f"from --speed-kmh {a.speed_kmh} vs flow speed {v_med*3.6:.1f} km/h at CAM_H {a.cam_h}"
    if a.lane:
        y, xl, xr = (float(v) for v in a.lane.split(","))
        dy = y - horizon * Hf
        if dy > 5:
            cam_h = round(a.lane_width * dy / (xr - xl), 2)
            cam_h_why = f"from a {a.lane_width} m lane measured {xr-xl:.0f} px wide at row {y:.0f}"

    os.makedirs(a.out, exist_ok=True)
    ov = overlay(full, horizon, foe_x, hfov, cam_h, show_lines, scale=Wf / W)
    cv2.imwrite(os.path.join(a.out, "calib_overlay.png"), ov)
    res = dict(HORIZON=round(horizon, 3), FOE_X=round(foe_x, 3), CAM_H=cam_h, HFOV=hfov,
               vp_horizon=vp_h and round(vp_h, 3), vp_spread=vp_iqr and round(vp_iqr, 3), vp_frames=len(vps),
               flow_horizon=fl_h and round(fl_h, 3), flow_frames=len(hz),
               flow_speed_kmh=v_med and round(v_med * 3.6, 1), source=str(a.source))
    json.dump(res, open(os.path.join(a.out, "calibration.json"), "w"), indent=2)

    print(f"\nsource: {a.source}  ({len(pairs)//2} frame pairs, {Wf}x{Hf})")
    print(f"vanishing points : {len(vps)} frames, horizon {vp_h and round(vp_h,3)} (IQR {vp_iqr and round(vp_iqr,3)}), VP x {vp_x and round(vp_x,3)}")
    print(f"optical flow     : {len(hz)} pairs, horizon {fl_h and round(fl_h,3)}, learned {est.n_lt} moving frames, "
          f"median speed {v_med and round(v_med*3.6,1)} km/h (at CAM_H {a.cam_h}, HFOV {hfov})")
    if vp_h and fl_h and abs(vp_h - fl_h) > 0.03:
        print("  ! the two horizon estimates differ by >3 % - hills / heavy pitching / few lines: check the overlay")
    if not moving:
        print("  (vehicle not moving in these frames -> no flow-based horizon/FOE/speed)")
    print("\nRecommended settings")
    print(f"  HORIZON={res['HORIZON']}   (web UI slider: {round(res['HORIZON']*100)} %)")
    print(f"  FOE_X={res['FOE_X']}")
    print(f"  CAM_H={cam_h}   <- {cam_h_why}")
    print(f"  HFOV={hfov}   <- {'--hfov' if a.hfov else 'preset ' + a.camera}")
    print(f"\noverlay: {os.path.join(a.out, 'calib_overlay.png')}   json: {os.path.join(a.out, 'calibration.json')}")


if __name__ == "__main__":
    main()
