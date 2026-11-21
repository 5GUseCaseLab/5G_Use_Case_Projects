"""Offline replay on the LAPTOP: video -> YOLO + motion.py + the board's tracker/advisor -> annotated MP4 + CSV.
Use it to tune HORIZON / CAM_H / HFOV / FOE_X on DriveCache clips before touching the board.

    python tools/replay.py videos/road_scenario_2_20_min_apart.mp4 --horizon 0.29 --cam-h 1.55 --hfov 68
    python tools/replay.py clip.mp4 --calib calib_out/calibration.json         # values from tools/calibrate.py
    python tools/replay.py clip.mp4 --rate 12      # imitate the board: only ~12 frames/s reach the laptop
    python tools/replay.py clip.mp4 --detector none   # ego-motion only (no ultralytics needed)

Output: <clip>_replay.mp4 (boxes with id / distance / closing speed / direction / km/h over ground, static and
dynamic horizon, FOE, own speed + turn rate, advisor banner) and <clip>_tracks.csv (one row per track per frame).
"""
import argparse, csv, json, math, sys, time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "laptop"))
sys.path.insert(0, str(ROOT / "unoq_app" / "python"))
from motion import MotionEstimator, FlowEngine          # noqa: E402
from tracking import Tracker                            # noqa: E402
from advisor import Advisor                             # noqa: E402

KEEP = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
COL = {"person": (62, 136, 240), "car": (255, 166, 88), "bus": (255, 168, 210), "truck": (255, 168, 210),
       "bicycle": (80, 185, 63), "motorcycle": (80, 185, 63)}
RISK = [None, (34, 153, 210), (12, 89, 232), (73, 81, 248)]


def make_detector(kind, model, imgsz, conf):
    if kind == "none":
        return lambda img: []
    from ultralytics import YOLO
    import torch
    m = YOLO(model)
    dev = 0 if torch.cuda.is_available() else "cpu"

    def det(img):
        r = m.predict(img, imgsz=imgsz, conf=conf, classes=list(KEEP), verbose=False, device=dev,
                      half=dev == 0)[0]
        b = r.boxes
        return [[KEEP[int(c)], float(p), *map(float, box)]
                for box, p, c in zip(b.xyxyn.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy())]
    return det


def draw(img, tracks, adv, ego, trk, cal, t_ms):
    H, W = img.shape[:2]
    s = W / 640
    th = max(1, int(round(s)))
    cv2.line(img, (0, int(cal["horizon"] * H)), (W, int(cal["horizon"] * H)), (200, 200, 200), th, cv2.LINE_AA)
    hy = int(trk.hy_now / trk.h * H)
    cv2.line(img, (0, hy), (W, hy), (0, 255, 255), th, cv2.LINE_AA)
    cv2.drawMarker(img, (int(cal["foe_x"] * W), hy), (0, 0, 255), cv2.MARKER_CROSS, int(16 * s), th)
    for tr in tracks:
        x1, y1, x2, y2 = (int(v * (W if i % 2 == 0 else H)) for i, v in enumerate(tr["box"]))
        c = RISK[tr["risk"]] or COL.get(tr["cls"], (255, 255, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), c, 2 * th if tr["risk"] >= 2 else th)
        l1 = f"#{tr['id']} {tr['cls']}" + (f" {tr['dist']}m" if tr["dist"] is not None else "")
        l2 = ""
        if tr["valid"]:
            l2 = f"{'+' if tr['speed'] > 0 else ''}{tr['speed']} m/s"
            if tr["ttc"] is not None: l2 += f" TTC {tr['ttc']}s"
        l3 = "" if tr["dir"] == "–" else (f"{tr['dir']} {tr['abs_kmh']} km/h" if tr["dir"] != "stopped" else "stopped")
        for k, txt in enumerate(t for t in (l1, l2, l3) if t):
            y = max(12, y1 - int(4 * s) - int(13 * s) * (2 - k))
            (tw, tht), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.38 * s, th)
            cv2.rectangle(img, (x1, y - tht - 2), (x1 + tw + 4, y + 3), c, -1)
            cv2.putText(img, txt, (x1 + 2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38 * s, (0, 0, 0), th, cv2.LINE_AA)
    hud = "ego: –"
    if ego:
        hud = f"ego {ego['v']*3.6:5.1f} km/h  turn {math.degrees(ego.get('yaw') or 0):+5.1f} deg/s"
    lvl = adv["level"]
    bar = [(60, 140, 60), (34, 153, 210), (12, 89, 232), (40, 40, 230)][lvl]
    cv2.rectangle(img, (0, H - int(22 * s)), (W, H), bar, -1)
    cv2.putText(img, f"{adv['title']}  {adv['detail'][:70]}", (int(6 * s), H - int(7 * s)), cv2.FONT_HERSHEY_SIMPLEX,
                0.42 * s, (255, 255, 255), th, cv2.LINE_AA)
    cv2.rectangle(img, (0, 0), (int(330 * s), int(20 * s)), (0, 0, 0), -1)
    cv2.putText(img, f"{hud}   {t_ms:.0f} ms", (int(6 * s), int(14 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.42 * s,
                (255, 255, 255), th, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("video")
    ap.add_argument("--calib", help="calibration.json from tools/calibrate.py")
    ap.add_argument("--hfov", type=float, default=68.0)
    ap.add_argument("--cam-h", type=float, default=1.55)
    ap.add_argument("--horizon", type=float, default=0.29)
    ap.add_argument("--foe-x", type=float, default=0.5)
    ap.add_argument("--ego-kmh", type=float, default=0.0, help="manual own speed if the flow fit fails")
    ap.add_argument("--detector", default="yolo", choices=["yolo", "none"])
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--imgsz", type=int, default=480)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--flow", default="dis", choices=["dis", "dis-fast", "raft"])
    ap.add_argument("--rate", type=float, default=0, help="process at most this many frames/s (0 = every frame)")
    ap.add_argument("--width", type=int, default=640, help="processing width (the board uses 640)")
    ap.add_argument("--out", help="output mp4")
    a = ap.parse_args()

    cal = dict(hfov=a.hfov, cam_h=a.cam_h, horizon=a.horizon, foe_x=a.foe_x)
    if a.calib:
        j = json.load(open(a.calib))
        cal.update(hfov=j["HFOV"], cam_h=j["CAM_H"], horizon=j["HORIZON"], foe_x=j["FOE_X"])
    print("calibration:", cal)
    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        sys.exit(f"cannot open {a.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out = Path(a.out or Path(a.video).with_name(Path(a.video).stem + "_replay.mp4"))
    det = make_detector(a.detector, a.model, a.imgsz, a.conf)
    est = MotionEstimator(FlowEngine(a.flow))
    est.set_cal(cal)
    adv = Advisor()
    trk = writer = None
    csv_path = out.with_name(out.stem.replace("_replay", "") + "_tracks.csv")
    fcsv = open(csv_path, "w", newline="")
    log = csv.writer(fcsv)
    log.writerow(["t", "ego_kmh", "yaw_dps", "id", "cls", "dist_m", "lat_m", "closing_mps", "ttc_s", "ttc_loom_s",
                  "abs_kmh", "heading_deg", "dir", "risk"])
    n, t_next, last_ego, last_use = 0, 0.0, None, None
    tracks, advice, t_ms = [], dict(level=0, title="CLEAR", detail=""), 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = n / fps
        n += 1
        h0, w0 = frame.shape[:2]
        img = cv2.resize(frame, (a.width, int(h0 * a.width / w0)), interpolation=cv2.INTER_AREA) if w0 != a.width else frame
        if trk is None:
            trk = Tracker(img.shape[1], img.shape[0], cal["hfov"], cal["cam_h"], cal["horizon"], cal["foe_x"])
            writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (img.shape[1], img.shape[0]))
        if a.rate <= 0 or t >= t_next - 1e-6:
            t_next = t + (1.0 / a.rate if a.rate > 0 else 0)
            t0 = time.perf_counter()
            dets = det(img)
            ego, ext = est.analyze(est.flow_step(img, t, 0), dets)
            if ext:
                dets = [d + [e] if e else d for d, e in zip(dets, ext)]
            if ego and ego.get("ok"):
                last_ego = ego
            use = ego if ego and ego.get("ok") else ({"v": a.ego_kmh / 3.6, "v_sd": 1.0, "yaw": 0.0} if a.ego_kmh > 0 else None)
            trk.update(dets, t, use)
            t_ms = (time.perf_counter() - t0) * 1000
            last_use = use
        tracks = trk.snapshot(t)
        advice = adv.update(tracks, t, (last_use or {}).get("v") or 0.0)
        for tr in tracks:
            log.writerow([round(t, 3), round((last_use or {}).get("v", 0) * 3.6, 1) if last_use else "",
                          round(math.degrees((last_use or {}).get("yaw") or 0), 2) if last_use else "",
                          tr["id"], tr["cls"], tr["dist"], tr["x"], tr["speed"], tr["ttc"], tr["ttc_loom"],
                          tr["abs_kmh"], tr["heading"], tr["dir"], tr["risk"]])
        vis = img.copy()
        draw(vis, tracks, advice, last_use, trk, cal, t_ms)
        writer.write(vis)
        if n % 60 == 0:
            e = f"{last_use['v']*3.6:.1f} km/h" if last_use else "–"
            print(f"  {t:6.1f} s  ego {e}  tracks {len(tracks)}  {t_ms:.0f} ms/frame", flush=True)
    if writer: writer.release()
    fcsv.close()
    print(f"\n{n} frames -> {out}\ntrack log -> {csv_path}")


if __name__ == "__main__":
    main()
