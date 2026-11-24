"""STEP: measure floor area from the camera view and save config.json.

Three methods:
  aruco        ArUco markers at the 4 corners of the zone; zone = the marker rectangle.
  aruco-click  (BEST FOR "any room, any camera view") markers laid anywhere on the floor with KNOWN relative
               positions only give the scale; you then click the floor corners of the room/area you want measured.
  manual       no markers: click a floor rectangle of known size, then click the zone.

Examples:
  python calibrate_area.py --source frame.jpg --method aruco --marker-m 0.167 --ref "0:0,0 1:3,0 2:3,2 3:0,2"
  python calibrate_area.py --source frame.jpg --method aruco-click --marker-m 0.167 --ref "0:0,0 1:3,0 2:3,2 3:0,2"
  python calibrate_area.py --source frame.jpg --method aruco-click --marker-m 0.167 --ref "..." --zone-px "120,700 1800,690 1500,300 400,310"
  python calibrate_area.py --source frame.jpg --method manual --rect-m 3.0x2.0
Clicking: left-click each point; ENTER (or SPACE) when finished; q = quit.
"""
import argparse

import cv2
import numpy as np

import common


def grab(source):
    src = int(source) if str(source).isdigit() else source
    img = cv2.imread(str(source))
    if img is not None:
        return img
    cap = cv2.VideoCapture(src)
    for _ in range(5):          # skip a few frames so exposure settles
        cap.read()
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("Could not read from source")
    return frame


def click_points(img, n, title):
    pts = []
    disp = img.copy()

    def cb(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN and (n is None or len(pts) < n):
            pts.append((x, y))
            cv2.circle(disp, (x, y), 5, (0, 0, 255), -1)
            if len(pts) > 1:
                cv2.line(disp, pts[-2], pts[-1], (0, 255, 255), 2)

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(title, cb)
    while True:
        cv2.imshow(title, disp)
        k = cv2.waitKey(20) & 0xFF
        if (n is not None and len(pts) == n) or (n is None and k in (13, 32)):
            break
        if k == ord("q"):
            raise SystemExit
    cv2.destroyWindow(title)
    return pts


def parse_ref(text):
    ref = {}
    for tok in text.split():
        mid, xy = tok.split(":")
        x, y = xy.split(",")
        ref[int(mid)] = (float(x), float(y))
    return ref


def parse_zone_px(text):
    return [tuple(float(v) for v in tok.split(",")) for tok in text.split()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="RTSP URL, camera index, video or image file")
    ap.add_argument("--method", choices=["aruco", "aruco-click", "manual"], default="aruco")
    ap.add_argument("--marker-m", type=float, default=0.20, help="printed black-square side in metres (measure it!)")
    ap.add_argument("--ref", default="0:0,0 1:3.8,0 2:3.8,2.8 3:0,2.8",
                    help="aruco: 'id:x,y ...' world position (m) of each marker's TOP-LEFT corner")
    ap.add_argument("--vertex-ids", default=None,
                    help="method aruco: ids that are zone corners, e.g. 0,1,2,3. Other ids in --ref are helper markers")
    ap.add_argument("--zone-px", default=None,
                    help="method aruco-click: zone polygon in image pixels 'x,y x,y ...' (skips the clicking)")
    ap.add_argument("--subtract-m2", type=float, default=0.0,
                    help="area (m2) of pillars/stage/furniture inside the zone to subtract (measure with a tape)")
    ap.add_argument("--dict", default="DICT_4X4_50")
    ap.add_argument("--rect-m", default="3.0x2.0", help="manual: width x height (m) of the reference rectangle")
    ap.add_argument("--out", default="config.json")
    a = ap.parse_args()

    frame = grab(a.source)

    if a.method == "aruco":
        H, poly_px, area = common.aruco_zone(
            frame, parse_ref(a.ref), a.marker_m, dict_name=a.dict,
            vertex_ids=[int(v) for v in a.vertex_ids.split(",")] if a.vertex_ids else None)
    elif a.method == "aruco-click":
        H, by_id = common.aruco_homography(frame, parse_ref(a.ref), a.marker_m, dict_name=a.dict)
        print(f"Scale from markers {sorted(m for m in by_id if m in parse_ref(a.ref))}")
        if a.zone_px:
            poly_px = parse_zone_px(a.zone_px)
        else:
            print("Click the corners of the FLOOR area to measure (where the floor meets the walls), "
                  "in order around it; press ENTER when done")
            poly_px = click_points(frame, None, "zone polygon")
        area = common.polygon_area_m2(H, poly_px)
    else:
        w, h = (float(v) for v in a.rect_m.lower().split("x"))
        print("Click the 4 corners of the reference rectangle: top-left, top-right, bottom-right, bottom-left")
        rect = click_points(frame, 4, "reference rectangle")
        H = common.compute_homography(rect, [(0, 0), (w, 0), (w, h), (0, h)])
        print("Now click the zone polygon vertices in order, press ENTER when done")
        poly_px = click_points(frame, None, "zone polygon")
        area = common.polygon_area_m2(H, poly_px)

    area -= a.subtract_m2
    print(f"\nMonitored area = {area:.2f} m^2   (compare with a tape-measure value!)")
    cfg = {"homography": np.asarray(H).tolist(), "zone_px": [list(map(float, p)) for p in poly_px],
           "obstacles_px": [], "area_m2": round(area, 3), "source_size": [frame.shape[1], frame.shape[0]]}
    common.save_config(a.out, cfg)
    vis = frame.copy()
    cv2.polylines(vis, [np.int32(poly_px)], True, (0, 255, 0), 2)
    cv2.putText(vis, f"{area:.2f} m2", tuple(np.int32(poly_px[0])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imwrite("calibration_preview.jpg", vis)
    print(f"Saved {a.out} and calibration_preview.jpg")


if __name__ == "__main__":
    main()
