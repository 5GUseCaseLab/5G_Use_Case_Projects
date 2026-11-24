"""CHECK your calibration: click two points on the FLOOR in a camera frame; prints the real distance in metres.
Lay a tape / 1 m stick somewhere in the zone (NOT used during calibration), click its two ends, compare.
  python measure_distance.py --config config.json --source frame.jpg
Click 2 points -> distance printed; press q to quit. Only valid for points touching the floor.
"""
import argparse

import cv2
import numpy as np

import common
from calibrate_area import grab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--source", required=True)
    a = ap.parse_args()
    H = np.array(common.load_config(a.config)["homography"])
    frame = grab(a.source)
    pts = []

    def cb(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
            if len(pts) == 2:
                d = common.distance_m(H, *pts)
                cv2.line(frame, pts[0], pts[1], (0, 255, 255), 2)
                cv2.putText(frame, f"{d:.3f} m", pts[1], cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                print(f"distance = {d:.3f} m")
                pts.clear()

    cv2.namedWindow("measure", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("measure", cb)
    while True:
        cv2.imshow("measure", frame)
        if cv2.waitKey(20) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    main()
