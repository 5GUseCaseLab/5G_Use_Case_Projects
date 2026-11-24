"""Quick check: can the camera see the ArUco markers well enough?
Grabs ONE frame from the Sparsh camera (or reads an image file), detects markers, prints ids and sizes in pixels,
and saves test.jpg (raw frame) and test_annotated.jpg (markers outlined).

Run in a terminal / Command Prompt on the laptop that is connected to the camera:
  python check_marker.py --password YOURPASSWORD
  python check_marker.py --password YOURPASSWORD --stream 2      (sub stream)
  python check_marker.py --image test.jpg                         (re-check a saved picture)
Needs:  pip install opencv-contrib-python numpy
"""
import argparse
import os

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")   # TCP is more reliable than UDP

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", default="192.168.128.10")
    ap.add_argument("--user", default="admin")
    ap.add_argument("--password", default="admin")
    ap.add_argument("--stream", type=int, default=1, help="1 = main stream (4K), 2 = sub stream")
    ap.add_argument("--image", default=None, help="use this image file instead of the camera")
    ap.add_argument("--dict", default="DICT_4X4_50")
    a = ap.parse_args()

    if a.image:
        frame = cv2.imread(a.image)
        if frame is None:
            raise SystemExit(f"Cannot read {a.image}")
    else:
        url = f"rtsp://{a.user}:{a.password}@{a.ip}:554/avstream/channel=1/stream={a.stream}.sdp"
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            raise SystemExit("Could not open the stream. Check IP, password, stream number and the cable.")
        for _ in range(15):          # let the picture settle
            cap.read()
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise SystemExit("Opened the stream but could not read a frame.")
        cv2.imwrite("test.jpg", frame)

    h, w = frame.shape[:2]
    print(f"Frame size: {w} x {h} pixels")
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, a.dict)),
                                  cv2.aruco.DetectorParameters())
    corners, ids, _ = det.detectMarkers(frame)
    if ids is None:
        print("NO markers detected. Check: whole marker + white border visible, in focus, not washed out,")
        print("not too small or too far. Open test.jpg and look at it.")
    else:
        print(f"Detected {len(ids)} marker(s):")
        for c, i in zip(corners, ids.flatten()):
            p = c[0]
            sides = [float(np.linalg.norm(p[k] - p[(k + 1) % 4])) for k in range(4)]
            s = min(sides)
            verdict = "good" if s >= 60 else "OK" if s >= 40 else "too small - move closer or print bigger"
            print(f"  id {int(i)}: shortest side {s:.0f} px  -> {verdict}")
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
    cv2.imwrite("test_annotated.jpg", frame)
    print("Saved test_annotated.jpg")


if __name__ == "__main__":
    main()
