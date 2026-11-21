"""Make a printable PDF of ArUco markers at an EXACT physical size (one marker per page, white margin around).

  python make_markers.py --ids 0 1 2 3 --size-cm 25 --paper A3 --out markers.pdf
Print at 100% / "actual size" (NOT 'fit to page'), then MEASURE the printed black square with a ruler
and use that number (in metres) as --marker-m in calibrate_area.py.
"""
import argparse

import cv2
import numpy as np
from PIL import Image, ImageDraw

PAPER_MM = {"A4": (210, 297), "A3": (297, 420), "A2": (420, 594)}
DPI = 300


def mm2px(mm):
    return int(round(mm / 25.4 * DPI))


def make_page(marker_id, size_cm, paper="A4", dict_name="DICT_4X4_50"):
    w_mm, h_mm = PAPER_MM[paper]
    if size_cm * 10 > min(w_mm, h_mm) - 20:
        raise SystemExit(f"{size_cm} cm marker does not fit {paper} with a white margin; use a bigger paper or smaller marker")
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    side = mm2px(size_cm * 10)
    marker = cv2.aruco.generateImageMarker(d, marker_id, side)
    page = np.full((mm2px(h_mm), mm2px(w_mm)), 255, np.uint8)
    y0, x0 = (page.shape[0] - side) // 2, (page.shape[1] - side) // 2
    page[y0:y0 + side, x0:x0 + side] = marker
    img = Image.fromarray(page).convert("RGB")
    ImageDraw.Draw(img).text((mm2px(10), page.shape[0] - mm2px(10)),
                             f"ArUco {dict_name}  id={marker_id}  nominal side={size_cm} cm  (keep this text away from the marker)",
                             fill=(0, 0, 0))
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--size-cm", type=float, default=20.0)
    ap.add_argument("--paper", choices=list(PAPER_MM), default="A4")
    ap.add_argument("--dict", default="DICT_4X4_50")
    ap.add_argument("--out", default="markers.pdf")
    a = ap.parse_args()
    pages = [make_page(i, a.size_cm, a.paper, a.dict) for i in a.ids]
    pages[0].save(a.out, save_all=True, append_images=pages[1:], resolution=DPI)
    print(f"Saved {a.out}: ids {a.ids}, {a.size_cm} cm markers on {a.paper}. Print at 100% and measure the black square!")


if __name__ == "__main__":
    main()
