"""Convert OFFICIAL CrowdHuman (.odgt annotations, human-labelled) to YOLO format.

Official layout you download:  annotation_train.odgt, annotation_val.odgt, Images/*.jpg
Each .odgt line is one JSON: {"ID": "273271,c9db000d5146c15", "gtboxes": [{"tag": "person",
  "fbox": [x,y,w,h] (full body, incl. hidden parts), "vbox": [x,y,w,h] (visible part only),
  "hbox": [x,y,w,h] (head), "extra": {"ignore": 0/1}}, ...]}

Run (once per split):
  python convert_crowdhuman.py --odgt annotation_train.odgt --images Images --out /content/crowdhuman --split train
  python convert_crowdhuman.py --odgt annotation_val.odgt   --images Images --out /content/crowdhuman --split val
Result: out/images/<split>/*.jpg (symlinks) and out/labels/<split>/*.txt  ("0 xc yc w h", normalised).

--box vbox (default) = visible body box. It stays accurate when people overlap, which suits crowd counting.
--box fbox = full-body box (includes hidden parts, so boxes overlap more -> harder for NMS).
"""
import argparse
import json
import os
from pathlib import Path

import cv2


def convert_line(rec, img_w, img_h, box="vbox", min_px=4):
    """Return YOLO label lines for one odgt record. Skips 'mask' regions and ignore=1 boxes."""
    out = []
    for g in rec.get("gtboxes", []):
        if g.get("tag") != "person" or g.get("extra", {}).get("ignore", 0) == 1:
            continue
        x, y, w, h = g[box]
        x1, y1, x2, y2 = max(x, 0), max(y, 0), min(x + w, img_w), min(y + h, img_h)   # clip to image
        if x2 - x1 < min_px or y2 - y1 < min_px:
            continue
        out.append(f"0 {(x1 + x2) / 2 / img_w:.6f} {(y1 + y2) / 2 / img_h:.6f} "
                   f"{(x2 - x1) / img_w:.6f} {(y2 - y1) / img_h:.6f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--odgt", required=True)
    ap.add_argument("--images", required=True, help="folder with the .jpg images")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", required=True, choices=["train", "val"])
    ap.add_argument("--box", default="vbox", choices=["vbox", "fbox"])
    ap.add_argument("--limit", type=int, default=0, help="only convert the first N images (0 = all)")
    a = ap.parse_args()

    img_out = Path(a.out, "images", a.split)
    lbl_out = Path(a.out, "labels", a.split)
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    n = persons = 0
    for line in open(a.odgt):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        src = Path(a.images, rec["ID"] + ".jpg")
        if not src.exists():
            continue
        h, w = cv2.imread(str(src)).shape[:2]
        rows = convert_line(rec, w, h, a.box)
        dst = img_out / src.name
        if not dst.exists():
            os.symlink(src.resolve(), dst)
        (lbl_out / (rec["ID"] + ".txt")).write_text("\n".join(rows) + ("\n" if rows else ""))
        n, persons = n + 1, persons + len(rows)
        if a.limit and n >= a.limit:
            break
    print(f"{a.split}: {n} images, {persons} person boxes ({persons / max(n, 1):.1f} per image)")


if __name__ == "__main__":
    main()
