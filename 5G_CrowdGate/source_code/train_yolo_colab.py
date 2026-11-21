"""Fine-tune YOLO on CrowdHuman (YOLO-format) in Google Colab (Runtime -> GPU).

Colab cells:
  !pip install -q ultralytics
  # upload / download the CrowdHuman YOLO-format dataset (e.g. from Mendeley Data) and unzip to /content/crowdhuman
  # expected: /content/crowdhuman/images/{train,val}  and  /content/crowdhuman/labels/{train,val}
  !python train_yolo_colab.py --data-root /content/crowdhuman --model yolo11s.pt --epochs 50

Result: runs/detect/crowdgate/weights/best.pt  -> copy to the MEC server.
Later, fine-tune again on ~200-500 labelled Sparsh frames (--data-root pointing at that set,
--model best.pt, fewer epochs) so the model learns your camera's angle and lighting.
"""
import argparse
import pathlib

ap = argparse.ArgumentParser()
ap.add_argument("--data-root", required=True)
ap.add_argument("--model", default="yolo11s.pt")
ap.add_argument("--epochs", type=int, default=50)
ap.add_argument("--imgsz", type=int, default=960, help="bigger = better for small/distant people")
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--names", default="person", help="comma-separated class names matching your label files")
ap.add_argument("--fraction", type=float, default=1.0, help="use only this fraction of train images (0.2 = 20%%) for a quick run")
ap.add_argument("--project", default="runs/detect", help="save folder; on Colab use /content/drive/MyDrive/crowdgate so a disconnect doesn't lose it")
ap.add_argument("--resume", action="store_true", help="continue an interrupted run (--model must be <project>/crowdgate/weights/last.pt)")
a = ap.parse_args()

root = pathlib.Path(a.data_root).resolve()
names = a.names.split(",")
yaml = root / "crowdgate.yaml"
yaml.write_text(f"path: {root}\ntrain: images/train\nval: images/val\nnames:\n" +
                "".join(f"  {i}: {n}\n" for i, n in enumerate(names)))

from ultralytics import YOLO
model = YOLO(a.model)
if a.resume:
    model.train(resume=True)
else:
    model.train(data=str(yaml), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, name="crowdgate",
                project=a.project, fraction=a.fraction, patience=15, mosaic=1.0,
                max_det=1000, exist_ok=True)    # max_det: allow up to 1000 detections per image
metrics = model.val()
print("mAP50:", metrics.box.map50, "mAP50-95:", metrics.box.map)
