"""Run on the LAPTOP (inside the venv, where ultralytics is installed):
       cd C:\\vpu\\laptop
       python ..\\tools\\export_onnx.py
   Creates tools/yolov8n_320.onnx  (~12 MB) -> copy it to the board's python/models/ folder."""
import shutil
from pathlib import Path
from ultralytics import YOLO

out_dir = Path(__file__).resolve().parent
model = YOLO("yolov8n.pt")                       # downloads the weights on first run
path = model.export(format="onnx", imgsz=320, opset=12, simplify=False, dynamic=False, half=False)
dst = out_dir / "yolov8n_320.onnx"
shutil.copy(path, dst)
print("\nDONE ->", dst, f"({dst.stat().st_size/1e6:.1f} MB)")
