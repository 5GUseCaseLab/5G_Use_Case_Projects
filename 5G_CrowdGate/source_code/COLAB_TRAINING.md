# Training on Google Colab - step by step

Goal: `best.pt` = YOLO11s fine-tuned on CrowdHuman (human-labelled crowds), then compared with the stock model.

## 0. Get the dataset (on your laptop)
Use the **official CrowdHuman** (human-labelled): https://www.crowdhumandataset.com/ (or a Hugging Face mirror such as `sshao0516/CrowdHuman`).
You need: `annotation_train.odgt`, `annotation_val.odgt` and the image zips (typically `CrowdHuman_train01/02/03.zip`, `CrowdHuman_val.zip`).
Upload them to **Google Drive** in a folder `crowdgate/raw/` (Colab reads Drive quickly; your laptop can then be switched off).
Do NOT use the Mendeley "CrowdHuman YOLO" set: its boxes were produced by a pretrained YOLOv8, not by people.
CrowdHuman licence: research / non-commercial - fine for a course project; cite the paper (Shao et al., 2018).

## 1. Colab setup  (colab.research.google.com -> New notebook)
Runtime -> Change runtime type -> **T4 GPU**. Then run each block as a cell:

```python
!nvidia-smi                      # must show a GPU
from google.colab import drive
drive.mount('/content/drive')
!pip install -q ultralytics
```

Upload `crowdgate.zip` to Colab (left Files panel) and run:
```python
!unzip -q crowdgate.zip && cd crowdgate && ls
```

## 2. Unzip the dataset onto Colab's local disk (faster than Drive)
```python
!mkdir -p /content/raw/Images
!for z in /content/drive/MyDrive/crowdgate/raw/CrowdHuman_*.zip; do unzip -q -o $z -d /content/raw/; done
!cp /content/drive/MyDrive/crowdgate/raw/*.odgt /content/raw/
!find /content/raw -name "*.jpg" | head -3      # note where the images ended up
```
If the images are not in `/content/raw/Images`, use the real folder in the next step.

## 3. Convert to YOLO format
```python
%cd /content/crowdgate
!python convert_crowdhuman.py --odgt /content/raw/annotation_train.odgt --images /content/raw/Images --out /content/crowdhuman --split train
!python convert_crowdhuman.py --odgt /content/raw/annotation_val.odgt   --images /content/raw/Images --out /content/crowdhuman --split val
```
It prints images and boxes per split (expect roughly 20+ people per image). If it says 0 images, the `--images` path is wrong.

## 4. Smoke test (2 minutes) - always do this first
```python
!python train_yolo_colab.py --data-root /content/crowdhuman --model yolo11s.pt --epochs 1 --fraction 0.02 --imgsz 640 --project /content/drive/MyDrive/crowdgate/runs
```
It should run an epoch and print mAP. If it crashes, fix that before the long run.

## 5. Real training
```python
!python train_yolo_colab.py --data-root /content/crowdhuman --model yolo11s.pt --epochs 30 --fraction 0.3 --imgsz 960 --batch 8 --project /content/drive/MyDrive/crowdgate/runs
```
- Free Colab limits GPU time and disconnects after a few hours, so **look at the time of the first epoch** and size the run (`--fraction`, `--epochs`) to fit. Checkpoints are saved to Drive after every epoch.
- Out-of-memory error -> `--batch 4` or `--imgsz 800`.
- If it disconnects: reconnect, redo steps 1-3 (the dataset on local disk is gone), then
  `!python train_yolo_colab.py --data-root /content/crowdhuman --resume --model /content/drive/MyDrive/crowdgate/runs/crowdgate/weights/last.pt`

Result: `/content/drive/MyDrive/crowdgate/runs/crowdgate/weights/best.pt`. Download it from Drive to your laptop.

## 6. Compare with the stock model (on the same CrowdHuman val images)
```python
from ultralytics import YOLO
y = '/content/crowdhuman/crowdgate.yaml'
print('stock  :', YOLO('yolo11s.pt').val(data=y, imgsz=960, classes=[0], max_det=1000).box.map50)
print('trained:', YOLO('/content/drive/MyDrive/crowdgate/runs/crowdgate/weights/best.pt').val(data=y, imgsz=960, max_det=1000).box.map50)
```
Caution: the stock model draws full-body boxes while we trained on visible-body boxes, so its mAP is partly lowered by the box style.

## 7. Use it
`python crowdgate_mec.py --model best.pt --openvino ...` (the OpenVINO export is made once from your `best.pt`).
Later: label 200-500 Sparsh frames and run step 5 again with `--model best.pt --epochs 20` on that small dataset.
