# Contactless rPPG Tele-Triage

Real-time contactless heart-rate estimation and START emergency triage using VitalLens POS and OpenCV, running on an **Arduino UNO Q** over a smartphone IP Webcam stream.

Developed under **EE-4032** at the **5G Usecase Laboratory, Department of Electronics and Electrical Engineering, IIT Guwahati** by **Vasudev Krishna (230102113)** and **Uditya Shekhawat (230102119)** under the guidance of **Prof. Salil Kashyap** and Teaching Assistants **Aditya Gupta**, **Abhay Saran**, and **Asif Hoque**. The course poster is kept in [docs/rPPG_5G_MEC_TeleTriage_Poster_A1-1.pdf](docs/rPPG_5G_MEC_TeleTriage_Poster_A1-1.pdf) for reference, while this README documents the actual implemented system and scripts.

---

## System architecture

```mermaid
flowchart LR
    CAM["Smartphone IP Webcam\n30 FPS MJPEG\nLocked AE, AF, WB"] -->|"HTTP /video via\nLaptop Wi-Fi Hotspot"| ARD["Arduino UNO Q\nPython + OpenCV + VitalLens POS\n30 Hz Resampling + 15 s Window"]
    ARD -->|"Display and Verification\nBPM, Confidence, START Tag"| PC["Base PC / Laptop\nHotspot + Display"]
```

### What is implemented

- **Network and compute topology.** Both the smartphone IP Webcam and the **Arduino UNO Q** (Linux-capable board) connect to the laptop's Wi-Fi hotspot. The Arduino UNO Q runs `vital_lens_ipcam.py` (OpenCV + ONNX face detection + VitalLens POS + HUD rendering), while the laptop serves as the hotspot and display/verification interface.
- **rPPG core (VitalLens POS, not CHROM).** Uses `VitalLens(method="pos")` (`POSRPPGMethod`, Plane-Orthogonal-to-Skin by Wang et al., 2017) with `vitallens_core.Session(..., "Global")`, **not CHROM**.
- **Real-time pipeline.** Separates HTTP MJPEG capture (`CAP_PROP_BUFFERSIZE = 1` with `time.monotonic()` timestamps), downscaled 3 Hz ONNX face detection with deadband plus EMA box smoothing, uniform 30 Hz linear timestamp resampling, and a configurable 15 s rolling estimation window (`8 s` minimum warmup).
- **Confidence scores and START triage.** Displays heart rate (BPM), **confidence score**, PPG waveform sparkline, and **START triage tags** (`RED`, `YELLOW`, `GREEN`, `CALIBRATING`, `NO_FACE`) in the OpenCV HUD and console output. No external LEDs, buzzers, or actuators are used. Note that classic `POSRPPGMethod` passes unit per-sample weights to `vitallens_core`, so displayed confidence stays `1.00` during valid POS solves rather than reflecting a neural signal-quality score.

---

## START triage rules

| Heart rate (BPM) | Triage tag | HUD / console status |
| :--- | :--- | :--- |
| `< 40` or `> 120` | `RED` | `RED - Immediate Attention` |
| `40 to 59` or `101 to 120` | `YELLOW` | `YELLOW - Abnormal / High` |
| `60 to 100` | `GREEN` | `GREEN - Normal / Stable` |
| No face active | `NO_FACE` | `NO FACE DETECTED` |
| Face active, buffer `< 8 s` | `CALIBRATING` | `CALIBRATING` |

---

## Repository layout

```text
Contactless-rPPG-Tele-Triage/
├── vital_lens_ipcam.py                  # Real-time IP Webcam & replay rPPG + OpenCV HUD
├── vital_lens_local.py                  # Offline batch VitalLens POS analyzer for local videos
├── requirements.txt                     # Python dependencies
├── .gitignore
├── README.md
└── docs/
    └── rPPG_5G_MEC_TeleTriage_Poster_A1-1.pdf
```

---

## Setup

Tested on Python 3.12.10 (Windows 11 verification and Linux-compatible fallback for Arduino UNO Q).

```bash
python3 -m venv .venv
source .venv/bin/activate   # On Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## Usage

### 1. Live IP Webcam stream (via laptop hotspot)

Configure Android IP Webcam before starting the server:
- **Resolution and framerate.** `1280x720` at `30` FPS limit (match orientation so your face appears upright).
- **Stream quality.** `85` to `90` JPEG quality, **Audio mode** disabled.
- **Optics locks.** **Focus mode** set to `Manually set` or `Auto, manually triggered`; **Antibanding** `50 Hz`; **White balance** fixed to room lighting; **Night mode** and **Text overlay** off.
- **Exposure.** Switch browser **Sensor control** to `Manual sensor control` with exposure at `33 ms` or less (or `Automatic exposure control` + `50 Hz` antibanding if LED tube flicker bands appear), then set browser **Video renderer** to `No video`.

Run the live pipeline on the Arduino UNO Q (or PC for local testing):

```bash
python3 vital_lens_ipcam.py --source http://192.168.137.35:8080
```

Interactive HUD controls: press `R` to reset the 15 s buffer, or `Q` / `Esc` to quit.

### 2. Local video replay and offline batch

Replay a local video through the 15 s rolling pipeline:

```bash
python3 vital_lens_ipcam.py --source path/to/video.mp4 --window-seconds 15 --expected-bpm 70 --fast-replay
```

Run the offline batch analyzer:

```bash
python3 vital_lens_local.py path/to/video.mp4 --expected-bpm 70
```

### Tunable options (`vital_lens_ipcam.py`)

| Flag | Default | Purpose |
| :--- | :--- | :--- |
| `--source` | `http://192.168.137.35:8080` | Stream URL or local video path |
| `--window-seconds` | `15.0` | Rolling POS estimation window |
| `--min-window-seconds` | `8.0` | Minimum window before first estimate |
| `--eval-interval` | `0.5` | Seconds between BPM updates |
| `--target-fps` | `30.0` | Uniform timestamp resampling rate |
| `--expected-bpm` | `None` | Optional reference BPM for live error readout |
| `--no-gui` | `False` | Run headless in console |
| `--fast-replay` | `False` | Process local video without wall-clock sleep |
| `--save-hud-frame` | `None` | Save a rendered HUD PNG snapshot |

---

## Verified benchmarks

- **Local clip `ayush_70.mp4` (`720x1280` at `30.06` FPS, reference `70.0` BPM):**
  - Batch `vital_lens_local.py` measured `70.19` BPM.
  - Streaming `vital_lens_ipcam.py` fast replay (`--window-seconds 15`) measured `69.8` BPM (`0.2` BPM error, `4.2 ms` solve time on PC).

---

## Documentation

- [docs/rPPG_5G_MEC_TeleTriage_Poster_A1-1.pdf](docs/rPPG_5G_MEC_TeleTriage_Poster_A1-1.pdf) is the course poster kept for reference.

---

## References

1. W. Verkruysse, L. O. Svaasand, and J. S. Nelson, "Remote plethysmographic imaging using ambient light," *Opt. Express*, vol. 16, no. 26, pp. 21434-21445, Dec. 2008.
2. W. Wang, A. C. den Brinker, S. Stuijk, and G. de Haan, "Algorithmic principles of remote PPG," *IEEE Trans. Biomed. Eng.*, vol. 64, no. 7, pp. 1479-1491, Jul. 2017.
3. M. Benson, K. L. Koenig, and C. H. Schultz, "Disaster triage: START, then SAVE," *Prehosp. Disaster Med.*, vol. 11, no. 2, pp. 117-124, 1996.
