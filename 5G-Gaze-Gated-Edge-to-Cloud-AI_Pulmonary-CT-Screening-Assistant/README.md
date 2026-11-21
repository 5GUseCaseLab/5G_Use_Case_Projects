# Gaze-Gated Edge-to-Cloud AI for Pulmonary CT Screening

A wearable, 5G-enabled diagnostic assistant that reduces **automation bias** in radiology. The AI analyzes each CT slice silently on edge hardware and reveals findings only when the clinician's gaze misses a suspicious region. Flagged regions are sent over 5G to a cloud ensemble for verification and a clinical description.

## How It Works

1. **Detect (Edge):** A lightweight CNN on an Arduino UNO Q locates candidate nodules in ~116 ms.
2. **Watch (Gaze):** A dual-camera eye tracker checks whether the clinician fixated on each candidate.
3. **Reveal (UI):** Bounding boxes appear only for regions that were missed.
4. **Verify and Describe (Cloud):** A flagged region is sent over 5G to the cloud, where a classifier filters false positives and a vision-language model writes a clinical narrative.

## Repository Structure

```text
project_root/
├── Inference-Pipeline/    # Edge AI, cloud ensemble, web dashboard
│   ├── index.html
│   ├── viewer.js
│   ├── unoq_service.py
│   ├── tier1_colab.ipynb
│   └── tier0_final.onnx
├── eye_tracker/        # Dual-camera gaze tracking
│   ├── pupil.py
│   ├── heatmap.py
│   └── gaze_model.joblib
└── future_work/        # Integrated wearable application
```

## Modules

### Inference Pipeline

| Tier | Component | Role | Details |
|------|-----------|------|---------|
| 0 | MobileNetV3-Small (ONNX) | Edge detection | CAM from a single forward pass; ~116 ms (8.5 FPS); 98.5% recall; served by FastAPI on the UNO Q |
| 1 | ResNet-18 (Gatekeeper) | False-positive filter | Trained on LIDC-IDRI crops; 99.9% validation AUC |
| 1 | Qwen2-VL 2B (Describer) | Clinical narrative | Template matching marks the target on the full scan to guide the prompt |

Tier 1 runs on a Colab T4 GPU behind a Cloudflare Tunnel, simulating a 5G multi-access edge computing (MEC) topology.

The web dashboard is a dark-mode PACS-style viewer. Gaze is currently simulated with the mouse, and AI boxes stay hidden until queried.

### eye_tracker

*Work in progress.*

- **Cameras:** _TBD_
- **Pupil detection:** _TBD_ (`pupil.py`)
- **Gaze mapping:** _TBD_ (`heatmap.py`, `gaze_model.joblib`)

### future_work

*Planned:* feed live gaze coordinates from `eye_tracker` into the dashboard to complete the gaze-gated loop.

## Quick Start

**1. Start the cloud server (Tier 1)**
- Open `tier1_colab.ipynb` in Google Colab with a T4 runtime.
- Upload `tier1_best.pth`, then run the cell to launch the server and tunnel.
- Copy the printed Cloudflare URL.

**2. Start the edge server (Tier 0, on the UNO Q)**
```bash
export TIER1_URL=<cloudflare-url>
python -m uvicorn unoq_service:app --host 0.0.0.0 --port 8000
```

**3. Open the dashboard**
```bash
python -m http.server 8080
```
Visit `http://<HOST-IP>:8080` in Chrome, upload a CT slice, and click a bounding box to query the cloud ensemble.