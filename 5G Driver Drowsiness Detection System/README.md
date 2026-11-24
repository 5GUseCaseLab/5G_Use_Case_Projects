# Driver Drowsiness Detection with Edge AI

**Arduino UNO Q** watches the driver's eyes and runs the ML model on the board.
A **laptop server** receives its verdicts, hosts the live dashboard and owns the alert.
A **NodeMCU car** slows down and beeps until the driver confirms they are awake.

> EE‑4032 · 5G Usecase Laboratory · IIT Guwahati
> Ayush Dixit (230102110) · Aaditya Pratap Shahi (230102124) · Guide: Prof. Salil Kashyap

![System concept](docs/images/concept.png)

---

## Contents

- [How it works](#how-it-works)
- [Architecture](#architecture)
- [How the devices communicate](#how-the-devices-communicate)
- [Detection algorithm](#detection-algorithm)
- [Hardware and wiring](#hardware-and-wiring)
- [Repository layout](#repository-layout)
- [Setup and run](#setup-and-run)
- [API reference](#api-reference)
- [Latency](#latency)
- [Troubleshooting](#troubleshooting)
- [References](#references)

---

## How it works

1. **Detect:** a USB webcam on the Arduino UNO Q feeds a MediaPipe face model. The board computes the Eye Aspect Ratio (EAR) and yawning, and flags drowsiness when any of these happens:
   - the eyes stay closed for ≥ 1.5 s;
   - the eyes are closed for ≥ 30 % of the last minute (PERCLOS);
   - the driver yawns 3 times within 60 s.
2. **Report:** the UNO Q sends a small JSON verdict to the laptop over Wi‑Fi (HTTP POST) every 0.3 s, and immediately when drowsiness starts.
3. **Decide:** the laptop server raises **one** alert and shows it on the dashboard. The dashboard calls the driver by name ("Ayush, please wake up!") and plays 3 beeps.
4. **Act:** the NodeMCU car polls the server every 150 ms. During an alert it slows from PWM 255 to 90 and sounds its buzzer.
5. **Confirm:** nothing recovers automatically. Only **Reset** on the dashboard restores full speed, followed by a 3 s cooldown.

| Normal | Alert |
|---|---|
| ![Dashboard, normal](docs/images/dashboard_normal.png) | ![Dashboard, alert](docs/images/dashboard_alert.png) |

---

## Architecture

```
 ┌──────────────┐  USB (UVC)  ┌──────────────────────────────┐
 │  USB webcam  │────────────▶│  Arduino UNO Q               │
 └──────────────┘             │  MPU: Debian Linux, Python   │
                              │       OpenCV + MediaPipe     │
                              │  MCU: STM32U585 LED matrix   │◀─ internal UART (RPC Bridge)
                              └──────────────┬───────────────┘
                                             │ Wi‑Fi · HTTP POST · JSON
                                             │ every 0.3 s + instantly on DROWSY
                                             ▼
 ┌──────────────┐  HTTP GET 400 ms  ┌──────────────────────────────┐
 │  Dashboard   │◀─────────────────▶│  Laptop server (Flask :5000) │
 │  (browser)   │  MJPEG · POST     │  single source of truth      │
 └──────────────┘  Reset / Drive    └──────────────┬───────────────┘
                                                   ▲ Wi‑Fi · HTTP POST /api/car/sync
                                                   │ every 150 ms (car polls, reply = orders)
                                    ┌──────────────┴───────────────┐
                                    │  NodeMCU 1.0 (ESP8266)       │
                                    │  GPIO/PWM → L298N → DC motor │
                                    │  GPIO → active buzzer        │
                                    └──────────────────────────────┘
```

**Design rule: the edge detects, the server decides, the car obeys.**
- The video never leaves the UNO Q. Only about 170 bytes of JSON per message go over the network, plus an optional low-rate preview JPEG.
- Only the server can raise or clear an alert, so no two devices can disagree about the state.
- Every board is an HTTP **client**, so no board runs a server or needs an open port, and the whole system works on a phone hotspot.

---

## How the devices communicate

| Link | Physical layer | Protocol | Who starts · how often | Carries |
|---|---|---|---|---|
| Webcam → UNO Q | USB 2.0 | UVC, read with OpenCV (V4L2) | UNO Q, every frame | 640 × 480 frames |
| UNO Q MPU ↔ MCU | On‑board UART, 115200 baud | MessagePack‑RPC (Arduino Bridge) | Python, on state change | State for the LED matrix |
| **UNO Q → laptop** | Wi‑Fi 2.4/5 GHz, TCP | **HTTP POST, JSON** (+ JPEG preview) | UNO Q: 0.3 s, instantly on DROWSY | Verdict, EAR, PERCLOS, yawns, fps |
| **NodeMCU ↔ laptop** | Wi‑Fi 2.4 GHz, TCP | **HTTP POST `/api/car/sync`, JSON both ways** | NodeMCU: every 150 ms | Telemetry up; alert, drive, speeds down |
| Browser ↔ laptop | Wi‑Fi or localhost | HTTP GET JSON, MJPEG, POST | Browser: every 400 ms | Status, event log, live video |
| NodeMCU → L298N / buzzer | Jumper wires (3.3 V logic) | GPIO, 1 kHz PWM | NodeMCU: 20 ms control tick | Speed, direction, beep |

### Why Wi‑Fi + HTTP and not USB between the UNO Q and the laptop

The UNO Q could also reach the laptop over its USB‑C cable, for example with a USB serial or USB network link. We chose **HTTP over Wi‑Fi** for four reasons:
- **Wireless:** the detector can sit in front of the driver, away from the laptop, just as it would in a real car.
- **Same pattern everywhere:** the UNO Q, the NodeMCU and the browser all use the same HTTP + JSON style, which is easy to debug with a browser or `curl`.
- **Fast enough:** a POST on a local network takes about 10–50 ms, which is small next to the deliberate 1.5 s detection window.
- **5G‑ready:** moving the server to a 5G MEC (edge computing) node only means changing one URL on each board.

USB is used only to **power** the boards, **flash** them and read **debug logs** (NodeMCU serial monitor at 115200 baud).

### Message examples

**UNO Q → laptop:** `POST /api/detector/status`
```json
{"state": "DROWSY", "drowsy": true, "reason": "eyes closed for 1.5 s",
 "ear": 0.112, "ear_threshold": 0.216, "closed_for": 1.5,
 "perclos": 0.21, "yawns": 0, "fps": 9.8}
```
Reply: `{"ok": true, "alert_active": true}`

**NodeMCU → laptop:** `POST /api/car/sync`
```json
{"state": "NORMAL", "drive": "forward", "speed": 255, "link_ok": true, "rssi": -60, "uptime_s": 312}
```
Reply (the car's orders):
```json
{"alert_active": true, "alert_id": 3, "drive": "forward", "speed_normal": 255, "speed_slow": 90}
```

---

## Detection algorithm

Pipeline on the UNO Q:

```
frame (640×480) → MediaPipe Face Landmarker (478 points + 52 blendshapes)
               → EAR, eyeBlink, jawOpen → 5 s personal calibration → 3 triggers → JSON verdict
```

**Eye Aspect Ratio** (Soukupová & Čech, 2016). Six landmarks are taken around each eye:

```
EAR = (|p2 − p6| + |p3 − p5|) / (2 · |p1 − p4|)
```

| | Value |
|---|---|
| Landmark indices | right eye `33, 160, 158, 133, 153, 144`; left eye `362, 385, 387, 263, 373, 380` |
| Typical EAR | ≈ 0.30 eyes open, ≈ 0.10 eyes closed |
| Calibration | median EAR of clearly‑open frames over 5 s × **0.72**, clamped to 0.14–0.30 (0.21 until calibrated) |
| A frame is "closed" when | `EAR < threshold` **or** MediaPipe `eyeBlink ≥ 0.60` |

**Triggers** (any one raises the alert):

| Trigger | Rule | Notes |
|---|---|---|
| Microsleep | eyes closed continuously ≥ **1.5 s** | A blink (0.1–0.4 s) never triggers it. An "open" flicker < 0.15 s doesn't reset the timer. |
| PERCLOS | eyes closed ≥ **30 %** of the last 60 s | Needs ≥ 20 s of data first. |
| Yawning | `jawOpen ≥ 0.55` held 1 s = one yawn; **3 yawns in 60 s** | PERCLOS and yawn histories reset after they fire. |

States reported: `CALIBRATING · AWAKE · EYES_CLOSED · YAWNING · DROWSY · NO_FACE`.

**Fallback:** if MediaPipe can't start on the board, the detector switches automatically to OpenCV Haar cascades (Viola–Jones face + eye detectors). In that mode, a visible face with no eyes found counts as closed.

---

## Hardware and wiring

**Parts**
- Arduino UNO Q (Qualcomm QRB2210 + STM32U585) and a USB webcam
- Laptop (Windows, macOS or Linux) with Python 3.10+
- NodeMCU 1.0 (ESP‑12E / ESP8266)
- L298N motor driver, DC motor, 3‑pin active buzzer module, battery pack

**NodeMCU pinout**

| NodeMCU | Connects to | Role |
|---|---|---|
| D1 (GPIO5) | L298N **ENA** (remove the ENA jumper) | Speed: PWM 0–255 at 1 kHz |
| D2 (GPIO4) | L298N **IN1** | Direction |
| D5 (GPIO14) | L298N **IN2** | Direction |
| D6 (GPIO12) | Buzzer **I/O** | `LOW` = beep (active‑low) |
| 3V3 | Buzzer **VCC** | Buzzer supply |
| GND | L298N **GND** + buzzer **GND** | **Common ground** (required) |
| USB | Laptop or power bank | Powers the NodeMCU, serial log |
| — | Battery **+/−** → L298N **+12V / GND** | Motor supply |
| — | L298N **OUT1/OUT2** → DC motor | Motor terminals |

Direction: `IN1=HIGH, IN2=LOW` forward · `LOW, HIGH` backward · `LOW, LOW` stop.

---

## Repository layout

```
.
├── laptop_server/
│   └── laptop_app.py              # Flask server + dashboard (single file)
├── unoq_detector/
│   └── unoq_detector.py           # runs on the UNO Q: MediaPipe/Haar + HTTP reporting
├── nodemcu_car/
│   └── nodemcu_car_follow/
│       └── nodemcu_car_follow.ino # car firmware (follows the server)
├── tools/
│   └── measure_latency.py         # measures alert → car latency over USB serial
└── docs/
    └── images/                    # screenshots used in this README
```

---

## Setup and run

All three devices must be on the **same Wi‑Fi network** (a phone hotspot works). The NodeMCU only supports **2.4 GHz**.

### 1. Laptop server

```bash
cd laptop_server
pip install flask opencv-python numpy
python laptop_app.py
```

The server prints its address, for example `http://172.20.10.2:5000`. Open `http://localhost:5000` for the dashboard, type the driver's name, and click once on the page so the browser allows sound.

> **Windows:** allow Python through the firewall on **private** networks, or the boards can't reach port 5000.

### 2. Arduino UNO Q (detector)

On the UNO Q's Linux side (App Lab terminal or `ssh arduino@<board>.local`):

```bash
sudo apt update
sudo apt install -y python3-opencv opencv-data python3-venv
python3 -m venv --system-site-packages ~/mpenv
~/mpenv/bin/pip install mediapipe        # optional; the Haar fallback works without it

# copy unoq_detector.py to the board, then:
~/mpenv/bin/python unoq_detector.py --check-mediapipe
~/mpenv/bin/python unoq_detector.py --server http://<laptop-IP>:5000 --camera 0
```

Look at the camera with your eyes open for the first 5 s while it calibrates. Type `r` + Enter to recalibrate. The dashboard's **Detector** badge turns **ONLINE**.

### 3. NodeMCU car

1. Arduino IDE → **File › Preferences › Additional boards manager URLs**: `http://arduino.esp8266.com/stable/package_esp8266com_index.json`
2. **Boards Manager** → install **esp8266 by ESP8266 Community**. **Library Manager** → install **ArduinoJson** (v7).
3. Open `nodemcu_car_follow.ino` and set:
   ```cpp
   const char* WIFI_SSID     = "YourHotspot";
   const char* WIFI_PASSWORD = "YourPassword";
   const char* SERVER_URL    = "http://<laptop-IP>:5000";
   ```
4. **Tools › Board** → *NodeMCU 1.0 (ESP‑12E Module)*, choose the port, and **Upload**.
5. Serial Monitor at **115200**: you should see `[LINK] laptop link OK`, and the dashboard's **Car** badge turns **ONLINE**.

### 4. Demo

| Do | Expect |
|---|---|
| Press **Forward** on the dashboard | Car at PWM 255 |
| Blink normally | Stays **AWAKE** |
| Close eyes for 2 s | **DROWSY**: car slows to 90, buzzer beeps, voice alert |
| Open eyes, touch nothing | Alert stays on |
| Click **Reset** (or press `R`) | Back to 255, buzzer off, 3 s cooldown |
| Stop the server | Car stops within 3 s (fail‑safe) |

No camera handy? **Simulate drowsiness** on the dashboard exercises the same server → car path.

---

## API reference

| Method | Endpoint | Called by | Purpose |
|---|---|---|---|
| POST | `/api/detector/status` | UNO Q, every 0.3 s | Verdict JSON; `drowsy: true` raises the alert |
| POST | `/api/detector/frame` | UNO Q, every 0.5 s | JPEG preview for the dashboard |
| POST | `/api/car/sync` | NodeMCU, every 150 ms | Car telemetry in, orders out |
| GET | `/api/status` | Dashboard, every 400 ms | Detector, car, alert and event log in one JSON |
| GET | `/video.mjpg` | Dashboard | Live MJPEG stream |
| POST | `/api/reset` | Dashboard | Clear the alert (the only way back to full speed) |
| POST | `/api/drive` | Dashboard | `forward`, `backward`, `left`, `right`, `stop` |
| POST | `/api/recalibrate` | Dashboard | Ask the UNO Q to redo the 5 s calibration |
| POST | `/api/test/drowsy` | Dashboard | Simulate an alert |

A device that stays silent for 3 s is shown as **OFFLINE**.

**Alert state machine (on the server):** `NORMAL → (drowsy) → ALERT → (Reset) → COOLDOWN 3 s → NORMAL`.
Independently, the car stops if it gets no reply for 3 s.

---

## Latency

| Stage | Time |
|---|---|
| Eyes closed until the rule fires | 1.5 s (by design) |
| Frame analysis on the UNO Q | 50–150 ms |
| UNO Q → laptop POST | 10–50 ms |
| Wait for the car's next poll | 0–150 ms |
| Wi‑Fi round trip of that poll | 20–100 ms |
| Motor control tick | ≤ 20 ms |
| **Car starts slowing after the rule fires** | **≈ 0.08–0.47 s** (estimate) |
| Ramp from PWM 255 to 90 | ≈ 0.84 s |

To measure the laptop → car part yourself, plug the NodeMCU into the laptop over USB, close the Serial Monitor, and run:

```bash
pip install pyserial
python tools/measure_latency.py --port COM6      # or /dev/ttyUSB0
```

It triggers 20 test alerts, times the car's `[ALERT]` / `[RESET]` serial lines against the same laptop clock, and saves `latency_results.csv`.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Car or detector stays OFFLINE | Same Wi‑Fi? Correct laptop IP? Windows firewall allows Python on private networks? |
| NodeMCU won't connect | It only supports 2.4 GHz; turn on "Maximise compatibility" on an iPhone hotspot. |
| `Illegal instruction` when MediaPipe starts on the UNO Q | That MediaPipe build needs CPU features the Cortex‑A53 lacks. Try `pip install "mediapipe<1"`; otherwise the detector falls back to Haar automatically. |
| `No module named mediapipe` | Run the script with `~/mpenv/bin/python`, not `python3`. |
| Motor hums but doesn't turn | Battery too weak, or PWM below about 70 (the firmware skips that range). |
| No voice on the dashboard | Click once anywhere on the page; browsers block sound until you interact. |
| Upload fails ("could not open port") | Close the Serial Monitor; install the CP2102/CH340 USB driver. |

---

## References

- T. Soukupová, J. Čech, *Real‑Time Eye Blink Detection using Facial Landmarks*, CVWW 2016.
- D. F. Dinges, R. Grace, *PERCLOS: A Valid Psychophysiological Measure of Alertness*, FHWA‑MCRT‑98‑006, 1998.
- Google, *MediaPipe Face Landmarker*.
- P. Viola, M. Jones, *Rapid Object Detection using a Boosted Cascade of Simple Features*, CVPR 2001.
- Arduino, *UNO Q documentation* and *Arduino RouterBridge* library.

## Acknowledgements

We thank Prof. Salil Kashyap and teaching assistants Aditya Gupta, Abhay Saran and Asif Hoque for their guidance.
