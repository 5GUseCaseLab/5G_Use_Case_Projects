# AI Pan-Tilt

**5G-ready AI camera tracking, automated pan-tilt control and intelligent camera handoff**

Proof of concept from the 5G Wireless Laboratory, Indian Institute of Technology Guwahati.

| Author | Roll no. |
|---|---|
| Bulbule Pranav Chandrakant | 230102019 |
| Chinmay Torvi | 230108014 |

Arduino UNO Q · YOLO · OpenCV · Private 5G + MEC (planned)

The project poster is in [`docs/AI_PanTilt_A1_Poster.pdf`](docs/AI_PanTilt_A1_Poster.pdf).

---

## Contents

1. [Abstract](#abstract)
2. [Motivation and objectives](#motivation-and-objectives)
3. [How it works](#how-it-works)
4. [Repository layout](#repository-layout)
5. [Hardware and wiring](#hardware-and-wiring)
6. [Software setup](#software-setup)
7. [Running the system](#running-the-system)
8. [Web pages](#web-pages)
9. [Tuning and calibration](#tuning-and-calibration)
10. [Test apps for the UNO Q](#test-apps-for-the-uno-q)
11. [Troubleshooting](#troubleshooting)
12. [5G lab deployment (planned)](#5g-lab-deployment-planned)
13. [Results, status and future scope](#results-status-and-future-scope)

---

## Abstract

An AI-enabled pan-tilt camera platform that detects a selected object, estimates its position in the camera frame and generates horizontal and vertical motion commands to keep it near the centre of view.

A second capability performs intelligent handoff between two camera feeds: when the tracked object leaves one camera's field of view and is detected by the other, the live display switches to that camera.

The system is built on computer vision (YOLO and OpenCV), Python control, an Arduino UNO Q and a dual-servo platform. Integration with a private 5G network and a MEC (multi-access edge computing) server is the planned deployment architecture.

## Motivation and objectives

Fixed cameras need manual repositioning and lose a moving subject. This project aims to build a camera system that:

- detects and tracks a selected object automatically,
- keeps the target near the camera's centre,
- controls pan and tilt with servo motors,
- switches between camera feeds to keep the subject in view,
- explores low-latency AI processing using 5G and MEC.

**Objectives**

1. Real-time object detection with YOLO.
2. Calculation of the target's displacement from the frame centre.
3. Conversion of horizontal and vertical errors into servo angle commands.
4. Stable two-axis pan-tilt control.
5. Automatic switching between two camera streams.
6. Integration with a private 5G network and a MEC server as the planned deployment.

## How it works

### Present prototype

```
 LAPTOP 2                                LAPTOP 1 (main processing unit)
 ┌─────────────────────┐   HTTP video    ┌────────────────────────────────────────────┐
 │ Camera B (webcam)   │ ──────────────► │ Camera A (webcam) + Camera B stream         │
 │ stream_webcam.py    │   (MJPEG, wifi) │ switcher.py: YOLO on both feeds             │
 │ no YOLO here        │                 │   ├─ servo branch: x/y error -> pan, tilt   │
 └─────────────────────┘                 │   └─ display branch: camera handoff logic   │
          ▲                              │ hosted web page (port 8000)                 │
          │ screen B page                └───────────────┬────────────────────────────┘
          └──────────────────────────────────────────────┤ USB cable (adb forward,
                                                         │ Socket.IO on port 7000)
                                         ┌───────────────▼────────────────────────────┐
                                         │ ARDUINO UNO Q                               │
                                         │ Linux side: python/main.py (WebUI brick)    │
                                         │   Bridge ──► STM32 side: sketch.ino          │
                                         │ D9 ──► pan servo (MG995)                     │
                                         │ D10 ─► tilt servo (DS3218)                   │
                                         └────────────────────────────────────────────┘
```

**Workflow**

1. Capture video from camera A (laptop 1's webcam) and camera B (laptop 2's webcam).
2. Laptop 2 streams camera B to laptop 1 over HTTP. No detection runs on laptop 2.
3. Laptop 1 runs YOLO on both feeds. Two branches then run in parallel:
   - **Servo control:** take the target's bounding-box centre, work out its offset from the frame centre, map it to pan and tilt angles, and send both angles to the UNO Q over USB.
   - **Camera switching:** check the target's visibility in both feeds and pick which camera is live. The selected camera is shown on the hosted web page.
4. The UNO Q passes the angles from its Linux side to its microcontroller, which drives the servos.

The default target is a **cell phone**, which is easy to test with. Any YOLO class works, for example `--classes person` to follow a presenter.

### Image-space coordinates and servo control

The origin is the centre of the frame. Positive x is to the right and positive y is up.

```
x = cx - W / 2          (cx, cy) = centre of the YOLO bounding box
y = H / 2 - cy          W, H     = frame width and height

pan  = clamp( centre      + (x / (W/2)) * span      )
tilt = clamp( tilt_centre + (y / (H/2)) * tilt_span )
```

This is the poster's `pan = clamp(90 + Kp * x)` with `Kp = span / (W/2)`, and likewise for tilt. `centre` defaults to 90° (the servo centre), `span` is how far the servo turns between the frame centre and its edge, and `clamp` keeps the angle inside safe limits. The result is smoothed before it is sent, and the sketch also limits how fast each servo can move.

### Intelligent camera handoff

```
Is the target visible in the active camera? ── yes ──► keep the current camera live
        │ no
        ▼
Does the other camera see the target? ── yes ──► confirm, then switch the live display
        │ no
        ▼
keep waiting for the target to come back
```

| Setting | Default | Option |
|---|---|---|
| Active camera must have lost the target for | 0.7 s | `--lost` |
| Other camera must hold the target for | 0.3 s | `--enter` |
| Minimum time between two cuts | 2 s | `--dwell` |
| Early handoff near the frame edge | off | `--edge 0.85` turns it on |

The first camera to see the target goes live immediately. After that a cut only happens when the target has really left the live camera, which stops flicker when a detection drops out for a frame or two. With `--edge` set, the system can also cut early when the target nears the edge of the live camera and the other camera already sees it.

### Which camera moves the servos

The servos can follow **camera A only**, **camera B only**, or **both** (A when it sees the target, B when only B does). This is chosen with three buttons on the video page while the system runs, or at start-up with `--servo-from A|B|both`. Camera B uses camera A's centre and span unless the `--b-...` options give it its own.

## Repository layout

```
ai-pan-tilt/
├── README.md
├── docs/
│   └── AI_PanTilt_A1_Poster.pdf
├── laptop1/                      main laptop, cabled to the UNO Q
│   ├── switcher.py               the main program: both cameras, YOLO, handoff, servos, web page
│   ├── live_switch.py            the camera handoff state machine
│   ├── viewer.py                 small web server for the video page (port 8000)
│   ├── viewer.html               overview page and per-laptop screens
│   ├── webcam_servo.py           single-camera tracker, plus the USB/adb link and angle maths
│   ├── xy_display.py             live detection with x/y offsets, also shared helpers
│   ├── webcam_check.py           checks that the webcam opens
│   ├── config.py                 default camera and model
│   └── requirements.txt
├── laptop2/                      second laptop, provides camera B
│   ├── stream_webcam.py          serves its webcam as an MJPEG stream (port 8081)
│   ├── start_laptop2.bat         one double-click start on Windows
│   └── requirements.txt
└── uno-q/                        Arduino App Lab apps for the UNO Q
    ├── laptop-camera-servo/      main app used with switcher.py
    │   ├── app.yaml
    │   ├── python/main.py        receives angles from the laptop, passes them to the sketch
    │   ├── sketch/sketch.ino     drives the pan and tilt servos
    │   ├── sketch/sketch.yaml
    │   └── assets/               web page with pan and tilt gauges (port 7000)
    └── tests/                    standalone apps for checking the servos
        ├── pan-sweep/            pan servo alone on D9
        ├── tilt-sweep/           tilt servo alone on D10
        ├── pan-tilt-sweep/       both servos taking turns
        └── pan-tilt-control/     sliders and a sweep button on a web page
```

## Hardware and wiring

| Part | Role |
|---|---|
| Arduino UNO Q | Linux MPU (App Lab Python app, web page) plus STM32 MCU (servo sketch) |
| MG995 servo | pan, horizontal |
| DS3218 servo | tilt, vertical (the stronger servo, since it carries the camera's weight) |
| Pan-tilt bracket and camera platform | mechanical mount, with a spirit level on the platform |
| External 5 to 6 V supply, 3 A or more (5 A recommended) | servo power |
| Laptop 1 and laptop 2 with webcams | cameras A and B, detection and display |
| USB-C cable | laptop 1 to the UNO Q |

### Connections

| From | To |
|---|---|
| Pan servo signal (orange) | UNO Q **D9** |
| Tilt servo signal (orange/yellow) | UNO Q **D10** |
| Both servo red wires | supply **+** (5 to 6 V) |
| Both servo brown/black wires | supply **−** |
| Supply **−** | UNO Q **GND** (common ground, required) |
| UNO Q USB-C | laptop 1 |
| Optional: 470 to 1000 µF capacitor, 16 V | across supply + and −, near the servos |
| Optional: 10 kΩ resistor | from each signal wire to GND |

- **Do not power the servos from the UNO Q's 5V or 3V3 pins.** One MG995 or DS3218 can draw over 1 A when starting or straining. That resets the board and can damage it.
- **6 V** suits both servos and gives close to full torque. 5 V works with a little less.
- **Wiring:** run each servo's power and ground straight to the supply terminals, with short, thick wires. Thin breadboard rails cause jitter.
- **Order:** switch the servo supply on after the App Lab app is running, so both servos start from 90°. Switch it off when no sketch is running: an undriven signal wire picks up noise and the servo twitches.

| Servo | Voltage | Moving current | Stall current | Torque at 6 V |
|---|---|---|---|---|
| MG995 | 4.8 to 7.2 V | 0.3 to 0.6 A | about 1.2 to 1.5 A | about 11 kg·cm |
| DS3218 | 4.8 to 6.8 V | 0.3 to 0.7 A | about 2 to 2.5 A | about 20 kg·cm |

Typical datasheet values. Clones vary, so check your own units.

## Software setup

### Laptop 1

1. Install Python 3 and adb:
   ```
   winget install Google.PlatformTools
   ```
2. In a terminal inside `laptop1/`:
   ```
   py -m pip install -r requirements.txt
   ```
   (`python` instead of `py` also works if it is on the PATH.) The YOLO model `yolov8n.pt` downloads by itself on the first run, so laptop 1 needs internet the first time.

### Laptop 2

Install Python 3, then in `laptop2/`:
```
py -m pip install -r requirements.txt
```
`start_laptop2.bat` does this step for you as well.

### Arduino UNO Q

1. Open Arduino App Lab with the board connected.
2. Import `uno-q/laptop-camera-servo` as a new app (zip the folder first if App Lab asks for a zip).
3. The first build downloads its libraries (Arduino_RouterBridge, Servo and their dependencies) and needs the board to have internet once. After that, nothing on the board needs the network: the laptop reaches it over the USB cable.

## Running the system

Start things in this order.

1. **UNO Q:** in App Lab, press **Run** on **Laptop Camera Servo** and wait for the build to finish. Then switch on the servo supply.
2. **Network:** connect both laptops to the same network. A phone hotspot is the most reliable choice, because lab and campus wifi often blocks devices from reaching each other.
3. **Laptop 2:** double-click `start_laptop2.bat`, or run:
   ```
   py stream_webcam.py
   ```
   It prints an address like `http://192.168.x.x:8081/video`. Open it in laptop 2's browser to check the picture.
4. **Laptop 1:** in `laptop1/`, run:
   ```
   py switcher.py --cam-b http://<laptop 2 IP>:8081/video --usb
   ```
   Replace `<laptop 2 IP>` with the real numbers, without the angle brackets. `--usb` sets up the adb link to the board on its own. The terminal should show `camera connected` twice and `connected to the board`.
5. **Open the pages** listed in the next section.

To stop: press `q` in the switcher window (or ctrl+c), ctrl+c on laptop 2, switch off the servo supply, then press Stop in App Lab.

### Other ways to run laptop 1

| Command | Use |
|---|---|
| `py webcam_check.py` | check the webcam opens |
| `py xy_display.py` | live detection with x/y offsets, no board |
| `py xy_display.py --source http://<laptop 2 IP>:8081/video` | check detection on camera B alone |
| `py webcam_servo.py --usb` | camera A and the servos only, no second camera |
| `py switcher.py` | camera A only, no switching |

## Web pages

| Page | Where | Shows |
|---|---|---|
| `http://localhost:8000` | laptop 1 | overview: live camera, both thumbnails with a LIVE badge, recent cuts, and the **Servos follow** buttons |
| `http://localhost:8000/?screen=A` | laptop 1 | camera A full screen while it is live, black standby otherwise. Click for full screen. |
| `http://<laptop 1 IP>:8000/?screen=B` | laptop 2 | the same for camera B. Add `&boxes=1` to see detection boxes. |
| `http://localhost:7000` | laptop 1 | pan and tilt gauges, x/y and position dot, served by the UNO Q |
| `http://<laptop 2 IP>:8081/video` | either | raw camera B stream |

Each laptop shows the video only while its own camera is live. If laptop 2 cannot open laptop 1's page, allow port 8000 through laptop 1's Windows firewall and set the network to Private.

## Tuning and calibration

All of these are options on `switcher.py`, so no code changes are needed.

| Option | Default | Meaning |
|---|---|---|
| `--centre` | 90 | pan angle when the target is in the middle of the frame |
| `--span` | 60 | pan travel from the frame centre to its edge |
| `--invert` | off | flip pan direction |
| `--tilt-centre` | 90 | tilt angle for the middle of the frame |
| `--tilt-span` | 30 | tilt travel from the frame centre to its top or bottom |
| `--tilt-invert` | off | flip tilt direction |
| `--smoothing` | 0.5 | 0.1 is slow and steady, 1.0 is instant and jittery |
| `--b-centre`, `--b-span`, `--b-invert`, `--b-tilt-centre`, `--b-tilt-span`, `--b-tilt-invert` | camera A's values | separate settings for camera B |
| `--servo-from` | both | which camera moves the servos at start: `A`, `B` or `both` |
| `--classes` | cell phone | YOLO classes to follow, for example `person` |
| `--conf` / `--conf-b` | 0.3 / 0.15 | detection confidence for camera A / camera B |
| `--imgsz` | model default | lower (416) for speed, higher (800) for small objects |
| `--lost`, `--enter`, `--dwell`, `--edge` | 0.7, 0.3, 2.0, 0 | handoff timing |

Camera B gets a lower confidence threshold by default because its picture arrives compressed over the network. `stream_webcam.py` asks the webcam for 1280x720 and sends 960-wide frames at JPEG quality 90. Lower these with `--width 640 --capture 640` if the stream lags.

### Calibrating to a PTZ camera with a different field of view

The laptop cameras are fixed and the PTZ camera is usually zoomed in, so the calibration maps "target at this spot in the laptop picture" to "these angles centre it in the PTZ view".

1. Run `py xy_display.py` to see the laptop picture with x and y offsets.
2. In App Lab, run `uno-q/tests/pan-tilt-control`, then run `adb forward tcp:7000 tcp:7000` and open `http://localhost:7000` to move the head with sliders. Watch the PTZ camera's own view.
3. Put the target at the real working distance, near x 0 and y 0, and move the sliders until it is centred in the PTZ view. These angles are `--centre` and `--tilt-centre`.
4. Move the target to the left and right edges of the laptop picture and centre it in the PTZ view each time. Then work out `--span = (right angle - left angle) / 2 × (320 / x used)`. If the right edge needed the smaller angle, add `--invert`.
5. Do the same at the top and bottom for `--tilt-span` (with 240 for a 480-tall picture) and `--tilt-invert`.
6. Go back to **Laptop Camera Servo**, run the switcher with the numbers, set **Servos follow** to camera A only, and check the middle, the edges and the corners. Repeat for camera B with the `--b-...` options.

For example:
```
py switcher.py --cam-b http://<laptop 2 IP>:8081/video --usb --centre 92 --span 26 --tilt-centre 88 --tilt-span 14 --smoothing 0.7
```

The map is exact only at the calibration distance. Keep each laptop camera close to the PTZ head and facing the same way to keep the error small.

The servo limits and maximum speed are set in `uno-q/laptop-camera-servo/sketch/sketch.ino`: pan 30° to 150°, tilt 60° to 120°, at most 120°/s. Narrow them if the mount reaches a hard stop, and lower `MAX_SPEED` if a loaded servo strains.

## Test apps for the UNO Q

The apps in `uno-q/tests/` check the hardware without the laptops. Run one at a time, and stop the main app first, since the web apps share port 7000.

| App | What it does |
|---|---|
| `pan-sweep` | pan servo alone on D9: 60, 90, 120, 90 every 1.5 s |
| `tilt-sweep` | tilt servo alone on D10: 70, 90, 110, 90 every 1.5 s |
| `pan-tilt-sweep` | both servos take turns, so only one starts at a time |
| `pan-tilt-control` | sliders for each servo from 0° to 180°, a **Start sweep** button and **Both to 90**. Open `http://localhost:7000` after `adb forward tcp:7000 tcp:7000`. |

The sweep apps do not use Bridge or the network, so they show whether the servos, pins and supply work on their own.

## Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `board NOT connected` on the video | the App Lab app is not running, or another app holds port 7000. Press Run on Laptop Camera Servo. The switcher reconnects by itself. |
| `can't reach the camera yet` for camera B | laptop 2's stream stopped, its IP changed, or the network blocks device-to-device traffic. Check `ipconfig` on laptop 2 and `ping` it from laptop 1. Use a phone hotspot if the ping fails. |
| Laptop 2 cannot open laptop 1's page | Windows firewall on laptop 1. Set the network to Private and allow port 8000. |
| Camera B detects the target worse than A | lighting or a soft webcam. Improve the light, hold the target larger, lower `--conf-b`, or raise `--imgsz`. |
| Servo jitters when idle | loose common ground, a floating signal wire, a weak supply, or a servo hunting under heavy load. Re-seat GND, add the capacitor and 10 kΩ pull-downs, and balance the load. |
| One servo slow or weak | measure the voltage at its connector. Swap the servos to see whether the fault follows the servo. Servos that strained against a load for long can fail. |
| Tilt struggles with the camera on it | not enough torque or current. Use 6 V and a 3 A+ supply, and move the load's centre of gravity closer to the tilt axis. |
| App Lab build fails with a corrupted file | clear the app's `.cache` folder or re-import the app. Do not press Stop during a build. |
| `couldn't reach the sketch` in the App Lab console | the sketch did not start. Press Stop, then Run again. |

## 5G lab deployment (planned)

*Planned architecture, not yet implemented.*

```
VIDEO PATH
5G scout / IP camera ──radio──► 5G RAN (gNB) ──user plane──► private 5G core + UPF ──► MEC / edge server
 (live RTSP stream)                                                                       YOLO detection
                                                                                          target tracking
CONTROL PATH                                                                              camera handoff
pan-tilt platform ◄── servos (MG995 + DS3218) ◄── Arduino UNO Q ◄── Teltonika RUTX50 ◄─── angle commands
```

The compute-heavy AI inference is meant to run on the MEC server, while the UNO Q handles the embedded servo control. This uses the low latency and edge computing of the lab's private 5G network.

**How 5G can help**

- Wireless camera link, with no laptop-hosted stream or camera cabling.
- High-bandwidth uplink for continuous live video.
- Edge processing: video reaches the MEC server on the private network.
- Low-latency connection between the camera and the edge AI.
- Scales to more cameras on a managed lab network.

The code is ready for parts of this. `xy_display.py` and `switcher.py` accept an `rtsp://` address in place of a webcam, and the angle commands already travel as network messages to the board.

## Results, status and future scope

**Implemented**

- Live YOLO detection with bounding boxes and x/y coordinates.
- Pan and tilt angle calculation and two-servo control through the UNO Q.
- Two-camera streaming over HTTP and software camera switching with confirmation delays.
- Hosted browser pages for the live video, per-laptop screens and servo gauges.
- Servo control from camera A, camera B or both, selectable on the page.

**Observations**

- The target's x/y offset from the frame centre drives the servo commands, with independent horizontal and vertical control.
- Camera handoff was observed during development. The confirmation delays cut down unnecessary switching.
- Detection quality depends on image sharpness, lighting and video compression.

Latency, accuracy and frame-rate figures have not been measured yet and are not claimed.

**Future scope**

1. Integrate the laboratory's 5G camera and its RTSP stream.
2. Run YOLO inference on a MEC / edge server.
3. Network-based control link to the UNO Q.
4. Measure end-to-end latency, frame rate and tracking error.
5. Better multi-target tracking and identity preservation.
6. Better mechanical stability and servo calibration.
7. Send the selected feed to a projector or HDMI display system.
8. Closed-loop tracking once the camera rides on the pan-tilt head, correcting from the centre error instead of a fixed map.

## Conclusion

AI Pan-Tilt combines computer vision, automated camera orientation, embedded servo control and intelligent camera handoff. It lays the groundwork for an intelligent camera-tracking system, with private 5G connectivity and MEC-based processing as its planned deployment architecture.
