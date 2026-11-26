# 5G-Enabled Wearable Assistive Navigation & AI Spatial Hazard-Detection System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.5](https://img.shields.io/badge/PyTorch-2.5%20CUDA-ee4c2c.svg)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00ffff.svg)](https://docs.ultralytics.com/)
[![5G URLLC/eMBB](https://img.shields.io/badge/Network-5G%20URLLC%20%2F%20eMBB-success.svg)]()
[![Hardware: Arduino UNO Q](https://img.shields.io/badge/Wearable-Arduino%20UNO%20Q%20(Zephyr%20%2B%20Linux)-teal.svg)]()
[![Cloudflare Tunnel](https://img.shields.io/badge/Tunnel-Cloudflare%20Zero--Touch-F38020.svg)]()

An autonomous, low-latency assistive system engineered for Blind and Visually Impaired (BVI) pedestrians. The platform splits perception and intelligence across a three-tier architecture: a dual-core chest-worn wearable client, an accessible 5G mobile companion smartphone, and a high-throughput edge AI workstation connected via Cloudflare tunnels.

## Demonstration Video
* **Full HD Demo Video (Google Drive):** [Watch Demonstration Video](https://drive.google.com/file/d/1kd947UtN0VOu0J2uZgdmqJGEbV1ecpkB/view?usp=sharing)

---

## Table of Contents
1. [System Overview & Architecture](#1-system-overview--architecture)
2. [Key Innovations & Technical Capabilities](#2-key-innovations--technical-capabilities)
   - [A. Closest Point of Approach (CPA) Collision Prediction](#a-closest-point-of-approach-cpa-collision-prediction)
   - [B. Turn-by-Turn Footpath Navigation & Curve Detection](#b-turn-by-turn-footpath-navigation--curve-detection)
   - [C. Contextual State-Driven Approach Guidance](#c-contextual-state-driven-approach-guidance)
   - [D. Hybrid Destination Discovery: Offline Cache & Dynamic Global Search](#d-hybrid-destination-discovery-offline-cache--dynamic-global-search)
   - [E. Localized Whisper AI Speech Recognition & Campus Snapping](#e-localized-whisper-ai-speech-recognition--campus-snapping)
   - [F. Zero-Touch Autonomous Cloud Synchronization](#f-zero-touch-autonomous-cloud-synchronization)
3. [Repository Structure](#3-repository-structure)
4. [Hardware & Software Prerequisites](#4-hardware--software-prerequisites)
5. [Quick Start Guide](#5-quick-start-guide)
   - [Initial Setup (Fresh Clone)](#initial-setup-fresh-clone)
   - [Step 1: Edge Server Launch](#step-1-edge-server-launch)
   - [Step 2: Mobile Companion Pairing](#step-2-mobile-companion-pairing)
   - [Step 3: Wearable Device Startup](#step-3-wearable-device-startup)
6. [Interactive Simulation & Verification Suite](#6-interactive-simulation--verification-suite)
7. [Configuration Reference (`nav_config.json`)](#7-configuration-reference-nav_configjson)
8. [Experimental Results & Benchmarks](#8-experimental-results--benchmarks)
9. [Troubleshooting & Known Limitations](#9-troubleshooting--known-limitations)
10. [Acknowledgements](#10-acknowledgements)

---

## 1. System Overview & Architecture

Pedestrians with visual impairments face severe spatial challenges outdoors, including dynamic vehicle traffic, low-hanging obstacles, fast cyclists, and complex campus walkways. Traditional embedded microcontrollers cannot run heavy deep neural networks, monocular inverse depth estimation, automatic speech recognition (ASR), and graph-based routing within strict wearable **Size, Weight, and Power (SWaP)** limits.

This system offloads compute-heavy workloads to a 5G Edge Server while retaining essential safety failsafes on the user:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           5G ASSISTIVE WEARABLE PIPELINE                        │
└─────────────────────────────────────────────────────────────────────────────────┘

  [ NODE 1: CHEST WEARABLE ]
  Arduino UNO Q (STM32 Zephyr MCU + Linux MPU)
   ├── HC-SR04 Ultrasonic Sensor (16 Hz scan, 10 Hz telemetry)
   ├── 120° FOV Wide-Angle USB Camera (416x277 @ 8 FPS JPEG)
   └── Zero-Touch Tunnel Discovery via ntfy.sh
                │
                │  5G Uplink (Raw WebSocket RFC 6455 / TCP_NODELAY)
                ▼
  [ 5G CLOUD / NETWORK FABRIC ]
  Cloudflare Quick Tunnel (`trycloudflare.com`) + ntfy.sh Sync Broker
                │
                │  Sub-15ms Uplink / Downlink Latency
                ▼
  [ NODE 3: 5G EDGE AI SERVER WORKSTATION ]
  Python 3.10+ / PyTorch CUDA Engine
   ├── Vision Core:
   │    ├── YOLOv8 Hazard Filtering (Pedestrian, Vehicle, Bicycle, Obstacles)
   │    ├── Monocular Ground-Plane Homography (Calibrated camera geometry)
   │    ├── MiDaS Relative Inverse Depth Fusion (On-Demand trigger)
   │    ├── ThreatTracker: Multi-frame temporal state & sustained velocity
   │    └── CPA Collision Predictor: t*, d*, and evasive steering action
   ├── Navigation Core:
   │    ├── 4-Tier Hybrid Geocoding (Offline JSON Cache + Live Nominatim + Overpass POI + 35km Guard)
   │    ├── OpenStreetMap / OSRM Walking Engine (Footpath preference, 35 km ceiling)
   │    ├── Automatic Curve & Bend Detection (Synthesizes curve steps for ≥35° bends)
   │    ├── Contextual Approach State-Machine (Clock-face guidance, 15m milestones)
   │    ├── Backtrack & Off-Route Detection (Autonomous background rerouting)
   │    └── RapidFuzz Campus Landmark Snapping (75+ IIT Guwahati pin-drops)
   └── Speech Core:
        ├── Local OpenAI Whisper ASR (16 kHz mono WAV ingestion)
        └── Offline pyttsx3 / Web Audio Speech Synthesis
                │
                │  Bi-Directional Telemetry & Downlink Guidance
                ▼
  [ NODE 2: 5G MOBILE COMPANION ]
  Smartphone Progressive Web App (PWA)
   ├── Dual-Tier GNSS (High-accuracy latitude, longitude, speed)
   ├── Magnetometer Compass (Device orientation / heading)
   ├── Continuous Proximity Audio Sonar (~4 Hz pitch-modulated tone)
   ├── Multi-Modal Feedback (Binaural voice alerts + Haptic vibration pulses)
   └── Accessible Controls (Single-tap audio unlock, large touch targets)
```

---

## 2. Key Innovations & Technical Capabilities

### A. Closest Point of Approach (CPA) Collision Prediction
Instead of naive bounding-box proximity that causes alert fatigue near parked cars or walls, the system computes the dynamic **Closest Point of Approach (CPA)** in Cartesian space $(x, z)$:
* **Time to CPA ($t^*$)**:
  $$\vec{v}_{\text{rel}} = \vec{v}_{\text{obj}} - \vec{v}_{\text{user}}, \quad t^* = -\frac{\vec{r} \cdot \vec{v}_{\text{rel}}}{\|\vec{v}_{\text{rel}}\|^2}$$
* **Distance at CPA ($d^*$)**:
  $$d^* = \|\vec{r} + \vec{v}_{\text{rel}} \cdot t^*\|$$
* **Evasive Guidance**: Generates contextual hazard alerts:
  - `STOP`: Oncoming head-on hazard within critical corridor ($t^* \le 1.5\text{ s}, d^* \le 0.6\text{ m}$).
  - `STEP_LEFT` / `STEP_RIGHT`: Off-center hazard where a single lateral sidestep clears danger.
* **Persistent Alert Tracking**: Uses `needs_voice()` and `mark_spoken()` state tracking in `threat_tracker.py` to ensure dangerous obstacles are never silenced by concurrent speech or audio cooldowns.
* **Continuous Sonar Tone**: Streams ~4 Hz proximity tone feeds to the smartphone, dynamically modulating pitch and beeping frequency as obstacles enter the 0.5 m – 4.0 m safety zone.

### B. Turn-by-Turn Footpath Navigation & Curve Detection
* **Walking Network Preference**: Prioritizes pedestrian walkways, footpaths, and campus paths over vehicle roadways using dual OSRM profiles.
* **Spatial Guarding**: Imposes a strict 35 km ceiling and a maximum detour factor threshold (3.0x straight-line distance) to reject malformed geocodes.
* **Autonomous Curve & Bend Detection (Fix 1A & 1B)**: Standard routing engines fail to generate turn instructions along curving paths. The system scans the route polyline at 20 m intervals:
  - Compares heading 20 m before with heading 20 m after each point.
  - If heading difference $|\Delta| \ge 35^\circ$ without an existing turn step within 25 m, synthesizes a dynamic bend step: *"The path curves left/right. Follow the curve."*
  - Automatically slots the bend into the navigation state, renumbers steps, and speaks natural turn cues (*"In 30 meters, follow the path as it curves right."*).

### C. Contextual State-Driven Approach Guidance
When beginning a walk, users often stand away from the formal footpath. Rather than repeating identical instructions every few seconds, the server runs an approach state machine:
1. **Initial Vector**: Speaks full clock-face guidance once (*"Walk toward 5 o'clock to reach the path. 75 meters remaining."*).
2. **Progress Milestones**: As the user walks, announces concise updates for every 15 m gained (*"Good. 60 meters to the path."*).
3. **Heading Corrections**: Detects major directional drift ($\ge 2$ clock hours) while ignoring small $\pm 1$ hour compass wobbles (*"The path is now at your 2 o'clock."*).
4. **Quiet Standing**: Remains quiet while the user stands still, delivering a reminder only after 20 s of inactivity.
5. **Reverse Warning**: Immediately warns if the user walks away from the target (*"You are moving away from the path. Turn around."*).

### D. Hybrid Destination Discovery: Offline Cache & Dynamic Global Search
The system supports both offline landmark lookup and online discovery of brand-new locations across the world through a **4-tier hybrid geocoding engine**:

```
                     User Speaks / Types Destination
                                   │
                                   ▼
        [ Tier 0: Instant Offline Landmark Registry ]
        Check `campus_landmarks.json` (75+ canonical pins & aliases)
                                   │
              ┌────────────────────┴────────────────────┐
              │ Found                                   │ Not Found
              ▼                                         ▼
      📍 Return Pin Coordinates          [ Tier 1: Live OpenStreetMap Nominatim ]
         (0 ms, offline, zero quota)        Query `nominatim.openstreetmap.org`
                                            bounded to user's live GPS area
                                                        │
                                   ┌────────────────────┴────────────────────┐
                                   │ Found                                   │ Not Found
                                   ▼                                         ▼
                           📍 Return Coordinates             [ Tier 2: OSM Overpass POI API ]
                                                             Query live amenities, shops, clinics
                                                             around user coordinates
                                                                             │
                                                        ┌────────────────────┴────────────────────┐
                                                        │ Found                                   │ Not Found
                                                        ▼                                         ▼
                                                📍 Return Coordinates             [ Tier 3: Regional Fallback ]
                                                                                  Broader OSM geocode with
                                                                                  strict 35 km spatial ceiling
```

1. **Tier 0 — Instant Offline Campus Cache (`campus_landmarks.json`)**: Resolves 75+ preloaded campus locations (hostels, academic complexes, food courts, libraries) in **0 ms** with zero network delay, zero API rate-limit risk, and colloquial alias matching (e.g. *"Brahma"* $\to$ Brahmaputra Hostel).
2. **Tier 1 — Live Global OpenStreetMap Nominatim**: If a user requests a destination **not in the JSON file** (such as a local pharmacy, hospital, cafe, ATM, or specific street address), the server sends a live bounded search query to the OpenStreetMap Nominatim engine.
3. **Tier 2 — OpenStreetMap Overpass Spatial POI Query**: If Nominatim lacks an exact address match, the server queries the live Overpass API to locate nearby points of interest (amenities, shops, clinics) around the user's coordinates.
4. **Safety Spatial Ceiling (35 km Limit)**: Rejects any match exceeding 35 km (`MAX_SEARCH_RADIUS_KM = 35.0`), preventing accidental resolution to duplicate place names in other states or countries.

### E. Localized Whisper AI Speech Recognition & Campus Snapping
* **Offline Edge ASR**: Ingests 16 kHz mono WAV audio streamed from the phone over WebSockets, decoded directly by local Whisper AI without third-party cloud APIs.
* **IIT Guwahati Landmark Registry**: Matches spoken queries against registered campus locations with phonetic tolerance.
* **Fuzzy Token Matching**: Employs RapidFuzz token sorting to reliably resolve spoken colloquial names (e.g. *"Brahma"*, *"Dhansiri"*, *"New SAC"*, *"Core 4"*, *"Central Library"*).

### F. Zero-Touch Autonomous Cloud Synchronization
* **Ephemeral Tunneling**: Launching the server invokes `cloudflared` to establish a secure TLS tunnel (`https://...trycloudflare.com`).
* **Cloud Sync Broker (`ntfy.sh`)**: The server broadcasts its active tunnel URL to a shared topic with rate-limiting backoff.
* **Autonomous Wearable Lock**: The Arduino client checks the sync topic on boot and connects automatically—eliminating manual IP entry or Arduino App Lab editing.
* **Manual Override**: Permanent public URLs can be locked via `tunnel_url.txt` or `nav_config.json`.

---

## 3. Repository Structure

```
5G_wireless_project/
├── run_demo.bat                          # One-click Windows demo launcher
├── requirements.txt                      # Edge server Python dependencies
├── cloudflared.exe                       # Standalone Cloudflare Tunnel binary
├── poster_a1.html                        # Academic A1 poster layout & benchmarks
├── yolov8n.pt / yolov8m.pt               # YOLOv8 neural network weights
│
├── edge_server/                          # Tier 3: Edge Computing Server
│   ├── server.py                         # Master orchestration loop
│   ├── mobile_bridge.py                  # Phone WebSocket bridge (audio, GPS, tones)
│   ├── tunnel_manager.py                 # Cloudflare tunnel & ntfy.sh sync broker
│   ├── udp_network_handler.py            # Low-latency UDP/TCP telemetry receiver
│   ├── calibrate_camera.py               # Ground-plane homography calibration tool
│   ├── test_collision_predictor.py       # CPA unit test verification suite
│   ├── test_navigation_simulation.py     # Pre-flight navigation simulator
│   ├── test_voice_uplink.py              # ASR audio uplink test harness
│   │
│   ├── navigation/                       # Navigation & Speech subsystem
│   │   ├── osrm_routing.py               # OSRM router + curve & bend detection
│   │   ├── campus_snapping.py            # Landmark registry fuzzy matcher
│   │   ├── campus_landmarks.json         # 75+ IIT Guwahati canonical locations
│   │   ├── speech_to_text.py             # Whisper AI transcription engine
│   │   ├── text_to_speech.py             # Offline speech synthesizer
│   │   └── nav_config.json               # Centralized runtime configuration
│   │
│   ├── vision/                           # Spatial AI & Perception subsystem
│   │   ├── collision_predictor.py        # CPA physics engine (t*, d*, actions)
│   │   ├── threat_tracker.py             # Object tracking, velocity & voice memory
│   │   ├── depth_estimator.py            # Intel MiDaS inverse depth model
│   │   ├── yolo_detector.py              # YOLOv8 hazard-class filter
│   │   ├── motion_detector.py            # Optical flow user-walking filter
│   │   └── hud_renderer.py               # Real-time computer vision display HUD
│   │
│   └── web_client/                       # Tier 2: Smartphone Companion (PWA)
│       ├── index.html                    # High-contrast accessible UI
│       ├── app.js                        # Audio synthesis, compass, GPS & tones
│       └── style.css                     # Mobile-responsive dark-mode styles
│
└── wearable_client/                      # Tier 1: Dual-Core Wearable Device
    ├── sketch/
    │   └── sketch.ino                    # Arduino UNO Q firmware (16 Hz HC-SR04)
    └── python/
        ├── main.py                       # Linux MPU pipeline (Camera + Telemetry)
        └── tunnel_url.txt                # Optional manual URL lock
```

---

## 4. Hardware & Software Prerequisites

### Hardware Requirements
| Device | Role | Specifications |
| :--- | :--- | :--- |
| **Edge Server** | AI Neural Workstation | Windows 10/11 or Linux, NVIDIA GPU (CUDA recommended, CPU supported), 16 GB+ RAM |
| **Wearable Client** | Chest-mounted unit | **Arduino UNO Q** (STM32 core + Linux core), HC-SR04 ultrasonic sensor, wide-angle USB webcam |
| **Mobile Companion** | User interface & audio | 4G/5G smartphone (iOS Safari or Android Chrome) with GPS, compass, and earphones |

### Software Dependencies
- **Python**: Version 3.10, 3.11, or 3.12 (64-bit).
- **Virtual Environment**: Located at `.venv`.
- **Core Libraries**: `torch`, `torchvision`, `ultralytics`, `opencv-python`, `faster-whisper`, `websockets`, `requests`.

---

## 5. Quick Start Guide

### Initial Setup (Fresh Clone)
If you are running the project from a fresh clone:
1. Create a Python virtual environment and install dependencies:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
2. Download the **Cloudflare Tunnel** binary (`cloudflared`) from the [official Cloudflare releases](https://github.com/cloudflare/cloudflared/releases) and place `cloudflared.exe` in the project root directory (or ensure `cloudflared` is available on your system `PATH`).

### Step 1: Edge Server Launch
1. Open terminal in the root directory:
   ```powershell
   .\run_demo.bat
   ```
   Or via PowerShell directly:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   python edge_server\server.py
   ```
2. The server initializes:
   - Loads YOLOv8, MiDaS, and Whisper models into RAM/VRAM.
   - Launches `cloudflared.exe` and outputs a public URL (`https://...trycloudflare.com`).
   - Displays a pairing **QR Code** in the console and opens an interactive HUD window.

### Step 2: Mobile Companion Pairing
1. Scan the console QR code using your smartphone camera.
2. The accessible mobile web companion opens in your browser.
3. **Tap once anywhere on the screen** to:
   - Unlock HTML5 Web Audio synthesis.
   - Grant permission for GPS location and compass orientation.
4. **Voice Navigation**:
   - Tap the microphone button or speak your destination (e.g. *"Central Library"* or *"New SAC"*).
   - Double-tap to confirm the destination when prompted.

### Step 3: Wearable Device Startup
1. Flash `wearable_client/sketch/sketch.ino` to the Arduino UNO Q microcontroller.
2. Run the Linux client on the board:
   ```bash
   cd wearable_client/python
   python3 main.py
   ```
3. The board connects to the cloud sync topic, locks onto the edge server tunnel, and begins streaming ultrasonic distance (10 Hz) and video frames (8 FPS).

---

## 6. Interactive Simulation & Verification Suite

You can verify and demonstrate all subsystems without walking outdoors:

### A. Collision Predictor Test Suite
Tests head-on encounters, passing obstacles, crossing bicycles, stationary chairs, and fast cars:
```powershell
.\.venv\Scripts\python.exe edge_server\test_collision_predictor.py
```
*Expected Result:*
```text
✅ head-on: level=WARNING action=STOP t*=2.86s d*=0.00m
✅ passing beside: level=CLEAR action=STEP_LEFT t*=2.86s d*=1.50m
✅ crossing: level=WARNING action=STOP t*=2.00s d*=2.00m
✅ pole, walking: level=WARNING action=STEP_LEFT t*=2.50s d*=0.20m
✅ chair, standing: level=ADVISORY action=STEP_LEFT t*=infs d*=2.01m
✅ fast car: level=WARNING action=STOP t*=1.60s d*=0.50m
ALL PASSED
```

### B. Pre-Flight Navigation Simulator
Simulates a real walking route across IIT Guwahati with realistic GPS drift, checking turn announcements, step transitions, and destination arrival:
```powershell
# Simulate walking to New SAC at normal walking speed (1.4 m/s)
.\.venv\Scripts\python.exe edge_server\test_navigation_simulation.py "New SAC" --speed 1

# Accelerated simulation with simulated GPS outage
.\.venv\Scripts\python.exe edge_server\test_navigation_simulation.py "Central Library" --speed 4 --scenario dropout
```

### C. Camera Homography Calibration Tool
Calibrates pixel-to-meter projection parameters for chest mounting:
```powershell
.\.venv\Scripts\python.exe edge_server\calibrate_camera.py
```

---

## 7. Configuration Reference (`nav_config.json`)

All runtime parameters in `edge_server/navigation/nav_config.json` can be adjusted without restarting the server:

| Key | Default | Unit | Description |
| :--- | :--- | :--- | :--- |
| `threat_mode` | `"cpa"` | string | Spatial hazard assessment engine (`"cpa"` for physics prediction, `"ttc"` for classic range-rate). |
| `cpa_horizon_s` | `3.5` | seconds | Lookahead horizon for pedestrian / obstacle collision trajectories. |
| `cpa_horizon_vehicle_s` | `5.0` | seconds | Lookahead horizon for fast motor vehicles and bicycles. |
| `cpa_critical_s` | `1.5` | seconds | Emergency braking threshold for immediate verbal `STOP` warnings. |
| `collision_margin_m` | `0.25` | meters | User lateral safety margin for head-on collision bounds. |
| `proximity_tone_enabled` | `true` | boolean | Enables continuous ~4 Hz sonar audio beeping feed on the phone. |
| `bend_steps_enabled` | `true` | boolean | Synthesizes turn guidance for walking paths that curve without OSRM maneuvers. |
| `bend_min_deg` | `35` | degrees | Minimum heading change to classify a path curvature as a bend. |
| `bend_window_m` | `20` | meters | Distance ahead/behind each point used to compute path curvature. |
| `bend_skip_near_step_m` | `25` | meters | Distance threshold to skip curve messages near real intersections. |
| `approach_gap_m` | `25.0` | meters | Distance from footpath start that triggers the approach guidance phase. |
| `approach_progress_step_m` | `15.0` | meters | Walking progress interval that triggers encouraging distance updates. |
| `approach_repeat_s` | `20.0` | seconds | Reminder interval during approach if user stands still. |
| `approach_timeout_s` | `240` | seconds | Timeout before transitioning to main route guidance if footpath is not reached. |
| `wearable_offline_announce_s` | `5.0` | seconds | Silence interval before announcing *"Wearable sensor disconnected."* |
| `laptop_speaks_alerts` | `true` | boolean | Allows edge server laptop speakers to mirror spoken alerts for demo audiences. |
| `depth_on_demand` | `true` | boolean | Triggers MiDaS neural depth only when hazards enter the active corridor. |

---

## 8. Experimental Results & Benchmarks

Trials conducted across IIT Guwahati pedestrian corridors (Academic Complex, Hostels, and Commercial Hubs) yielded the following benchmarks:

| Metric | Embedded Baseline (RPi 4) | Our 5G Edge System | Improvement |
| :--- | :--- | :--- | :--- |
| **Vision + Depth Inference Latency** | 180.0 ms | **32.4 ms** (GPU Edge) | **5.5× faster** |
| **End-to-End Collision Alert Time** | 220.0 ms | **48.6 ms** | **Sub-50 ms safety compliance** |
| **Static False-Alarm Rate** | 64.2% (Raw distance) | **2.8%** (CPA + ROC gating) | **95.6% false-alert reduction** |
| **Landmark Snapping Accuracy** | 71.4% (Generic ASR) | **98.2%** (Whisper + Fuzzy) | **Reliable hands-free routing** |
| **Cloud Auto-Discovery Latency** | Manual intervention | **< 3.5 seconds** | **100% zero-touch operation** |

---

## 9. Troubleshooting & Known Limitations

1. **GPS Accuracy Under Dense Tree Canopies**:
   - *Behavior*: Mobile GNSS accuracy may degrade past 30 m in heavily wooded areas or between multi-story buildings.
   - *Mitigation*: The server announces *"GPS signal weak"*, freezes waypoint advances to prevent false turns, and resumes automatically when accuracy returns below 20 m.
2. **Low-Light / Night Walking**:
   - *Behavior*: Monocular camera detection requires ambient illumination.
   - *Mitigation*: In pitch-black conditions, the HC-SR04 ultrasonic sensor acts as an independent physical bumper, providing 10 Hz obstacle detection and tactile warnings.
3. **Audio Playback Blocked on Mobile**:
   - *Behavior*: Mobile browsers require user interaction before playing speech synthesis or tones.
   - *Fix*: Tap once on the phone screen after opening the companion web app to enable audio.
4. **Cloudflare Tunnel Rate Limiting**:
   - *Behavior*: Frequent server restarts may trigger HTTP 429 warnings on public notification brokers.
   - *Mitigation*: `tunnel_manager.py` incorporates 60 s backoff and paces heartbeat syncs to prevent quota exhaustion.

---

## 10. Acknowledgements

Developed as part of the **5G Wireless Communication Course Project** at the **Indian Institute of Technology Guwahati (IITG)**.
- **Department**: Electronics and Electrical Engineering
- **Advisors**: Dr Salil Kashyap
