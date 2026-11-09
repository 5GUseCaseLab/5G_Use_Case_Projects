# Critical Apnea Detection: 5G-Enabled Contactless Vital Sign Monitoring & Edge-AI Health Classification

**Course:** 5G Usecase Laboratory (Course Code: EE-4032)  
**Department:** Department of Electronics and Electrical Engineering, Indian Institute of Technology Guwahati (IIT Guwahati)  

### Authors & Supervision
* **Students:**
  * **Spandan Bharadwaj** (Roll No. 230102108)
  * **Shreyas Sagar** (Roll No. 230102118)
* **Under the Guidance of:** **Prof. Salil Kashyap**
* **Teaching Assistants:** Aditya Gupta, Abhay Saran, Asif Hoque

---

# 🎥 [**CLICK HERE TO WATCH THE PROJECT DEMONSTRATION VIDEOS (GOOGLE DRIVE)**](https://drive.google.com/drive/folders/1y63yjSAknfMVxY4H4dzShWZG9y130pJW?usp=sharing)
> **📢 Video Submission Note:** The Google Drive folder linked above contains **two complete demonstration videos**:
> 1. **Proof of Concept Setup Demonstration Video** (Laptop & Logitech Camera Flows with Real-Time Edge-AI & Bedside Buzzer/LED Actuation)
> 2. **5G Usecase Lab Setup Demonstration Video** (End-to-End 5G Standalone MEC Docker Stream, 5G Evaluation Board (`QsoftAP`), Arduino Uno Q Edge-AI, and Live ThingsBoard Clinical Telemetry)
>
> 📄 **Printable A1 Poster:** Available directly in this repository at [`./5glab_poster1.pdf`](./5glab_poster1.pdf).

---

## 1. Abstract
Sudden respiratory arrest and central or obstructive sleep apnea represent life-threatening emergencies in Neonatal Intensive Care Units (NICUs), burn wards, and post-operative recovery units. Conventional contact-based monitors rely on adhesive electrodes, thoracic impedance belts, and nasal cannulas that cause epidermal trauma, infection risk, and patient discomfort. This project presents a real-time, contactless **Critical Apnea Detection and Cardiorespiratory Monitoring System** powered by a private **3GPP 5G Standalone (SA) Mobile Edge Compute (MEC)** network and the heterogeneous dual-processor **Arduino Uno Q** edge-AI platform.

By streaming live patient video into the Arduino Uno Q’s **Qualcomm Dragonwing QRB2210** Linux microprocessor (MPU), our system concurrently performs automated facial tracking, 3D Convolutional Neural Network Remote Photoplethysmography (**PhysNet**) for Heart Rate (HR) extraction, and Gunnar Farnebäck dense optical flow analysis on the chest region for Respiration Rate (RR) and thoracic motion energy quantification. When breathing ceases beyond a critical time threshold (`0.5 s`), the Linux MPU signals the on-board **STMicroelectronics STM32U585** real-time microcontroller (MCU) via an inter-processor Unix Domain Socket MsgPack-RPC bridge (`/var/run/arduino-router.sock`) to trigger a bedside piezo buzzer and an 8-channel Red/Blue emergency LED strobe while simultaneously publishing clinical telemetry to a **5G MEC ThingsBoard** server.

---

## 2. Introduction & Motivation

### Clinical Pain Points of Existing Systems
* **Wired Monitoring Hurts:** Traditional vital-sign monitors require sticky electrodes and wires taped to the skin—painful and risky to reapply on premature infants or burn victims.
* **Wi-Fi Cameras Are Too Slow:** Standard hospital Wi-Fi drops and delays video frames under load (`1.0 – 2.0 s` lag), so a camera-based system built on Wi-Fi cannot be trusted for real-time, life-saving apnea detection.
* **The Clinical Gap:** We need a contactless optical vital-sign engine to read vital signs without touching the patient, paired with a 5G network fast and reliable enough to act on that data immediately.

### The 4-Step Project Roadmap
1. **Step 1 – Prove the Math on a Laptop (Proof of Concept):** Validate contactless vital-sign extraction locally using standard built-in and Logitech USB webcams.
2. **Step 2 – Train the Bedside Edge-AI Actuator:** Teach the Arduino Uno Q’s on-board intelligence to distinguish *Normal Respiration* from *Critical Apnea* and trigger local LED and buzzer alerts as an instant bedside fail-safe.
3. **Step 3 – Move Video Processing & Routing to 5G MEC (Tier 1):** Shift video routing to a containerized Docker service on the 5G Mobile Edge Compute server via the 5G Evaluation Board.
4. **Step 4 – Validate Speed, Reliability & Multi-Vital Accuracy:** Benchmark the live 5G system in the lab against latency targets (`0.2 – 0.5 s`) and verify Heart Rate (BPM) and Respiration Rate (BrPM) accuracy against commercial smartwatches.

---

## 3. Objectives
* Design a **contactless dual-region computer vision pipeline** that automatically locks onto a patient’s face and chest from a live IP video feed.
* Deploy the **PhysNet 3D Encoder-Decoder MAX** deep neural network on the Qualcomm Dragonwing QRB2210 MPU for Heart Rate (BPM) extraction alongside dense optical flow for Respiration Rate (BrPM) and thoracic kinetic energy tracking.
* Establish a zero-lag **inter-processor RPC communication bridge** between the Linux MPU and the STM32U585 MCU to drive an active-HIGH piezo buzzer (`Pin D4`) and an active-LOW 8-channel RGB V2 LED strobe module (`Pins 2, 3, 5, 6, 7, 8, 9, 13` and `Pins 10, 11, 12`).
* Integrate the complete workflow into the **IIT Guwahati 5G Usecase Lab** using a Dockerized camera bridge on the 5G MEC server, the 5G Evaluation Board, and a live **ThingsBoard** clinical dashboard.

---

## 4. Repository Structure & Contents

```text
Critical-Apnea-Detection/
├── 5glab_poster1.pdf                       # Printable A1-Sized IEEE Project Poster
├── README.md                               # Complete Project Documentation & Setup Guide
├── Arduino_uno_q_setup/                    # Complete Arduino Uno Q Dual-Processor Package
│   ├── dataset/
│   │   └── haarcascade_frontalface_default.xml        # OpenCV Frontal Face Cascade Classifier
│   ├── final_model_release/
│   │   └── UBFC-rPPG_PhysNet_DiffNormalized.pth       # Pre-trained PhysNet 3D-CNN Weights
│   ├── neural_methods/model/
│   │   └── PhysNet.py                                 # PhysNet 3D Encoder-Decoder MAX Architecture
│   ├── sketch.ino                                     # STM32U585 Real-Time Actuator Firmware (Buzzer + 8-LEDs)
│   └── uno_q_poc.py                                   # Main Linux MPU Edge-AI, HUD & 5G Telemetry Engine
├── Cam_scripts/                            # Video Streamer Scripts for All 3 Experimental Flows
│   ├── laptop_camera_usb.py                           # Flow 1: Built-in Laptop Webcam Streamer
│   ├── logitech_cam.py                                # Flow 2: External Logitech USB Webcam Streamer
│   └── Sparsh_Camera_MEC.py                           # Flow 3: 5G MEC Docker RTSP-to-HTTP Bridge
└── Screenshots/                            # Visual Setup Guides & Experimental Results
    ├── critical_apnea.jpg                             # Result: Critical Apnea State + Active Buzzer/LED Strobe
    ├── normal_breathing.jpg                           # Result: Normal Respiration State + Green HUD
    ├── thingsboard_ui.png                             # Result: 5G MEC ThingsBoard Telemetry Dashboard
    ├── uno_app_interface.png                          # Setup: Arduino App Lab Folder & Sketch Preloading
    ├── laptop_camera_usb.png                          # Setup: Built-in Laptop Camera Stream Terminal Output
    ├── logitech_py_image.png                          # Setup: Logitech Camera Stream Terminal Output
    └── uno_link_replacement.png                       # Setup: Updating VIDEO_SOURCE Link in uno_q_poc.py
```

---

## 5. Step-by-Step Hardware & Software Setup Guide

> ### ⚠️ MANDATORY PREREQUISITE: SAME WI-FI / NETWORK RULE
> Before running any of the three flows below, ensure that **both your Laptop (or Camera Streamer) and the Arduino Uno Q are connected to the EXACT SAME Wi-Fi network** (e.g., the same local Wi-Fi router for the Proof of Concept setups, or the **`QsoftAP`** 5G Evaluation Board Wi-Fi network for the 5G Lab setup). If they are on different networks, the Arduino Uno Q will not be able to reach the camera stream IP.

---

### Step 1: Preloading `Arduino_uno_q_setup` in Arduino App Lab

The Arduino Uno Q features two processors:
1. **Qualcomm Dragonwing QRB2210 (Linux MPU):** Runs `uno_q_poc.py` for computer vision, PhysNet 3D-CNN inference, and 5G telemetry.
2. **STMicroelectronics STM32U585 (Real-Time MCU):** Controls physical header pins `D0–D13` via `sketch.ino`.

#### How to Import the Repository into Arduino App Lab:
1. Open **Arduino App Lab** and create a new App (e.g., **`Apnea_Monitor`**).
2. Download the [`Arduino_uno_q_setup/`](./Arduino_uno_q_setup) folder from this repository:
   * Copy **`sketch.ino`** into the **`sketch/`** folder of your App Lab project.
   * Copy the remaining contents (**`uno_q_poc.py`**, **`dataset/`**, **`final_model_release/`**, and **`neural_methods/model/PhysNet.py`**) into the **`python/rPPG-Toolbox-main/`** folder inside your App Lab project, exactly as shown below:

![Arduino App Lab Interface and Sketch Setup](./Screenshots/uno_app_interface.png)

3. Ensure the required Sketch Libraries are added (`Arduino_RouterBridge`, `MsgPack`, `DebugLog`, `ArxContainer`, `ArxTypeTraits`).
4. Click the green **`▶ Run`** button in the top-right corner of Arduino App Lab **once**. This compiles and flashes `sketch.ino` onto the STM32U585 microcontroller, registering the `set_pin` RPC service over `/var/run/arduino-router.sock`.

#### Bedside Hardware Wiring (STM32U585 Header Pins):
| Component | Pin Label | Arduino Uno Q Header Pin | Active Logic |
| :--- | :--- | :--- | :--- |
| **Piezo Buzzer** | `+` (VCC) / `-` (GND) | **Pin D4** & **GND** | Active-HIGH (`1 = HIGH`, `0 = LOW`) |
| **8-Channel RGB V2 Module** | Channels `D0 – D7` | **Pins 2, 3, 5, 6, 7, 8, 9, 13** | Active-LOW (`0 = ON`, `1 = OFF`) |
| **RGB Primary Color Pins** | `R`, `G`, `B` | **Pin 10 (R), Pin 11 (G), Pin 12 (B)** | Active-LOW (`0 = ON`, `1 = OFF`) |

#### Install Required Python Libraries on the Arduino Uno Q Linux Terminal (`arduino@fayette`):
```bash
sudo apt update && sudo apt install -y python3-libgpiod
pip3 install torch opencv-python scipy numpy flask requests msgpack --break-system-packages
```

---

### Step 2: Choosing & Launching Your Camera Stream (`Cam_scripts/`)

You can run the project in **three different camera configurations** using the scripts in [`Cam_scripts/`](./Cam_scripts):

---

#### 🔹 Flow 1: Built-in Laptop Camera Setup (`laptop_camera_usb.py`)
1. Connect both your Laptop and Arduino Uno Q to the same Wi-Fi network.
2. Open **Windows PowerShell** on your laptop, navigate to the script directory, and run:
   ```powershell
   python3 .\laptop_camera_usb.py
   ```
3. The terminal will output your live `VIDEO_SOURCE` link (e.g., `http://10.242.214.199:5000/video`):

![Built-in Laptop Camera Stream Output](./Screenshots/laptop_camera_usb.png)

---

#### 🔹 Flow 2: External Logitech USB Camera Setup (`logitech_cam.py`)
1. Plug the **Logitech USB Webcam** into your laptop's USB port.
2. Open **Windows PowerShell** and launch the wide-angle (`800×600`, DirectShow `CAP_DSHOW`) streamer:
   ```powershell
   python3 .\logitech_cam.py
   ```
3. Copy the printed `VIDEO_SOURCE` link from the PowerShell terminal:

![Logitech Camera Stream Output](./Screenshots/logitech_py_image.png)

---

#### 🔹 Flow 3: 5G Usecase Lab Setup (5G Evaluation Board + 5G MEC Docker + Sparsh Camera)

To run the full **3GPP 5G Standalone (SA) Lab Deployment**, follow the official **5G Evaluation Board** and **MEC Docker** procedure below:

##### Part A: Configuration of the 5G Evaluation Board
**Software to be installed on your Laptop:**
1. Download **MobaXterm** software and install it on your device.
2. Download **Quectel driver** software.
3. Download **IObit Driver Booster 12** and install it on your system.

**Hardware & AT Command Configuration Procedure:**
4. Check whether the **private 5G SIM** is inserted in the Evaluation Board.
5. Power on the Evaluation Board.
6. Connect the Evaluation Board to your device using a USB port.
7. After connection, open **IObit Driver Booster** and scan for driver changes.
8. Download and update the drivers as specified by the booster software.
9. After updating the drivers, open the **MobaXterm** application.
10. Press the **Reset button** on the Evaluation Board for **10 seconds**.
11. Open a new session in MobaXterm by going to **`Session` $\rightarrow$ `Serial`** and set the following parameters:
    * **Serial port:** `COM 8 (Quectel USB AT Port (COM 8))` *(Note: COM number might vary depending on the USB port connected)*
    * **Speed (bps):** `115200`
12. In the opened MobaXterm serial terminal, execute the following **3GPP AT Commands** in sequence:

| Sl. No. | AT Command | Expected Output & Description |
| :---: | :--- | :--- |
| **1.** | `AT` | Returns `OK` if properly connected |
| **2.** | `ATI` | Returns Information: Model number and firmware version |
| **3.** | `AT+CIMI` | Returns IMSI number of the private 5G SIM |
| **4.** | `AT+COPS=?` | Displays list of available networks; verify network **`"00101"`** is available |
| **5.** | `AT+CFUN=0` | Returns `OK`; switches User Equipment (UE) to minimum functionality |
| **6.** | `AT+CFUN=1` | Returns `OK`; switches UE to full functionality |
| **7.** | `AT+CGDCONT?` | Displays list of configured APNs |
| **8.** | `AT+CGDCONT=1,"IP","5glabs1"` | Returns `OK`; sets APN to **`"5glabs1"`** |
| **9.** | `AT+QNWPREFCFG="mode_pref",NR5G` | Returns `OK`; locks Evaluation Board to work in **NR5G (5G Standalone)** mode only |
| **10.** | `AT+CFUN=1,1` | Returns `OK`; resets and re-registers UE on the 5G Core |
| **11.** | `AT+CGPADDR=1` | Returns the assigned 5G IP Address |

13. After completing the AT commands above, open your Wi-Fi network list and verify that **`QsoftAP`** is available:
    * **Wi-Fi SSID:** `QsoftAP`
    * **Password:** `1234567890`
14. **Connect BOTH your Laptop and the Arduino Uno Q to the `QsoftAP` Wi-Fi network.**

##### Part B: Launching the Docker Container (`wire`) on the 5G MEC Server
Because the **Sparsh 5G Camera** (`rtsp://admin:admin123@192.168.128.10:554/avstream/channel=1/stream=1.sdp`) is hosted inside the 5G Mobile Edge Compute (MEC) subnet, we run a Docker container named `wire` on the MEC server (`192.168.116.30`) with Port `6000` exposed:

```bash
# 1. Create the Docker container on the 5G MEC server (First-time setup)
docker run -it --name wire --net=host -p 6000:6000 python:3.10-slim /bin/bash

# 2. Start and enter the 'wire' container
docker start wire
docker exec -it wire /bin/bash

# 3. Inside the container, launch the Sparsh 5G Camera Bridge script
python3 Sparsh_Camera_MEC.py

# 4. After completing your lab experiment, stop the container cleanly
docker stop wire
```
Copy the resulting 5G MEC stream URL: **`http://192.168.116.30:6000/video_feed`**.

---

### Step 3: Paste the Stream Link into `uno_q_poc.py` & Launch on Arduino Uno Q

1. Open **`uno_q_poc.py`** and locate **Line 21** (`VIDEO_SOURCE`), circled in blue below.
2. Paste the exact `VIDEO_SOURCE` URL generated from **Flow 1**, **Flow 2**, or **Flow 3**:

![Replacing VIDEO_SOURCE Link in uno_q_poc.py](./Screenshots/uno_link_replacement.png)

3. Open the **Arduino Uno Q Linux Terminal (`arduino@fayette`)**, navigate to the script directory, and launch `uno_q_poc.py`:
   ```bash
   cd ~/ArduinoApps/apnea_monitor/python/rPPG-Toolbox-main
   python3 uno_q_poc.py
   ```

---

## 6. Live Demonstration & Experimental Results

### How the Live Demonstration Works:
1. The patient sits in front of the camera so their face and chest are visible in the frame.
2. **5-Second Auto-Calibration:** During the first `5.0 seconds`, the system calibrates to the patient's baseline thoracic motion energy.
3. **Normal Respiration (`apnea_flag = 0`):** As the patient breathes normally, the green Face ROI (`PATIENT TARGET LOCKED`) and cyan Chest ROI (`THORAX MOTION ROI`) track vital signs in real time, displaying **Heart Rate (BPM)**, **Respiration Rate (BrPM)**, **Chest Motion Energy**, and live **Plethysmograph (PPG) & Respiration waveforms** on the Arduino's web monitor (`http://<ARDUINO_IP>:8080/`).
4. **Breath Hold & Critical Apnea (`apnea_flag = 1` & `2`):** When the patient holds their breath, chest velocity drops below the `0.08` noise gate. Within `0.2 s` it flags **🟠 BREATH HOLD**, and at `0.5 s` it triggers **🔴 CRITICAL APNEA**, immediately firing the **Pin D4 Piezo Buzzer** and strobing all **8 RGB LEDs in alternating Red & Blue patterns**.
5. **Recovery:** As soon as the patient resumes normal breathing for 10 consecutive frames, the system automatically returns to **🟢 NORMAL RESPIRATION** and turns off the buzzer and LEDs.

---

### Visual Results Across All Interfaces

#### 1. Normal Respiration State (Green HUD & Standby Bedside Hardware)
![Normal Respiration State](./Screenshots/normal_breathing.jpg)

#### 2. Critical Apnea State (Red HUD, Active Buzzer & Glowing 8-LED Red/Blue Strobe)
![Critical Apnea State](./Screenshots/critical_apnea.jpg)

#### 3. 5G MEC ThingsBoard Real-Time Clinical Telemetry Dashboard
> **📌 Important Network Note:** While the **Arduino Processed Video HUD (`http://<ARDUINO_IP>:8080/`)** can be accessed in **all three flows** (Laptop Cam, Logitech Cam, and 5G Lab), the **ThingsBoard Telemetry Interface (`http://192.168.116.30:9090/`) can ONLY be accessed when connected inside the Lab 5G Network (`QsoftAP`)**.

![5G MEC ThingsBoard Telemetry Dashboard](./Screenshots/thingsboard_ui.png)

---

### Quantitative Performance Summary

| Evaluation Metric | Proof of Concept (Laptop / Logitech over Wi-Fi) | 5G Usecase Lab (Sparsh + MEC Docker + Evaluation Board) |
| :--- | :--- | :--- |
| **End-to-End Detection Latency** | `1.0 – 2.0 seconds` | **`0.2 – 0.5 seconds` (Ultra-Low Latency)** |
| **Heart Rate Accuracy (MAE)** | `± 1.8 BPM` (Verified with Smartwatch) | `± 1.8 BPM` (Verified with Smartwatch) |
| **Respiration Rate Accuracy (MAE)** | `± 1.1 BrPM` | `± 1.1 BrPM` (98.4% Apnea Sensitivity) |
| ** Processed HUD Dashboard** | Accessible (`http://<ARDUINO_IP>:8080/`) | Accessible (`http://<ARDUINO_IP>:8080/`) |
| **ThingsBoard 5G MEC Dashboard** | Local CSV Logging (`apnea_uno_q_5g_log.csv`) | **Live 5G Telemetry (`http://192.168.116.30:9090/`)** |

---

## 7. Conclusion
This project successfully demonstrates an end-to-end, contactless **5G Edge-AI Critical Apnea and Vital Sign Monitoring System**. Compared to Wi-Fi Proof of Concept trials (`1.0 – 2.0 s` latency), deploying over the **3GPP 5G Standalone MEC Server and 5G Evaluation Board (`QsoftAP`)** reduced end-to-end latency to **`0.2 – 0.5 seconds`** while keeping patient video private on the local edge server. Combined with the dual-processor **Arduino Uno Q** (Qualcomm Dragonwing QRB2210 MPU + STM32U585 MCU), the system delivers both instantaneous bedside audio-visual fail-safe alarms and centralized ThingsBoard hospital telemetry without a single wire touching the patient.

---

## 8. Acknowledgement
The authors express their sincere gratitude to **Prof. Salil Kashyap** and Teaching Assistants **Aditya Gupta, Abhay Saran, and Asif Hoque** for their guidance, support during preliminary experiments, and assistance with the 5G Usecase Laboratory infrastructure at IIT Guwahati.

---

## 9. References
1. **UBFC-rPPG Dataset:** S. Bobbia, R. Macwan, Y. Benezeth, A. Mansouri, and J. Dubois, "Unsupervised skin tissue segmentation for remote photoplethysmography," *Pattern Recognition Letters*, vol. 124, pp. 82–90, 2019. [https://sites.google.com/view/ybenezeth/ubfcrppg](https://sites.google.com/view/ybenezeth/ubfcrppg)
2. **rPPG-Toolbox:** X. Liu, J. Fromm, S. Patel, and D. McDuff, *rPPG-Toolbox: Deep Learning Toolbox for Remote Photoplethysmography*, GitHub Repository: [https://github.com/ubicomplab/rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox)
3. **Arduino Uno Q Datasheet:** Arduino s.r.l., *Arduino® UNO Q Dual-Processor Hardware Datasheet (ABX00162 / ABX00173)*: [https://docs.arduino.cc/resources/datasheets/ABX00162-ABX00173-datasheet.pdf](https://docs.arduino.cc/resources/datasheets/ABX00162-ABX00173-datasheet.pdf)
4. **5G Usecase Lab Manual:** Department of EEE, Indian Institute of Technology Guwahati, *5G Usecase Laboratory Manual (Course Code: EE-4032)* & 3GPP TS 23.501 System Architecture for the 5G System.
