# CrowdGate

**Real-time crowd-density monitoring and automatic gate control on a 5G edge server (MEC)**

EE4032: 5G Use-Case Lab course project, Department of Electronics and Electrical Engineering, Indian Institute of Technology Guwahati.

|            |                                                      |
| ---------- | ---------------------------------------------------- |
| Team       | Aditya Jain (230108002), Anisha Majumder (230102006) |
| Supervisor | Dr. Salil Kashyap                                    |

---

## 1. Project description

Overcrowding at gates and halls is hard to notice early, and counting people by hand is slow. CrowdGate watches a floor zone
with a Sparsh 5G camera, counts the people standing inside it with a fine-tuned YOLO11s model running on the lab's MEC
(Multi-access Edge Computing) server, and divides the count by the real floor area of the zone to get a density in
people per m². The density sets a safety state, and an Arduino UNO Q gate reacts to it automatically.

| State                    | Density (people/m²) | Gate action                                       |
| ------------------------ | ------------------- | ------------------------------------------------- |
| GREEN                    | below 2             | green LED, barrier open                           |
| YELLOW                   | 2 to 4              | yellow LED blinks, barrier open                   |
| RED                      | above 4             | red LED, buzzer, barrier closed                   |
| CAMERA_FAILURE / no data | -                   | fail-safe: red LED blinks, buzzer, barrier closed |

For the lab demo the YELLOW and RED limits were lowered (`--yellow 0.25 --red 0.55`) so that one or two people can trigger them.
The real limits are 2 and 4 people/m².

## 2. Overall workflow

```
Sparsh 5G camera --RTSP--> MEC server --------------------------------------------> web dashboard (:5000)
                           1. read newest frame                                        ^
                           2. YOLO11s (OpenVINO, CPU) finds people                     |
                           3. keep people whose feet are inside the calibrated zone    |
                           4. density = people / zone area (m2), smoothed, hysteresis  |
                           5. state GREEN / YELLOW / RED / CAMERA_FAILURE  --> GET /status (JSON)
                                                                                       |
                                                    Arduino UNO Q gate (polls once per second over Wi-Fi)
                                                    LEDs + buzzer + servo barrier, fail-safe after 5 s without data
```

1. **Calibrate (once per camera position).** Four printed ArUco markers of known size are placed on the floor. A homography
   (a 3x3 matrix that maps image pixels on the floor to metres) gives the real area of the zone.
2. **Count.** The MEC reads the camera stream, detects people with YOLO11s and keeps those standing inside the zone.
3. **Decide.** Density = people / area. The count is smoothed over about 10 frames and the state uses hysteresis so it does not flicker.
4. **Publish.** The state is shown on a live web dashboard and served as JSON at `/status`.
5. **Act.** The UNO Q reads `/status` every second and drives the LEDs, buzzer and barrier. If no valid reply arrives for 5 s, or the
   camera stops delivering frames for 3 s, the gate goes to the fail-safe state.

## 3. Repository structure

```
CrowdGate/
  README.md              this file (a copy is also kept in README/)
  README/                README file
  source_code/           all code
    crowdgate_mec.py       main service on the MEC: YOLO -> zone -> density -> dashboard + /status
    common.py              geometry (homography, area, zone test) and the density state machine
    make_markers.py        prints ArUco markers at an exact size
    check_marker.py        grabs a camera frame and checks that the markers are detected
    calibrate_area.py      computes the zone area and writes the calibration file
    measure_distance.py    checks a calibration by measuring a known distance
    convert_crowdhuman.py, train_yolo_colab.py, COLAB_TRAINING.md   fine-tuning YOLO11s on CrowdHuman
    infer_server.py, latency_client.py     MEC vs cloud latency measurement
    run_tests.py, tests/   offline tests
    uno_q/                 gate: python/main.py (HTTP poller), sketch/ (microcontroller code), tools/fake_mec.py (test stand-in)
  model/                 best.pt, the fine-tuned YOLO11s weights
  calibration/           config_floor.json (3.33 m2 zone), config_markers.json (1.71 m2 marker rectangle), preview image
  results/               latency runs and the log of a live run
  poster/                A1 poster (PDF and PPTX)
  demo_video/            demonstration video
```

## 4. Requirements

**Hardware**

- Sparsh 5G camera (RTSP stream) and a 5G network, as in the lab
- MEC server (here: CPU-only 12-core Intel Xeon Gold 5317, Ubuntu 22.04)
- Arduino UNO Q, 3 LEDs (green, yellow, red) with 220 to 330 ohm resistors, a buzzer module, a servo (barrier), jumper wires, breadboard
- Wi-Fi shared by the MEC and the UNO Q (the lab's 5G CPE is enough, internet is not needed)
- 4 printed ArUco markers (DICT_4X4_50, ids 0 to 3), about 16.7 cm

**Software**

- Python 3.10 or newer
- Python packages (see `source_code/requirements.txt`): ultralytics, opencv-contrib-python, numpy, flask, requests, pillow, openvino
  (`paho-mqtt` only if you want the optional MQTT output)
- Arduino App Lab on a laptop for the UNO Q, with the library `Arduino_RouterBridge` (and `Servo` if your board has it)
- The UNO Q Python script uses only the standard library

## 5. Setup

```
cd source_code
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_tests.py                  # all tests should print PASS
```

## 6. How to run

### A. Quick check without any hardware

Use any short video of people walking and a manual area in m²:

```
cd source_code
python crowdgate_mec.py --source people.mp4 --area 20 --model yolo11s.pt --imgsz 640 --port 5000
```

Open `http://localhost:5000` for the dashboard and `http://localhost:5000/status` for the JSON state.

### B. Calibrate the floor zone (with the camera)

1. Print the markers: `python make_markers.py --ids 0 1 2 3 --size-cm 17 --paper A4` (print at 100 percent, then measure the black square).
2. Tape the four markers flat on the floor as a rectangle and measure its width W and height H with a tape.
3. Check that all four are detected: `python check_marker.py --password YOUR_CAMERA_PASSWORD --stream 0`
4. Compute the area from a saved frame `frame.jpg`:

```
python calibrate_area.py --source frame.jpg --method aruco-click --marker-m 0.167 \
       --ref "0:0,0 1:W,0 2:W,H 3:0,H" --out config_floor.json
```

(replace W and H with the measured metres; with `--method aruco-click` you click the corners of the floor area to monitor, then press Enter.
Use `--method aruco` to take the marker rectangle itself as the zone.) Ready-made results are in `calibration/`.

### C. Run the live system on the MEC

```
cd source_code
export CAMERA_URL="rtsp://USER:PASSWORD@192.168.128.10:554/avstream/channel=1/stream=0.sdp"
python crowdgate_mec.py --config ../calibration/config_floor.json --model ../model/best.pt \
       --openvino --imgsz 640 --port 5000
```

- Dashboard: `http://MEC_IP:5000` State as JSON: `http://MEC_IP:5000/status`
- Demo thresholds: add `--yellow 0.25 --red 0.55`
- Camera failure: unplug the camera. After 3 s the state becomes CAMERA_FAILURE (purple on the dashboard).
- The first `--openvino` run exports the model to OpenVINO format automatically.

### D. Run the gate (Arduino UNO Q)

Wiring: green LED D4, yellow LED D5, red LED D6 (each through a resistor to GND), buzzer module I/O D7 (VCC 3.3 V, GND),
servo signal D9 (power 5 V, GND).

1. Test the parts: open `source_code/uno_q/sketch/test_parts.ino` in Arduino App Lab and run it. The LEDs, buzzer and servo should cycle.
2. Create an App Lab app. Put `source_code/uno_q/sketch/sketch.ino` into its sketch folder (use `sketch_no_servo_lib.ino` if `Servo.h` is missing)
   and `source_code/uno_q/python/main.py` into its python folder. Add the library `Arduino_RouterBridge`.
3. Join the UNO Q and the MEC to the same Wi-Fi. In `main.py` set `MEC_HOST` to the MEC's IP address (default `192.168.116.30`).
4. Run the app. The console prints `state -> ...` whenever the state changes.
5. To test the gate without the camera or the MEC, run `python source_code/uno_q/tools/fake_mec.py` on a laptop on the same Wi-Fi, set `MEC_HOST`
   to that laptop's IP, and type G, Y, R or C to simulate GREEN, YELLOW, RED and CAMERA_FAILURE.

### E. Optional: MEC vs cloud latency

Start `python infer_server.py --model best.pt --openvino --port 8000` on the MEC and on a cloud machine (copy `common.py` next to it), then
run `python latency_client.py --image frame.jpg --url MEC=http://MEC_IP:8000 --url CLOUD=http://CLOUD_HOST:8000 --n 100`.

### F. Optional: retrain the model

`source_code/COLAB_TRAINING.md` explains how `model/best.pt` was fine-tuned on CrowdHuman in Google Colab.

## 7. Results

- **Model:** YOLO11s fine-tuned on CrowdHuman (500 validation images): mAP50 0.732 to 0.811, recall 0.629 to 0.708 compared with the stock model.
- **Floor area:** the ArUco calibration measured 1.71 m² for the marker rectangle against 1.735 m² with a tape (1.4 % error).
  The click-defined demo zone is 3.33 m².
- **Latency** (same image, request to response, 2 runs of 100 requests): MEC mean 94.9 ms (p95 110.0 ms), cloud CPU instance mean
  1298.6 ms (p95 1535.2 ms), about 13.7 times faster at the edge. Raw numbers are in `results/`.
- **Speed:** about 11 to 19 AI frames per second on the CPU-only MEC with OpenVINO at `--imgsz 640`.
- **Fail-safe:** a frozen or unplugged camera gives CAMERA_FAILURE within 3 s; the gate also goes to the safe state if the MEC does not answer for 5 s.

## 8. Limitations

- The camera is mounted low, so people overlap and counts fall in dense groups.
- The wide-angle lens curves the picture edges; keep the zone in the central part of the image.
- The demo thresholds are scaled down; the real safety limits are 2 and 4 people/m².
- The UNO Q Python code follows the Arduino App Lab examples; check the Bridge calls against your App Lab version.

## 9. Credits

YOLO11 (Ultralytics), CrowdHuman dataset, OpenCV (ArUco), Intel OpenVINO, Arduino UNO Q and App Lab.
Thanks to Dr. Salil Kashyap and the 5G Use-Case Lab, IIT Guwahati, for the MEC server, the Sparsh 5G camera and the Kenstel 5G CPE.
