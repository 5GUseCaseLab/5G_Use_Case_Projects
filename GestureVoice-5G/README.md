# GestureVoice-5G: Real-Time Sign-to-Speech Translation

**Mir Maiti, Pranav Vishal Deshmukh**, Department of Electronics and Electrical Engineering, IIT Guwahati

GestureVoice-5G turns hand signs into speech in real time. An Arduino UNO Q reads
a CCTV camera, extracts hand landmarks with MediaPipe and sends only those
landmarks, never video, to a multi-access edge computing (MEC) server. A small
GRU classifier on the MEC recognises the sign, and the result is spoken and
shown on a live web page.

📄 **Poster:** [poster/GestureVoice5G_A1_Poster.pdf](poster/GestureVoice5G_A1_Poster.pdf)

## Demo

https://github.com/user-attachments/assets/00a1c049-2a2a-4ff8-87cf-f694a476da3b

## How it works

```
 IP CCTV camera ──RTSP──▶ Arduino UNO Q ──Wi-Fi──▶ 5G CPE ──Ethernet──▶ MEC server
   (15 fps)               MediaPipe hands          (local network)       GRU classifier
                          42 landmarks/frame                               (PyTorch, CPU)
                          568 bytes, no video                                    │
                                ▲                                                │
                                └──────────── sign + confidence ◀────────────────┘
                                                                                 │
                                                  live web viewer (any browser) ◀┘
```

## Results

| Metric | Result |
|---|---|
| Accuracy, 411 held-out validation clips | **96.8%** (398 / 411) |
| Macro F1, 27 signs | **95.8%** |
| Main confusion | "eat" predicted as "food" in 5 of 10 clips; every other error occurs once |
| Mean round trip, UNO Q → 5G CPE local network → MEC (incl. inference) | **34.2 ms** (n = 343) |
| Mean round trip, UNO Q → campus network → laptop (baseline) | 165.9 ms (n = 348) |
| Model inference on the MEC (CPU) | 6–27 ms per prediction |
| Frames processed on the UNO Q (decode + hand tracking + send) | ≈ 2.5 fps |
| Payload | 568 bytes per frame, about 50× smaller than a JPEG frame |

The raw numbers behind the poster figures are in [results/](results/).

**Read these results with their limits:**

- **The two round-trip figures come from different setups:** different servers
  (MEC on CPU vs a MacBook) and different days. They are not a pure network
  comparison.
- **The validation clips come from the same signers and sessions as the
  training clips**, and were also used to pick the best epoch. Accuracy for new
  signers will likely be lower.
- **The UNO Q processes about 2.5 fps, against 30 fps in the training data**, so
  each 45-frame window covers about 18 s of motion instead of 1.5 s. This is the
  main trade-off and hurts fast signs.
- End-to-end delay from sign to speech was not measured.

## Vocabulary (27 signs)

bathroom, come, drink, eat, food, friend, go, hello, help, home, how, i, like,
my, need, no, school, stop, thankyou, want, water, we, what, where, who, yes, you

## Repository layout

```
GestureVoice-5G/
├── poster/                  A1 poster (PDF)
├── results/                 confusion matrix, per-sign metrics, round-trip and frame-rate data
├── signbridge/
│   ├── capture/             MediaPipe landmark source: webcam or RTSP camera; clip replay
│   ├── host/                TCP client, text-to-speech, MJPEG camera preview
│   ├── server/              TCP server, inference sessions, live web viewer
│   ├── model/               GRU, MLP and Transformer classifiers
│   ├── training/            dataset loading and training loop
│   └── cli/                 collect, train, serve, run_host and other commands
├── data/train/              2,089 landmark clips (.npz), one folder per sign
├── models/                  trained classifier (best.pt), its metrics, MediaPipe models
├── deploy/unoq/run_live.sh  keeps the UNO Q streaming the camera to the MEC
├── web/index.html           optional browser client (laptop webcam → WebSocket bridge)
├── tests/                   pytest suite (109 tests)
└── config.yaml              all settings in one place
```

The software package is called `signbridge`, the project's working name.

## Quick start

You need Python 3.11. The two ends have separate requirements, so each machine
installs only what it uses.

### 1. MEC server

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-server.txt
python -m signbridge.cli.serve --config config.yaml --view-port 8090
```

The server listens for capture devices on TCP port 9009 and serves the live web
viewer on port 8090. It never imports MediaPipe or OpenCV, because it only ever
receives landmarks.

### 2. Capture device

On a laptop with a webcam:

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-host.txt
python -m signbridge.cli.run_host --server <mec-ip>:9009 --video-port 8081
```

From an IP camera, pass the stream URL:

```bash
python -m signbridge.cli.run_host --server <mec-ip>:9009 \
    --camera rtsp://<camera-ip>:554/h264/ch1/sub/av_stream --video-port 8081 --no-preview
```

Without a camera, replay recorded clips instead:

```bash
python -m signbridge.cli.run_host --server <mec-ip>:9009 --source replay --clips data/train/hello
```

`--video-port` is optional. It lets the web viewer show the camera feed, which
goes straight from the capture device to the browser and never passes through
the MEC. Frames are only encoded while someone is watching.

### 3. Live web viewer

Open `http://<mec-ip>:8090` from any laptop or phone on the same network. It
shows the camera (as video, or as the hand skeleton the model receives), the
recognised sign and its confidence, a running caption, frame rate and model
time. **Speak words** reads each new sign aloud. The page loads nothing from the
internet, so it also works on an isolated lab network.

### Check the network path first

To test the link without the trained model:

```bash
python -m signbridge.cli.serve --smoke-test
python -m signbridge.cli.run_host --source synthetic --count 150
```

## Running on the Arduino UNO Q

The UNO Q (Qualcomm QRB2210, 4× Cortex-A53) needs two adjustments:

- **Use MediaPipe 0.10.9 on Python 3.11.** Newer MediaPipe aarch64 wheels use CPU
  instructions the Cortex-A53 lacks and crash with `SIGILL`. Install
  `mediapipe==0.10.9`, `numpy==1.26.4` and `opencv-contrib-python==4.10.0.84`
  into a Python 3.11 virtual environment.
- **Use the camera's sub-stream.** The board decodes the 1280×720 sub-stream at
  about 20 fps but the 4K main stream at only 8.7 fps. Streams are read over TCP,
  and a background thread always hands MediaPipe the newest frame, so the board
  never falls behind the camera.

[deploy/unoq/run_live.sh](deploy/unoq/run_live.sh) keeps the board streaming.
It waits until the MEC and the camera answer, starts the capture host, logs
predictions to `~/live.log`, and restarts after any drop. Edit the addresses at
the top, then run:

```bash
nohup bash deploy/unoq/run_live.sh > /dev/null 2>&1 &
tail -f ~/live.log
```

## Signing a sentence

The system recognises one sign at a time and adds a word to the caption when the
recognised sign changes. Sign in sign order (for example HELLO · HOW · YOU, or
BATHROOM · WHERE), hold each sign for about 5 seconds at the UNO Q's frame rate,
and drop your hands for about 2 seconds between signs.

## Record data and retrain

```bash
python -m signbridge.cli.collect --label hello --samples 30
python -m signbridge.cli.train --config config.yaml
python precision_recall.py --metrics models/metrics.json
```

`collect` records 45-frame clips of landmarks only into `data/train/<label>/`;
no video is saved. `train` holds out a stratified 20% validation split, trains
the GRU and writes `models/best.pt` and `models/metrics.json`.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The suite covers the wire protocol, features, sessions, the TCP and WebSocket
servers, the web viewer, the camera preview and the camera helpers. It also
checks that the server never imports a camera stack and the capture device never
imports PyTorch.

## Future work

- Run end to end through the 5G core and the UPF.
- Measure the end-to-end delay from sign to speech.
- Evaluate on signers held out from training.
- Raise the capture rate on the UNO Q, or make the model's window time-based,
  so recognition holds up at low frame rates.
- Turn the sequence of signs into grammatical sentences.

## Authors and contributors

- **Authors:** Mir Maiti, Pranav Vishal Deshmukh
- **Contributors:** Arjav Sethi, Vishal Shashidhar, Abhay Sharan, Asif Hosque
- **Training data:** clips recorded by the team, plus clips merged from
  sarbaniAi/mir-gesture-train-data

## Acknowledgement

We sincerely thank Prof. Salil Kashyap for teaching the course and for his
guidance, encouragement and valuable feedback throughout this project. We also
thank the Department of Electronics and Electrical Engineering, IIT Guwahati, for
the laboratory facilities used in this work.

## License

GNU General Public License v2.0. See [LICENSE](LICENSE).
