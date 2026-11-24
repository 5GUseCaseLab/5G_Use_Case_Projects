# WiFeye: Device-Free Fall & Activity Detection with Two ESP32 Wi-Fi Receivers

**EE 4032: 5G Wireless Networks, course project**
Devansh Jangid (230102027) · Debasis Patnaik (230102025)
Department of EEE, Indian Institute of Technology Guwahati

---

## 1. What this project does

WiFeye detects **falls and daily activities without a camera and without anything worn**. An ordinary 2.4 GHz Wi-Fi router transmits as usual. Two low-cost ESP32 boards record the router's **channel state information (CSI)**: the amplitude of 64 OFDM subcarriers for every packet. A person moving in the room changes how the signal reaches each receiver. A cleaned, event-based pipeline turns those changes into one of five classes: **fall, walk, posture change (sitting down or standing up), hand gesture, empty room**. A fall raises an alarm.

All results below come from leave-one-subject-out testing (LOSO), so every number is measured on a person the model never saw:

| Result | Value |
|---|---|
| Fall-vs-activity AUC: original pipeline → cleaned data → adopted system | **0.48 → 0.71–0.73 → 0.78** |
| Adopted system, 5 classes (chance 20%) | **66.2% accuracy, macro-F1 0.66** |
| Gain from the accepted CSI domain features + two heads | 66.2% vs 63.5%: 26 trials better, 12 worse. McNemar p = 0.034 (0.10 after Holm over 3 tests). ΔF1 95% CI +0.014 to +0.067 |
| Fall alarm, threshold chosen inside each training fold | 2.8% false alarms → 38% of falls (95% CI 28–49%); 8.8% → 48% |
| One ESP32 vs two | Two receivers ≈ the better single receiver (macro-F1 0.66 vs 0.65, p = 0.74) and far above the weaker one (0.54, p < 0.001). Placement matters more than count |
| Synthetic falls for the subject who recorded none | They only shift the threshold, so they were not adopted |
| Subject-wise feature pruning; per-person calibration | No significant gain, so neither was adopted |
| Live engine vs offline pipeline | Identical decisions on 60/60 recorded trials |

The poster is in `Poster/`, the demonstration video in `Demo_Video/`. The methodology notes and the literature survey are in `Source_Code/docs/`.

## 2. Repository layout

```
WiFeye_.../
├── README.md                  this file
├── Poster/                    WiFeye_poster_A1.pdf (print-ready A1) + preview
├── Demo_Video/                demonstration video (see Demo_Video/README.md)
└── Source_Code/
    ├── requirements.txt
    ├── firmware/ESP32-CSI-Tool/      ESP32 firmware (ESP-IDF v4.3; active_sta used as receiver)
    │   -- acquisition and cleaning
    ├── dataset_collector.py          synchronized 2-laptop CSI data collection over the router's Wi-Fi
    ├── merge_dual_rx.py              packet filter (router→ESP32 frames) + 100 Hz alignment of RX1 and RX2
    ├── csi_preprocess.py             Hampel + Butterworth + per-subcarrier median removal
    ├── run_pipeline.py               one command: merge → preprocess → window features (v1 / v2 / v3)
    │   -- features
    ├── feature_extraction.py         2 s window features (per receiver)
    ├── event_localizer.py            fused, reliability-weighted, gap-masked event localiser
    ├── event_features.py             event-segment features (shared with the live engine)
    ├── build_fused_events.py         localiser noise reference + event features for every trial
    ├── domain_features.py            CSI domain features (speed spectrogram, PCA, ACF, wavelet, Fresnel, RSSI, DTW)
    ├── eda_features.py, eda_domain_features.py, acceptance_domain.py   bottom-up feature analysis + acceptance test
    │   -- validation
    ├── train_classifier.py           window-level LOSO / grouped-CV evaluation
    ├── evaluate_event_model.py       event models, paired tests, per-person calibration curve
    ├── evaluate_localizer.py, localizer_ablation.py                     localiser evaluation
    ├── validate_features.py, validate_twohead.py, validate_adopted.py   feature / two-head / adopted-system LOSO
    ├── model_audit.py, diagnostics_ceiling.py                           model + fusion audit, accuracy-ceiling study
    ├── compare_receivers.py          1 vs 2 ESP32 (earlier model; the adopted one is in validate_adopted.py)
    ├── fall_operating_point.py       nested-LOSO fall threshold + ROC (earlier model)
    ├── augment_csi.py, missing_cell_experiment.py   subject-3 augmentation test
    ├── calibrate_tiers.py            Tier-1/2 gate calibration
    │   -- deployment
    ├── train_final.py                trains and packages the deployed two-head model
    ├── live_demo_engine.py           live 2-ESP32 demo (master / worker), replay, LOSO replay, self-test
    ├── ui_dashboard.py, dashboard.html   browser dashboard for the live engine (standard library only)
    ├── serial_proc.py                ESP32 reader + traffic pump in a separate process (master side)
    ├── make_poster.py                regenerates all figures and the poster from saved results
    ├── make_video_assets.py          renders the title, diagram and result clips/cards used in the demo video
    ├── test_suite.py                 end-to-end pipeline test with synthetic data
    ├── run_live_master.bat, run_live_worker.bat, run_replay_demo.bat
    ├── run_collector_master.bat, run_collector_worker.bat
    ├── data/                         deployed model (6.1 MB), feature tables, saved results, 11 sample trials
    └── docs/                         research reports (methodology, survey)
```

## 3. Overall workflow

```
 Router (TX, 2.4 GHz ch 6) ──Wi-Fi frames──►  ESP32 RX1 ──USB──► Laptop A (master)
                           ──Wi-Fi frames──►  ESP32 RX2 ──USB──► Laptop B (worker)
                                                         Laptop B ──UDP over the router's Wi-Fi──► Laptop A

 1. Data collection   dataset_collector.py (master + worker, NTP-style clock sync, scheduled 10 s trials)
 2. Cleaning (v3)     keep only router→ESP32 HT frames; drop subcarriers 0 and 27–37 (52 kept)
 3. Conditioning      one 100 Hz grid → Hampel → 10 Hz zero-phase low-pass → per-subcarrier median removal
 4. Localisation      per-receiver motion curves, packet-gap masking, soft time prior; receivers weighted by
                      how far their burst rises above their empty-room noise level → onset / peak / offset
 5. Features          event-segment + window features (49 per receiver) + 42 CSI domain features that passed
                      a bottom-up acceptance test (separability, subject-independence, consistency, redundancy)
 6. Two heads         fall head and activity head: one random forest per receiver each, late fusion;
                      P(fall) from the fall head, the other classes rescaled from the activity head
 7. Decision          FALL iff P(fall) ≥ τ (τ for ~3% false alarms, nested LOSO), else the most likely activity
 8. Live demo         live_demo_engine.py runs steps 2–7 every second on a rolling 10 s buffer
```

## 4. Requirements

**Software**

- Windows 10/11 and Python 3.11. The project was developed in the ESP-IDF Python environment `D:\Espressif\IDF\python_env\idf4.3_py3.11_env`; any Python 3.11 works.
- Install the Python packages:
  ```
  cd Source_Code
  pip install -r requirements.txt
  ```
- To build or flash the firmware you need ESP-IDF v4.3 (see `firmware/ESP32-CSI-Tool/README.md`). It is only needed for new hardware. Before building `active_sta`, put your router's Wi-Fi password in `active_sta/main/main.cc` (`ESP_WIFI_PASS`) and `active_sta/sdkconfig.defaults`; the repository contains the placeholder `YOUR_WIFI_PASSWORD`.
- Optional, only for `make_video_assets.py`: `pip install imageio-ffmpeg`.

**Hardware (live demo / data collection)**

- TP-Link TL-WR845N router (SSID `CSI`, 2.4 GHz channel 6, 20 MHz).
- 2 × ESP32-WROOM-32 flashed with `active_sta` (921 600 baud, FreeRTOS tick 1000 Hz).
- 2 laptops, both connected to the Wi-Fi `CSI`.

## 5. How to run

Run all commands from `Source_Code/`.

### 5.1 Quick checks (no hardware)
```
python test_suite.py                          # end-to-end pipeline test on synthetic data
python live_demo_engine.py --selftest 11      # live code path vs offline pipeline on the sample trials
run_replay_demo.bat                           # replays one recorded trial per activity in the browser dashboard
run_replay_demo.bat trial 1_fall_t06          # one specific recorded trial (here: a fall)
python make_poster.py                         # regenerates data/visualizations/* and reports/WiFeye_poster_A1.pdf
```

### 5.2 Live demo with two ESP32s
1. Power the router. Connect **both laptops to the Wi-Fi `CSI`**, exactly as during data collection.
2. Plug ESP32 RX1 into Laptop A and ESP32 RX2 into Laptop B. The COM port is detected automatically.
3. On **Laptop B**, run `run_live_worker.bat`. It announces itself on the Wi-Fi every second and waits for the master.
4. On **Laptop A**, run `run_live_master.bat`. The two laptops find each other automatically (no IP addresses needed); the worker's console then shows `master: 192.168.0.x` and the master prints `Worker heard at …`. If they do not, check that both are on Wi-Fi `CSI`, or start the master with `--worker-ip <Laptop B's IP>`.
5. A browser dashboard opens at `http://127.0.0.1:8050/` (press F11 for full screen). Wait about 12 s while the 10 s buffer fills; the room does not need to be empty.
6. Perform activities. The dashboard shows:
   - the current detection in large type, with a **FALL DETECTED** banner and an audio alarm about 3.5–5 s after a fall;
   - P(fall) against the alarm threshold τ, and the class probabilities of the last event;
   - packets/s, online status and a live CSI waterfall (52 subcarriers) for each receiver;
   - a live motion trace and a timeline of recent events.

   If Laptop B stops, the engine continues on RX1 alone. The console dashboard still runs behind the browser; add `--no-display` to turn it off, or leave out `--ui` to use the console only.

Options:
- `run_live_master.bat single` uses one ESP32 only.
- `--fall-far 5` or `--fall-far 10` gives a more sensitive alarm with more false alarms.
- `--tier1` turns on the power-saving Tier-1 gate. It loses about 5% of falls and often skips steady walking, so it is off by default.

### 5.3 Data collection (to record new trials)
- Laptop B: `run_collector_worker.bat`
- Laptop A: `run_collector_master.bat`

Both laptops must be on Wi-Fi `CSI`. Trials are saved to `data/raw/<activity>/<subject>_<activity>_t<NN>_RX1|RX2.csv`.

### 5.4 Reproducing all results (needs the full dataset)
The complete dataset (1 018 raw CSV files, about 0.9 GB) is too large for this repository and is available from the authors. Place it in `data/raw/`, then run:
```
python run_pipeline.py --version v3                 # merge, preprocess, window features (RX1, RX2, both)
python event_features.py --in-dir data/processed_v3 --out data/event_features_v3.csv
python build_fused_events.py                        # fused localiser: data/event_features_v3_fused.csv
python eda_domain_features.py                       # CSI domain features + acceptance test
python validate_adopted.py                          # LOSO of the adopted system, pruning, calibration, 1 vs 2 ESP32
python train_final.py --version v3                  # writes data/models/safesense_live_bundle.pkl (two-head)
python make_poster.py
```
The earlier experiments shown on the poster (receiver comparison of the first model, augmentation, tiers) are reproduced with:
- `train_classifier.py`
- `compare_receivers.py`
- `evaluate_event_model.py`
- `fall_operating_point.py`
- `missing_cell_experiment.py`
- `calibrate_tiers.py`

`python live_demo_engine.py --loso-subject 4` gives an honest out-of-sample demo. It rebuilds the model without subject 4 (about a minute) and replays that subject's trials through the live engine. With the full dataset this gives 45% accuracy for subject 4, the hardest person; the offline LOSO result for the same subject is 49%.

## 6. Notes and limitations

- Results are **cross-subject on 5 subjects in one room**, with staged falls onto a mattress and no fall look-alikes such as lying down or sitting on the floor. They are not person-independent. Fall detection (38% of falls at 2.8% false alarms) is far from a ≥95% / ≤3% target.
- Raw ESP32 captures mix CSI from several transmitters. Always use the v3 packet filter (`merge_dual_rx.py --source-filter own`).
- Sit and stand are merged into "posture change" because amplitude-only CSI barely tells their direction apart; all sit and stand recordings are kept.
- ESP32 firmware: [ESP32-CSI-Tool](https://github.com/StevenMHernandez/ESP32-CSI-Tool) by S. M. Hernandez (MIT licence, see `firmware/ESP32-CSI-Tool/LICENSE`), with a modified send-rate limiter.

## 7. Use cases and future plans

- **Use cases:**
  - elderly people living alone, with an instant alert to family or a carer;
  - hospital and care-home wards, for fall and bed-exit watch;
  - bathrooms and bedrooms, where cameras are unacceptable;
  - smart homes, for occupancy, activity and rehabilitation logs.
- **Edge deployment:** run the whole pipeline on an **Arduino UNO Q**. Its STM32U585 microcontroller reads the ESP32 CSI stream, and its Qualcomm QRB2210 Linux processor runs preprocessing, the localiser and the forests. No laptops are needed, and the Tier-1 gate saves power.
- **5G alerting:** send fall alarms and activity summaries over a 5G modem, matched to the application:
  - low-latency emergency alerts (URLLC) to a carer's phone or a nurse station;
  - low-rate reports from many rooms (mMTC) to a care dashboard.
- **Data:** more people, rooms and fall look-alikes; CSI phase or CSI-ratio features; a few labelled trials per new home.

## 8. References
1. S. M. Hernandez, E. Bulut, "Lightweight and standalone IoT based WiFi sensing for active repositioning and mobility" (ESP32-CSI-Tool), IEEE WoWMoM, 2020.
2. W. Wang et al., "Understanding and modeling of WiFi signal based human activity recognition" (CARM), ACM MobiCom, 2015.
3. Y. Wang, K. Wu, L. M. Ni, "WiFall: Device-free fall detection by wireless networks," IEEE Trans. Mobile Computing, 2017.
4. Y. Hu et al., "DeFall: Environment-independent passive fall detection using WiFi," IEEE Internet of Things Journal, 2022.
5. D. Zhang, H. Wang, D. Wu, "Toward centimeter-scale human activity sensing with Wi-Fi signals," IEEE Computer, 2017.
6. J. Yang et al., "SenseFi: A library and benchmark on deep-learning-empowered WiFi human sensing," Patterns, 2023.
7. S. Kapoor, A. Narayanan, "Leakage and the reproducibility crisis in ML-based science," Patterns, 2023.
8. L. Breiman, "Random forests," Machine Learning, 2001.
9. Arduino, "UNO Q" board documentation, 2025.
