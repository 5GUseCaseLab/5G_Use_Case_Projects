# Demonstration video

**File:** `WiFeye_demo.mp4` (1280×720, 6 min 18 s, with voice-over)

## Contents

| Section | What it shows |
|---|---|
| Opening | A live fall and the **FALL DETECTED** alarm on the dashboard |
| Title | Project, course and authors |
| Why WiFeye | Falls go unseen; cameras and wearables have drawbacks; Wi-Fi is already in the home |
| What the Wi-Fi sees | Real CSI from recorded trials: an empty room, walking, and a fall |
| Setup | Router (2.4 GHz, channel 6), two ESP32 receivers, two laptops |
| Pipeline | The seven processing steps, one at a time: acquire, clean, condition, localise, features, classify, decide |
| Dashboard | What each panel of the live dashboard means |
| **Live demo** | Two people perform the activities in front of the system, with the camera view and the live dashboard side by side: empty room, walking, sitting down and standing up (both "posture change"), a hand gesture, and a fall onto a mattress. Each run is continuous; where waiting time was shortened, a "skipped" marker is shown. Labels appear 3.5–5 s after a movement because the engine waits to see the whole event |
| Results | Leave-one-subject-out results (people the model never saw): fall AUC 0.48 → 0.78, 66.2 % accuracy over five classes, the fall-alarm trade-off, one vs two ESP32s |
| Future work | The pipeline on an Arduino UNO Q at the edge, with alerts over 5G (URLLC for fall alarms, mMTC for activity logs) |
| Conclusion | What was shown, and the limits: one room, five people, staged falls |

The title, diagram and result visuals were rendered by `Source_Code/make_video_assets.py` from the saved results; the live footage was recorded with the live engine (`run_live_master.bat` and `run_live_worker.bat`) and the browser dashboard.
