# How it works

## Pipeline and router
Camera thread keeps only the newest frame -> **frame-diff gate** (64x36 thumbnail vs last *processed* frame; skip if
diff < threshold, forced refresh every 0.5 s) -> **router** -> detector -> **tracker** (always on UNO Q) -> UI + MCU alert.
Router: remote_est = RTT + GPU time + JPEG encode; local_est = measured local inference. Auto mode switches to the laptop
only if remote_est < min(800 ms, 4 x local_est); back to local only if remote_est > 1.3 x that limit (hysteresis). RTT is pinged every 1 s while local, measured from real
frames while remote. Remote call has a hard timeout (3 x estimate, clamped to 0.6-2 s); on timeout that same frame is run locally.
Tracking, distance, speed and direction run on the UNO Q for both paths, so switching never resets IDs.

Distance = fusion of road-plane geometry (camera height + horizon) and object size (class heights in tracking.py).
Speed: per-track Kalman filter; see the motion section below for what the laptop adds. Calibrate HORIZON / CAM_H / HFOV / FOE_X ([calibration.md](calibration.md)).

## Laptop motion analysis (`laptop/motion.py`, on by default)
For every frame pair from the board (protocol v2, negotiated automatically; old boards/servers still work):
* **Ego-motion**: dense optical flow (DIS on CPU in parallel with YOLO, or RAFT on CUDA) on the road pixels, fitted with
  a ground-plane + rotation model -> own speed, yaw rate (turning), pitch rate. Pitch is integrated into a dynamic
  horizon, so braking dives and bumps no longer throw distances off. `--auto-horizon` also lets it re-learn the
  static horizon/FOE slowly (off by default: on slow curvy roads the flow fit is biased; the UI shows "Flow horizon"
  as a suggestion either way).
* **Looming** per box: the scale-change rate of the object = 1/TTC, with no calibration at all. The tracker turns it
  into a closing-speed measurement, so in the self-test closing speed settles in ~0.4 s instead of ~0.8 s from differencing distances.
* **Rotation-free image motion** per box -> lateral speed.
* With own speed + yaw known, the tracker's Kalman filters run in ground coordinates (own motion is a control input):
  each object gets speed over ground, heading, and a label **same dir / oncoming / crossing L->R / R->L / stopped**.
* `--depth` (experimental): Depth Anything V2 metric depth per box, re-scaled every frame to the road plane, fused as a
  third distance estimate. Needs `pip install transformers` and a model download.
Self-tests with rendered roads and known truth (no GPU needed): `cd laptop && python selftest_motion.py` and
`python selftest_tracking.py`. Motion works best when the laptop gets >= 10 frames/s from the board.
