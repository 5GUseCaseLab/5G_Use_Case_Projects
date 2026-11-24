"""Self-test for motion.py on a rendered road with KNOWN ground truth (no GPU / no model needed).

    python selftest_motion.py            # prints estimated vs true ego speed, yaw rate, horizon, TTC

A textured ground plane + far backdrop is rendered from a camera at height h with a given HFOV and pitch,
driving at V with a yaw rate and a pitch bounce; one 'car' (textured billboard) ahead closes in.
The estimator starts with a deliberately wrong horizon to show that it learns the real one."""
import math, sys
import cv2
import numpy as np
from motion import MotionEstimator, FlowEngine

W, H = 480, 270
HFOV, CAM_H, PITCH0 = 68.0, 1.55, math.radians(4.0)       # camera tilted 4 deg down -> horizon above centre
FPS, V, YAW_RATE = 15.0, 11.0, math.radians(6.0)          # 39.6 km/h, gentle right turn
CAR_Z0, CAR_V = 22.0, 4.0                                  # car 22 m ahead driving 4 m/s -> closing 7 m/s
f = (W / 2) / math.tan(math.radians(HFOV) / 2)
rng = np.random.default_rng(1)
TEX = cv2.GaussianBlur((rng.random((2048, 2048)) * 255).astype(np.float32), (0, 0), 2.0)
TEX = cv2.normalize(TEX, None, 0, 255, cv2.NORM_MINMAX)
CARTEX = cv2.resize((rng.random((24, 40)) * 200 + 30).astype(np.float32), (400, 240), interpolation=cv2.INTER_NEAREST)
PPM = 60.0                                                  # texture pixels per metre


def render(cxw, cz, yaw, pitch, car_d):
    ss = 2
    xs, ys = np.meshgrid((np.arange(W * ss) + 0.5) / ss - W / 2, (np.arange(H * ss) + 0.5) / ss - H / 2)
    dx, dy, dz = xs / f, ys / f, np.ones_like(xs)
    cp, sp = math.cos(pitch), math.sin(pitch)                # camera nose down by pitch
    dy, dz = dy * cp + dz * sp, -dy * sp + dz * cp
    lx, lz = dx, dz                                          # level camera frame (car lives here)
    cy_, sy_ = math.cos(yaw), math.sin(yaw)                  # heading to the right by yaw
    dx, dz = dx * cy_ + dz * sy_, -dx * sy_ + dz * cy_
    t = np.where(dy > 1e-6, CAM_H / np.maximum(dy, 1e-6), np.inf)
    ground = t < 120
    gx, gz = t * dx + cxw, t * dz + cz
    ang = np.arctan2(dx, dz); el = np.arctan2(-dy, np.hypot(dx, dz))
    mx = np.where(ground, gx * PPM, ang * 800 + 5000)
    my = np.where(ground, gz * PPM, el * 800 + 3000)
    img = cv2.remap(TEX, (mx % 2048).astype(np.float32), (my % 2048).astype(np.float32), cv2.INTER_LINEAR)
    # car: vertical billboard 1.8 m wide, 1.5 m tall, standing on the road at depth car_z (world), lane centre
    # (rendered in camera coords: transform its plane intersection)
    car_w, car_h = 1.8, 1.5
    tz = np.where(lz > 1e-6, car_d / lz, np.inf)             # plane at depth car_d straight ahead
    hx, hy = tz * lx, tz * dy
    hit = (np.abs(hx) < car_w / 2) & (hy > CAM_H - car_h) & (hy < CAM_H) & (tz < t)
    u = ((hx + car_w / 2) / car_w * 399).clip(0, 399).astype(np.float32)
    v = ((hy - (CAM_H - car_h)) / car_h * 239).clip(0, 239).astype(np.float32)
    carimg = cv2.remap(CARTEX, u, v, cv2.INTER_NEAREST)
    img = np.where(hit, carimg, img)
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    ys_, xs_ = np.nonzero(cv2.resize(hit.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST))
    box = None if len(xs_) == 0 else ["car", 0.9, xs_.min() / W, ys_.min() / H, (xs_.max() + 1) / W, (ys_.max() + 1) / H]
    return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR), box


def main(kind="dis"):
    est = MotionEstimator(FlowEngine(kind), auto_horizon=True)
    true_h = 0.5 - f * math.tan(PITCH0) / H
    est.set_cal(dict(hfov=HFOV, cam_h=CAM_H, horizon=true_h + 0.04, foe_x=0.5))   # start 4 % off on purpose
    print(f"true: V={V:.2f} m/s  yaw={math.degrees(YAW_RATE):.1f} deg/s  horizon={true_h:.4f}  (start guess {true_h+0.04:.4f})")
    print(" t     V_est  yaw_est  hz_true  hz_dyn  learned   TTC_true TTC_loom   inl   res")
    errs_v, errs_t = [], []
    for i in range(45):
        t = i / FPS
        bounce = math.radians(0.6) * math.sin(2 * math.pi * 1.3 * t)
        yaw = YAW_RATE * t                                     # drive along a circle (heading = travel direction)
        cxw, cz = V / YAW_RATE * (1 - math.cos(yaw)), V / YAW_RATE * math.sin(yaw)
        car_d = CAR_Z0 - (V - CAR_V) * t                       # car ahead on our path, slower than us
        img, box = render(cxw, cz, yaw, PITCH0 + bounce, car_d)
        dets = [box] if box else []
        ctx = est.flow_step(img, t, 0)
        ego, ext = est.analyze(ctx, dets)
        if ego and ego.get("ok"):
            ttc_true = car_d / (V - CAR_V)
            e = ext[0] if ext else None
            ttc_l = 1 / e["loom"] if e and e["loom"] > 1e-3 else float("nan")
            if i > 8:
                errs_v.append(ego["v"] - V); errs_t.append(ttc_l / ttc_true - 1)
            hz_t = 0.5 - f * math.tan(PITCH0 + bounce) / H
            print(f"{t:4.2f}  {ego['v']:6.2f}  {math.degrees(ego['yaw']):6.2f}  {hz_t:.4f}  {ego['horizon']:.4f}  {ego['horizon_lt']:.4f}"
                  f"   {ttc_true:6.2f}   {ttc_l:6.2f}   {ego['inl']:.2f}  {ego['res']:.2f}")
    ev, et = np.array(errs_v), np.array(errs_t)
    print(f"\nego speed error: mean {ev.mean():+.2f} m/s, rms {np.sqrt((ev**2).mean()):.2f} m/s  (true {V} m/s)")
    print(f"looming TTC error: median {np.nanmedian(np.abs(et))*100:.1f} %")
    print(f"learned horizon {est.cal['horizon'] + est.lt_dy/ (H*320/W) :.4f} vs true {true_h:.4f}")
    ok = abs(ev.mean()) < 0.08 * V and np.nanmedian(np.abs(et)) < 0.2
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "dis"))
