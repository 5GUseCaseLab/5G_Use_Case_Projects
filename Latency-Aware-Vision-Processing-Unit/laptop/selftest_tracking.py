"""End-to-end self-test: rendered road -> motion.py (ego + looming) -> board tracker -> speed / direction.

    python selftest_tracking.py

Three scenarios with known truth: a car ahead driving the same way (slower), a stopped car, an oncoming car.
Checks own speed, the car's distance, closing speed, speed over ground and the direction label, and compares
how fast the closing speed settles with and without the laptop's looming measurement."""
import math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "unoq_app" / "python"))
import numpy as np
import selftest_motion as R
from motion import MotionEstimator, FlowEngine
from tracking import Tracker

FPS = 15.0


def run(car_v, label, use_ext=True, use_ego=True, n=36):
    est = MotionEstimator(FlowEngine("dis"))
    true_h = 0.5 - R.f * math.tan(R.PITCH0) / R.H
    cal = dict(hfov=R.HFOV, cam_h=R.CAM_H, horizon=true_h, foe_x=0.5)
    est.set_cal(cal)
    trk = Tracker(R.W, R.H, R.HFOV, R.CAM_H, true_h, 0.5)
    V, z0 = R.V, 30.0
    first_ok = None
    rows = []
    for i in range(n):
        t = i / FPS
        car_d = z0 - (V - car_v) * t
        img, box = R.render(0.0, V * t, 0.0, R.PITCH0, car_d)
        dets = [box] if box else []
        ego, ext = est.analyze(est.flow_step(img, t, 0), dets)
        if use_ext and ext:
            dets = [d + [e] if e else d for d, e in zip(dets, ext)]
        trk.update(dets, t, ego if (use_ego and ego and ego.get("ok")) else None)
        snap = trk.snapshot(t)
        if snap:
            s = snap[0]
            if s["valid"] and first_ok is None and abs(s["speed"] - (V - car_v)) < 0.15 * abs(V - car_v):
                first_ok = t
            rows.append((t, car_d, s))
    t, car_d, s = rows[-1]
    print(f"\n[{label}]  truth: car {car_d:.1f} m ahead, closing {V-car_v:.1f} m/s, over ground {abs(car_v)*3.6:.0f} km/h"
          f"  | ego {'on' if use_ego else 'off'}, looming {'on' if use_ext else 'off'}")
    print(f"   tracker: dist {s['dist']} m, closing {s['speed']} m/s, TTC {s['ttc']} s (loom {s['ttc_loom']}),"
          f" over ground {s['abs_kmh']} km/h, heading {s['heading']}, dir '{s['dir']}'")
    print(f"   closing speed within 15 % of truth after {first_ok if first_ok is None else round(first_ok, 2)} s")
    return s, first_ok


def main():
    ok = True
    s, t1 = run(4.0, "car ahead, same direction, 4 m/s")
    ok &= s["dir"] == "same dir" and abs(s["abs_kmh"] - 14.4) < 5 and abs(s["speed"] - 7.0) < 1.2
    s, _ = run(0.0, "stopped car")
    ok &= s["dir"] == "stopped" and abs(s["speed"] - 11.0) < 1.5
    s, _ = run(-6.0, "oncoming car, 6 m/s", n=22)
    ok &= s["dir"] == "oncoming" and abs(s["abs_kmh"] - 21.6) < 7
    _, t0 = run(4.0, "same car, positions only (old behaviour)", use_ext=False, use_ego=False)
    print(f"\nsettling time of the closing speed: {t1} s with looming vs {t0} s from positions only")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
