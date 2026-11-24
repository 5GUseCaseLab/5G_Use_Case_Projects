"""Regenerate docs/images/ui_*.png - screenshots of the REAL web UI (unoq_app/assets) fed by the REAL pipeline.

    pip install playwright && playwright install chromium
    python tools/make_screenshots.py

What is real: index.html / app.js / style.css, laptop/motion.py (optical-flow ego-motion + looming), the board's
tracking.py and advisor.py, run on a rendered road with known truth (laptop/selftest_motion.py).
What is simulated: the camera picture (a rendered textured road + one "car" billboard, not a real photo), the detector
(its box comes from the renderer) and the link/latency numbers (RTT, GPU ms ...), which are placeholders.
To get real screenshots, run the app on the board, open its web page and save a browser screenshot over these files.
"""
import base64, http.server, json, math, sys, threading
from functools import partial
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "laptop"))
sys.path.insert(0, str(ROOT / "unoq_app" / "python"))
import selftest_motion as R                      # noqa: E402
from motion import MotionEstimator, FlowEngine   # noqa: E402
from tracking import Tracker                     # noqa: E402
from advisor import Advisor                      # noqa: E402

OUT = ROOT / "docs" / "images"
FPS, N = 15.0, 40


def simulate():
    true_h = 0.5 - R.f * math.tan(R.PITCH0) / R.H
    cal = dict(hfov=R.HFOV, cam_h=R.CAM_H, horizon=true_h, foe_x=0.5)
    est = MotionEstimator(FlowEngine("dis"))
    est.set_cal(cal)
    trk = Tracker(R.W, R.H, R.HFOV, R.CAM_H, true_h, 0.5)
    adv = Advisor()
    ego = None
    for i in range(N):
        t = i / FPS
        car_d = R.CAR_Z0 + 8 - (R.V - R.CAR_V) * t
        img, box = R.render(0.0, R.V * t, 0.0, R.PITCH0, car_d)
        dets = [box] if box else []
        ego, ext = est.analyze(est.flow_step(img, t, 0), dets)
        if ext:
            dets = [d + [e] if e else d for d, e in zip(dets, ext)]
        use = ego if ego and ego.get("ok") else None
        trk.update(dets, t, use)
        tracks = trk.snapshot(t)
        a = adv.update(tracks, t, (use or {}).get("v") or 0.0)
    ok, buf = cv2.imencode(".jpg", cv2.resize(img, (320, int(R.H * 320 / R.W))), [cv2.IMWRITE_JPEG_QUALITY, 70])
    frame = {"img": base64.b64encode(buf).decode(), "tracks": tracks, "advice": a,
             "horizon": round(trk.hy_now / trk.h, 4)}
    return frame, ego, true_h


def stats(frame, ego, hz, remote=True):
    s = dict(mode="remote" if remote else "local", cfg_mode="auto", link=remote, rtt=22.4, infer=9.8, remote_est=41.0,
             local_ms=185.0, e2e=52.0, fps=14.6, skip=18, diff=6.2, alert=frame["advice"]["level"], local_ok=True,
             url="ws://100.101.102.103:8765", thr=3.0, max_age=0.5, hfov=R.HFOV, cam_h=R.CAM_H, horizon=round(hz, 3),
             ego_kmh=0, foe_x=0.5, ego_mode="auto", ego_src="laptop flow", ego_v=round(ego["v"] * 3.6, 1),
             ego_yaw=round(math.degrees(ego.get("yaw") or 0), 1), hz_fit=ego.get("horizon_fit"), foe_fit=ego.get("foe_x"),
             motion_ms=14.0, v2=True, r_sent=412, r_ok=409, r_to=2, r_busy=1, r_err=0, send_w=480, kb=21.5,
             link_state="up", link_err="", tries=0, bias=4.0,
             source="file: simulated_road.mp4 @ 15.0 fps")
    if not remote:
        s.update(link=False, link_state="failed", link_err="timed out", tries=3, rtt=None, infer=None,
                 remote_est=None, r_to=17, fps=5.1, e2e=190.0)
    return s


MOCK = """
window.__h = {};
window.io = () => ({on: (e, f) => { window.__h[e] = f; }, once: () => {}, emit: () => {}});
window.__fire = (e, d) => window.__h[e](d);
"""


def main():
    from playwright.sync_api import sync_playwright
    frame, ego, hz = simulate()
    print("simulated: ego %.1f km/h, tracks %d, advice %s" % (ego["v"] * 3.6, len(frame["tracks"]), frame["advice"]["title"]))
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(ROOT / "unoq_app" / "assets"))
    handler.log_message = lambda *a, **k: None
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/index.html"
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=1.5)
        pg.route("**/libs/socket.io.min.js", lambda r: r.fulfill(body="", content_type="text/javascript"))
        pg.add_init_script(MOCK)
        for name, remote, adv_level in (("ui_main", True, None), ("ui_local_fallback", False, None)):
            pg.goto(url)
            pg.evaluate("([f, s]) => { __fire('frame', f); __fire('stats', s); }", [frame, stats(frame, ego, hz, remote)])
            pg.wait_for_timeout(400)
            pg.screenshot(path=str(OUT / f"{name}.png"), full_page=True)
            print("wrote", OUT / f"{name}.png")
        b.close()
    srv.shutdown()


if __name__ == "__main__":
    main()
