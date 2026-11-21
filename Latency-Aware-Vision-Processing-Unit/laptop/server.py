"""Detection + motion server. Picks the fastest detector backend it finds:
   NVIDIA GPU (CUDA)  ->  Intel GPU / CPU via OpenVINO  ->  plain CPU (PyTorch)
and, per connected board, runs motion.py (optical flow -> ego speed / yaw / pitch / horizon, and per-object
looming = 1/TTC plus rotation-compensated image velocity) in parallel with the detector.

Protocol
  v1  client -> binary [uint32 frame_id][jpeg]
  v2  client -> binary b"VPU2" [uint32 frame_id][float64 capture_time_s][uint32 source_generation][jpeg]
      client -> {"cal": {"hfov":..,"cam_h":..,"horizon":..,"foe_x":..}}     (whenever the calibration changes)
  server -> {"id", "infer_ms" (total server time), "yolo_ms", "motion_ms",
             "dets": [[cls,conf,x1,y1,x2,y2],..] (normalised),
             "ext":  [ {loom, loom_sd, u, v, u_sd, n[, z, z_rel]} | null, ..]  (v2 only, one per det)
             "ego":  {ok, v, v_sd, yaw, pitch, horizon, horizon_fit, foe_x, ...} | null }   (v2 only)
  client -> {"ping":n}       server -> {"pong":n, "v":2}
v1 clients keep working unchanged (they never get ext/ego).
"""
import asyncio, json, struct, time, argparse, threading
from pathlib import Path
import cv2, numpy as np, torch, websockets
from ultralytics import YOLO
import logging
from motion import MotionEstimator, FlowEngine, MetricDepth
logging.getLogger("ultralytics").setLevel(logging.ERROR)   # hides the repeated 'half is deprecated' warning

KEEP = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
MAGIC = b"VPU2"
ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8765)
ap.add_argument("--model", default="yolov8n.pt")
ap.add_argument("--imgsz", type=int, default=480)
ap.add_argument("--conf", type=float, default=0.35)
ap.add_argument("--token", default="", help="shared secret; clients must connect with ?token=SECRET")
ap.add_argument("--backend", default="auto", choices=["auto", "cuda", "openvino", "cpu"])
ap.add_argument("--motion", default="on", choices=["on", "off"], help="ego-motion + looming (v2 clients)")
ap.add_argument("--flow", default="dis", choices=["dis", "dis-fast", "raft"],
                help="optical flow: dis (CPU, default), dis-fast (slow CPUs), raft (CUDA, needs torchvision)")
ap.add_argument("--auto-horizon", action="store_true",
                help="let the flow fit slowly re-learn the static horizon/FOE (off: only pitch bounce is followed)")
ap.add_argument("--depth", action="store_true",
                help="EXPERIMENTAL: Depth Anything V2 metric depth per box (pip install transformers)")
args = ap.parse_args()

model, backend, ov_device = None, "cpu", None
flow_engine = FlowEngine(args.flow) if args.motion == "on" else None
flow_lock = threading.Lock()            # one flow computation at a time (shared DIS object / RAFT model)
depth_net = None
if args.depth:
    try:
        depth_net = MetricDepth()
    except Exception as e:
        print("[depth] disabled:", e)
gpu_lock = threading.Lock()


def infer(img):
    kw = dict(imgsz=args.imgsz, conf=args.conf, classes=list(KEEP), verbose=False)
    if backend == "cuda":       kw.update(device=0, half=True)
    elif backend == "openvino": kw.update(device=ov_device)
    else:                       kw.update(device="cpu")
    with gpu_lock:
        r = model.predict(img, **kw)[0]
    b = r.boxes
    xyxy, cf, cl = b.xyxyn.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy()
    return [[KEEP[int(c)], round(float(p), 3), *[round(float(v), 4) for v in box]]
            for box, p, c in zip(xyxy, cf, cl)]


def depth_map(img):
    with gpu_lock:
        return depth_net(img)


def blank():
    return np.zeros((360, 480, 3), np.uint8)


def setup():
    global model, backend, ov_device
    if args.backend in ("auto", "cuda") and torch.cuda.is_available():
        model, backend = YOLO(args.model), "cuda"
        return
    if args.backend in ("auto", "openvino"):
        try:
            import openvino as ov
            devs = ov.Core().available_devices
            print("OpenVINO devices:", devs)
            ov_device = "intel:gpu" if any(d.startswith("GPU") for d in devs) else "intel:cpu"
            folder = Path(f"{Path(args.model).stem}_{args.imgsz}_openvino_model")
            if not folder.exists():
                print("exporting model to OpenVINO (first run only, ~1 min)...")
                out = YOLO(args.model).export(format="openvino", imgsz=args.imgsz, half=False)
                Path(out).rename(folder)
            model, backend = YOLO(str(folder), task="detect"), "openvino"
            try:
                infer(blank())
            except Exception as e:
                if ov_device == "intel:gpu":
                    print("Intel GPU failed (", e, ") -> OpenVINO on CPU")
                    ov_device = "intel:cpu"; infer(blank())
                else:
                    raise
            return
        except Exception as e:
            print("OpenVINO not usable:", e, "-> plain CPU")
    model, backend = YOLO(args.model), "cpu"


setup()
infer(blank())                                          # warm-up (kernel compile / cudnn autotune)
t0 = time.perf_counter()
for _ in range(5): infer(blank())
print(f"ready on :{args.port}  backend={backend} device={ov_device or ('cuda' if backend == 'cuda' else 'cpu')}"
      f"  ~{(time.perf_counter()-t0)/5*1000:.0f} ms/frame  motion={args.motion}"
      f"{' flow=' + flow_engine.kind if flow_engine else ''}{'  depth=on' if depth_net else ''}", flush=True)


def flow_step(est, img, ts, gen):
    with flow_lock:
        return est.flow_step(img, ts, gen)


async def _handle(ws):
    est = MotionEstimator(flow_engine, auto_horizon=args.auto_horizon) if flow_engine else None
    async for msg in ws:
        if isinstance(msg, str):
            d = json.loads(msg)
            if "ping" in d:
                await ws.send(json.dumps({"pong": d["ping"], "v": 2}))
            if "cal" in d and est is not None:
                est.set_cal(d["cal"])
            continue
        v2 = msg[:4] == MAGIC
        if v2:
            fid, ts, gen = struct.unpack("!IdI", msg[4:20])
            jpg = msg[20:]
        else:
            fid, ts, gen, jpg = struct.unpack("!I", msg[:4])[0], None, 0, msg[4:]
        t0 = time.perf_counter()
        img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            await ws.send(json.dumps({"id": fid, "infer_ms": 0, "dets": [], "err": "bad jpeg"}))
            continue
        jobs = [asyncio.to_thread(infer, img)]
        use_motion = v2 and est is not None
        if use_motion:
            jobs.append(asyncio.to_thread(flow_step, est, img, ts, gen))
        if v2 and depth_net is not None:
            jobs.append(asyncio.to_thread(depth_map, img))
        t_y = time.perf_counter()
        res = await asyncio.gather(*jobs, return_exceptions=True)
        dets = res[0] if not isinstance(res[0], Exception) else []
        if isinstance(res[0], Exception):
            print("detector error:", res[0], flush=True)
        yolo_ms = (time.perf_counter() - t_y) * 1000
        out = {"id": fid, "dets": dets}
        if use_motion:
            t_m = time.perf_counter()
            ctx = res[1] if not isinstance(res[1], Exception) else None
            try:
                ego, ext = est.analyze(ctx, dets)
            except Exception as e:
                print("motion error:", type(e).__name__, e, flush=True)
                ego, ext = None, None
            if v2 and depth_net is not None and not isinstance(res[-1], Exception):
                zs = MetricDepth.per_box(res[-1], dets, est.cal, ego["horizon"] if ego and ego.get("ok") else None)
                ext = ext or [None] * len(dets)
                ext = [dict(e or {}, **z) if z else e for e, z in zip(ext, zs)]
            out.update(ego=ego, ext=ext, motion_ms=round((ctx or {}).get("flow_ms", 0) + (time.perf_counter() - t_m) * 1000, 1))
        out["yolo_ms"] = round(yolo_ms, 1)
        out["infer_ms"] = round((time.perf_counter() - t0) * 1000, 2)   # all server-side time (client subtracts it from the RTT)
        await ws.send(json.dumps(out))


async def handler(ws):
    if args.token:
        req = getattr(ws, "request", None)
        path = (getattr(req, "path", None) or getattr(ws, "path", "") or "")
        if f"token={args.token}" not in path:
            print("rejected a client with a missing/wrong token", flush=True)
            await ws.close(1008, "bad token")
            return
    print("client connected:", ws.remote_address, flush=True)
    try:
        await _handle(ws)
    except websockets.exceptions.ConnectionClosed:
        print("client connection lost (board restarted or link dropped) - waiting for it to reconnect", flush=True)
    except Exception as e:
        print("client error:", type(e).__name__, e, flush=True)
    else:
        print("client disconnected", flush=True)


async def main():
    async with websockets.serve(handler, "0.0.0.0", args.port, max_size=2**22, compression=None):
        await asyncio.Future()

asyncio.run(main())
