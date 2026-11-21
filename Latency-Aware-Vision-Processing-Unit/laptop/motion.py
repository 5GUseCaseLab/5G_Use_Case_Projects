"""Laptop-side motion analysis: dense optical flow between consecutive frames of one client, then

  * EGO-MOTION   fit of a rigid ground-plane flow model to the road pixels ->
                 own speed (m/s), yaw rate (turning), pitch rate (bumps), and where the horizon and the
                 focus of expansion really are in this frame (dynamic horizon + online calibration).
  * PER OBJECT   inside each detection box: looming (scale-change) rate = 1/TTC, which needs NO
                 calibration at all, and the box's image velocity with the ego ROTATION removed.

The tracker on the board turns these into closing speed, lateral speed, absolute speed and heading
(same direction / oncoming / crossing / stopped). Everything is optional on the protocol level: if a frame
pair is unusable (first frame, big time gap, video looped, too little texture) the fields are simply null.

Geometry (pinhole, small angles; x right, y down, pixels relative to the principal point (W/2, H/2)):
  rotation      yaw w (+ = camera turns right), pitch p (+ = nose down), roll r, per frame:
                  u = -(f + x^2/f) w - (x y / f) p + r y        v = -(x y / f) w - (f + y^2/f) p - r x
  translation   camera moves D forward; a ground point at depth Z has  y_t = y - y_h = f h / Z,
                s = D / Z = T y_t / f  with  T = D / h  ->   u = x_t s/(1-s),  v = y_t s/(1-s)
                (x_t, y_t measured from the focus of expansion, which sits on the horizon)
  unknowns      T, w, p, r, dy (horizon offset), dx (FOE offset) - Gauss-Newton + Huber IRLS, weak priors
                on dy/dx. Ego speed V = T h / dt.  Note V scales with the camera height h and with f (HFOV):
                the same calibration that sets distances also sets this speed, so they stay consistent.
"""
import math, time
import cv2
import numpy as np

MOTION_W = 320            # flow is computed at this width whatever size the client sends
MAX_DT = 0.35             # s: larger gaps between frames -> no flow (motion too large / stale)
MIN_DT = 0.008


class ScalarKF:
    """Random-walk filter for one scalar (ego speed, yaw rate)."""
    def __init__(self, q):
        self.q, self.x, self.P = q, None, None

    def update(self, z, r, dt):
        if self.x is None:
            self.x, self.P = z, r
            return self.x
        self.P += self.q * self.q * dt
        if z is not None:
            k = self.P / (self.P + r)
            self.x += k * (z - self.x)
            self.P *= (1 - k)
        return self.x

    def reset(self):
        self.x = self.P = None


# ----------------------------------------------------------------------------------------- flow
class FlowEngine:
    """'dis'      = OpenCV DIS down to full resolution (CPU, ~10-25 ms per fwd+bwd pair at 320 px). Default.
       'dis-fast' = DIS stopping one pyramid level early (~3x faster; under-reads large motion -> ego speed
                    ~7 % low at 15 fps in the self-test, vs ~2 % for 'dis'). For slow laptop CPUs.
       'raft'     = torchvision RAFT-small on CUDA (better on smooth asphalt / at night; ~15-25 ms on an RTX 3050).
       If RAFT cannot be set up or fails once, it falls back to DIS for good."""
    def __init__(self, kind="dis"):
        self.kind = kind
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        if kind != "dis-fast":
            self.dis.setFinestScale(0)
        self.raft = None
        if kind == "raft":
            try:
                import torch
                from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA not available")
                self.torch = torch
                self.raft = raft_small(weights=Raft_Small_Weights.DEFAULT, progress=False).cuda().eval()
                print("[motion] optical flow: RAFT-small on CUDA")
            except Exception as e:
                print("[motion] RAFT unavailable (", e, ") -> DIS on CPU")
                self.kind = "dis"
        if self.kind != "raft":
            print("[motion] optical flow: OpenCV DIS" + (" (fast)" if kind == "dis-fast" else ""))

    def _raft(self, g0, g1):
        torch = self.torch
        h, w = g0.shape
        H8, W8 = (h + 7) // 8 * 8, (w + 7) // 8 * 8

        def prep(a, b):
            t = torch.from_numpy(np.stack([a, b])).cuda().float()[:, None].repeat(1, 3, 1, 1)
            t = t / 127.5 - 1.0
            return torch.nn.functional.pad(t, (0, W8 - w, 0, H8 - h), mode="replicate")
        with torch.inference_mode():
            fl = self.raft(prep(g0, g1), prep(g1, g0), num_flow_updates=8)[-1]
        fl = fl[:, :, :h, :w].permute(0, 2, 3, 1).float().cpu().numpy()
        return fl[0], fl[1]

    def __call__(self, g0, g1):
        if self.raft is not None:
            try:
                return self._raft(g0, g1)
            except Exception as e:
                print("[motion] RAFT failed (", e, ") -> DIS from now on")
                self.raft, self.kind = None, "dis"
                self.dis.setFinestScale(0)
        return self.dis.calc(g0, g1, None), self.dis.calc(g1, g0, None)


def fb_valid(fwd, bwd, xs, ys):
    """Forward-backward consistency at integer sample points -> (u, v, ok)."""
    u, v = fwd[ys, xs, 0], fwd[ys, xs, 1]
    x1 = np.clip(np.rint(xs + u).astype(np.int32), 0, fwd.shape[1] - 1)
    y1 = np.clip(np.rint(ys + v).astype(np.int32), 0, fwd.shape[0] - 1)
    err = np.hypot(u + bwd[y1, x1, 0], v + bwd[y1, x1, 1])
    inside = (xs + u >= 0) & (xs + u < fwd.shape[1]) & (ys + v >= 0) & (ys + v < fwd.shape[0])
    return u, v, inside & (err < 0.6 + 0.06 * np.hypot(u, v))


# ----------------------------------------------------------------------------------------- ego fit
def ego_model(th, xp, yp, xt0, yt0, f):
    T, w, p, r, dy, dx = th
    xt, yt = xt0 - dx, yt0 - dy
    s = np.clip(T * yt / f, -0.5, 0.75)
    g = s / (1 - s)
    u = xt * g - (f + xp * xp / f) * w - (xp * yp / f) * p + r * yp
    v = yt * g - (xp * yp / f) * w - (f + yp * yp / f) * p - r * xp
    return u, v


def fit_ego(xp, yp, xt0, yt0, u, v, f, T0, sd_dy, sd_dx, iters=6):
    """Robust Gauss-Newton. Returns (theta, cov, inlier_mask, median_residual_px)."""
    n = len(u)
    # linear initial guess (first-order translation, no offsets)
    A = np.zeros((2 * n, 4))
    A[:n, 0] = xt0 * yt0 / f;  A[n:, 0] = yt0 * yt0 / f
    A[:n, 1] = -(f + xp * xp / f); A[n:, 1] = -(xp * yp / f)
    A[:n, 2] = -(xp * yp / f);  A[n:, 2] = -(f + yp * yp / f)
    A[:n, 3] = yp;              A[n:, 3] = -xp
    b = np.concatenate([u, v])
    try:
        th4 = np.linalg.lstsq(A, b, rcond=None)[0]
    except np.linalg.LinAlgError:
        th4 = np.array([T0, 0, 0, 0])
    th = np.array([th4[0] if np.isfinite(th4[0]) else T0, th4[1], th4[2], th4[3], 0.0, 0.0])
    prior_w = np.array([0, 0, 0, 0, 1 / sd_dy, 1 / sd_dx])
    steps = np.array([1e-4, 1e-6, 1e-6, 1e-6, 0.05, 0.05])
    wts = np.ones(n)
    for _ in range(iters):
        mu, mv = ego_model(th, xp, yp, xt0, yt0, f)
        ru, rv = u - mu, v - mv
        rn = np.hypot(ru, rv)
        sig = max(0.25, 1.4826 * np.median(rn))
        c = 1.5 * sig
        wts = np.where(rn <= c, 1.0, c / np.maximum(rn, 1e-9))
        wts[rn > 4 * sig] = 0.0
        J = np.empty((2 * n, 6))
        for k in range(6):
            t2 = th.copy(); t2[k] += steps[k]
            u2, v2 = ego_model(t2, xp, yp, xt0, yt0, f)
            J[:n, k] = (u2 - mu) / steps[k]; J[n:, k] = (v2 - mv) / steps[k]
        W = np.concatenate([wts, wts])
        JW = J * W[:, None]
        Hm = J.T @ JW + np.diag(prior_w ** 2) * sig * sig
        g = JW.T @ np.concatenate([ru, rv]) - prior_w ** 2 * sig * sig * th * (np.arange(6) >= 4)
        try:
            d = np.linalg.solve(Hm + 1e-9 * np.eye(6), g)
        except np.linalg.LinAlgError:
            break
        th = th + d
        if np.abs(d[0]) < 1e-5 and np.abs(d[4]) < 0.01:
            break
    mu, mv = ego_model(th, xp, yp, xt0, yt0, f)
    rn = np.hypot(u - mu, v - mv)
    inl = rn < max(0.6, 3 * 1.4826 * np.median(rn))
    try:
        cov = np.linalg.inv(Hm) * sig * sig
    except Exception:
        cov = np.full((6, 6), np.inf)
    return th, cov, inl, float(np.median(rn))


# ----------------------------------------------------------------------------------------- estimator
class MotionEstimator:
    """One per client connection (it keeps the previous frame)."""
    W_LT0 = 15 / (0.006 * 180) ** 2          # prior on the user's horizon ~ 15 good frames' worth
    def __init__(self, flow=None, auto_horizon=False):
        self.flow = flow or FlowEngine("dis")
        self.auto_horizon = auto_horizon
        self.prev = None                     # (gray, t, gen)
        self.cal = dict(hfov=70.0, cam_h=1.3, horizon=0.45, foe_x=0.5)
        self.lt_dy = self.lt_dx = 0.0        # learned horizon / FOE offsets (px at MOTION_W)
        self.n_lt, self.w_lt = 0, self.W_LT0
        self.kv, self.kw = ScalarKF(3.0), ScalarKF(0.6)
        self.t_last_ok = None
        self.dyn_dy = None                   # smoothed per-frame horizon offset of the fit (diagnostic)
        self.pitch_off = 0.0                 # px: horizon bounce from integrated pitch rate (high-passed)

    def set_cal(self, cal):
        new = {k: float(cal[k]) for k in ("hfov", "cam_h", "horizon", "foe_x") if cal.get(k) is not None}
        if any(abs(new[k] - self.cal.get(k, 0)) > 1e-6 for k in new):
            changed_geo = any(k in new and abs(new[k] - self.cal[k]) > 1e-6 for k in ("horizon", "foe_x", "hfov"))
            self.cal.update(new)
            if changed_geo:                  # user moved the horizon / FOV: forget what was learned
                self.lt_dy = self.lt_dx = 0.0; self.n_lt, self.w_lt = 0, self.W_LT0
            self.kv.reset()

    # step 1 (can run in parallel with the detector)
    def flow_step(self, frame, t, gen=0):
        h0, w0 = frame.shape[:2]
        H = int(round(h0 * MOTION_W / w0)); H -= H % 2
        g = cv2.cvtColor(cv2.resize(frame, (MOTION_W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        prev, self.prev = self.prev, (g, t, gen)
        if prev is None or prev[2] != gen or prev[0].shape != g.shape:
            return None
        dt = t - prev[1]
        if not (MIN_DT <= dt <= MAX_DT):
            return None
        t0 = time.perf_counter()
        fwd, bwd = self.flow(prev[0], g)
        eig = cv2.cornerMinEigenVal(prev[0].astype(np.float32) / 255.0, 5, 3)
        return dict(fwd=fwd, bwd=bwd, eig=eig, dt=dt, W=MOTION_W, H=H, t=t, flow_ms=(time.perf_counter() - t0) * 1000)

    # step 2 (needs the detections)
    def analyze(self, ctx, dets):
        if ctx is None:
            return None, None
        W, H, dt = ctx["W"], ctx["H"], ctx["dt"]
        fwd, bwd, eig = ctx["fwd"], ctx["bwd"], ctx["eig"]
        c = self.cal
        f = (W / 2) / math.tan(math.radians(c["hfov"]) / 2)
        cx, cy = W / 2, H / 2
        yh = c["horizon"] * H + self.lt_dy
        xf = c["foe_x"] * W + self.lt_dx
        h_cam = c["cam_h"]

        # --- road samples: below the horizon, inside +-9 m laterally, not on a detected object
        step = 5
        ys, xs = np.mgrid[int(yh + 4):H - 2:step, 4:W - 4:step]
        xs, ys = xs.ravel(), ys.ravel()
        yt0, xt0 = ys - yh, xs - xf
        keep = (yt0 > 3) & (np.abs(xt0 / np.maximum(yt0, 1e-3)) < 9.0 / h_cam)
        bmask = np.zeros((H, W), bool)
        for d in dets:
            x1, y1, x2, y2 = d[2:6]
            ex, ey = 0.1 * (x2 - x1), 0.1 * (y2 - y1)
            bmask[max(0, int((y1 - ey) * H)):int((y2 + ey) * H) + 1, max(0, int((x1 - ex) * W)):int((x2 + ex) * W) + 1] = True
        keep &= ~bmask[ys, xs]
        keep &= eig[ys, xs] > 2e-4
        xs, ys = xs[keep], ys[keep]
        ego = None
        rot = (0.0, 0.0, 0.0)
        if len(xs) >= 60:
            u, v, ok = fb_valid(fwd, bwd, xs, ys)
            xs, ys, u, v = xs[ok], ys[ok], u[ok], v[ok]
            if len(xs) >= 60:
                xp, yp = xs - cx, ys - cy
                T0 = (self.kv.x or 0.0) * dt / h_cam
                th, cov, inl, res = fit_ego(xp, yp, xs - xf, ys - yh, u, v, f, T0,
                                            sd_dy=0.03 * H, sd_dx=0.05 * W)
                T, w, p, r, dy, dx = th
                v_raw = T * h_cam / dt
                v_sd = math.sqrt(max(cov[0, 0], 0)) * h_cam / dt if np.isfinite(cov[0, 0]) else 99.0
                inl_ratio = float(inl.mean())
                good = inl_ratio > 0.45 and inl.sum() >= 50 and res < 1.5 and v_sd < 3.0 and -3 < v_raw < 70
                rot = (w, p, r)
                dy_sd = math.sqrt(max(cov[4, 4], 0)) if np.isfinite(cov[4, 4]) else 1e9
                if good:
                    V = self.kv.update(v_raw, max(v_sd, 0.3) ** 2, dt)
                    yaw = self.kw.update(w / dt, (0.02 / dt) ** 2, dt)
                    # learn the static horizon / FOE slowly - only when moving and the offset is well determined
                    # (slow driving, hills and sharp curves make the ground-plane model ambiguous)
                    if self.auto_horizon and V > 3.0 and dy_sd < 0.012 * H:
                        wgt = 1.0 / max(dy_sd, 0.002 * H) ** 2
                        self.w_lt = 0.97 * self.w_lt + wgt
                        a = wgt / self.w_lt
                        self.lt_dy = float(np.clip(self.lt_dy + a * dy, -0.08 * H, 0.08 * H))
                        self.lt_dx = float(np.clip(self.lt_dx + a * dx, -0.08 * W, 0.08 * W))
                        self.n_lt += 1
                    dd = float(np.clip(dy, -0.06 * H, 0.06 * H)) if dy_sd < 0.02 * H else 0.0
                    self.dyn_dy = dd if self.dyn_dy is None else 0.8 * self.dyn_dy + 0.2 * dd
                    # dynamic horizon = calibrated horizon + pitch bounce. Pitch (a uniform vertical image shift) is
                    # well determined by the fit; integrate it and let it relax back (tau 0.6 s), so braking dives
                    # and bumps are followed but the calibrated horizon stays the reference.
                    lam = math.exp(-dt / 0.6)
                    self.pitch_off = float(np.clip(lam * (self.pitch_off - f * p), -0.05 * H, 0.05 * H))
                    ego = dict(ok=True, v=round(V, 2), v_raw=round(v_raw, 2), v_sd=round(v_sd, 2),
                               yaw=round(yaw, 4), pitch=round(p / dt, 4),
                               horizon=round((yh + self.pitch_off) / H, 4),
                               horizon_fit=round((yh + self.dyn_dy) / H, 4),
                               horizon_lt=round((c["horizon"] * H + self.lt_dy) / H, 4),
                               foe_x=round((c["foe_x"] * W + self.lt_dx) / W, 4), n_lt=self.n_lt,
                               inl=round(inl_ratio, 2), n=int(inl.sum()), res=round(res, 2), dt=round(dt, 3))
                else:
                    self.kv.update(None, 0, dt)
                    self.pitch_off *= math.exp(-dt / 0.6)
                    ego = dict(ok=False, inl=round(inl_ratio, 2), n=int(inl.sum()), res=round(res, 2),
                               v_raw=round(v_raw, 2), dt=round(dt, 3))

        # --- per object: looming + rotation-compensated image velocity
        w_, p_, r_ = rot
        ext = []
        for d in dets:
            x1, y1, x2, y2 = d[2] * W, d[3] * H, d[4] * W, d[5] * H
            bw, bh = x2 - x1, y2 - y1
            if bw < 6 or bh < 6:
                ext.append(None); continue
            sx = max(1, int(min(bw, bh) / 12))
            gy, gx = np.mgrid[int(y1 + 0.15 * bh):int(y2 - 0.15 * bh) + 1:sx, int(x1 + 0.15 * bw):int(x2 - 0.15 * bw) + 1:sx]
            gx, gy = gx.ravel().clip(0, W - 1), gy.ravel().clip(0, H - 1)
            if len(gx) < 12:
                ext.append(None); continue
            u, v, ok = fb_valid(fwd, bwd, gx, gy)
            ok &= eig[gy, gx] > 5e-5
            if ok.sum() < 10:
                ext.append(None); continue
            gx, gy, u, v = gx[ok], gy[ok], u[ok], v[ok]
            xc, yc = gx.mean(), gy.mean()
            ax, ay = gx - xc, gy - yc
            wts = np.ones(len(u))
            for _ in range(4):                                     # robust fit of  u=u0+k*ax, v=v0+k*ay
                sw = wts.sum()
                u0, v0 = (wts * u).sum() / sw, (wts * v).sum() / sw
                den = (wts * (ax * ax + ay * ay)).sum()
                k = (wts * (ax * (u - u0) + ay * (v - v0))).sum() / max(den, 1e-6)
                rn = np.hypot(u - u0 - k * ax, v - v0 - k * ay)
                sig = max(0.2, 1.4826 * np.median(rn))
                wts = np.where(rn < 2.5 * sig, 1.0, 0.0)
                if wts.sum() < 8: break
            if wts.sum() < 8:
                ext.append(None); continue
            k_sd = sig / math.sqrt(max(den, 1e-6))
            xp, yp = xc - cx, yc - cy
            ur = -(f + xp * xp / f) * w_ - (xp * yp / f) * p_ + r_ * yp
            vr = -(xp * yp / f) * w_ - (f + yp * yp / f) * p_ - r_ * xp
            c_sd = sig / math.sqrt(wts.sum())                      # px: uncertainty of the centre motion
            ext.append(dict(loom=round(k / dt, 4), loom_sd=round(k_sd / dt, 4),
                            u=round((u0 - ur) / f / dt, 4), v=round((v0 - vr) / f / dt, 4),
                            u_sd=round(max(c_sd, 0.15) / f / dt, 4), n=int(wts.sum())))
        return ego, ext


# ----------------------------------------------------------------------------------------- optional metric depth
class MetricDepth:
    """Optional (--depth): Depth Anything V2 metric (outdoor) on the GPU, rescaled per frame so that the
    road in front matches the ground-plane geometry (removes the network's camera-dependent scale error).
    Needs `pip install transformers` and downloads the model from Hugging Face on first use."""
    def __init__(self, model="depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"):
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.proc = AutoImageProcessor.from_pretrained(model)
        self.net = AutoModelForDepthEstimation.from_pretrained(model).to(self.dev).eval()
        if self.dev == "cuda":
            self.net = self.net.half()
        print(f"[depth] {model} on {self.dev}")

    def __call__(self, frame_bgr):
        torch = self.torch
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        inp = self.proc(images=rgb, return_tensors="pt").to(self.dev)
        if self.dev == "cuda":
            inp = {k: v.half() for k, v in inp.items()}
        with torch.inference_mode():
            d = self.net(**inp).predicted_depth
        d = torch.nn.functional.interpolate(d[:, None].float(), size=rgb.shape[:2], mode="bilinear")[0, 0]
        return d.cpu().numpy()

    @staticmethod
    def per_box(depth, dets, cal, horizon=None):
        """-> list of {'z','z_rel'} (or None) per det, after aligning the depth map to the road plane."""
        H, W = depth.shape
        f = (W / 2) / math.tan(math.radians(cal["hfov"]) / 2)
        yh = (horizon if horizon is not None else cal["horizon"]) * H
        ys = np.arange(int(yh + 0.06 * H), int(0.92 * H), 3)
        xs = np.arange(int(0.35 * W), int(0.65 * W), 3)
        if len(ys) < 3:
            return [None] * len(dets)
        gy, gx = np.meshgrid(ys, xs, indexing="ij")
        z_ground = f * cal["cam_h"] / (gy - yh)
        z_net = depth[gy, gx]
        okm = (z_net > 0.5) & (z_ground < 40)
        for d in dets:                                   # do not align on objects
            x1, y1, x2, y2 = d[2] * W, d[3] * H, d[4] * W, d[5] * H
            okm &= ~((gx >= x1) & (gx <= x2) & (gy >= y1) & (gy <= y2))
        if okm.sum() < 30:
            return [None] * len(dets)
        scale = float(np.median(z_ground[okm] / z_net[okm]))
        spread = float(np.median(np.abs(np.log(z_ground[okm] / (scale * z_net[okm])))))
        out = []
        for d in dets:
            x1, y1, x2, y2 = d[2] * W, d[3] * H, d[4] * W, d[5] * H
            r = depth[int(y1 + 0.5 * (y2 - y1)):int(y1 + 0.9 * (y2 - y1)) + 1,
                      int(x1 + 0.3 * (x2 - x1)):int(x1 + 0.7 * (x2 - x1)) + 1]
            if r.size < 4:
                out.append(None); continue
            out.append(dict(z=round(float(np.median(r)) * scale, 2), z_rel=round(0.12 + 1.5 * spread, 3)))
        return out
