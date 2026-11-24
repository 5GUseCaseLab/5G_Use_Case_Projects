"""Tracking + monocular depth / closing speed / lateral position / heading. Detector-agnostic.
Detections are [cls, conf, x1, y1, x2, y2] (+ optional 7th element `ext` dict from the laptop), coords 0..1.

Depth: up to three independent estimates, fused by inverse variance (log domain)
  * road plane : Z = f * cam_height / (y_bottom - y_horizon)    (camera height + horizon line; the horizon
                                                                   follows pitch bounce when the laptop sends it)
  * object size: Z = f * real_height / box_height_px             (class height prior)
  * GPU depth  : ext["z"] (laptop --depth, optional)

Velocity: one constant-velocity Kalman filter per axis and track.
  * Without ego-motion the state is RELATIVE (as seen from the camera).
  * With ego-motion (laptop optical flow, or the manual 'my speed' box) the state is the object's velocity
    over the GROUND, and our own motion enters as a known control input:
        dZ/dt = vz - V + yaw*X          dX/dt = vx - yaw*Z          (V = own speed, yaw + = turning right)
    so turning and accelerating no longer look like the other car moving, and stopped things stay stopped.
  * Besides positions, the laptop gives two VELOCITY measurements per box:
        looming  ext["loom"] = 1/TTC  ->  closing speed = Z * loom          (responds within 1-2 frames)
        ext["u"] = image velocity with the ego rotation removed  ->  lateral speed = u*Z + X*(dZ/dt)/Z
Outputs per track: distance, lateral offset, closing speed, TTC, and (with ego-motion) absolute speed, heading
and a direction label: same dir / oncoming / crossing L->R / crossing R->L / stopped."""
import math

REAL_H = {"person": 1.7, "bicycle": 1.1, "motorcycle": 1.3, "car": 1.5, "bus": 3.0, "truck": 3.2}
REL_H  = {"person": 0.10, "bicycle": 0.25, "motorcycle": 0.20, "car": 0.15, "bus": 0.12, "truck": 0.25}
HOLD_S = 2.0        # a track is kept (and drawn) this long without a fresh detection
STALE_S = 0.6       # ...and drawn dashed once it is older than this
LANE_HALF = 1.8     # m: half width of "my lane"


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class Kalman1D:
    """Constant-velocity filter for one axis. State [pos, vel]; pos' = vel + ctrl (ctrl = known ego term)."""
    def __init__(self, pos, var_pos, vel=0.0, var_vel=225.0):
        self.x = [pos, vel]
        self.P = [[var_pos, 0.0], [0.0, var_vel]]

    def predict(self, dt, acc_sigma, ctrl=0.0, ctrl_var=0.0):
        p, v = self.x
        P = self.P
        self.x = [p + (v + ctrl)*dt, v]
        p00 = P[0][0] + dt*(P[1][0] + P[0][1]) + dt*dt*P[1][1]
        p01 = P[0][1] + dt*P[1][1]
        p10 = P[1][0] + dt*P[1][1]
        p11 = P[1][1]
        q = acc_sigma * acc_sigma
        p00 += q*dt**4/4 + ctrl_var*dt*dt; p01 += q*dt**3/2; p10 += q*dt**3/2; p11 += q*dt*dt
        self.P = [[p00, p01], [p10, p11]]

    def update(self, z, var_z):
        P = self.P
        S = P[0][0] + var_z
        K0, K1 = P[0][0]/S, P[1][0]/S
        y = z - self.x[0]
        self.x = [self.x[0] + K0*y, self.x[1] + K1*y]
        self.P = [[(1-K0)*P[0][0], (1-K0)*P[0][1]],
                  [P[1][0] - K1*P[0][0], P[1][1] - K1*P[0][1]]]

    def update_vel(self, z, var_z):
        """Direct measurement of the velocity component."""
        P = self.P
        S = P[1][1] + var_z
        K0, K1 = P[0][1]/S, P[1][1]/S
        y = z - self.x[1]
        self.x = [self.x[0] + K0*y, self.x[1] + K1*y]
        self.P = [[P[0][0] - K0*P[1][0], P[0][1] - K0*P[1][1]],
                  [P[1][0] - K1*P[1][0], P[1][1] - K1*P[1][1]]]

    def vel_gate(self, z, var_z, n_sigma=4.0):
        return (z - self.x[1])**2 <= n_sigma*n_sigma*(self.P[1][1] + var_z)


class Track:
    def __init__(self, tid, cls, conf, box, t):
        self.id, self.cls, self.conf = tid, cls, conf
        self.box, self.vel = list(box), [0.0]*4
        self.t = self.t_seen = self.t_f = self.t_first = t
        self.hits, self.n_upd, self.n_vel, self.outliers = 1, 0, 0, 0
        self.kd = self.kx = None
        self.ego_known, self.v_ref, self.w_ref = False, 0.0, 0.0
        self.motion, self.dir, self.loom = "steady", "–", None

    def predict_box(self, t, horizon):
        dt = max(0.0, min(horizon, t - self.t))
        return [self.box[i] + self.vel[i]*dt for i in range(4)]

    def correct(self, box, conf, t):
        dt = t - self.t
        nb = [0.7*box[i] + 0.3*self.box[i] for i in range(4)]       # light smoothing of the drawn box
        if dt > 1e-3:
            nv = [(nb[i] - self.box[i])/dt for i in range(4)]
            v = [0.7*self.vel[i] + 0.3*nv[i] for i in range(4)]
            self.vel = [0.0 if abs(x) < 0.02 else max(-1.0, min(1.0, x)) for x in v]   # dead-band: static stays static
        self.box, self.conf = nb, max(conf, 0.5*self.conf + 0.5*conf)
        self.t = self.t_seen = t
        self.hits += 1


def _ego_terms(ego):
    """-> (known, V m/s, V variance, yaw rad/s)."""
    if not ego or ego.get("v") is None:
        return False, 0.0, 0.0, 0.0
    return True, float(ego["v"]), float(ego.get("v_sd") or 0.5)**2, float(ego.get("yaw") or 0.0)


class Tracker:
    def __init__(self, frame_w, frame_h, hfov_deg=70.0, cam_h=1.3, horizon=0.45, foe_x=0.5):
        self.w, self.h = frame_w, frame_h
        self.f = (frame_w/2) / math.tan(math.radians(hfov_deg)/2)
        self.cam_h, self.hy = cam_h, horizon*frame_h
        self.hy_now = self.hy                      # follows pitch bounce when the laptop sends a dynamic horizon
        self.cx = foe_x*frame_w                    # image column we drive towards
        self.tracks, self._next = [], 1
        self.ego = None

    # ---------- geometry ----------
    def measure(self, cls, box, ext=None):
        """-> (depth m, relative error, lateral m) or None if the box is too cut off to trust."""
        x1, y1, x2, y2 = box
        W, H, f = self.w, self.h, self.f
        hpx = (y2 - y1)*H
        top_cut, bot_cut = y1 < 0.01, y2 > 0.985
        d_h = rel_h = d_g = rel_g = None
        if hpx >= 6 and not top_cut and not bot_cut:
            d_h = f*REAL_H.get(cls, 1.5)/hpx
            rel_h = math.hypot(REL_H.get(cls, 0.2), 2.0/hpx)
        dy = y2*H - self.hy_now
        if dy > 3 and not bot_cut:
            d_g = f*self.cam_h/dy
            rel_g = math.hypot(2.0, 5.0*H/360.0)/dy                  # box jitter + ~5 px horizon uncertainty (calibration, pitch)
        if d_g and d_h:
            if 0.4 <= d_g/d_h <= 2.5:                              # consistent: inverse-variance fusion (log domain)
                wg, wh = 1/rel_g**2, 1/rel_h**2
                d = math.exp((wg*math.log(d_g) + wh*math.log(d_h))/(wg + wh))
                rel = (wg + wh) ** -0.5
            else:                                                  # not on the road plane / horizon off: trust the size
                d, rel = d_h, rel_h*1.5
        elif d_g:
            d, rel = d_g, rel_g
        elif d_h:
            d, rel = d_h, rel_h
        else:
            d = rel = None
        if ext and ext.get("z"):                                   # GPU metric depth (optional)
            d_n, rel_n = float(ext["z"]), float(ext.get("z_rel") or 0.2)
            if d is None:
                d, rel = d_n, rel_n
            elif 0.5 <= d_n/d <= 2.0:
                w1, w2 = 1/rel**2, 1/rel_n**2
                d = math.exp((w1*math.log(d) + w2*math.log(d_n))/(w1 + w2))
                rel = (w1 + w2) ** -0.5
        if d is None:
            return None
        lat = d*((x1 + x2)/2*W - self.cx)/f
        return d, rel, lat

    def _filter(self, tr, d, rel, lat, t, ext=None):
        known, V, V_var, w = _ego_terms(self.ego)
        var_d = (d*rel)**2
        var_x = (d*2.0/self.f)**2 + (abs(lat)*rel)**2
        if tr.kd is None:
            # start as "moving like us" (relative speed 0): no false alarm at first sight
            tr.kd = Kalman1D(d, var_d, V if known else 0.0)
            tr.kx = Kalman1D(lat, var_x, w*d if known else 0.0, 25.0)
            tr.t_f = tr.t_first = t; tr.n_upd, tr.n_vel = 1, 0
            tr.ego_known, tr.v_ref, tr.w_ref = known, V, w
            return
        if known != tr.ego_known:                                   # switch relative <-> over-ground velocity
            sgn = 1.0 if known else -1.0
            Vr, wr = (V, w) if known else (tr.v_ref, tr.w_ref)
            tr.kd.x[1] += sgn*(Vr - wr*tr.kx.x[0])
            tr.kx.x[1] += sgn*(wr*tr.kd.x[0])
            tr.ego_known = known
        if known:
            tr.v_ref, tr.w_ref = V, w
        dt = max(1e-3, t - tr.t_f)
        z0, x0 = tr.kd.x[0], tr.kx.x[0]
        if known:
            tr.kd.predict(dt, 4.0, ctrl=-V + w*x0, ctrl_var=V_var)
            tr.kx.predict(dt, 2.0, ctrl=-w*z0, ctrl_var=(0.02*z0)**2)
        else:
            tr.kd.predict(dt, 4.0); tr.kx.predict(dt, 2.0)
        tr.t_f = t
        y, S = d - tr.kd.x[0], tr.kd.P[0][0] + var_d
        if y*y/S > 16.0:                                           # > 4 sigma: ignore, re-start if it persists
            tr.outliers += 1
            if tr.outliers >= 3:
                tr.kd = tr.kx = None
                self._filter(tr, d, rel, lat, t, ext)
            return
        tr.outliers = 0
        tr.kd.update(d, var_d); tr.kx.update(lat, var_x)
        tr.n_upd += 1
        if not ext:
            return
        Z, X = tr.kd.x[0], tr.kx.x[0]
        dz_trans = tr.kd.x[1] - V if known else tr.kd.x[1]           # dZ/dt from translation only
        loom = ext.get("loom")
        if loom is not None and ext.get("n", 0) >= 10:             # looming -> closing speed
            vc = Z*loom
            var = (Z*float(ext.get("loom_sd") or 0.1))**2 + (vc*rel)**2 + 0.3**2
            meas = -vc + (V - w*X if known else 0.0)               # in state units (over ground if known)
            if tr.kd.vel_gate(meas, var):
                tr.kd.update_vel(meas, var); tr.n_vel += 1
            tr.loom = loom
        u = ext.get("u")
        if u is not None and ext.get("n", 0) >= 10:                # rotation-free image motion -> lateral speed
            meas = u*Z + X*dz_trans/max(Z, 0.5)
            var = (Z*float(ext.get("u_sd") or 0.02))**2 + (meas*rel)**2 + 0.2**2
            if tr.kx.vel_gate(meas, var):
                tr.kx.update_vel(meas, var)

    # ---------- association ----------
    def update(self, dets, t, ego=None):
        """ego: dict with v (m/s), v_sd, yaw (rad/s, + = turning right), horizon (dynamic, 0..1) - or None."""
        self.ego = ego if ego and ego.get("v") is not None else None
        hz = (ego or {}).get("horizon")
        self.hy_now = hz*self.h if hz is not None and abs(hz*self.h - self.hy) < 0.08*self.h else self.hy
        used_t, used_d = set(), set()

        def ext_of(d):
            return d[6] if len(d) > 6 and isinstance(d[6], dict) else None

        def take(cands):
            cands.sort(key=lambda c: -c[0])
            for _, tr, i in cands:
                if id(tr) in used_t or i in used_d:
                    continue
                used_t.add(id(tr)); used_d.add(i)
                d = dets[i]
                tr.correct(d[2:6], d[1], t)
                m = self.measure(tr.cls, d[2:6], ext_of(d))
                if m: self._filter(tr, *m, t, ext_of(d))

        c1 = []
        for tr in self.tracks:
            pb = tr.predict_box(t, 0.6)
            for i, d in enumerate(dets):
                if d[0] == tr.cls:
                    s = iou(pb, d[2:6])
                    if s > 0.10: c1.append((s, tr, i))
        take(c1)
        c2 = []                                                    # second chance: centre distance (fast movers, slow link)
        for tr in self.tracks:
            if id(tr) in used_t: continue
            pb = tr.predict_box(t, 0.6)
            pcx, pcy = (pb[0]+pb[2])/2, (pb[1]+pb[3])/2
            size = max(pb[2]-pb[0], pb[3]-pb[1], 0.02)
            for i, d in enumerate(dets):
                if i in used_d or d[0] != tr.cls: continue
                dist = math.hypot((d[2]+d[4])/2 - pcx, (d[3]+d[5])/2 - pcy)/size
                if dist < 0.6: c2.append((1 - dist, tr, i))
        take(c2)
        for i, d in enumerate(dets):
            if i not in used_d:
                tr = Track(self._next, d[0], d[1], d[2:6], t); self._next += 1
                m = self.measure(tr.cls, d[2:6], ext_of(d))
                if m: self._filter(tr, *m, t)                      # no velocity measurement on the first sight
                self.tracks.append(tr)
        # drop old tracks, and duplicates (same class, nearly same box)
        self.tracks = [tr for tr in self.tracks if t - tr.t_seen < HOLD_S]
        self.tracks.sort(key=lambda tr: -tr.hits)
        keep = []
        for tr in self.tracks:
            if not any(k.cls == tr.cls and iou(k.box, tr.box) > 0.7 for k in keep):
                keep.append(tr)
        self.tracks = keep

    # ---------- output ----------
    @staticmethod
    def _direction(tr, vz, vx):
        """Heading over the ground -> label, with hysteresis so it does not flicker."""
        sp = math.hypot(vz, vx)
        if tr.dir == "stopped" and sp < 2.0 or sp < 1.2:
            return "stopped"
        hd = math.degrees(math.atan2(vx, vz))                      # 0 = same way as us, 180 = towards us
        a = abs(hd)
        prev = tr.dir
        m = 8 if prev in ("same dir", "oncoming") else 0            # sticky bands
        if a < 35 + m: return "same dir"
        if a > 145 - m: return "oncoming"
        return "crossing L→R" if vx > 0 else "crossing R→L"

    def snapshot(self, t, ego_mps=0.0):
        """Everything the UI / advisor needs, for time t. Works between detections: tracks are held, boxes
        are extrapolated only a little, distance is advanced with the filter's speed.
        (ego_mps is accepted for compatibility; ego-motion now comes in through update())"""
        known, V, _, w = _ego_terms(self.ego)
        out = []
        for tr in self.tracks:
            age = t - tr.t_seen
            if age > HOLD_S or (tr.hits < 2 and tr.conf < 0.55):
                continue
            rec = dict(id=tr.id, cls=tr.cls, conf=round(tr.conf, 2), age=round(age, 1), stale=age > STALE_S,
                       box=[round(v, 4) for v in tr.predict_box(t, 0.3)],
                       valid=False, dist=None, x=None, speed=None, vx=None, ttc=None, bearing=None,
                       motion="–", zone="–", abs_kmh=None, heading=None, dir="–", ttc_loom=None, risk=0)
            if tr.kd is not None:
                dt = max(0.0, min(1.0, t - tr.t_f))
                z0, x0 = tr.kd.x[0], tr.kx.x[0]
                if tr.ego_known:                                    # relative rates from over-ground state
                    dzdt = tr.kd.x[1] - tr.v_ref + tr.w_ref*x0
                    dxdt = tr.kx.x[1] - tr.w_ref*z0
                else:
                    dzdt, dxdt = tr.kd.x[1], tr.kx.x[1]
                d = max(0.5, z0 + dzdt*dt)
                x = x0 + dxdt*dt
                vc, vx = -dzdt, dxdt
                settled = (tr.t_f - tr.t_first) >= 0.5 or tr.n_vel >= 3
                valid = tr.n_upd >= 3 and settled and math.sqrt(max(tr.kd.P[1][1], 0)) < 4.0
                rec.update(dist=round(d, 1), x=round(x, 1), bearing=round(math.degrees(math.atan2(x, d)), 1),
                           zone="ahead" if abs(x) < LANE_HALF else ("left" if x < 0 else "right"))
                if tr.loom and tr.loom > 0.05 and age < 0.5:
                    rec["ttc_loom"] = round(1/tr.loom, 1)
                if valid:
                    if tr.motion == "steady":
                        tr.motion = "closing" if vc > 1.0 else "opening" if vc < -1.0 else "steady"
                    elif tr.motion == "closing" and vc < 0.4: tr.motion = "steady"
                    elif tr.motion == "opening" and vc > -0.4: tr.motion = "steady"
                    rec.update(valid=True, speed=round(vc, 1), vx=round(vx, 1), motion=tr.motion,
                               ttc=round(d/vc, 1) if vc > 0.5 else None)
                    if tr.ego_known and known:
                        vz_g, vx_g = tr.kd.x[1], tr.kx.x[1]
                        tr.dir = self._direction(tr, vz_g, vx_g)
                        rec.update(abs_kmh=round(math.hypot(vz_g, vx_g)*3.6),
                                   heading=round(math.degrees(math.atan2(vx_g, vz_g))), dir=tr.dir)
            out.append(rec)
        return out
