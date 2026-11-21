"""Turns tracked objects into driver advice: CLEAR / CAUTION / BRAKE / BRAKE HARD.
Uses closing speed (relative), time-to-collision and the deceleration needed to stop before the object.
Raises quickly, lowers slowly, so the banner does not flicker."""
from tracking import LANE_HALF

TITLES = ["CLEAR", "CAUTION", "BRAKE", "BRAKE HARD"]
ACTIONS = ["Road ahead is clear", "Be ready to brake", "Brake now", "EMERGENCY - brake hard"]
STOP_GAP = 2.0       # m we want left between us and the object when stopped


class Advisor:
    def __init__(self):
        self.level, self.t_up, self.t_down, self.info = 0, None, None, None

    def evaluate(self, tracks, ego_mps):
        best, best_t = 0, None
        for t in tracks:
            t["risk"] = 0
            if not t["valid"] or t["age"] > 1.2:
                continue
            d, vc, x, vx = t["dist"], t["speed"], t["x"], t["vx"]
            in_path = abs(x) < LANE_HALF
            ttc = d/vc if vc > 0.5 else None
            cut_in = bool(ttc is not None and not in_path and abs(vx) > 0.4 and x*vx < 0
                          and abs(x + vx*min(ttc, 3.0)) < 0.9*LANE_HALF)
            risk, a_req, why = 0, None, ""
            if (in_path or cut_in) and ttc is not None:
                a_req = vc*vc/(2*max(d - STOP_GAP, 0.5))
                if ttc < 1.2 or a_req > 7: risk = 3
                elif ttc < 2.5 or a_req > 4: risk = 2
                elif ttc < 4.5 or a_req > 2: risk = 1
                if cut_in and not in_path: risk, why = min(risk, 2), "cutting in"
            if in_path and ego_mps > 1.0 and d/ego_mps < 0.8 and d < 30:
                if risk < 1: risk, why = 1, "following too close"
            t["risk"] = risk
            if risk > best or (risk == best and risk > 0 and best_t and d < best_t["dist"]):
                best, best_t = risk, dict(t, a_req=a_req, why=why)
        return best, best_t

    def update(self, tracks, now, ego_mps=0.0):
        raw, tgt = self.evaluate(tracks, ego_mps)
        if raw > self.level:
            self.t_down = None
            self.t_up = now if self.t_up is None else self.t_up
            if now - self.t_up >= (0.1 if raw == 3 else 0.25):
                self.level, self.t_up = raw, None
        else:
            self.t_up = None
            if raw < self.level:
                self.t_down = now if self.t_down is None else self.t_down
                if now - self.t_down >= 1.0:
                    self.level, self.t_down = raw, None
            else:
                self.t_down = None
        if tgt: self.info = tgt
        if self.level == 0: self.info = None
        detail, target = "", None
        if self.info:
            i = self.info
            target = i["id"]
            detail = f'{i["cls"]} #{i["id"]} · {i["dist"]:.0f} m · closing {i["speed"]:.1f} m/s'
            if i["ttc"] is not None: detail += f' · {i["ttc"]:.1f} s to impact'
            if i.get("a_req"): detail += f' · needs {i["a_req"]:.1f} m/s² braking'
            if i.get("why"): detail += f' · {i["why"]}'
        return dict(level=self.level, title=TITLES[self.level], action=ACTIONS[self.level], detail=detail, target=target)
