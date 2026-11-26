import math

SEVERITY = {'car': 3, 'bus': 3, 'truck': 3, 'motorcycle': 3, 'bicycle': 3, 'train': 3,
            'person': 2, 'dog': 2, 'horse': 2, 'cow': 2}

def assess(t, cfg):
    x, z = t.get('x_ref', t.get('x', 0.0)), t.get('z_ref', t['distance'])
    vx, vz = t.get('vx', 0.0), t.get('vz', 0.0)
    sev = SEVERITY.get(t['name'], 1)
    R = cfg.get("user_half_width_m", 0.35) + min(t.get('half_w', 0.3), 1.2) + cfg.get("collision_margin_m", 0.25)
    horizon = cfg.get("cpa_horizon_vehicle_s", 5.0) if sev == 3 else cfg.get("cpa_horizon_s", 3.5)
    v2 = vx * vx + vz * vz
    if v2 < 0.04:
        t_star, d_star = float('inf'), math.hypot(x, z)
    else:
        t_star = max(0.0, -(x * vx + z * vz) / v2)
        d_star = math.hypot(x + vx * t_star, z + vz * t_star)

    crossing = abs(vx) > 0.5 and abs(vx) > abs(vz) and not t.get('edge') and t.get('hits', 0) >= 6
    in_path_now = abs(x) <= R and z <= cfg.get("vision_immediate_m", 1.5)
    crossing_in_path = crossing and abs(x + vx * t_star) <= R and z <= cfg.get("cpa_crossing_range_m", 3.5)
    on_course = (d_star <= R or crossing_in_path) and t_star <= horizon
    near_miss = d_star <= R + cfg.get("near_miss_extra_m", 0.5) and t_star <= horizon + 1.0
    if in_path_now or (on_course and t_star <= cfg.get("cpa_critical_s", 1.5)):
        level = "CRITICAL"
    elif on_course:
        level = "WARNING"
    elif near_miss or (abs(x) <= R and z <= 2.0):
        level = "ADVISORY"
    else:
        level = "CLEAR"

    side = "left" if x < -0.3 else ("right" if x > 0.3 else "ahead")
    if level == "CRITICAL" or sev == 3 or crossing or abs(x) <= 0.15:
        action = "STOP"
    else:
        action = "STEP_LEFT" if x > 0 else "STEP_RIGHT"
    rank = {"CRITICAL": 3, "WARNING": 2, "ADVISORY": 1, "CLEAR": 0}[level]
    risk = rank * 10 + sev - min(t_star, 9) * 0.1
    return {"level": level, "action": action, "side": side, "t_star": t_star,
            "d_star": d_star, "crossing": crossing, "risk": risk, "distance": z}

def phrase(t, a, dist_words):
    name = t['name']
    if a["crossing"]:
        return f"Stop. {name} crossing from your {'right' if t.get('vx', 0) < 0 else 'left'}."
    if a["action"] == "STOP":
        return f"Stop! {name} {a['side']}, {dist_words(a['distance'])}."
    return f"{name} {a['side']}, {'step left' if a['action'] == 'STEP_LEFT' else 'step right'}."
