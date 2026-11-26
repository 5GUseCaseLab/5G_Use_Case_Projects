"""
Route Optimizer Module for 5G Assistive Wearable
=================================================
Calculates route cost (combining path distance and turn penalties),
evaluates remaining cost on active navigation, and determines if a newly
discovered route offers clear, significant savings before switching.
"""

TURN_TYPES = ("turn", "fork", "end of road", "roundabout", "rotary")

def route_cost(steps, distance_m, turn_penalty=15.0):
    turns = sum(1 for s in steps if s.get("maneuver") in TURN_TYPES and (s.get("modifier") or "") not in ("straight", ""))
    return distance_m + turn_penalty * turns

def remaining_cost(nav_state, turn_penalty=15.0, pt_spacing_m=5.0):
    pts, idx = nav_state.get("route_points", []), nav_state.get("user_route_idx", 0)
    steps, cur = nav_state.get("structured_steps", []), nav_state.get("step_index", 0)
    return route_cost(steps[cur + 1:], max(0, len(pts) - 1 - idx) * pt_spacing_m, turn_penalty)

def is_clearly_better(new_cost, old_cost, min_gain_m=30.0, min_gain_pct=0.15):
    return new_cost < old_cost - max(min_gain_m, min_gain_pct * old_cost)
