"""Offline tests for the non-camera logic:  python tests/test_logic.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import common


def test_zone_and_feet():
    poly = [(0, 0), (100, 0), (100, 100), (0, 100)]
    boxes = [[10, 10, 30, 50], [90, 60, 130, 140], [200, 10, 220, 50]]  # 2nd & 3rd have feet outside
    assert common.in_zone(common.foot_points(boxes), poly) == [True, False, False]


def test_state_machine():
    s = common.DensityState(area_m2=10, window=3)
    seq = [5, 5, 5, 25, 25, 25, 45, 45, 45, 45, 45, 30, 30, 30, 30, 30, 5, 5, 5, 5, 5]
    states = [s.update(c)[0] for c in seq]
    assert states[0] == "GREEN" and "YELLOW" in states and "RED" in states and states[-1] == "GREEN"
    s2 = common.DensityState(area_m2=10, window=1)
    assert s2.update(21)[0] == "YELLOW"
    assert s2.update(19)[0] == "YELLOW"     # 1.9/m2 -> still yellow (hysteresis)
    assert s2.update(15)[0] == "GREEN"


def test_polygon_area_with_obstacle():
    H = np.eye(3)
    sq = [(0, 0), (4, 0), (4, 3), (0, 3)]
    ob = [(1, 1), (2, 1), (2, 2), (1, 2)]
    assert abs(common.polygon_area_m2(H, sq, [ob]) - 11.0) < 1e-4


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
