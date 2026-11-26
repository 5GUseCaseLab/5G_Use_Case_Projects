import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vision.collision_predictor import assess

cfg = {}
cases = [
    ("head-on",        {'name': 'person', 'distance': 4, 'x_ref': 0.0, 'z_ref': 4, 'vx': 0, 'vz': -1.4, 'half_w': 0.3, 'hits': 6}, {"WARNING", "CRITICAL"}),
    ("passing beside", {'name': 'person', 'distance': 4, 'x_ref': 1.5, 'z_ref': 4, 'vx': 0, 'vz': -1.4, 'half_w': 0.3, 'hits': 6}, {"CLEAR", "ADVISORY"}),
    ("crossing",       {'name': 'bicycle', 'distance': 2, 'x_ref': -3.0, 'z_ref': 2, 'vx': 1.5, 'vz': 0, 'half_w': 0.4, 'hits': 6}, {"WARNING", "CRITICAL"}),
    ("pole, walking",  {'name': 'fire hydrant', 'distance': 3, 'x_ref': 0.2, 'z_ref': 3, 'vx': 0, 'vz': -1.2, 'half_w': 0.2, 'hits': 6}, {"WARNING", "CRITICAL"}),
    ("chair, standing",{'name': 'chair', 'distance': 2, 'x_ref': 0.2, 'z_ref': 2, 'vx': 0, 'vz': 0, 'half_w': 0.3, 'hits': 6}, {"ADVISORY"}),
    ("fast car",       {'name': 'car', 'distance': 8, 'x_ref': 0.5, 'z_ref': 8, 'vx': 0, 'vz': -5.0, 'half_w': 0.9, 'hits': 6}, {"WARNING", "CRITICAL"}),
]
ok = True
for name, t, expected in cases:
    a = assess(t, cfg)
    passed = a["level"] in expected
    ok &= passed
    print(f"{'✅' if passed else '❌'} {name}: level={a['level']} action={a['action']} t*={a['t_star']:.2f}s d*={a['d_star']:.2f}m")
print("ALL PASSED" if ok else "SOME FAILED")
