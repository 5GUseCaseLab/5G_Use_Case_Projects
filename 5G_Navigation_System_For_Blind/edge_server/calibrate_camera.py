"""
Camera Homography Calibration Helper for 5G Assistive Wearable
==============================================================
Calibrates ground-plane projective homography parameters using two physical distance measurements:
1. Have a person stand at exactly 2.0 meters, record bottom bounding box pixel (by2_2).
2. Have the person stand at exactly 5.0 meters, record bottom bounding box pixel (by2_5).

Calculates:
- Horizon y-coordinate: y_h = (5 * by2_5 - 2 * by2_2) / 3
- Focal length (px):     f   = 2.0 * (by2_2 - y_h) / cam_height
- cam_horizon_frac = y_h / frame_height
- cam_focal_frac_y = f / frame_height
"""

import os
import json
import sys

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "navigation", "nav_config.json")

def calibrate(by2_2, by2_5, cam_height_m=1.25, frame_height=320, save=False):
    y_h = (5.0 * by2_5 - 2.0 * by2_2) / 3.0
    f = 2.0 * (by2_2 - y_h) / cam_height_m
    horizon_frac = round(y_h / frame_height, 3)
    focal_frac = round(f / frame_height, 3)

    print("\n" + "=" * 50)
    print("      CAMERA CALIBRATION RESULTS")
    print("=" * 50)
    print(f"Inputs:")
    print(f"  Bottom edge at 2.0m (by2_2) : {by2_2:.1f} px")
    print(f"  Bottom edge at 5.0m (by2_5) : {by2_5:.1f} px")
    print(f"  Camera height (h)          : {cam_height_m:.2f} m")
    print(f"  Frame height (H)           : {frame_height} px")
    print("-" * 50)
    print(f"Calculated Parameters:")
    print(f"  Horizon Line (y_h)         : {y_h:.1f} px")
    print(f"  Focal Length (f_y)         : {f:.1f} px")
    print(f"  cam_horizon_frac           : {horizon_frac}")
    print(f"  cam_focal_frac_y           : {focal_frac}")
    print("=" * 50)

    if save and os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f_cfg:
                cfg = json.load(f_cfg)
            cfg["cam_height_m"] = cam_height_m
            cfg["cam_horizon_frac"] = horizon_frac
            cfg["cam_focal_frac_y"] = focal_frac
            with open(CONFIG_PATH, "w", encoding="utf-8") as f_cfg:
                json.dump(cfg, f_cfg, indent=2)
            print(f"✅ Successfully updated {CONFIG_PATH} with calibrated values!")
        except Exception as e:
            print(f"⚠️ Could not update {CONFIG_PATH}: {e}")

    return horizon_frac, focal_frac

if __name__ == "__main__":
    if len(sys.argv) >= 3:
        b2 = float(sys.argv[1])
        b5 = float(sys.argv[2])
        h = float(sys.argv[3]) if len(sys.argv) > 3 else 1.25
        calibrate(b2, b5, cam_height_m=h, save=True)
    else:
        print("Usage: python calibrate_camera.py <by2_at_2m> <by2_at_5m> [cam_height_m]")
        print("Example: python calibrate_camera.py 260 170 1.25")
