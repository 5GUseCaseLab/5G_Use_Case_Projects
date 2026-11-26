"""
5G Assistive Wearable - Edge Server Orchestrator
================================================
Central orchestration engine for the 5G Assistive Wearable system.
Connects all decoupled subsystems:
- Network Ingestion (UDP Wearable + 5G Mobile Web/WebSocket Bridge)
- Vision AI Pipeline (YOLOv8 Detection, MiDaS Depth, Optical Motion)
- Navigation & Voice Guidance (OSRM Routing, Campus Snapping, Whisper STT, SAPI TTS)
- HUD & Live Map Synchronization
"""

import os
import sys
import time
import json
import copy
import threading
import collections
import math
import cv2
import numpy as np

# Ensure terminal uses UTF-8
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Ensure sibling packages can be resolved
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 1. Network Handlers
from udp_network_handler import UDPNetworkHandler
from mobile_bridge import MobileBridgeServer
from tunnel_manager import CloudflareTunnelManager, print_terminal_qr

# 2. Vision Subsystem (Decoupled Package)
from vision import (
    YoloDetector,
    DepthEstimator,
    MotionDetector,
    ThreatTracker,
    assess,
    phrase,
    draw_navigation_hud,
    draw_threat_boxes,
    draw_qr_overlay
)

# 3. Navigation & Audio AI Subsystem (Decoupled Package)
from navigation import (
    get_detailed_route,
    snap_to_campus_landmark,
    clean_destination_query,
    get_voice_command,
    get_whisper_model,
    speak_text,
    get_live_location,
    haversine_distance_km,
    route_cost,
    remaining_cost,
    is_clearly_better
)

# File paths
MAP_HTML_PATH = os.path.join(current_dir, "live_map.html")
NAV_STATE_PATH = os.path.join(current_dir, "navigation_state.json")

# Real-world pedestrian collision hazard classes and threat priorities (COCO classes)
HAZARD_PRIORITY = {   # higher = more dangerous
    'car': 3, 'bus': 3, 'truck': 3, 'motorcycle': 3, 'bicycle': 3, 'train': 3,
    'person': 2, 'dog': 2, 'horse': 2, 'cow': 2,
    'bench': 1, 'chair': 1, 'couch': 1, 'potted plant': 1, 'dining table': 1, 'bed': 1,
    'fire hydrant': 1, 'stop sign': 1, 'parking meter': 1, 'suitcase': 1,
    'refrigerator': 1, 'tv': 1, 'toilet': 1
}
PEDESTRIAN_COLLISION_CLASSES = set(HAZARD_PRIORITY)   # existing name kept, so other code is unaffected

# Global navigation state and synchronization locks
nav_lock = threading.RLock()
is_listening_nav = False
is_mic_capturing = False
MOBILE_AUDIO_ONLY = True  # Direct all audio instructions strictly to mobile earphones when connected
EXTERNAL_MAPS_NAV = False  # Full autonomous edge server audio guidance; Google Maps is not required

nav_state = {
    "is_active": False,
    "destination_name": "Standby (Press 'T')",
    "start_coords": (26.1885, 91.6917),
    "dest_coords": None,
    "current_gps": None,
    "current_step": "Press 'T' or Push Button on Wearable for Voice Navigation",
    "structured_steps": [],
    "all_steps": [],
    "step_index": 0,
    "total_steps": 0,
    "total_distance_m": 0,
    "route_points": [],
    "last_step_advance_time": 0.0,
    "pre_turn_announced": False,
    "last_milestone_announced": 99999,
    "last_progress_announce_time": 0.0,
    "gps_buf": [],
    "arrive_count": 0,
    "gps_was_weak": False,
    "good_gps_start_time": None,
    "last_weak_gps_warn": 0.0,
    "last_restored_gps_warn": 0.0,
    "phone_disconnect_time": None,
    "approach_prev_gap": 0.0,
    "approach_away_count": 0,
    "last_approach_hint": 0.0,
    "route_peak_idx": 0,
    "backtrack_count": 0,
    "last_backtrack_warn": 0.0,
    "last_any_guidance": 0.0,
    "last_dist_to_target_m": None,
    "last_updated": time.time()
}

def sync_navigation_state_file():
    """
    Saves the live navigation state snapshot to navigation_state.json
    so that live_map.html (Leaflet) updates in real-time.
    """
    try:
        with nav_lock:
            state_snapshot = copy.deepcopy(nav_state)
            state_snapshot.pop("gps_buf", None)
        with open(NAV_STATE_PATH, "w") as f:
            json.dump(state_snapshot, f, indent=2, default=list)
    except Exception:
        pass

def calculate_bearing(p1, p2):
    """Calculates forward compass bearing between two coordinates in degrees (0-360) (Problem 4b & 9)."""
    lat1, lon1 = math.radians(p1[0]), math.radians(p1[1])
    lat2, lon2 = math.radians(p2[0]), math.radians(p2[1])
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0

NAV_CONFIG_PATH = os.path.join(current_dir, "navigation", "nav_config.json")
def load_nav_config():
    defaults = {
        "step_length_m": 0.7,
        "quiet_period_sec": 1.5,
        "off_route_min_distance_m": 35.0,
        "off_route_accuracy_multiplier": 1.5,
        "off_route_readings_threshold": 8,
        "wrong_dir_readings_threshold": 3,
        "turn_execution_radius_min_m": 5.0,
        "turn_execution_radius_max_m": 10.0,
        "turn_execution_closeness_m": 15.0,
        "get_ready_distance_m": 22.0,
        "pre_turn_warning_min_m": 25.0,
        "pre_turn_warning_max_m": 55.0,
        "reassurance_bucket_interval_m": 40.0,
        "reassurance_cooldown_sec": 15.0,
        "gps_weak_acc_threshold_m": 30.0,
        "gps_restored_acc_threshold_m": 20.0,
        "stationary_max_gps_age_sec": 30.0,
        "moving_max_gps_age_sec": 10.0,
        "destination_confirm_timeout_sec": 15.0,
        "steps_vs_meters_limit_m": 30.0,
        "arrival_radius_min_m": 10.0,
        "step_advance_cooldown_sec": 3.0,
        "pre_turn_audio_cooldown_sec": 8.0,
        "reroute_cooldown_sec": 20.0,
        "wrong_dir_cooldown_sec": 12.0,
        "corridor_catchup_points_ahead": 6,
        "passed_min_dist_m": 12.0,
        "ultra_contact_distance_cm": 25.0,
        "ultra_critical_distance_cm": 50.0,
        "laptop_speaks_alerts": True,
        "depth_on_demand": True,
        "ttc_alerts_enabled": True,
        "ttc_warning_s": 3.0,
        "ttc_critical_s": 1.5,
        "vision_immediate_m": 1.0,
        "approach_gap_m": 25.0,
        "reopt_enabled": False,
        "max_silence_sec": 30,
        "reopt_timer_sec": 150,
        "reopt_min_interval_sec": 45,
        "reopt_switch_cooldown_sec": 120,
        "reopt_no_switch_near_turn_m": 30,
        "reopt_min_gain_m": 30,
        "reopt_min_gain_pct": 0.15,
        "vision_min_hits": 3,
        "path_half_width_m": 0.75,
        "cam_focal_frac_x": 0.9,
        "depth_fusion_weight": 0.0,
        "cam_height_m": 1.25,
        "cam_focal_frac_y": 0.85,
        "cam_horizon_frac": 0.40
    }
    if os.path.exists(NAV_CONFIG_PATH):
        try:
            with open(NAV_CONFIG_PATH, "r", encoding="utf-8") as f:
                defaults.update(json.load(f))
        except Exception:
            pass
    return defaults

NAV_CONFIG = load_nav_config()
STEP_LENGTH_M = NAV_CONFIG.get("step_length_m", 0.7)

def dist_words(m):
    """Speaks short distances in pedestrian steps and longer distances in meters (Problem 10)."""
    s = int(round(m / STEP_LENGTH_M))
    limit = NAV_CONFIG.get("steps_vs_meters_limit_m", 30.0)
    return f"{s} steps" if m < limit else f"{int(round(m / 5.0) * 5)} meters"

def announce_guidance(text, mobile_bridge=None, async_mode=True, priority="normal", haptic=None):
    """
    Centralized Audio Guidance Engine.
    - If 5G mobile phone client is active: Dispatches instruction directly to user's mobile earphones with optional haptic cue.
    - If MOBILE_AUDIO_ONLY is True and mobile phone is connected: Mutes laptop speakers for clean earphone operation.
    - If no phone client is connected: Speaks via laptop speakers (fallback for local workstation testing or standalone wearable).
    """
    nav_state["last_any_guidance"] = time.time()
    has_phone = (mobile_bridge is not None and hasattr(mobile_bridge, 'has_phone_clients') and mobile_bridge.has_phone_clients())
    if has_phone:
        print(f"🎧 [AUDIO -> 5G EARPHONES ({priority.upper()})]: \"{text}\"")
        mobile_bridge.send_speech(text, priority=priority, haptic=haptic)
        if MOBILE_AUDIO_ONLY:
            return
    else:
        if mobile_bridge and getattr(mobile_bridge, 'has_active_clients', lambda: False)():
            print("⚠️ [AUDIO NOTICE]: Wearable/Camera connected but no phone detected. Speaking via Laptop speakers.")
            
    speak_text(text, async_mode=async_mode)

def terminal_key_listener(network, mobile_bridge=None):
    """
    Listens for keyboard shortcuts pressed directly inside the VS Code terminal.
    Allows user to trigger voice navigation without needing mouse focus on the monitor.
    """
    try:
        import msvcrt
        while True:
            if msvcrt.kbhit():
                ch = msvcrt.getch().decode('utf-8', errors='ignore').lower()
                if ch in ('t', ' ', '\r', '\n'):  # 't', Spacebar, or Enter
                    threading.Thread(target=handle_voice_navigation, args=(network, mobile_bridge), daemon=True).start()
            time.sleep(0.04)
    except Exception:
        pass

def densify_polyline(points, step_m=5.0):
    """
    Densifies a polyline so adjacent coordinates are at most ~step_m meters apart (Bug 1).
    Reduces worst-case waypoint arrival mapping error from ~27m down to ~2.5m.
    """
    if not points or len(points) < 2:
        return points or []
    out = [points[0]]
    for a, b in zip(points, points[1:]):
        d = haversine_distance_km((a[0], a[1]), (b[0], b[1])) * 1000.0
        n = max(1, math.ceil(d / step_m))
        for i in range(1, n + 1):
            t = i / float(n)
            out.append([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t])
    return out

def reopt_check(network, mobile_bridge, gps, rid):
    """Periodically evaluates whether a substantially shorter route exists."""
    try:
        res = get_detailed_route(gps, nav_state["destination_name"])
        if not res.get("success") or nav_state.get("route_id") != rid:
            return
        pts = res.get("route_points") or []
        if pts and haversine_distance_km(gps, pts[0]) * 1000.0 > NAV_CONFIG.get("approach_gap_m", 25.0):
            return
        new_c = route_cost(res.get("structured_steps", []), res["total_distance_m"])
        old_c = remaining_cost(nav_state)
        if is_clearly_better(new_c, old_c, NAV_CONFIG.get("reopt_min_gain_m", 30.0), NAV_CONFIG.get("reopt_min_gain_pct", 0.15)):
            nav_state["last_route_switch"] = time.time()
            handle_voice_navigation(network, mobile_bridge, nav_state["destination_name"], True, res, "optimize")
    except Exception as e:
        print(f"⚠️ [REOPT SKIPPED]: {e}")
    finally:
        nav_state["reopt_running"] = False

def handle_voice_navigation(network, mobile_bridge=None, direct_destination=None,
                            is_exact=False, precomputed_route=None, quiet_reason=None):
    """
    Asynchronous voice navigation handler.
    Runs on a background worker thread so real-time camera inference never pauses.
    """
    global is_listening_nav, is_mic_capturing, nav_state
    
    with nav_lock:
        if is_listening_nav and not direct_destination and not precomputed_route:
            print("⏳ Voice Navigation is currently active. Please speak your destination...")
            return
        is_listening_nav = True

    try:
        if precomputed_route:
            destination = target_dest = direct_destination or precomputed_route.get("destination_name")
            needs_confirm = False
        elif direct_destination:
            destination_raw = str(direct_destination).strip()
            print(f"\n🎙️  [IPHONE 5G VOICE COMMAND]: '{destination_raw}'")
            cleaned = clean_destination_query(destination_raw)
            snapped, conf, corrected = snap_to_campus_landmark(cleaned)
            target_dest = snapped if snapped else (cleaned or destination_raw)
            needs_confirm = not is_exact
        else:
            nav_state["current_step"] = "🎙️ Listening for destination..."
            sync_navigation_state_file()
            
            # Announce prompt before opening mic to avoid feedback loop
            announce_guidance("Where do you want to go?", mobile_bridge=mobile_bridge, async_mode=False)
            time.sleep(0.1)
            
            with nav_lock:
                is_mic_capturing = True
            try:
                destination_raw = get_voice_command()
            finally:
                with nav_lock:
                    is_mic_capturing = False

            if not destination_raw:
                announce_guidance("No destination heard. Navigation cancelled.", mobile_bridge=mobile_bridge)
                nav_state["current_step"] = "Press 'T' or Push Button to Retry Voice Navigation"
                sync_navigation_state_file()
                print("\n👉 Ready! Press 'T' or Enter in the terminal to try again.")
                return
            cleaned = clean_destination_query(destination_raw)
            snapped, conf, corrected = snap_to_campus_landmark(cleaned)
            target_dest = snapped if snapped else (cleaned or destination_raw)
            needs_confirm = True

        # Bug 3: Spoken destinations require double-tap confirmation; typed entries (is_exact=True) skip
        if needs_confirm:
            if mobile_bridge:
                mobile_bridge.pop_destination_confirmation()  # Clear any old double-tap
            confirm_msg = f"{target_dest}. Double-tap the screen to confirm, or tap the speak button to try again."
            announce_guidance(confirm_msg, mobile_bridge=mobile_bridge)
            nav_state["pending_destination"] = (target_dest, time.time())
            start_wait = time.time()
            confirmed = False
            timeout_sec = NAV_CONFIG.get("destination_confirm_timeout_sec", 15.0)
            while (time.time() - start_wait) < timeout_sec:
                if mobile_bridge and mobile_bridge.pop_destination_confirmation():
                    confirmed = True
                    break
                time.sleep(0.15)
            if not confirmed:
                announce_guidance("Destination not confirmed. Navigation cancelled.", mobile_bridge=mobile_bridge)
                nav_state["pending_destination"] = None
                return
            destination = target_dest
            nav_state["pending_destination"] = None
        else:
            destination = target_dest

        nav_state["current_step"] = f"🧭 Routing to: {destination}"
        sync_navigation_state_file()
        if quiet_reason not in ("optimize", "offroute"):
            announce_guidance(f"Calculating walking route to {destination}...", mobile_bridge=mobile_bridge, async_mode=False)
        
        # 1. Determine Starting Location with GPS Quality Verification (Bug 9)
        if precomputed_route:
            current_gps = precomputed_route.get("start_coords") or get_live_location()
            current_acc = mobile_bridge.get_latest_accuracy() if mobile_bridge else 5.0
            route_res = precomputed_route
        else:
            server_ip = UDPNetworkHandler.get_local_ip()
            gps_tuple, gps_source, gps_age = (None, None, 999.0)
            current_acc = 999.0
            if mobile_bridge:
                gps_tuple, gps_source, gps_age = mobile_bridge.get_latest_gps_with_metadata()
                current_acc = mobile_bridge.get_latest_accuracy()
            if not gps_tuple or gps_age >= 10.0 or current_acc > 30.0:
                if network:
                    n_tuple, n_src, n_age = network.get_latest_gps_with_metadata()
                    if n_tuple and n_age < 10.0:
                        gps_tuple, gps_source, gps_age = n_tuple, n_src, n_age
                        current_acc = 15.0

            is_preset = (gps_source in ("preset", "manual"))
            is_gps_valid = is_preset or (gps_tuple and gps_age < 10.0 and current_acc <= 30.0)

            # Retry up to 10 seconds if GPS is settling (Bug 9)
            if not is_gps_valid and not is_preset:
                announce_guidance("Getting your location, please wait.", mobile_bridge=mobile_bridge, async_mode=False)
                t_start = time.time()
                while (time.time() - t_start) < 10.0:
                    time.sleep(1.0)
                    if mobile_bridge:
                        gps_tuple, gps_source, gps_age = mobile_bridge.get_latest_gps_with_metadata()
                        current_acc = mobile_bridge.get_latest_accuracy()
                        if gps_tuple and gps_age < 10.0 and current_acc <= 30.0:
                            is_gps_valid = True
                            break

            if is_gps_valid and gps_tuple:
                current_gps = gps_tuple
                mode_label = "5G REMOTE SATELLITE" if gps_source in ("live", "live_watch", "network", "iphone_5g") else ("CAMPUS PRESET" if is_preset else "LIVE SATELLITE")
                nav_state["gps_mode"] = mode_label
                print("\n" + "="*58)
                print(f"🛰️  [LOCATION SOURCE: {mode_label} GPS (±{current_acc:.1f}m)]")
                print(f"📍 Active Coordinates : ({current_gps[0]:.5f}, {current_gps[1]:.5f})")
                print(f"⏱️  Satellite Fix Age  : {gps_age:.1f}s ago")
                print("="*58 + "\n")
            else:
                current_gps = get_live_location()
                nav_state["gps_mode"] = "FALLBACK"
                reason = f"GPS degraded (Acc: ±{current_acc:.1f}m, Age: {gps_age:.1f}s)" if gps_tuple else "No live GPS packets received yet"
                print("\n" + "="*58)
                print("⚠️  [LOCATION SOURCE: FALLBACK LOCATION]")
                print(f"📍 Active Coordinates : ({current_gps[0]:.5f}, {current_gps[1]:.5f})")
                print(f"ℹ️  Fallback Reason    : {reason}")
                print(f"👉 Target in iPhone App: Host {server_ip}, Port 5006 (UDP) or 5G Web Link")
                print("="*58 + "\n")
                if not is_preset:
                    announce_guidance("I can't find your location. Please move to an open area and try again.", mobile_bridge=mobile_bridge)
                    nav_state["current_step"] = "⚠️ Location unavailable. Please move to an open area."
                    sync_navigation_state_file()
                    return
                
            # 2. Compute Complete Navigation Route & Map Geometry
            route_res = get_detailed_route(current_gps, destination)
        
        if route_res["success"] and len(route_res.get("steps", [])) > 0:
            steps_list = route_res.get("steps", [])
            structured_steps = route_res.get("structured_steps", [])
            if not structured_steps:
                structured_steps = [
                    {"step_number": i + 1, "instruction": s, "distance_m": 0, "waypoint": [0, 0]}
                    for i, s in enumerate(steps_list)
                ]
            first_step = structured_steps[0]["instruction"]
            
            # Bug 1 & Problem 6: Densify route polyline to ~5m intervals and match turns sequentially
            r_points = densify_polyline(route_res.get("route_points", []), step_m=5.0)
            route_res["route_points"] = r_points
            
            # Map each step waypoint to nearest index in densified route_points in order (Problem 6)
            start_search_idx = 0
            for step_obj in structured_steps:
                wp = step_obj.get("waypoint")
                if wp and wp[0] != 0 and r_points:
                    best_i, best_d = start_search_idx, float('inf')
                    for r_idx in range(start_search_idx, len(r_points)):
                        rp = r_points[r_idx]
                        d = haversine_distance_km((wp[0], wp[1]), (rp[0], rp[1]))
                        if d < best_d:
                            best_d = d
                            best_i = r_idx
                    step_obj["route_idx"] = best_i
                    start_search_idx = best_i
                else:
                    step_obj["route_idx"] = start_search_idx

            # Approach phase check (Group C2)
            gap_to_path_m = haversine_distance_km(current_gps, r_points[0]) * 1000.0 if r_points else 0.0

            # Update Global Navigation State (Thread-safe with route_id tracking to prevent race conditions)
            with nav_lock:
                nav_state["is_active"] = True
                nav_state["route_id"] = nav_state.get("route_id", 0) + 1
                nav_state["destination_name"] = route_res["destination_name"]
                nav_state["start_coords"] = route_res["start_coords"]
                nav_state["dest_coords"] = route_res["end_coords"]
                nav_state["current_gps"] = current_gps
                nav_state["structured_steps"] = structured_steps
                nav_state["all_steps"] = [s["instruction"] for s in structured_steps]
                nav_state["step_index"] = 0
                nav_state["total_steps"] = len(structured_steps)
                nav_state["total_distance_m"] = route_res["total_distance_m"]
                nav_state["route_points"] = r_points
                nav_state["current_step"] = f"Step 1 of {len(structured_steps)}: {first_step}"
                nav_state["last_step_advance_time"] = time.time()
                nav_state["pre_turn_announced"] = False
                nav_state["ready_announced"] = False
                nav_state["arrival_announced"] = False
                nav_state["pending_turn"] = None
                nav_state["pending_turn_msg"] = None
                nav_state["min_dist_to_target"] = 99999.0
                nav_state["user_route_idx"] = 0
                nav_state["passed_count"] = 0
                nav_state["route_passed_count"] = 0
                nav_state["arrive_count"] = 0
                nav_state["off_route_count"] = 0
                nav_state["wrong_dir_count"] = 0
                nav_state["approach_phase"] = (gap_to_path_m > NAV_CONFIG.get("approach_gap_m", 25.0)) if quiet_reason != "optimize" else False
                nav_state["approach_start"] = time.time()
                nav_state["approach_prev_gap"] = gap_to_path_m
                nav_state["approach_away_count"] = 0
                nav_state["last_approach_hint"] = 0.0
                nav_state["last_approach_clock"] = None
                nav_state["last_approach_gap_spoken"] = 1e9
                nav_state["orienting"] = (quiet_reason != "optimize") and not nav_state["approach_phase"]
                nav_state["orient_start"] = time.time()
                nav_state["last_orient_hint"] = 0.0
                nav_state["route_peak_idx"] = 0
                nav_state["backtrack_count"] = 0
                nav_state["last_backtrack_warn"] = 0.0
                nav_state["last_any_guidance"] = time.time()
                nav_state["last_dist_to_target_m"] = None
                nav_state["post_turn_checked"] = False
                nav_state["last_reroute"] = 0.0
                nav_state["last_wrong_dir_warn"] = 0.0
                nav_state["last_eval_gps_ts"] = 0.0
                nav_state["last_milestone_announced"] = 99999
                nav_state["last_progress_announce_time"] = time.time()
                nav_state["gps_buf"] = []
                if quiet_reason != "optimize":
                    nav_state["gps_was_weak"] = False
                nav_state["good_gps_start_time"] = None
                nav_state["last_weak_gps_warn"] = 0.0
                nav_state["last_restored_gps_warn"] = 0.0
                nav_state["phone_disconnect_time"] = None
                nav_state["start_acc"] = current_acc
                nav_state["reopt_gps_done"] = False
                nav_state["reopt_running"] = False
                nav_state["last_updated"] = time.time()

            # Construct direct Universal Google Maps walking navigation URL
            start_lat, start_lon = current_gps
            dest_coords = route_res.get("end_coords") or route_res.get("dest_coords") or (26.1928, 91.6985)
            dest_lat, dest_lon = dest_coords
            google_maps_url = f"https://www.google.com/maps/dir/?api=1&destination={dest_lat:.6f},{dest_lon:.6f}&travelmode=walking&dir_action=navigate"
            with nav_lock:
                nav_state["google_maps_url"] = google_maps_url

            # Sync navigation state file outside the main lock block
            sync_navigation_state_file()
            
            # Dispatch full route package over 5G to mobile client
            if mobile_bridge:
                mobile_bridge.latest_google_maps_url = google_maps_url
                mobile_bridge.send_nav_route(
                    destination=route_res["destination_name"],
                    steps=structured_steps,
                    total_distance_m=route_res["total_distance_m"],
                    google_maps_url=google_maps_url,
                    dest_coords=dest_coords,
                    start_coords=current_gps,
                    route_points=r_points
                )
                mobile_bridge.send_nav_step(f"Step 1 of {len(structured_steps)}: {first_step}", speak=False)
            
            # Announce the route found autonomously over 5G earphones (Group C3)
            if quiet_reason == "optimize":
                step_txt = first_step
                if step_txt.lower().startswith("head out on"):
                    step_txt = "Continue straight on" + step_txt[11:]
                elif step_txt.lower().startswith("head out"):
                    step_txt = "Continue straight" + step_txt[8:]
                announce_guidance(f"Found a shorter way. {step_txt}", mobile_bridge=mobile_bridge)
            else:
                announce_guidance(f"Route to {route_res['destination_name']} found. Step 1 of {len(structured_steps)}: {first_step}", mobile_bridge=mobile_bridge)

            # Lead the user onto the path first if gap exists (Group C2)
            if nav_state.get("approach_phase"):
                announce_guidance(f"First walk to the path, about {dist_words(gap_to_path_m)} away.", mobile_bridge=mobile_bridge)
            
            # Optional operator verification link (valid google.com URL, clickable in terminal)
            print(f"🗺️  [OPTIONAL GOOGLE MAPS LINK]: {google_maps_url}")
            
            # Send downlink packet over UDP to wearable client buzzer/display
            route_payload = {
                "type": "NAVIGATION_ROUTE",
                "destination": destination,
                "first_step": first_step,
                "all_steps": nav_state["all_steps"],
                "total_distance_m": route_res["total_distance_m"],
                "timestamp": time.time()
            }
            network.send_downlink_alert(route_payload)
            print(f"\n🗺️ [5G ROUTE SENT]: {len(structured_steps)} steps dispatched to mobile.")
        else:
            nav_state["current_step"] = "❌ Could not find walking route."
            sync_navigation_state_file()
            announce_guidance("Sorry, I could not find a walking route to that destination.", mobile_bridge=mobile_bridge)
            
    except Exception as e:
        print(f"⚠️ Navigation error: {e}")
        announce_guidance("An error occurred during navigation.", mobile_bridge=mobile_bridge)
    finally:
        with nav_lock:
            is_listening_nav = False
            is_mic_capturing = False

def run_edge_server():
    """
    Main edge server process. Initializes models and executes the real-time processing loop.
    """
    global nav_state
    
    print("==================================================")
    print("      5G WEARABLE ASSISTIVE SYSTEM - EDGE SERVER   ")
    print("==================================================")
    
    # 0. Calibrate Starting Location
    fallback_gps = get_live_location()
    nav_state["start_coords"] = fallback_gps
    sync_navigation_state_file()
    print(f"📍 Fallback starting location calibrated to: {fallback_gps}")
    
    # 1. Initialize Network Ingestion
    network = UDPNetworkHandler(host="0.0.0.0", video_port=5005, telemetry_port=5006)
    network.start()
    
    mobile_bridge = MobileBridgeServer(host="0.0.0.0", port=8080)
    mobile_bridge.start()
    
    tunnel_manager = CloudflareTunnelManager(port=8080, auto_start=True)
    tunnel_manager.start()
    show_qr_overlay = False
    
    server_ip = UDPNetworkHandler.get_local_ip()
    threading.Thread(target=terminal_key_listener, args=(network, mobile_bridge), daemon=True).start()
    
    # 2. Initialize Vision AI & Preload Speech Model
    print("\n⏳ Initializing Edge AI Vision & Speech Models...")
    # Using Nano model (yolov8n) for real-time 25ms inference latency (10x faster than medium)
    detector = YoloDetector('yolov8n.pt')
    HAZARD_IDS = [i for i, n in detector.model.names.items() if n in PEDESTRIAN_COLLISION_CLASSES]
    depth_model = DepthEstimator()
    motion_detector = MotionDetector(walk_threshold=4.5)
    threat_tracker = ThreatTracker()
    last_proximity_sent = 0.0
    threading.Thread(target=get_whisper_model, daemon=True).start()
    print("✅ All AI models loaded into Server Memory!")
    
    # Ground-Plane Homography & Collision Warning Configuration
    COLLISION_ALERT_DISTANCE_M = 3.5    # Warning zone (<= 3.5m)
    CRITICAL_STOP_DISTANCE_M = 1.5      # Critical alert zone (<= 1.5m)
    ALERT_COOLDOWN_WARNING_SEC = 3.0    # 3.0s cooldown gives clear, quiet space for turn navigation
    ALERT_COOLDOWN_CRITICAL_SEC = 2.0   # 2.0s cooldown for critical stop notices
    ALERT_COOLDOWN_IMMEDIATE_SEC = 3.5  # 3.5s cooldown for immediate physical contact bumper
    LEVEL = {"warning": 1, "critical": 2, "contact": 3}
    last_alert_time = 0
    last_alert_level = 0
    last_nav_sync_time = 0
    last_telemetry_sync_time = 0
    last_gps_state = "INITIAL"
    
    # Rate-of-Closure sliding window queues (tracks whether obstacle is actively closing in)
    ultra_history = collections.deque(maxlen=6)
    vision_history = collections.deque(maxlen=6)
    
    # Camera Source Management (Laptop webcam disabled by default to prioritize 5G mobile feed)
    allow_laptop_webcam = ("--webcam" in sys.argv)
    local_cam = None
    if allow_laptop_webcam:
        try:
            local_cam = cv2.VideoCapture(0)
            if not local_cam.isOpened():
                local_cam = None
            else:
                print("💻 [CAMERA MODE]: Laptop webcam enabled via --webcam flag.")
        except Exception:
            local_cam = None
    else:
        print("📱 [CAMERA MODE]: Dedicated 5G Mobile Phone Camera prioritized (Laptop webcam disabled).")
        print("ℹ️ Tip: Connect phone via QR Code on monitor or pass --webcam (or press 'w') for laptop webcam fallback.")
        
    print("\n🟢 Edge Server fully active and awaiting 5G packets!")
    print(f"📡 LOCAL UDP RECEIVER     : Listening on UDP Ports 5005 (Video) & 5006 (Sensors)")
    print(f"🌐 5G REMOTE WEBSOCKET    : Active on Port 8080 (Forwarded via Cloudflare Tunnel)")
    print(f"📱 IPHONE NMEA APP TARGET : Set Host = {server_ip}, Port = 5006 (UDP)")
    print(f"🗺️ GOOGLE MAPS NAVIGATION: Dynamic walking route generated automatically upon voice query")
    print("👉 Press 't', Space, or Enter in the TERMINAL or tap button on iPhone to trigger Voice Navigation.")
    print("👉 Press 'c' in the monitor window to reprint or toggle the 5G QR Code.")
    print("👉 Press 'w' in monitor to toggle laptop webcam fallback on/off.")
    print("👉 Press 'q' in the monitor window or Ctrl+C in terminal to terminate.\n")
    
    cv2.namedWindow("5G Edge Server - Master AI Monitor", cv2.WINDOW_NORMAL)
    
    wearable_has_connected_once = False
    wearable_disconnect_announced = False
    last_recorded_ultra_ts = 0.0
    phone_was_connected = False

    try:
        while True:
            try:
                # Phone reconnection recovery: resend route & step update if phone reconnects during active navigation
                phone_now = (mobile_bridge is not None and hasattr(mobile_bridge, 'has_phone_clients') and mobile_bridge.has_phone_clients())
                if phone_now and not phone_was_connected and nav_state.get("is_active") \
                        and (time.time() - nav_state.get("last_eval_gps_ts", 0)) < 300:
                    print("📱 [PHONE RECONNECTED]: Resending route geometry and active step to client...")
                    if mobile_bridge:
                        dest_c = nav_state.get("dest_coords") or (26.1928, 91.6985)
                        mobile_bridge.send_nav_route(
                            destination=nav_state.get("destination_name", "Destination"),
                            steps=nav_state.get("structured_steps", []),
                            total_distance_m=nav_state.get("total_distance_m", 0),
                            google_maps_url=nav_state.get("google_maps_url", ""),
                            dest_coords=dest_c,
                            start_coords=nav_state.get("start_coords"),
                            route_points=nav_state.get("route_points", [])
                        )
                        s_idx = nav_state.get("step_index", 0)
                        s_tot = nav_state.get("total_steps", 1)
                        s_steps = nav_state.get("structured_steps", [])
                        s_inst = s_steps[s_idx]["instruction"] if s_idx < len(s_steps) else ""
                        mobile_bridge.send_nav_step_update(s_idx, s_tot, s_inst, speak=False, resync=True)
                    resumed_msg = f"Guidance resumed. {nav_state.get('current_step', '')}"
                    announce_guidance(resumed_msg, mobile_bridge=mobile_bridge, priority="normal")
                phone_was_connected = phone_now

                # 3. Check for Voice Navigation Trigger
                if mobile_bridge.has_voice_command():
                    direct_cmd, is_exact = mobile_bridge.get_voice_command_metadata()
                    mobile_bridge.get_voice_command()
                    print(f"\n🎙️ Voice Navigation destination received directly from iPhone: '{direct_cmd}' (Exact: {is_exact})")
                    threading.Thread(target=handle_voice_navigation, args=(network, mobile_bridge, direct_cmd, is_exact), daemon=True).start()
                elif network.is_button_pressed() or mobile_bridge.is_button_pressed():
                    print("\n🔘 Push-to-Talk button triggered! Listening on microphone...")
                    threading.Thread(target=handle_voice_navigation, args=(network, mobile_bridge, None), daemon=True).start()
                    
                # 4. Retrieve Video Frame (Priority 1: Local UDP Wearable -> Priority 2: 5G Cloudflare Wearable -> Priority 3: Laptop Webcam)
                frame, frame_time = network.get_latest_frame()
                cam_source = "WEARABLE CAMERA (ARDUINO UNO Q - LOCAL UDP)" if frame is not None else None
                
                # Check 5G Cloudflare Tunnel if not on local Wi-Fi UDP
                if frame is None and mobile_bridge is not None:
                    cloud_frame, cloud_time = mobile_bridge.get_latest_frame()
                    if cloud_frame is not None:
                        frame = cloud_frame
                        frame_time = cloud_time
                        cam_source = "WEARABLE CAMERA (ARDUINO UNO Q - 5G CLOUDFLARE)"
                
                if frame is None and local_cam is not None and local_cam.isOpened():
                    ret, cam_frame = local_cam.read()
                    if ret and cam_frame is not None and cam_frame.size > 0:
                        frame = cam_frame
                        cam_source = "LAPTOP WEBCAM (LOCAL DEBUG)"
                
                if frame is not None and (frame.size == 0 or frame.shape[0] < 10 or frame.shape[1] < 10):
                    frame = None
                    cam_source = None
                    
                has_live_camera = (frame is not None)
                
                # If no camera is streaming, display dynamic QR code standby screen
                if not has_live_camera:
                    frame = tunnel_manager.render_standby_frame((480, 640))
                    time.sleep(0.02)
                
                frame_height, frame_width = frame.shape[:2]
                
                # 5. Optical Motion Energy Tracking
                motion_energy = motion_detector.update_frame(frame) if has_live_camera else 0.0
                
                # 6. Retrieve Sensor Telemetry (GPS + Speed + Ultrasonic)
                gps_tuple, gps_source, gps_age = mobile_bridge.get_latest_gps_with_metadata()
                current_speed = mobile_bridge.get_latest_speed()
                if not gps_tuple or gps_age >= 35.0:
                    gps_tuple, gps_source, gps_age = network.get_latest_gps_with_metadata()
                    current_speed = network.get_latest_speed()
                now = time.time()
                
                if gps_tuple and gps_age < 35.0:
                    current_gps = gps_tuple
                    nav_state["current_gps"] = gps_tuple
                    active_mode = "5G SATELLITE" if gps_source == "iphone_5g" else "LIVE SATELLITE"
                else:
                    current_gps = fallback_gps
                    nav_state["current_gps"] = current_gps
                    active_mode = "FALLBACK"
                    
                nav_state["gps_mode"] = active_mode
                # Prioritize ultrasonic proximity from wearable BLE link via mobile_bridge, fallback to local UDP
                current_ultra = mobile_bridge.get_latest_ultrasonic() if mobile_bridge else None
                ultra_ts = (mobile_bridge.get_latest_ultrasonic_ts() if mobile_bridge else 0.0) or 0.0
                if current_ultra is None and network:
                    net_ultra = network.get_latest_ultrasonic()
                    if net_ultra is not None:
                        current_ultra = net_ultra
                        ultra_ts = network.get_latest_ultrasonic_ts() or ultra_ts

                # Track wearable ultrasonic connection status transitions (Fix 1: announce only real disconnections)
                gap = (time.time() - ultra_ts) if ultra_ts else 999.0
                offline_s = NAV_CONFIG.get("wearable_offline_announce_s", 5.0)

                if gap <= 1.5:
                    if wearable_disconnect_announced:
                        announce_guidance("Wearable sensor reconnected.", mobile_bridge=mobile_bridge)
                        wearable_disconnect_announced = False
                    wearable_has_connected_once = True
                elif gap > offline_s and wearable_has_connected_once and not wearable_disconnect_announced:
                    announce_guidance("Wearable sensor disconnected.", mobile_bridge=mobile_bridge)
                    wearable_disconnect_announced = True
                
                # 7. AI Object Detection, Depth Estimation & Dual-Stage Threat Gating
                valid_threats = []
                is_walking = False
                if not has_live_camera:
                    phone_walking = mobile_bridge.is_walking() if mobile_bridge else False
                    is_walking = phone_walking or (current_speed >= 0.8)
                
                if has_live_camera:
                    # T8: Pass classes=HAZARD_IDS and imgsz=416 to run 1.5-2x faster and isolate relevant hazards
                    detections = detector.detect_objects(frame, confidence_threshold=0.25, classes=HAZARD_IDS, imgsz=416)
                    
                    # Consolidate walking evaluation using MotionDetector
                    phone_walking = mobile_bridge.is_walking() if mobile_bridge else False
                    has_gps_fix = bool(gps_tuple and gps_age < 35.0)
                    walk_eval = motion_detector.evaluate_walking(current_speed, phone_walking, detections, has_gps=has_gps_fix)
                    is_walking = walk_eval["is_walking"]
                    
                    # Estimate depth on demand if fusion is active and a hazard exists in path (Group B2 & T6)
                    depth_w = NAV_CONFIG.get("depth_fusion_weight", 0.0)
                    need_depth = (depth_w > 0.0) and any(
                        str(d.get('class_name', '')).lower() in PEDESTRIAN_COLLISION_CLASSES
                        and d['bbox'][0] <= frame_width * 0.85 and d['bbox'][2] >= frame_width * 0.15
                        for d in detections)
                    raw_depth_map = None
                    if need_depth and NAV_CONFIG.get("depth_on_demand", True):
                        depth_res = depth_model.estimate_depth(frame)
                        raw_depth_map = depth_res[0] if isinstance(depth_res, (tuple, list)) else depth_res
                        if hasattr(raw_depth_map, 'cpu'):
                            raw_depth_map = raw_depth_map.cpu().numpy()
                        if hasattr(raw_depth_map, 'ndim') and raw_depth_map.ndim == 3:
                            raw_depth_map = raw_depth_map.squeeze()
                    elif (depth_w > 0.0) and not NAV_CONFIG.get("depth_on_demand", True):
                        depth_res = depth_model.estimate_depth(frame)
                        raw_depth_map = depth_res[0] if isinstance(depth_res, (tuple, list)) else depth_res
                        if hasattr(raw_depth_map, 'cpu'):
                            raw_depth_map = raw_depth_map.cpu().numpy()
                        if hasattr(raw_depth_map, 'ndim') and raw_depth_map.ndim == 3:
                            raw_depth_map = raw_depth_map.squeeze()
                        
                    # Walking corridor boundaries (15% to 85% width)
                    walking_corridor_xmin = int(frame_width * 0.15)
                    walking_corridor_xmax = int(frame_width * 0.85)
                    ground_cutoff_ymin    = int(frame_height * 0.82)
                    ceiling_cutoff_ymax   = int(frame_height * 0.12)
                    
                    # Camera Ground-Plane Homography Parameters (T7)
                    cam_h = NAV_CONFIG.get("cam_height_m", 1.25)
                    f_y = frame_height * NAV_CONFIG.get("cam_focal_frac_y", 0.85)
                    y_horizon = int(frame_height * NAV_CONFIG.get("cam_horizon_frac", 0.40))
                    corridor_has_yolo_threat = False
                    
                    for det in detections:
                        x1, y1, x2, y2 = det['bbox']
                        confidence = det.get('confidence', 0.5)
                        class_name = str(det.get('class_name') or '').lower().strip()
                        
                        if confidence < 0.25:
                            continue
                            
                        bx1 = max(0, min(frame_width - 1, int(x1)))
                        by1 = max(0, min(frame_height - 1, int(y1)))
                        bx2 = max(0, min(frame_width, int(x2)))
                        by2 = max(0, min(frame_height, int(y2)))
                        
                        if bx2 <= bx1 or by2 <= by1:
                            continue
                            
                        is_hazard = class_name in PEDESTRIAN_COLLISION_CLASSES
                        box_color = (60, 60, 60)
                        cv2.rectangle(frame, (bx1, by1), (bx2, by2), box_color, 1)
                        cv2.putText(frame, f"{class_name} {int(confidence*100)}%", (bx1, max(14, by1 - 4)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)
                        
                        # Only real pedestrian collision hazards are evaluated for audio threats
                        if not is_hazard:
                            continue
                            
                        in_path = True if NAV_CONFIG.get("threat_mode", "ttc") == "cpa" else (bx1 <= walking_corridor_xmax and bx2 >= walking_corridor_xmin)
                        bbox_height = by2 - by1
                        is_floor = (bbox_height < int(frame_height * 0.10)) and (by1 >= ground_cutoff_ymin)
                        is_ceiling = (by1 <= ceiling_cutoff_ymax and by2 <= int(frame_height * 0.22))
                        
                        if in_path and not is_floor and not is_ceiling:
                            corridor_has_yolo_threat = True
                            
                            # 1. Ground-Plane Projective Homography (T7)
                            delta_y = max(1.0, float(by2 - y_horizon))
                            dist_homography = (cam_h * f_y) / delta_y
                            
                            # 2. Nearest surface depth from MiDaS 85th percentile (T6)
                            dist_depth = dist_homography
                            if raw_depth_map is not None and hasattr(raw_depth_map, 'shape') and raw_depth_map.ndim == 2:
                                obj_depth_crop = raw_depth_map[by1:by2, bx1:bx2]
                                if obj_depth_crop.size > 0:
                                    closest_depth_val = float(np.percentile(obj_depth_crop, 85))
                                    if closest_depth_val > 1.0:
                                        dist_depth = float(1000.0 / (closest_depth_val + 1e-4))
                            
                            # Stable fused distance (T6): default weight 0.0 uses ground-geometry only
                            w = NAV_CONFIG.get("depth_fusion_weight", 0.0)
                            distance_meters = round((1.0 - w) * dist_homography + w * dist_depth, 2)
                            # Fix 4: Make the "feet not visible" distance less aggressive using height model
                            if by2 >= int(frame_height * 0.96):
                                if class_name == 'person':
                                    est = (1.7 * f_y) / max(1.0, float(by2 - by1))
                                    distance_meters = min(distance_meters, max(0.5, round(est, 2)))
                                else:
                                    distance_meters = min(distance_meters, 1.25)
                                
                            # D3 + D4: Lateral offset, ultrasonic fusion & CPA / TTC filtering
                            f_x = frame_width * NAV_CONFIG.get("cam_focal_frac_x", 0.9)
                            cx = (bx1 + bx2) / 2.0
                            x_m = ((cx - frame_width / 2.0) / f_x) * distance_meters            # − left, + right
                            half_w_m = ((bx2 - bx1) / 2.0 / f_x) * distance_meters

                            # D4: use ultrasonic range for objects straight ahead
                            if current_ultra is not None and current_ultra < 300 and abs(x_m) < 0.3 \
                                    and abs(current_ultra / 100.0 - distance_meters) < 1.0:
                                distance_meters = round(current_ultra / 100.0, 2)

                            if NAV_CONFIG.get("threat_mode", "ttc") == "cpa":
                                keep = abs(x_m) <= NAV_CONFIG.get("track_lateral_max_m", 4.0) \
                                       and 0.3 <= distance_meters <= NAV_CONFIG.get("track_range_max_m", 8.0)
                            else:
                                near_x = 0.0 if bx1 <= frame_width / 2.0 <= bx2 else min(abs(bx1 - frame_width / 2.0), abs(bx2 - frame_width / 2.0))
                                keep = (near_x / f_x) * distance_meters <= NAV_CONFIG.get("path_half_width_m", 0.75) \
                                       and 0.3 <= distance_meters <= 6.0                          # existing behaviour

                            # Fix 2: Mark partial edge objects so movement is only estimated from fully visible positions
                            if keep:
                                valid_threats.append({
                                    'name': class_name,
                                    'distance': distance_meters,
                                    'x': x_m,
                                    'half_w': half_w_m,
                                    'bbox': (bx1, by1, bx2, by2),
                                    'conf': confidence,
                                    'edge': bx1 <= 2 or bx2 >= frame_width - 2
                                })
                                
                    # Stage 2: Spatial Occupancy Grid (Muted for road navigation to prevent bare pavement/floor false alarms)
                    # Real road hazards (pedestrians, vehicles, obstacles) are accurately tracked by YOLOv8,
                    # and curbs/barriers are directly detected by the chest ultrasonic sensor.
                    # if not corridor_has_yolo_threat and raw_depth_map is not None:
                    #     ... (Occupancy grid bypassed for clean outdoor road operation)
                                            
                    valid_threats.sort(key=lambda item: item['distance'])
                    
                    # Group B3: Per-object tracking and Time-to-Collision (TTC)
                    if NAV_CONFIG.get("threat_mode", "ttc") != "cpa" and NAV_CONFIG.get("ttc_alerts_enabled", True) and valid_threats:
                        threat_tracker.update(valid_threats, now)
                        def danger(t):  # sort by soonest collision, then by importance, then distance
                            return (min(t.get('ttc_s', 99), 99), -HAZARD_PRIORITY.get(t['name'], 1), t['distance'])
                        valid_threats.sort(key=danger)
                    
                    # Draw threat boxes on monitor HUD
                    draw_threat_boxes(frame, valid_threats, is_walking=is_walking, 
                                      alert_distance_m=COLLISION_ALERT_DISTANCE_M, 
                                      critical_distance_m=CRITICAL_STOP_DISTANCE_M)
                    
                    # Track Vision AI Rate-of-Closure
                    vision_is_closing = False
                    if valid_threats:
                        p_dist = valid_threats[0]['distance']
                        vision_history.append((p_dist, now))
                        if len(vision_history) >= 3:
                            old_v_dist, old_v_t = vision_history[0]
                            v_dt = now - old_v_t
                            if 0.25 <= v_dt <= 1.5:
                                v_delta = p_dist - old_v_dist
                                # Shrunk by >= 0.15m within window -> closing in!
                                if v_delta <= -0.15:
                                    vision_is_closing = True

                with nav_lock:
                    nav_speaking = is_mic_capturing
                
                is_voice_rec = mobile_bridge.is_recording_voice() if mobile_bridge else False
                user_speaking = (nav_speaking or is_voice_rec)
                    
                alert_dispatched = False

                # -------------------------------------------------------------
                # RATE-OF-CLOSURE (APPROACHING HAZARD) EVALUATION
                # An obstacle is an active hazard ONLY IF the distance is
                # continuously shrinking over successive readings (closing in),
                # AND either the user is walking or the obstacle is moving.
                # -------------------------------------------------------------
                
                # 1. Track Ultrasonic Rate-of-Closure (Runs every loop, sensor-driven)
                ultra_is_closing = False
                if current_ultra is not None and 2.0 <= current_ultra <= 250.0:
                    # Issue 4a: Only append to history on a fresh new reading from wearable hardware
                    if ultra_ts != last_recorded_ultra_ts:
                        last_recorded_ultra_ts = ultra_ts
                        ultra_history.append((current_ultra, now))
                        if len(ultra_history) >= 3:
                            old_u_dist, old_u_t = ultra_history[0]
                            u_dt = now - old_u_t
                            if 0.25 <= u_dt <= 1.5:
                                u_delta = current_ultra - old_u_dist
                                # Shrunk by >= 6.0 cm within window -> closing in!
                                if u_delta <= -6.0:
                                    ultra_is_closing = True

                obstacle_moving = walk_eval.get("obstacle_moving", False) if (has_live_camera and 'walk_eval' in locals()) else False
                motion_active = (is_walking or obstacle_moving)

                # Reset alert priority level after 4s cooldown window passes (T1)
                if now - last_alert_time > 4.0:
                    last_alert_level = 0

                # -------------------------------------------------------------
                # LAYER 1: Wearable Ultrasonic Proximity Alarm (Camera-Independent)
                # -------------------------------------------------------------
                # Fires ONLY IF:
                # 1. An object is actively closing in (ultra_is_closing) AND motion is active, OR
                # 2. Emergency contact bumper exception: physically touching/immediate (<= contact_cm)
                contact_cm = NAV_CONFIG.get("ultra_contact_distance_cm", 25.0)
                crit_cm = NAV_CONFIG.get("ultra_critical_distance_cm", 50.0)
                if current_ultra is not None and 2.0 <= current_ultra <= 100.0:
                    is_immediate_contact = (current_ultra <= contact_cm)
                    
                    # Ultrasonic Time-to-Collision (TTC) evaluation (Group B4)
                    u_speed = 0.0
                    if len(ultra_history) >= 3:
                        (d0, t0), (d1, t1) = ultra_history[0], ultra_history[-1]
                        if t1 - t0 >= 0.3:
                            u_speed = (d0 - d1) / 100.0 / (t1 - t0)       # m/s
                    u_ttc = (current_ultra / 100.0) / u_speed if u_speed > 0.15 else float('inf')
                    
                    is_approaching_hazard = (ultra_is_closing and motion_active) or \
                        (NAV_CONFIG.get("ttc_alerts_enabled", True) and u_ttc <= NAV_CONFIG.get("ttc_warning_s", 3.0))
                    
                    if is_immediate_contact or is_approaching_hazard:
                        is_ultra_critical = (current_ultra <= crit_cm) or \
                            (NAV_CONFIG.get("ttc_alerts_enabled", True) and u_ttc <= NAV_CONFIG.get("ttc_critical_s", 1.5))
                        active_cooldown = ALERT_COOLDOWN_IMMEDIATE_SEC if is_immediate_contact else (
                            ALERT_COOLDOWN_CRITICAL_SEC if is_ultra_critical else ALERT_COOLDOWN_WARNING_SEC
                        )
                        level = 3 if is_immediate_contact else (2 if is_ultra_critical else 1)
                        cooldown_ok = (now - last_alert_time) > active_cooldown or level > last_alert_level
                        
                        if cooldown_ok and (not user_speaking or level >= 2):
                            # Fuse semantic obstacle name from Vision AI if an object is in corridor
                            fused_name = "obstacle"
                            if valid_threats and valid_threats[0]['distance'] <= 2.2:
                                fused_name = valid_threats[0]['name']
                                
                            urgency_level = "critical" if is_ultra_critical else "warning"
                            dist_m = round(current_ultra / 100.0, 2)
                            
                            # T9: For touching distance (<= 25cm), "Stop! obstacle immediate" without cm digits is enough
                            if current_ultra <= 25.0:
                                warning_msg = f"Stop! {fused_name} immediate"
                            elif is_ultra_critical:
                                warning_msg = f"Stop! {fused_name} {int(current_ultra)} centimeters"
                            else:
                                warning_msg = f"Caution! {fused_name} {int(current_ultra)} centimeters"
                                
                            alert_payload = {
                                "type": "OBSTACLE_WARNING",
                                "obstacle": str(fused_name),
                                "distance_m": float(dist_m),
                                "urgency": urgency_level,
                                "text": warning_msg,
                                "source": "ultrasonic",
                                "timestamp": float(now)
                            }
                            network.send_downlink_alert(alert_payload)
                            mobile_bridge.send_threat_alert(warning_msg, obstacle=fused_name, distance_m=dist_m, urgency=urgency_level)
                            if NAV_CONFIG.get("laptop_speaks_alerts", True) or not (mobile_bridge and mobile_bridge.has_phone_clients()):
                                speak_text(warning_msg, async_mode=True)
                            
                            last_alert_time, last_alert_level = now, level
                            alert_dispatched = True
                            print(f"🚨 [ULTRASONIC APPROACH ALERT ({urgency_level.upper()})]: {warning_msg} (Closing: {ultra_is_closing}, Delta: {current_ultra - ultra_history[0][0]:.1f}cm)")

                # -------------------------------------------------------------
                # LAYER 2: Vision AI Threat Evaluation (Camera + Depth Model)
                # -------------------------------------------------------------
                if has_live_camera and NAV_CONFIG.get("threat_mode", "ttc") == "cpa":
                    hdg = None
                    if mobile_bridge and hasattr(mobile_bridge, 'get_latest_heading_source') and mobile_bridge.get_latest_heading_source() == "gps":
                        hdg = mobile_bridge.get_latest_heading()
                    threat_tracker.update(valid_threats, now, heading_deg=hdg)
                    min_hits = NAV_CONFIG.get("vision_min_hits", 3)
                    scored = []
                    for t in valid_threats:
                        if t.get('hits', 0) >= min_hits:
                            a = assess(t, NAV_CONFIG)
                            a["level"], a["rose"] = threat_tracker.confirm_level(t['track_id'], a["level"], now)
                            scored.append((t, a))
                    scored.sort(key=lambda ta: ta[1]["risk"], reverse=True)

                    # Change 2: Draw CPA-colored threat boxes matching voice urgency
                    COL = {"CRITICAL": (0, 0, 255), "WARNING": (0, 140, 255),
                           "ADVISORY": (0, 220, 255), "CLEAR": (0, 200, 0)}
                    for t, a in scored:
                        bx1, by1, bx2, by2 = t['bbox']
                        cv2.rectangle(frame, (bx1, by1), (bx2, by2), COL[a["level"]], 2)
                        cv2.putText(frame, a["level"], (bx1, by2 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL[a["level"]], 1)

                    # D10: proximity tone feed (~4 Hz)
                    if now - last_proximity_sent > 0.25 and mobile_bridge and NAV_CONFIG.get("proximity_tone_enabled", True):
                        top = scored[0] if scored and scored[0][1]["level"] != "CLEAR" else None
                        mobile_bridge.send_proximity(top[1]["level"] if top else "CLEAR",
                                                     top[1]["distance"] if top else None,
                                                     top[1]["side"] if top else None)
                        last_proximity_sent = now

                    # Bug 1: Persistent voice alert for any newly dangerous or persisting CRITICAL object
                    cand = next(((t, a) for t, a in scored
                                 if a["level"] in ("WARNING", "CRITICAL")
                                 and threat_tracker.needs_voice(t['track_id'], a["level"], now)), None)
                    if not alert_dispatched and cand:
                        t, a = cand
                        level = 2 if a["level"] == "CRITICAL" else 1
                        active_cooldown = ALERT_COOLDOWN_CRITICAL_SEC if level == 2 else ALERT_COOLDOWN_WARNING_SEC
                        cooldown_ready = (now - last_alert_time) > active_cooldown or level > last_alert_level
                        if cooldown_ready and (not user_speaking or level == 2) and threat_tracker.should_alert(t, now):
                            many = sum(1 for _, b in scored if b["level"] in ("WARNING", "CRITICAL")) >= 2
                            msg = "Stop. Several obstacles ahead, wait." if many else phrase(t, a, dist_words)
                            haptic = {"STOP": "turn_now", "STEP_LEFT": "left", "STEP_RIGHT": "right"}[a["action"]]
                            urg = "critical" if level == 2 else "warning"
                            network.send_downlink_alert({"type": "OBSTACLE_WARNING", "obstacle": t['name'],
                                                         "distance_m": float(a["distance"]), "urgency": urg,
                                                         "text": msg, "source": "vision_cpa", "timestamp": float(now)})
                            mobile_bridge.send_threat_alert(msg, obstacle=t['name'], distance_m=a["distance"], urgency=urg, haptic=haptic)
                            if NAV_CONFIG.get("laptop_speaks_alerts", True) or not (mobile_bridge and mobile_bridge.has_phone_clients()):
                                speak_text(msg, async_mode=True)
                            last_alert_time, last_alert_level = now, level
                            threat_tracker.mark_spoken(t['track_id'], a["level"], now)
                            print(f"🚨 [CPA ALERT ({urg.upper()})]: {msg} (t*={a['t_star']:.1f}s, d*={a['d_star']:.2f}m)")
                elif has_live_camera and not alert_dispatched and valid_threats:
                    # R1: Select the most dangerous confirmed object (>= min_hits) so unconfirmed detections never hide confirmed ones
                    min_hits = NAV_CONFIG.get("vision_min_hits", 3)
                    primary = next((t for t in valid_threats if t.get('hits', 0) >= min_hits), None)
                    if primary is not None:
                        dist = primary['distance']
                        p_name = primary['name']
                        
                        if dist <= COLLISION_ALERT_DISTANCE_M:
                            if NAV_CONFIG.get("ttc_alerts_enabled", True):
                                ttc = primary.get('ttc_s', float('inf'))
                                is_critical = dist <= CRITICAL_STOP_DISTANCE_M or ttc <= NAV_CONFIG.get("ttc_critical_s", 1.5)
                                should_alert = ((ttc <= NAV_CONFIG.get("ttc_warning_s", 3.0) and dist <= COLLISION_ALERT_DISTANCE_M)
                                                or dist <= NAV_CONFIG.get("vision_immediate_m", 1.0))
                            else:
                                is_critical = (dist <= CRITICAL_STOP_DISTANCE_M)
                                closing_in = (vision_is_closing or ultra_is_closing)
                                should_alert = closing_in and motion_active
                            
                            active_cooldown = ALERT_COOLDOWN_CRITICAL_SEC if is_critical else ALERT_COOLDOWN_WARNING_SEC
                            level = 2 if is_critical else 1
                            cooldown_ready = (now - last_alert_time) > active_cooldown or level > last_alert_level
                            
                            # T3: Check threat_tracker.should_alert last so object is not marked alerted if cooldown blocks it
                            if should_alert and (not user_speaking or is_critical) and cooldown_ready and threat_tracker.should_alert(primary, now):
                                urgency_level = "critical" if is_critical else "warning"
                                bbox_center_x = (primary['bbox'][0] + primary['bbox'][2]) / 2.0
                                if bbox_center_x < frame_width * 0.38:
                                    dir_label = "on your left"
                                elif bbox_center_x > frame_width * 0.62:
                                    dir_label = "on your right"
                                else:
                                    dir_label = "ahead"

                                vehicle_classes = {'car', 'bus', 'truck', 'motorcycle', 'bicycle'}
                                is_fast_vehicle = primary.get('closing_mps', 0.0) > 1.5 and p_name.lower() in vehicle_classes
                                fast_desc = " approaching fast" if is_fast_vehicle else ""

                                # T9: Use dist_words(dist) for humanized natural speech
                                if is_critical:
                                    warning_msg = f"Stop! {p_name}{fast_desc} {dir_label}, {dist_words(dist)}"
                                else:
                                    warning_msg = f"{p_name}{fast_desc} {dir_label}, {dist_words(dist)}"
                                    
                                alert_payload = {
                                    "type": "OBSTACLE_WARNING",
                                    "obstacle": str(p_name),
                                    "distance_m": float(round(dist, 1)),
                                    "urgency": urgency_level,
                                    "text": warning_msg,
                                    "source": "vision",
                                    "timestamp": float(now)
                                }
                                network.send_downlink_alert(alert_payload)
                                mobile_bridge.send_threat_alert(warning_msg, obstacle=p_name, distance_m=dist, urgency=urgency_level)
                                if NAV_CONFIG.get("laptop_speaks_alerts", True) or not (mobile_bridge and mobile_bridge.has_phone_clients()):
                                    speak_text(warning_msg, async_mode=True)
                                
                                last_alert_time, last_alert_level = now, level
                                print(f"🚨 [VISION APPROACH ALERT ({urgency_level.upper()})]: {warning_msg} (Closing: {vision_is_closing}, Walking: {is_walking})")
                
                # 8. Sequential Turn-by-Turn Waypoint Advancement & Continuous Pedestrian Guidance
                if nav_state["is_active"] and nav_state.get("structured_steps"):
                    rid = nav_state.get("route_id", 0)
                    steps = nav_state["structured_steps"]
                    cur_idx = nav_state.get("step_index", 0)
                    now_ts = time.time()
                    
                    # Auto-cancel route if phone has been disconnected for > 2 minutes
                    if mobile_bridge and not mobile_bridge.has_phone_clients():
                        phone_disc_ts = nav_state.get("phone_disconnect_time")
                        if phone_disc_ts is None:
                            nav_state["phone_disconnect_time"] = now_ts
                        elif (now_ts - phone_disc_ts) > 120.0:
                            nav_state["is_active"] = False
                            nav_state["phone_disconnect_time"] = None
                            sync_navigation_state_file()
                            print("\n📱 [NAVIGATION AUTO-CANCEL]: Phone disconnected for > 2 minutes. Ending active route.")
                    else:
                        nav_state["phone_disconnect_time"] = None
                    
                    # Periodic guidance if quiet too long (B4)
                    max_silence = NAV_CONFIG.get("max_silence_sec", 30.0)
                    time_since_guidance = now_ts - nav_state.get("last_any_guidance", now_ts)
                    quiet_sec = NAV_CONFIG.get("quiet_period_sec", 1.5)
                    if time_since_guidance > max_silence and (now_ts - last_alert_time) >= quiet_sec:
                        if nav_state.get("gps_was_weak", False):
                            announce_guidance("Still waiting for a clear GPS signal. Continue carefully.", mobile_bridge=mobile_bridge)
                        elif not nav_state.get("approach_phase", False):
                            d = nav_state.get("last_dist_to_target_m")
                            msg = f"Continue. {dist_words(d)} to the next instruction." if d else "Guidance active. Continue on the path."
                            announce_guidance(msg, mobile_bridge=mobile_bridge)

                    # Bug 2 & Problem 5: Deliver pending turn message + haptic cue as soon as obstacle quiet period passes
                    pending_turn = nav_state.get("pending_turn") or nav_state.get("pending_turn_msg")
                    if pending_turn and (now_ts - last_alert_time) >= quiet_sec:
                        nav_state.pop("pending_turn", None)
                        nav_state.pop("pending_turn_msg", None)
                        pt_entry = pending_turn
                        pt_msg = pt_entry[0] if isinstance(pt_entry, (tuple, list)) else pt_entry
                        pt_haptic = pt_entry[1] if isinstance(pt_entry, (tuple, list)) and len(pt_entry) > 1 else "turn_now"
                        pt_prio = pt_entry[2] if isinstance(pt_entry, (tuple, list)) and len(pt_entry) > 2 else "high"
                        announce_guidance(pt_msg, mobile_bridge=mobile_bridge, priority=pt_prio, haptic=pt_haptic)

                    # Handle manual step request from phone (phone speaks tapped step, server syncs state)
                    pending_step = mobile_bridge.get_pending_step_change() if mobile_bridge else None
                    if pending_step is not None and 0 <= pending_step < len(steps):
                        if nav_state.get("route_id", 0) == rid:
                            nav_state["step_index"] = pending_step
                            nav_state["pre_turn_announced"] = False
                            nav_state["ready_announced"] = False
                            nav_state["pending_turn"] = None
                            nav_state["pending_turn_msg"] = None
                            nav_state["passed_count"] = 0
                            nav_state["route_passed_count"] = 0
                            nav_state["post_turn_checked"] = False
                            nav_state["missed_turn_warned"] = False
                            nav_state["min_dist_to_target"] = 99999.0
                            nav_state["last_milestone_announced"] = 99999
                            nav_state["last_progress_announce_time"] = now_ts
                            nav_state["gps_buf"] = []
                            step_obj = steps[pending_step]
                            step_text = f"Step {pending_step + 1} of {len(steps)}: {step_obj['instruction']}"
                            nav_state["current_step"] = step_text
                            sync_navigation_state_file()
                            print(f"\n📱 [MANUAL STEP CHANGE]: User navigated to {step_text}")
                            if mobile_bridge:
                                mobile_bridge.send_nav_step_update(pending_step, len(steps), step_obj['instruction'], speak=False)
                            
                    # Automatic GPS Waypoint Progression & Responsive Assistive Guidance
                    elif current_gps and cur_idx < len(steps):
                        # Problem 6 & Bug 1: Standing still allowance based on speed at last fix
                        current_speed = mobile_bridge.get_latest_speed() if mobile_bridge else 0.0
                        last_fix_speed = mobile_bridge.get_last_fix_speed() if mobile_bridge else 0.0
                        current_acc = mobile_bridge.get_latest_accuracy() if mobile_bridge else 999.0
                        stat_age = NAV_CONFIG.get("stationary_max_gps_age_sec", 30.0)
                        mov_age = NAV_CONFIG.get("moving_max_gps_age_sec", 10.0)
                        restored_acc = NAV_CONFIG.get("gps_restored_acc_threshold_m", 20.0)
                        weak_acc = NAV_CONFIG.get("gps_weak_acc_threshold_m", 30.0)
                        was_stationary = (last_fix_speed < 0.5 and current_acc < restored_acc)
                        max_allowed_age = stat_age if was_stationary else mov_age
                        is_weak = (current_acc > weak_acc or gps_age > max_allowed_age)
                        
                        # Bug 2: Update internal weak state immediately, independent of audio cooldown
                        if is_weak:
                            nav_state["good_gps_start_time"] = None
                            nav_state["gps_buf"] = []
                            if not nav_state.get("gps_was_weak", False):
                                nav_state["gps_was_weak"] = True
                                if (now_ts - nav_state.get("last_weak_gps_warn", 0)) > 30.0:
                                    nav_state["last_weak_gps_warn"] = now_ts
                                    ping_age = (now_ts - getattr(mobile_bridge, 'last_ping_time', now_ts)) if mobile_bridge else 0.0
                                    if ping_age > 5.0:
                                        print(f"🌐 [NETWORK STALL]: No phone WebSocket packets/pings for {ping_age:.1f}s.")
                                    else:
                                        print(f"🛰️ [GPS STALL]: Network active, but GPS reading is stale/degraded (Age: {gps_age:.1f}s, Acc: ±{current_acc:.1f}m).")
                                    announce_guidance("GPS signal weak, continue carefully.", mobile_bridge=mobile_bridge)
                        else:
                            # Hysteresis recovery: accuracy must stay < 20m and age <= 4s for at least 5.0 continuous seconds
                            if nav_state.get("gps_was_weak", False):
                                if current_acc < restored_acc and gps_age <= 4.0:
                                    if nav_state.get("good_gps_start_time") is None:
                                        nav_state["good_gps_start_time"] = now_ts
                                    elif (now_ts - nav_state["good_gps_start_time"]) >= 5.0:
                                        nav_state["gps_was_weak"] = False
                                        if (now_ts - nav_state.get("last_restored_gps_warn", 0)) > 30.0:
                                            nav_state["last_restored_gps_warn"] = now_ts
                                            announce_guidance("GPS signal restored.", mobile_bridge=mobile_bridge)
                                else:
                                    nav_state["good_gps_start_time"] = None
                            else:
                                nav_state["good_gps_start_time"] = now_ts

                        is_gps_degraded = nav_state.get("gps_was_weak", False) or is_weak
                                
                        # Skip advancing waypoints while GPS is degraded
                        if not is_gps_degraded:
                            # Problem 1: Fresh reading timestamp evaluation
                            gps_reading_ts = now - gps_age if gps_age is not None else 0.0
                            last_eval_ts = nav_state.get("last_eval_gps_ts", 0.0)
                            is_fresh_gps = abs(gps_reading_ts - last_eval_ts) >= 0.4
                            if is_fresh_gps:
                                nav_state["last_eval_gps_ts"] = gps_reading_ts
                                buf = nav_state.setdefault("gps_buf", [])
                                buf.append((current_gps, gps_reading_ts))
                                buf[:] = [(p, t) for (p, t) in buf if (gps_reading_ts - t) <= 3.0][-3:]

                            buf = nav_state.get("gps_buf", [])
                            smooth_gps = (sum(p[0] for p, t in buf)/len(buf), sum(p[1] for p, t in buf)/len(buf)) if buf else current_gps

                            # Track user's progress along densified route polyline (sliding window up to +60 points)
                            r_points = nav_state.get("route_points", [])
                            last_u_idx = nav_state.get("user_route_idx", 0)
                            user_route_idx = last_u_idx
                            best_dist = float('inf')
                            if r_points:
                                search_start = max(0, last_u_idx - 20)
                                search_end = min(len(r_points), last_u_idx + 60)
                                best_idx = last_u_idx
                                for rp_i in range(search_start, search_end):
                                    rp = r_points[rp_i]
                                    d = haversine_distance_km(smooth_gps, (rp[0], rp[1]))
                                    if d < best_dist:
                                        best_dist = d
                                        best_idx = rp_i
                                if nav_state.get("route_id", 0) == rid:
                                    nav_state["user_route_idx"] = best_idx
                                user_route_idx = best_idx

                                # Track furthest route index reached (peak) and check for backtracking (B3)
                                if is_fresh_gps and not nav_state.get("approach_phase") and nav_state.get("off_route_count", 0) == 0:
                                    peak = nav_state.get("route_peak_idx", 0)
                                    if best_idx > peak:
                                        nav_state["route_peak_idx"] = best_idx
                                        nav_state["backtrack_count"] = 0
                                    else:
                                        back_m = haversine_distance_km(r_points[peak], r_points[best_idx]) * 1000.0
                                        nav_state["backtrack_count"] = nav_state.get("backtrack_count", 0) + 1 if back_m >= 20.0 else 0
                                        if nav_state["backtrack_count"] >= 2 and (now_ts - nav_state.get("last_backtrack_warn", 0)) > 12.0:
                                            nav_state["last_backtrack_warn"] = now_ts
                                            nav_state["route_peak_idx"] = best_idx
                                            nav_state["backtrack_count"] = 0
                                            print(f"⚠️ [BACKTRACKING]: User moved {back_m:.1f}m back along route (peak={peak}, now={best_idx}). Announcing turn around.")
                                            announce_guidance("You are walking back along the route. Turn around.",
                                                              mobile_bridge=mobile_bridge, priority="high", haptic="turn_now")

                                # Approach phase check & Off-route check on fresh readings (Group C2 & B1)
                                if nav_state.get("approach_phase"):
                                    path_start = r_points[0]
                                    gap_to_path = haversine_distance_km(smooth_gps, path_start) * 1000.0
                                    reached = gap_to_path < NAV_CONFIG.get("approach_gap_m", 25.0)
                                    if reached or (now_ts - nav_state.get("approach_start", now_ts)) > NAV_CONFIG.get("approach_timeout_s", 150.0):
                                        nav_state["approach_phase"] = False
                                        if reached:
                                            announce_guidance("You are on the path.", mobile_bridge=mobile_bridge, haptic="arrive")
                                        else:
                                            announce_guidance("Continuing guidance from here.", mobile_bridge=mobile_bridge)
                                    else:
                                        # Active approach guidance cues (B1)
                                        prev_gap = nav_state.get("approach_prev_gap", gap_to_path)
                                        if is_fresh_gps:
                                            if gap_to_path > prev_gap + 2.0:
                                                nav_state["approach_away_count"] = nav_state.get("approach_away_count", 0) + 1
                                            elif gap_to_path < prev_gap - 1.0:
                                                nav_state["approach_away_count"] = 0
                                            nav_state["approach_prev_gap"] = gap_to_path

                                        if nav_state.get("approach_away_count", 0) >= 3 and (now_ts - nav_state.get("last_approach_hint", 0)) > 8.0:
                                            nav_state["last_approach_hint"] = now_ts
                                            nav_state["approach_away_count"] = 0
                                            announce_guidance("You are moving away from the path. Turn around.",
                                                              mobile_bridge=mobile_bridge, priority="high", haptic="turn_now")
                                        else:
                                            bearing_to_path = calculate_bearing(smooth_gps, path_start)
                                            u_heading = mobile_bridge.get_latest_heading() if mobile_bridge else None
                                            clock = None
                                            if u_heading is not None:
                                                rel = (bearing_to_path - u_heading + 360.0) % 360.0
                                                clock = round(rel / 30.0) % 12 or 12

                                            last_t = nav_state.get("last_approach_hint", 0.0)
                                            since = now_ts - last_t
                                            last_clock = nav_state.get("last_approach_clock")
                                            last_gap = nav_state.get("last_approach_gap_spoken", 1e9)
                                            first = (last_t == 0.0)
                                            clock_changed = (clock is not None and last_clock is not None and
                                                             min(abs(clock - last_clock), 12 - abs(clock - last_clock)) >= 2)
                                            progressed = (last_gap - gap_to_path) >= NAV_CONFIG.get("approach_progress_step_m", 15.0)
                                            reminder = since >= NAV_CONFIG.get("approach_repeat_s", 20.0)

                                            hint = None
                                            if first:
                                                hint = (f"Walk toward {clock} o'clock to reach the path. {dist_words(gap_to_path)} remaining."
                                                        if clock is not None else f"Walk toward the path. {dist_words(gap_to_path)} remaining.")
                                            elif since > 8.0 and clock_changed:
                                                hint = f"The path is now at your {clock} o'clock."
                                            elif since > 8.0 and progressed:
                                                hint = f"Good. {dist_words(gap_to_path)} to the path."
                                            elif reminder:
                                                hint = (f"The path is at your {clock} o'clock, {dist_words(gap_to_path)} away."
                                                        if clock is not None else f"The path is {dist_words(gap_to_path)} away.")

                                            if hint:
                                                nav_state["last_approach_hint"] = now_ts
                                                nav_state["last_approach_clock"] = clock
                                                nav_state["last_approach_gap_spoken"] = gap_to_path
                                                print(f"🚶 [APPROACH HINT]: {hint}")
                                                announce_guidance(hint, mobile_bridge=mobile_bridge, priority="normal")

                                    nav_state["off_route_count"] = 0  # no false "left the route" while walking to the path
                                else:
                                    off_thresh = NAV_CONFIG.get("off_route_readings_threshold", 5)
                                    if is_fresh_gps:
                                        off_m = best_dist * 1000.0
                                        off_limit = max(NAV_CONFIG.get("off_route_min_distance_m", 35.0),
                                                        current_acc * NAV_CONFIG.get("off_route_accuracy_multiplier", 1.5))
                                        nav_state["off_route_count"] = nav_state.get("off_route_count", 0) + 1 if off_m > off_limit else 0

                                    reroute_cool = NAV_CONFIG.get("reroute_cooldown_sec", 20.0)
                                    if nav_state.get("off_route_count", 0) >= off_thresh and (now_ts - nav_state.get("last_reroute", 0)) > reroute_cool:
                                        nav_state["last_reroute"] = now_ts
                                        nav_state["off_route_count"] = 0
                                        print(f"\n⚠️  [OFF ROUTE]: Distance {off_m:.1f}m > {off_limit:.1f}m limit. Rerouting in background...")
                                        announce_guidance("You have left the route. Please stop. Finding a new route.",
                                                          mobile_bridge=mobile_bridge, priority="high", haptic="turn_now")
                                        threading.Thread(target=handle_voice_navigation,
                                                         args=(network, mobile_bridge, nav_state["destination_name"], True, None, "offroute"),
                                                         daemon=True).start()

                            # Problem 9 & Bug 4: User heading / orientation guidance relative to route
                            user_heading = mobile_bridge.get_latest_heading() if mobile_bridge else None
                            heading_source = mobile_bridge.get_latest_heading_source() if mobile_bridge else "none"

                            # 4a: Initial orientation guidance loop until aligned or 30s timeout
                            if cur_idx == 0 and nav_state.get("orienting", False) and user_heading is not None and r_points and len(r_points) >= 2:
                                path_bearing = calculate_bearing(r_points[0], r_points[min(4, len(r_points) - 1)])
                                rel = (path_bearing - user_heading + 360.0) % 360.0
                                clock = round(rel / 30.0) % 12 or 12
                                if clock in (11, 12, 1):
                                    nav_state["orienting"] = False
                                    announce_guidance("Facing along the path. Walk straight.",
                                                      mobile_bridge=mobile_bridge, priority="normal", haptic="arrive")
                                elif is_fresh_gps and (now_ts - nav_state.get("last_orient_hint", 0)) > 6.0:
                                    nav_state["last_orient_hint"] = now_ts
                                    announce_guidance(f"Hold the phone flat. The path is at your {clock} o'clock.",
                                                      mobile_bridge=mobile_bridge, priority="high")
                                
                                if (now_ts - nav_state.get("orient_start", now_ts)) > 30.0:
                                    nav_state["orienting"] = False

                            # 4c & 9b: Post-turn confirmation (~5s after turning, GPS heading only)
                            if cur_idx > 0 and (5.0 <= (now_ts - nav_state.get("last_step_advance_time", 0)) <= 8.0) and not nav_state.get("post_turn_checked", False):
                                if heading_source == "gps" and user_heading is not None and r_points:
                                    nav_state["post_turn_checked"] = True
                                    path_bearing = calculate_bearing(r_points[user_route_idx], r_points[min(user_route_idx + 4, len(r_points) - 1)])
                                    rel = (path_bearing - user_heading + 360.0) % 360.0
                                    clock = round(rel / 30.0) % 12 or 12
                                    if clock in (11, 12, 1):
                                        announce_guidance("Good, you are on track.", mobile_bridge=mobile_bridge, priority="normal")
                                    else:
                                        announce_guidance(f"Path is at your {clock} o'clock.", mobile_bridge=mobile_bridge, priority="normal")

                            # 4c & 9c: Wrong way detection while walking (GPS heading only)
                            if is_walking and heading_source == "gps" and user_heading is not None and r_points and is_fresh_gps:
                                path_bearing = calculate_bearing(r_points[user_route_idx], r_points[min(user_route_idx + 4, len(r_points) - 1)])
                                rel = (path_bearing - user_heading + 360.0) % 360.0
                                if 120.0 <= rel <= 240.0:
                                    wrong_dir_thresh = NAV_CONFIG.get("wrong_dir_readings_threshold", 5)
                                    wrong_cool = NAV_CONFIG.get("wrong_dir_cooldown_sec", 20.0)
                                    nav_state["wrong_dir_count"] = nav_state.get("wrong_dir_count", 0) + 1
                                    if nav_state["wrong_dir_count"] >= wrong_dir_thresh and (now_ts - nav_state.get("last_wrong_dir_warn", 0)) > wrong_cool:
                                        nav_state["last_wrong_dir_warn"] = now_ts
                                        nav_state["wrong_dir_count"] = 0
                                        announce_guidance("You are walking away from the route. Turn around.",
                                                          mobile_bridge=mobile_bridge, priority="high", haptic="turn_now")
                                else:
                                    nav_state["wrong_dir_count"] = 0

                            # Identify final approach step (destination door)
                            is_final_step = (cur_idx >= len(steps) - 1) or (cur_idx + 1 < len(steps) and steps[cur_idx + 1].get("maneuver") == "arrive")
                            if is_final_step:
                                route_end = nav_state["route_points"][-1] if nav_state.get("route_points") else (nav_state.get("dest_coords") or steps[-1].get("waypoint"))
                                target_wp = route_end
                            else:
                                next_step_obj = steps[cur_idx + 1]
                                target_wp = next_step_obj.get("waypoint")
                                
                            if target_wp and target_wp[0] != 0 and target_wp[1] != 0:
                                dist_to_target_m = haversine_distance_km(smooth_gps, (target_wp[0], target_wp[1])) * 1000.0
                                nav_state["last_dist_to_target_m"] = dist_to_target_m
                                nav_state["min_dist_to_target"] = min(nav_state.get("min_dist_to_target", 1e9), dist_to_target_m)
                                min_dist = nav_state["min_dist_to_target"]

                                # Group C4: Better-route check at sensible moments
                                cfg = NAV_CONFIG
                                if cfg.get("reopt_enabled", True) and not nav_state.get("reopt_running") and not is_final_step:
                                    since_switch = now_ts - nav_state.get("last_route_switch", 0)
                                    since_check  = now_ts - nav_state.get("last_reopt_check", 0)
                                    gps_improved = nav_state.get("start_acc", 0) > 15 and current_acc < 10 and not nav_state.get("reopt_gps_done")
                                    after_turn   = 5 < (now_ts - nav_state.get("last_step_advance_time", 0)) < 15
                                    timer        = since_check > cfg.get("reopt_timer_sec", 150)
                                    safe_moment  = dist_to_target_m > cfg.get("reopt_no_switch_near_turn_m", 30) and since_switch > cfg.get("reopt_switch_cooldown_sec", 120)
                                    if safe_moment and since_check > cfg.get("reopt_min_interval_sec", 45) and (gps_improved or after_turn or timer):
                                        nav_state["last_reopt_check"] = now_ts
                                        nav_state["reopt_gps_done"] = nav_state.get("reopt_gps_done") or gps_improved
                                        nav_state["reopt_running"] = True
                                        threading.Thread(target=reopt_check, args=(network, mobile_bridge, smooth_gps, rid), daemon=True).start()

                                quiet_period = (now_ts - last_alert_time) >= NAV_CONFIG.get("quiet_period_sec", 1.5)

                                # -------------------------------------------------------------
                                # PHASE 0: Destination Arrival Check
                                # -------------------------------------------------------------
                                pre_turn_min = NAV_CONFIG.get("pre_turn_warning_min_m", 25.0)
                                pre_turn_max = NAV_CONFIG.get("pre_turn_warning_max_m", 55.0)
                                pre_turn_cool = NAV_CONFIG.get("pre_turn_audio_cooldown_sec", 8.0)
                                arrival_min = NAV_CONFIG.get("arrival_radius_min_m", 10.0)
                                if is_final_step:
                                    arrival_r = max(arrival_min, current_acc)
                                    at_route_end = bool(r_points and user_route_idx >= len(r_points) - 2)
                                    close_enough = (dist_to_target_m <= arrival_r) or (at_route_end and dist_to_target_m <= (arrival_r + 3.0))
                                    if is_fresh_gps:
                                        nav_state["arrive_count"] = nav_state.get("arrive_count", 0) + 1 if close_enough else 0
                                    arrived = (nav_state.get("arrive_count", 0) >= 2)
                                    if arrived:
                                        if not nav_state.get("arrival_announced", False):
                                            if nav_state.get("route_id", 0) == rid:
                                                arrival_msg = f"You have arrived at your destination: {nav_state['destination_name']}."
                                                
                                                # Point to the building entrance if it is set back from the walkable route end
                                                try:
                                                    dest_coords = nav_state.get("dest_coords")
                                                    if dest_coords and route_end and len(r_points) >= 2:
                                                        gap_m = haversine_distance_km(route_end, dest_coords) * 1000.0
                                                        if gap_m > 8.0:
                                                            rel = (calculate_bearing(route_end, dest_coords) - calculate_bearing(r_points[-2], route_end) + 360.0) % 360.0
                                                            clock = round(rel / 30.0) % 12 or 12
                                                            arrival_msg += f" The entrance is about {dist_words(gap_m)} at your {clock} o'clock."
                                                except Exception as hint_err:
                                                    print(f"⚠️ [ARRIVAL HINT SKIPPED]: {hint_err}")

                                                print(f"\n🏁 [DESTINATION ARRIVAL]: {arrival_msg}")
                                                announce_guidance(arrival_msg, mobile_bridge=mobile_bridge, haptic="arrive")
                                                last_idx = len(steps) - 1
                                                if mobile_bridge:
                                                    mobile_bridge.send_nav_step_update(last_idx, len(steps), arrival_msg, speak=False)

                                                nav_state["arrival_announced"] = True
                                                nav_state["is_active"] = False
                                                nav_state["step_index"] = last_idx
                                                nav_state["current_step"] = f"🏁 Arrived at {nav_state['destination_name']}"
                                                nav_state["pending_turn"] = None
                                                nav_state["pending_turn_msg"] = None
                                                sync_navigation_state_file()
                                    elif pre_turn_min < dist_to_target_m <= pre_turn_max and not nav_state.get("pre_turn_announced", False):
                                        time_since_last_audio = now_ts - nav_state.get("last_progress_announce_time", 0)
                                        if time_since_last_audio > pre_turn_cool and quiet_period:
                                            prep_msg = f"In {dist_words(dist_to_target_m)}, you will reach {nav_state['destination_name']}."
                                            print(f"\n🏁 [APPROACHING DESTINATION ({dist_to_target_m:.1f}m)]: {prep_msg}")
                                            announce_guidance(prep_msg, mobile_bridge=mobile_bridge, priority="normal")
                                            nav_state["pre_turn_announced"] = True
                                            nav_state["last_progress_announce_time"] = now_ts
                                else:
                                    # Intermediate Turn Maneuver Logic
                                    next_step_obj = steps[cur_idx + 1]
                                    PHRASES = {
                                        "left": "Turn left",
                                        "right": "Turn right",
                                        "slight left": "Bear slightly left",
                                        "slight right": "Bear slightly right",
                                        "sharp left": "Turn sharp left",
                                        "sharp right": "Turn sharp right",
                                        "uturn": "Turn around",
                                        "straight": "Continue straight"
                                    }
                                    mod = (next_step_obj.get("modifier") or "").replace("-", " ").strip().lower()
                                    maneuver_type = (next_step_obj.get("maneuver") or "").lower()
                                    is_straight_or_continue = (mod in ("straight", ""))

                                    turn_phrase = PHRASES.get(mod)
                                    if not turn_phrase:
                                        inst_low = next_step_obj.get("instruction", "").lower()
                                        if "left" in inst_low:
                                            turn_phrase = "Turn left"
                                        elif "right" in inst_low:
                                            turn_phrase = "Turn right"
                                        else:
                                            turn_phrase = "Continue straight"

                                    if maneuver_type == "bend":
                                        turn_phrase = f"Follow the path as it curves {'left' if 'left' in mod else 'right'}"

                                    turn_r = max(NAV_CONFIG.get("turn_execution_radius_min_m", 5.0),
                                                 min(NAV_CONFIG.get("turn_execution_radius_max_m", 10.0), current_acc * 0.6))
                                    turn_haptic = "left" if "left" in turn_phrase.lower() else ("right" if "right" in turn_phrase.lower() else "turn_now")

                                    get_ready_d = NAV_CONFIG.get("get_ready_distance_m", 22.0)
                                    turn_close_m = NAV_CONFIG.get("turn_execution_closeness_m", 15.0)

                                    # Fix 2b: Stricter "passed the closest point" check
                                    passed_margin = max(6.0, current_acc * 1.2)
                                    passed_min_d = NAV_CONFIG.get("passed_min_dist_m", 12.0)
                                    if is_fresh_gps:
                                        if min_dist <= passed_min_d and dist_to_target_m > (min_dist + passed_margin):
                                            nav_state["passed_count"] = nav_state.get("passed_count", 0) + 1
                                        else:
                                            nav_state["passed_count"] = 0

                                        # Fix 2c: Stricter "passed along the route" check (distance along route)
                                        along_to_turn_m = (next_step_obj.get("route_idx", 99999) - user_route_idx) * 5.0  # negative = past the turn
                                        turn_max_r = NAV_CONFIG.get("turn_execution_radius_max_m", 10.0)
                                        if along_to_turn_m <= 0 and dist_to_target_m <= turn_max_r:
                                            nav_state["route_passed_count"] = nav_state.get("route_passed_count", 0) + 1
                                        else:
                                            nav_state["route_passed_count"] = 0

                                    passed_closest = (nav_state.get("passed_count", 0) >= 2)
                                    route_passed = (nav_state.get("route_passed_count", 0) >= 2)
                                    turn_reason = "radius" if dist_to_target_m <= turn_r else ("passed_closest" if passed_closest else ("route_passed" if route_passed else "unknown"))
                                    turn_now_condition = (dist_to_target_m <= turn_r or passed_closest or route_passed)

                                    # Bug 2 & Bug A: Quietly advance to next step if already on subsequent corridor
                                    catchup_pts = NAV_CONFIG.get("corridor_catchup_points_ahead", 6)
                                    step_cool = NAV_CONFIG.get("step_advance_cooldown_sec", 3.0)
                                    caught_up = False
                                    if not is_final_step and user_route_idx >= (next_step_obj.get("route_idx", 99999) + catchup_pts) \
                                            and (now_ts - nav_state.get("last_step_advance_time", 0)) > step_cool:
                                        if nav_state.get("route_id", 0) == rid:
                                            cur_idx += 1
                                            nav_state["step_index"] = cur_idx
                                            nav_state["last_step_advance_time"] = now_ts
                                            nav_state["pre_turn_announced"] = False
                                            nav_state["ready_announced"] = False
                                            nav_state["passed_count"] = 0
                                            nav_state["route_passed_count"] = 0
                                            nav_state["gps_buf"] = []
                                            nav_state["min_dist_to_target"] = 99999.0
                                            nav_state["last_progress_announce_time"] = now_ts
                                            nav_state["last_milestone_announced"] = 99999
                                            nav_state["post_turn_checked"] = False
                                            sync_navigation_state_file()
                                            if mobile_bridge:
                                                next_s = steps[cur_idx]
                                                mobile_bridge.send_nav_step_update(cur_idx, len(steps), next_s.get("instruction", ""), speak=False)
                                        caught_up = True

                                    if not caught_up:
                                        # -------------------------------------------------------------
                                        # PHASE 1A: "Get ready to turn X" (<= get_ready_d, once) (Skip for straight)
                                        # -------------------------------------------------------------
                                        if dist_to_target_m <= get_ready_d and not nav_state.get("ready_announced", False) and not turn_now_condition:
                                            if not is_straight_or_continue:
                                                if quiet_period:
                                                    ready_msg = f"Get ready to {turn_phrase.lower()}."
                                                    print(f"\n🚶 [PRE-TURN READY ({dist_to_target_m:.1f}m)]: {ready_msg}")
                                                    announce_guidance(ready_msg, mobile_bridge=mobile_bridge, priority="normal", haptic=turn_haptic)
                                                    nav_state["ready_announced"] = True
                                            else:
                                                nav_state["ready_announced"] = True

                                        # -------------------------------------------------------------
                                        # PHASE 1B: "Turn X now" & Step Advance
                                        # -------------------------------------------------------------
                                        if turn_now_condition:
                                            if (now_ts - nav_state.get("last_step_advance_time", 0)) > step_cool:
                                                if nav_state.get("route_id", 0) == rid:
                                                    nav_state["last_step_advance_time"] = now_ts
                                                    nav_state["step_index"] = cur_idx + 1
                                                    nav_state["ready_announced"] = False
                                                    nav_state["pre_turn_announced"] = False
                                                    nav_state["min_dist_to_target"] = 99999.0
                                                    nav_state["passed_count"] = 0
                                                    nav_state["route_passed_count"] = 0
                                                    nav_state["gps_buf"] = []
                                                    nav_state["post_turn_checked"] = False
                                                    nav_state["missed_turn_warned"] = False
                                                    nav_state["last_milestone_announced"] = 99999
                                                    nav_state["last_progress_announce_time"] = now_ts
                                                    
                                                    next_dist = next_step_obj.get("distance_m", 0)
                                                    walk_msg = f", then walk {dist_words(next_dist)}" if next_dist > 0 else ""
                                                    if is_straight_or_continue:
                                                        turn_now_msg = f"{turn_phrase}{walk_msg}."
                                                    else:
                                                        turn_now_msg = f"{turn_phrase} now{walk_msg}."

                                                    step_text = f"Step {cur_idx + 2} of {len(steps)}: {next_step_obj['instruction']}"
                                                    nav_state["current_step"] = step_text
                                                    sync_navigation_state_file()
                                                    
                                                    print(f"\n🚶 [TURN EXECUTION (±{dist_to_target_m:.1f}m, reason={turn_reason})]: {turn_now_msg}")
                                                    # Issue 2: Straight steps use None haptic and normal priority
                                                    exec_haptic = None if is_straight_or_continue else "turn_now"
                                                    exec_prio = "normal" if is_straight_or_continue else "high"
                                                    nav_state["pending_turn"] = (turn_now_msg, exec_haptic, exec_prio)
                                                    if quiet_period:
                                                        pt_entry = nav_state.pop("pending_turn")
                                                        pt_msg = pt_entry[0]
                                                        pt_hap = pt_entry[1]
                                                        pt_pr = pt_entry[2] if len(pt_entry) > 2 else "high"
                                                        announce_guidance(pt_msg, mobile_bridge=mobile_bridge, priority=pt_pr, haptic=pt_hap)
                                                    if mobile_bridge:
                                                        mobile_bridge.send_nav_step_update(cur_idx + 1, len(steps), next_step_obj['instruction'], speak=False)
                                                    
                                        # -------------------------------------------------------------
                                        # PHASE 2: Advance Pre-Turn Preparation Warning (pre_turn_min to pre_turn_max)
                                        # -------------------------------------------------------------
                                        elif pre_turn_min < dist_to_target_m <= pre_turn_max and not nav_state.get("pre_turn_announced", False) and not is_straight_or_continue:
                                            time_since_last_audio = now_ts - nav_state.get("last_progress_announce_time", 0)
                                            if time_since_last_audio > pre_turn_cool and quiet_period:
                                                prep_msg = f"In {dist_words(dist_to_target_m)}, {turn_phrase.lower()}."
                                                print(f"\n🚶 [PRE-TURN WARNING ({dist_to_target_m:.1f}m)]: {prep_msg}")
                                                announce_guidance(prep_msg, mobile_bridge=mobile_bridge, priority="normal", haptic=turn_haptic)
                                                nav_state["pre_turn_announced"] = True
                                                nav_state["last_progress_announce_time"] = now_ts
                                                
                                        # -------------------------------------------------------------
                                        # PHASE 3: Periodic Reassurance Progress Countdown (> pre_turn_max)
                                        # -------------------------------------------------------------
                                        elif dist_to_target_m > pre_turn_max:
                                            # Problem 7: Skip straight steps to compute distance to next real turn
                                            j = cur_idx + 1
                                            while j < len(steps) - 1 and ((steps[j].get("modifier") or "") in ("straight", "") or (steps[j].get("maneuver") or "").lower() in ("continue", "new name")):
                                                j += 1
                                            real_turn_wp = steps[j].get("waypoint")
                                            if real_turn_wp and real_turn_wp[0] != 0:
                                                reassurance_dist_m = haversine_distance_km(current_gps, (real_turn_wp[0], real_turn_wp[1])) * 1000.0
                                            else:
                                                reassurance_dist_m = dist_to_target_m

                                            base_iv = NAV_CONFIG.get("reassurance_bucket_interval_m", 40.0)
                                            reassurance_interval = 200.0 if reassurance_dist_m > 600.0 else (100.0 if reassurance_dist_m > 200.0 else base_iv)
                                            reassurance_cooldown = NAV_CONFIG.get("reassurance_cooldown_sec", 15.0)
                                            milestone_bucket = int(reassurance_dist_m // reassurance_interval) * int(reassurance_interval)
                                            last_bucket = nav_state.get("last_milestone_announced", 99999)
                                            time_since_last_audio = now_ts - nav_state.get("last_progress_announce_time", 0)
                                            
                                            if milestone_bucket >= int(reassurance_interval) and milestone_bucket < last_bucket and time_since_last_audio > reassurance_cooldown and quiet_period:
                                                prog_msg = f"Continue straight. {dist_words(reassurance_dist_m)} remaining to next turn."
                                                print(f"\n🚶 [NAVIGATION PROGRESS ({reassurance_dist_m:.1f}m)]: {prog_msg}")
                                                announce_guidance(prog_msg, mobile_bridge=mobile_bridge, priority="normal")
                                                nav_state["last_milestone_announced"] = milestone_bucket
                                                nav_state["last_progress_announce_time"] = now_ts
                
                # Signal Status Transitions
                if active_mode in ("5G SATELLITE", "LIVE SATELLITE") and last_gps_state != "LIVE":
                    last_gps_state = "LIVE"
                    print(f"\n🛰️  [GPS CONNECTED]: Mode '{active_mode}' active at ({current_gps[0]:.5f}, {current_gps[1]:.5f}).")
                elif active_mode == "FALLBACK" and last_gps_state == "LIVE":
                    last_gps_state = "FALLBACK"
                    print(f"\n⚠️  [GPS SIGNAL LOST]: No packets for {gps_age:.1f}s. Switching to fallback location.")
                elif active_mode in ("5G SATELLITE", "LIVE SATELLITE") and last_gps_state == "FALLBACK":
                    last_gps_state = "LIVE"
                    print(f"\n🛰️  [GPS RECONNECTED]: Satellite lock restored at ({current_gps[0]:.5f}, {current_gps[1]:.5f}).")
                
                # Sync state file for Leaflet live map
                if now - last_nav_sync_time > 1.5:
                    sync_navigation_state_file()
                    last_nav_sync_time = now
                
                # Periodic telemetry sync to mobile web client (Wearable ultrasonic distance & ground speed)
                if now - last_telemetry_sync_time > 0.4:
                    if mobile_bridge and mobile_bridge.has_active_clients():
                        mobile_bridge.send_telemetry_update(
                            ultra_cm=current_ultra if (current_ultra is not None and current_ultra < 400.0) else None,
                            speed_kmh=current_speed,
                            gps_lat=current_gps[0] if current_gps else None,
                            gps_lon=current_gps[1] if current_gps else None
                        )
                    last_telemetry_sync_time = now
                
                # 9. Render Navigation HUD and Overlays
                if has_live_camera:
                    draw_navigation_hud(frame, current_gps, current_ultra, nav_state, 
                                        gps_mode=active_mode, current_speed=current_speed, 
                                        cam_source=cam_source)
                    if show_qr_overlay:
                        draw_qr_overlay(frame, tunnel_manager.get_qr_image())
                        
                cv2.imshow("5G Edge Server - Master AI Monitor", frame)
                
                # 10. Process Monitor Window Keyboard Shortcuts
                key = cv2.waitKey(20) & 0xFF
                if key in (ord('q'), ord('Q'), 27):
                    break
                elif key in (ord('t'), ord('T'), ord(' ')):
                    print("\n⌨️ Keyboard trigger pressed: Starting Voice Navigation!")
                    threading.Thread(target=handle_voice_navigation, args=(network, mobile_bridge), daemon=True).start()
                elif key in (ord('c'), ord('C')):
                    show_qr_overlay = not show_qr_overlay
                    url = tunnel_manager.get_tunnel_url()
                    if url:
                        print_terminal_qr(url)
                    else:
                        print("\n⏳ Cloudflare tunnel is initializing... URL will appear in a moment.")
                elif key in (ord('w'), ord('W')):
                    if local_cam is None:
                        try:
                            local_cam = cv2.VideoCapture(0)
                            if local_cam.isOpened():
                                print("\n💻 [LAPTOP WEBCAM]: Fallback webcam turned ON.")
                            else:
                                local_cam = None
                                print("\n⚠️ [LAPTOP WEBCAM]: Could not open webcam.")
                        except Exception as cam_ex:
                            local_cam = None
                            print(f"\n⚠️ [LAPTOP WEBCAM ERROR]: {cam_ex}")
                    else:
                        local_cam.release()
                        local_cam = None
                        print("\n📱 [CAMERA MODE]: Laptop webcam turned OFF. Returning to dedicated 5G phone camera.")
                        
            except Exception as loop_err:
                print(f"⚠️ [EDGE SERVER MAIN LOOP RECOVERED]: {loop_err}")
                time.sleep(0.04)
                
    except KeyboardInterrupt:
        print("\nStopping Edge Server...")
    finally:
        if local_cam is not None:
            try:
                local_cam.release()
            except Exception:
                pass
        network.stop()
        mobile_bridge.stop()
        tunnel_manager.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_edge_server()
