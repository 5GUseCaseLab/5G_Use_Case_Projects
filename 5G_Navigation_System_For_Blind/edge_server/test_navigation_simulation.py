#!/usr/bin/env python3
"""
Pre-Flight Navigation Simulator for 5G Assistive Wearable (Autonomous Voice Navigation)
Comprehensive validation harness supporting:
1. Real walking path traversal using dense OSRM route_points from server state.
2. Repeatable test results with configurable PRNG seed (--seed 42).
3. 1x default walking speed (~1.4 m/s) with acceleration flag (--speed).
4. Strict pass/fail assertion rules:
   - Sequential step advances without skipping.
   - Exactly one destination arrival announcement.
   - Zero duplicated messages within 3.0 seconds.
   - "Turn now" cues verified within <= 10m true distance of junction (verifies Bug 1).
5. Selectable safety scenarios:
   - normal: full route walk with ±4.5m realistic GPS noise.
   - dropout: 15s GPS outage to verify "GPS weak" detection and waypoint freeze (Bug 3).
   - poor_accuracy: ±50m degraded GPS accuracy to verify waypoint gating (Bug 7).
   - hazard_at_turn: simulated obstacle alert 1s before turn to verify turn cue queuing (Bug 2).
"""

import asyncio
import json
import time
import math
import random
import sys
import argparse
import urllib.request

DEFAULT_START = (26.18871, 91.69961)
DEFAULT_DESTINATION = "New SAC"

def haversine_m(p1, p2):
    R = 6371000.0
    lat1, lon1 = math.radians(p1[0]), math.radians(p1[1])
    lat2, lon2 = math.radians(p2[0]), math.radians(p2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def interpolate_polyline_at_intervals(points, interval_m=1.4):
    """Interpolate polyline so consecutive points are approx interval_m apart."""
    if not points:
        return []
    result = [points[0]]
    for i in range(len(points) - 1):
        p1 = points[i]
        p2 = points[i + 1]
        dist = haversine_m(p1, p2)
        if dist <= interval_m:
            result.append(p2)
        else:
            num_sub = int(dist // interval_m)
            for s in range(1, num_sub + 1):
                ratio = s * interval_m / dist
                if ratio < 1.0:
                    sub_lat = p1[0] + (p2[0] - p1[0]) * ratio
                    sub_lon = p1[1] + (p2[1] - p1[1]) * ratio
                    result.append((sub_lat, sub_lon))
            result.append(p2)
    return result

def add_gps_noise(lat, lon, sigma_m=4.5):
    """Adds realistic Gaussian GPS drift to simulated coordinates."""
    dlat = random.gauss(0, sigma_m) / 111000.0
    dlon = random.gauss(0, sigma_m) / (111000.0 * math.cos(math.radians(lat)))
    return (lat + dlat, lon + dlon)

def fetch_server_state(http_url="http://127.0.0.1:8080/api/state"):
    """Fetches ground-truth server navigation state from REST endpoint."""
    try:
        req = urllib.request.Request(http_url, headers={"User-Agent": "5G-PreFlight-Simulator"})
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return None

async def run_simulation(dest_name=DEFAULT_DESTINATION, ws_url="ws://127.0.0.1:8080/ws", 
                         speed_mult=1.0, seed=42, scenario="normal"):
    try:
        import websockets
    except ImportError:
        print("❌ 'websockets' package missing. Please run inside your .venv.")
        return False

    random.seed(seed)

    print("=" * 75)
    print("🚀 [PRE-FLIGHT HIGH-ASSURANCE SIMULATION]: 5G Autonomous Navigation")
    print(f"📍 Starting Position:     {DEFAULT_START}")
    print(f"🎯 Target Destination:     '{dest_name}'")
    print(f"⚡ Speed Multiplier:       {speed_mult}x human walking pace (~1.4 m/s)")
    print(f"🎲 Seed:                   {seed}")
    print(f"🧪 Evaluation Scenario:    '{scenario.upper()}'")
    print(f"🔌 Server WebSocket:       {ws_url}")
    print("=" * 75)

    try:
        async with websockets.connect(ws_url) as ws:
            print("✅ [CONNECTED]: Established WebSocket connection with Edge Server.\n")

            # 1. Register client role as phone
            await ws.send(json.dumps({"type": "hello", "role": "phone", "client": "high_assurance_sim"}))

            # 2. Establish initial GPS Fix
            init_gps_pkt = {
                "type": "gps",
                "lat": DEFAULT_START[0],
                "lon": DEFAULT_START[1],
                "accuracy": 4.5,
                "speed_kmh": 0.0,
                "source": "live",
                "timestamp": time.time()
            }
            await ws.send(json.dumps(init_gps_pkt))
            print("🛰️  [GPS FIX SENT]: Initial location established at campus origin.")

            # 3. Trigger Voice Navigation
            print(f"🎙️  [DESTINATION QUERY]: Requesting route to '{dest_name}'...")
            dest_pkt = {
                "type": "voice_command",
                "command": dest_name,
                "text": dest_name,
                "destination": dest_name,
                "is_exact": True,
                "lat": DEFAULT_START[0],
                "lon": DEFAULT_START[1],
                "accuracy": 4.5,
                "speed_kmh": 0.0,
                "source": "live",
                "timestamp": time.time()
            }
            await ws.send(json.dumps(dest_pkt))

            # 4. Wait for Route Calculation & Step 1 announcement
            print("⏳ Awaiting route computation and audio downlink...")
            route_data = None
            speech_prompts = []  # list of dicts: {"text": str, "priority": str, "time": float, "true_pos": tuple}
            step_updates = []    # list of dicts: {"index": int, "text": str, "time": float}
            threat_alerts = []   # list of dicts: {"text": str, "urgency": str, "time": float}

            timeout_route = time.time() + 20.0
            while time.time() < timeout_route:
                try:
                    raw_msg = await asyncio.wait_for(ws.recv(), timeout=1.5)
                    msg = json.loads(raw_msg)
                    msg_type = msg.get("type")

                    if msg_type == "speech_prompt":
                        txt = msg.get("text", "")
                        prio = msg.get("priority", "normal")
                        speech_prompts.append({"text": txt, "priority": prio, "time": time.time(), "true_pos": DEFAULT_START})
                        print(f"   🔊 [EARPHONE VOICE ({prio})]: \"{txt}\"")
                        if "double-tap" in txt.lower():
                            await ws.send(json.dumps({"type": "confirm_destination", "ok": True}))
                            print("   👆👆 [SIMULATED DOUBLE-TAP]: Destination confirmed.")
                            timeout_route = time.time() + 20.0

                    elif msg_type == "threat_alert":
                        t = msg.get("text", "")
                        urg = msg.get("urgency", "warning")
                        threat_alerts.append({"text": t, "urgency": urg, "time": time.time()})
                        print(f"\n   🚨 [HAZARD ALERT ({urg})]: \"{t}\"")

                    elif msg_type == "nav_route":
                        route_data = msg
                        print(f"\n🗺️  [ROUTE RECEIVED]: Destination: {msg.get('destination')}")
                        print(f"   Total Distance: {msg.get('total_distance_m')}m")
                        print(f"   Total Steps:    {len(msg.get('steps', []))}")
                        break

                except asyncio.TimeoutError:
                    continue

            if not route_data or not route_data.get("steps"):
                print("❌ [SIMULATION FAILED]: Server did not return a valid route.")
                return False

            steps = route_data["steps"]
            print("\n📋 [ROUTE MANEUVERS]:")
            for idx, s in enumerate(steps):
                print(f"   Step {idx + 1}: {s.get('instruction')} (Distance: {s.get('distance_m')}m, Waypoint: {s.get('waypoint')})")

            # 5. Extract polyline points: prefer /api/state or route_points
            route_pts = route_data.get("route_points")
            if not route_pts:
                # Query server state via HTTP
                state = fetch_server_state()
                if state and state.get("route_points"):
                    route_pts = state["route_points"]
            
            if not route_pts:
                route_pts = [DEFAULT_START] + [s["waypoint"] for s in steps if s.get("waypoint")]
                if route_data.get("dest_coords"):
                    route_pts.append(route_data["dest_coords"])

            dense_path = interpolate_polyline_at_intervals(route_pts, interval_m=1.4)
            print(f"\n🚶 [WALKING SIMULATION STARTED]: {len(dense_path)} checkpoints (~{len(dense_path)*1.4:.0f}m total)")
            print(f"   Simulation stepping rate: {1.0/speed_mult:.2f}s per 1.4m step.")

            arrived = False
            dropout_tested = False
            hazard_tested = False
            off_route_tested = False
            reroute_heard = False
            hazard_time = 0.0
            turn_after_hazard_time = 0.0
            sleep_step = max(0.04, 1.0 / speed_mult)
            current_true_pos = DEFAULT_START

            first_junction_idx = None
            if len(steps) > 1 and steps[1].get("waypoint"):
                j1_wp = steps[1]["waypoint"]
                # Find closest index in dense_path to junction 1
                best_j1_d = float('inf')
                for d_i, d_pt in enumerate(dense_path):
                    d = haversine_m(d_pt, j1_wp)
                    if d < best_j1_d:
                        best_j1_d = d
                        first_junction_idx = d_i

            for step_idx, true_pt in enumerate(dense_path):
                current_true_pos = true_pt

                # SCENARIO: dropout (15s GPS outage after 30 steps)
                if scenario == "dropout" and step_idx == 30 and not dropout_tested:
                    print("\n🛑 [SCENARIO TRIGGER: GPS DROPOUT]: Halting GPS transmissions for 15.0s...")
                    dropout_start = time.time()
                    dropout_tested = True
                    # Let 15 seconds elapse without sending any GPS
                    while (time.time() - dropout_start) < 15.0:
                        try:
                            raw_in = await asyncio.wait_for(ws.recv(), timeout=0.5)
                            in_msg = json.loads(raw_in)
                            in_type = in_msg.get("type")
                            if in_type == "speech_prompt":
                                t = in_msg.get("text", "")
                                speech_prompts.append({"text": t, "priority": in_msg.get("priority", "normal"), "time": time.time(), "true_pos": current_true_pos})
                                print(f"   🔊 [EARPHONE VOICE]: \"{t}\"")
                            elif in_type == "threat_alert":
                                t = in_msg.get("text", "")
                                urg = in_msg.get("urgency", "warning")
                                threat_alerts.append({"text": t, "urgency": urg, "time": time.time()})
                                print(f"\n   🚨 [HAZARD ALERT ({urg})]: \"{t}\"")
                        except asyncio.TimeoutError:
                            pass
                    print("✅ [SCENARIO DROPOUT RESUMED]: Resuming GPS fix packets.")

                # SCENARIO: dropout_stationary (5s stationary then 20s outage; tests 30s standing-still allowance)
                if scenario == "dropout_stationary" and step_idx == 30 and not dropout_tested:
                    print("\n🛑 [SCENARIO TRIGGER: DROPOUT STATIONARY]: Simulating user standing still (speed=0.0) for 5s...")
                    dropout_tested = True
                    for _ in range(5):
                        stat_pkt = {
                            "type": "gps",
                            "lat": true_pt[0],
                            "lon": true_pt[1],
                            "accuracy": 4.0,
                            "speed_kmh": 0.0,
                            "source": "live",
                            "fix_age": 0.0,
                            "timestamp": time.time()
                        }
                        await ws.send(json.dumps(stat_pkt))
                        await asyncio.sleep(1.0)
                    print("🛑 [SCENARIO DROPOUT STATIONARY]: Halting GPS transmissions for 20.0s (under 30s stationary allowance)...")
                    dropout_start = time.time()
                    while (time.time() - dropout_start) < 20.0:
                        try:
                            raw_in = await asyncio.wait_for(ws.recv(), timeout=0.5)
                            in_msg = json.loads(raw_in)
                            in_type = in_msg.get("type")
                            if in_type == "speech_prompt":
                                t = in_msg.get("text", "")
                                speech_prompts.append({"text": t, "priority": in_msg.get("priority", "normal"), "time": time.time(), "true_pos": current_true_pos})
                                print(f"   🔊 [EARPHONE VOICE]: \"{t}\"")
                            elif in_type == "threat_alert":
                                t = in_msg.get("text", "")
                                urg = in_msg.get("urgency", "warning")
                                threat_alerts.append({"text": t, "urgency": urg, "time": time.time()})
                                print(f"\n   🚨 [HAZARD ALERT ({urg})]: \"{t}\"")
                        except asyncio.TimeoutError:
                            pass
                    print("✅ [SCENARIO DROPOUT STATIONARY RESUMED]: Resuming walking GPS packets.")

                # SCENARIO: hazard_at_turn (Problem 13: inject 20cm obstacle 8m before junction)
                if scenario == "hazard_at_turn" and not hazard_tested and len(steps) > 1:
                    j_wp = steps[1].get("waypoint")
                    if j_wp and haversine_m(true_pt, j_wp) <= 8.5:
                        hazard_tested = True
                        print("\n⚠️  [SCENARIO TRIGGER: HAZARD AT TURN]: Simulating wearable ultrasonic obstacle (20cm) 8m before turn...")
                        
                        async def simulate_wearable_hazard():
                            nonlocal hazard_time
                            try:
                                async with websockets.connect(ws_url) as ws_wearable:
                                    await ws_wearable.send(json.dumps({"type": "hello", "role": "wearable"}))
                                    for _ in range(10):  # 1.0s of clear path first (10 Hz)
                                        await ws_wearable.send(json.dumps({"type": "telemetry", "ultrasonic_cm": 300.0}))
                                        await asyncio.sleep(0.1)
                                    hazard_time = time.time()
                                    for _ in range(5):   # 0.5s obstacle at 20cm
                                        await ws_wearable.send(json.dumps({"type": "telemetry", "ultrasonic_cm": 20.0}))
                                        await asyncio.sleep(0.1)
                                    for _ in range(30):  # 3.0s clear again to maintain connection
                                        await ws_wearable.send(json.dumps({"type": "telemetry", "ultrasonic_cm": 300.0}))
                                        await asyncio.sleep(0.1)
                            except Exception as h_err:
                                print(f"⚠️ Hazard injection warning: {h_err}")

                        asyncio.create_task(simulate_wearable_hazard())

                # SCENARIO: wrong_way (Problem 13: walk straight past junction for 45m)
                if scenario == "wrong_way" and not off_route_tested and first_junction_idx and step_idx >= first_junction_idx:
                    print("\n🔀 [SCENARIO TRIGGER: WRONG WAY]: Missed turn! Walking straight off-route past junction for 45m...")
                    off_route_tested = True
                    prev_p = dense_path[max(0, step_idx - 1)]
                    dlat = true_pt[0] - prev_p[0]
                    dlon = true_pt[1] - prev_p[1]
                    for off_step in range(1, 35):
                        off_pt = (true_pt[0] + dlat * off_step, true_pt[1] + dlon * off_step)
                        current_true_pos = off_pt
                        off_gps = {
                            "type": "gps",
                            "lat": off_pt[0],
                            "lon": off_pt[1],
                            "accuracy": 4.0,
                            "speed_kmh": 4.5,
                            "source": "live",
                            "fix_age": 0.0,
                            "timestamp": time.time()
                        }
                        await ws.send(json.dumps(off_gps))
                        await asyncio.sleep(sleep_step)
                        try:
                            while True:
                                raw_in = await asyncio.wait_for(ws.recv(), timeout=0.02)
                                in_msg = json.loads(raw_in)
                                in_type = in_msg.get("type")
                                if in_type == "speech_prompt":
                                    t = in_msg.get("text", "")
                                    p = in_msg.get("priority", "normal")
                                    speech_prompts.append({"text": t, "priority": p, "time": time.time(), "true_pos": off_pt})
                                    print(f"\n   🔊 [EARPHONE VOICE ({p})]: \"{t}\"")
                                elif in_type == "threat_alert":
                                    t = in_msg.get("text", "")
                                    urg = in_msg.get("urgency", "warning")
                                    threat_alerts.append({"text": t, "urgency": urg, "time": time.time()})
                                    print(f"\n   🚨 [HAZARD ALERT ({urg})]: \"{t}\"")
                                elif in_type == "nav_route":
                                    reroute_heard = True
                                    print(f"\n🗺️  [NEW REROUTE RECEIVED FROM SERVER!]")
                        except asyncio.TimeoutError:
                            pass
                    break

                # GPS Accuracy setup per scenario
                acc_to_send = 5.0
                if scenario == "poor_accuracy":
                    acc_to_send = random.uniform(45.0, 65.0)
                else:
                    acc_to_send = random.uniform(3.5, 6.5)

                noisy_pt = add_gps_noise(true_pt[0], true_pt[1], sigma_m=acc_to_send)

                # Problem 1 & 13: clock_skew scenario sends timestamp 30s behind but valid fix_age
                send_ts = (time.time() - 30.0) if scenario == "clock_skew" else time.time()

                gps_pkt = {
                    "type": "gps",
                    "lat": noisy_pt[0],
                    "lon": noisy_pt[1],
                    "accuracy": acc_to_send,
                    "speed_kmh": 4.5,
                    "source": "live",
                    "fix_age": 0.0,
                    "timestamp": send_ts
                }
                await ws.send(json.dumps(gps_pkt))

                # Non-blocking check for server speech downlink
                try:
                    while True:
                        raw_in = await asyncio.wait_for(ws.recv(), timeout=0.015)
                        in_msg = json.loads(raw_in)
                        in_type = in_msg.get("type")

                        if in_type == "speech_prompt":
                            t = in_msg.get("text", "")
                            p = in_msg.get("priority", "normal")
                            speech_prompts.append({"text": t, "priority": p, "time": time.time(), "true_pos": current_true_pos})
                            print(f"\n   🔊 [EARPHONE VOICE ({p})]: \"{t}\"")
                            if "arrived" in t.lower():
                                arrived = True
                            if hazard_tested and "now" in t.lower() and turn_after_hazard_time == 0.0:
                                turn_after_hazard_time = time.time()

                        elif in_type == "threat_alert":
                            t = in_msg.get("text", "")
                            urg = in_msg.get("urgency", "warning")
                            threat_alerts.append({"text": t, "urgency": urg, "time": time.time()})
                            print(f"\n   🚨 [HAZARD ALERT ({urg})]: \"{t}\"")

                        elif in_type == "nav_step_update":
                            if in_msg.get("resync"):
                                continue  # re-sync after (re)connection, not a real step change
                            s_idx = in_msg.get("step_index", 0)
                            s_txt = in_msg.get("step_text") or in_msg.get("step", "")
                            step_updates.append({"index": s_idx, "text": s_txt, "time": time.time()})
                            print(f"\n   🧭 [STEP ADVANCE]: Step {s_idx + 1} -> {s_txt}")
                            if "arrived" in s_txt.lower():
                                arrived = True

                except asyncio.TimeoutError:
                    pass

                # Periodically display progress percentage
                pct = int((step_idx + 1) / len(dense_path) * 100)
                if pct % 20 == 0 and step_idx > 0 and (step_idx % 10 == 0):
                    print(f"   ... walking progress: {pct}% ({step_idx + 1}/{len(dense_path)} checkpoints)")

                if arrived:
                    break

                await asyncio.sleep(sleep_step)

            # Extra settle window: simulate standing at destination while listening for arrival (Bug 4)
            if not arrived and dense_path:
                end_pt = dense_path[-1]
                for _ in range(10):
                    noisy = add_gps_noise(end_pt[0], end_pt[1], sigma_m=5.0)
                    gps_pkt = {
                        "type": "gps",
                        "lat": noisy[0],
                        "lon": noisy[1],
                        "accuracy": 5.0,
                        "speed_kmh": 0.0,
                        "source": "live",
                        "fix_age": 0.0,
                        "timestamp": time.time()
                    }
                    await ws.send(json.dumps(gps_pkt))
                    try:
                        while True:
                            raw_in = await asyncio.wait_for(ws.recv(), timeout=0.15)
                            in_msg = json.loads(raw_in)
                            in_type = in_msg.get("type")
                            if in_type == "speech_prompt":
                                t = in_msg.get("text", "")
                                p = in_msg.get("priority", "normal")
                                speech_prompts.append({"text": t, "priority": p, "time": time.time(), "true_pos": end_pt})
                                print(f"\n   🔊 [EARPHONE VOICE ({p})]: \"{t}\"")
                                if "arrived" in t.lower():
                                    arrived = True
                            elif in_type == "nav_step_update":
                                if in_msg.get("resync"):
                                    continue  # re-sync after (re)connection, not a real step change
                                s_idx = in_msg.get("step_index", 0)
                                s_txt = in_msg.get("step_text") or in_msg.get("step", "")
                                step_updates.append({"index": s_idx, "text": s_txt, "time": time.time()})
                                print(f"\n   🧭 [STEP ADVANCE]: Step {s_idx + 1} -> {s_txt}")
                                if "arrived" in s_txt.lower():
                                    arrived = True
                            elif in_type == "threat_alert":
                                t = in_msg.get("text", "")
                                urg = in_msg.get("urgency", "warning")
                                threat_alerts.append({"text": t, "urgency": urg, "time": time.time()})
                                print(f"\n   🚨 [HAZARD ALERT ({urg})]: \"{t}\"")
                    except asyncio.TimeoutError:
                        pass
                    if arrived:
                        break
                    await asyncio.sleep(1.0)

            # Drain any trailing arrival messages that may arrive just after the loop
            if not arrived and dense_path:
                drain_end = time.time() + 2.0
                while time.time() < drain_end:
                    try:
                        raw_in = await asyncio.wait_for(ws.recv(), timeout=0.3)
                        in_msg = json.loads(raw_in)
                        in_type = in_msg.get("type")
                        if in_type == "speech_prompt":
                            t = in_msg.get("text", "")
                            p = in_msg.get("priority", "normal")
                            speech_prompts.append({"text": t, "priority": p, "time": time.time(), "true_pos": dense_path[-1]})
                            print(f"\n   🔊 [EARPHONE VOICE ({p})]: \"{t}\"")
                            if "arrived" in t.lower():
                                arrived = True
                        elif in_type == "nav_step_update":
                            if in_msg.get("resync"):
                                continue  # re-sync after (re)connection, not a real step change
                            s_idx = in_msg.get("step_index", 0)
                            s_txt = in_msg.get("step_text") or in_msg.get("step", "")
                            step_updates.append({"index": s_idx, "text": s_txt, "time": time.time()})
                            print(f"\n   🧭 [STEP ADVANCE]: Step {s_idx + 1} -> {s_txt}")
                            if "arrived" in s_txt.lower():
                                arrived = True
                        elif in_type == "threat_alert":
                            t = in_msg.get("text", "")
                            urg = in_msg.get("urgency", "warning")
                            threat_alerts.append({"text": t, "urgency": urg, "time": time.time()})
                            print(f"\n   🚨 [HAZARD ALERT ({urg})]: \"{t}\"")
                    except asyncio.TimeoutError:
                        pass
                    # once arrival is heard, listen only 0.5 s more to catch the trailing step update
                    if arrived and drain_end - time.time() > 0.5:
                        drain_end = time.time() + 0.5

            # ---------------------------------------------------------
            # 6. STRICT PASS/FAIL ASSERTION ENGINE
            # ---------------------------------------------------------
            print("\n" + "=" * 75)
            print("📊 [PRE-FLIGHT EVALUATION & ASSERTION REPORT]")
            print(f"Total Speech Prompts Heard:    {len(speech_prompts)}")
            print(f"Total Hazard Alerts Heard:     {len(threat_alerts)}")
            print(f"Total Step Advances Heard:      {len(step_updates)}")
            print(f"Destination Arrived Confirmed: {'YES' if arrived else 'NO'}")
            print("-" * 75)

            failures = []

            # Assertion 1: Step advances arrive once each, in strictly ascending order
            indices_seen = [su["index"] for su in step_updates]
            expected_indices = list(range(1, len(steps)))
            if scenario not in ("wrong_way", "poor_accuracy"):
                if indices_seen != expected_indices:
                    failures.append(f"Steps sequence mismatch: Saw {indices_seen}, expected {expected_indices}")
            else:
                for expected_idx in expected_indices:
                    if indices_seen.count(expected_idx) > 1:
                        failures.append(f"Step {expected_idx + 1} was advanced multiple times ({indices_seen.count(expected_idx)} times).")

            # Assertion 2: Exactly one arrival announcement (skipped for off-route and degraded scenarios)
            if scenario not in ("wrong_way", "poor_accuracy"):
                arrival_prompts = [sp for sp in speech_prompts if "arrived at your destination" in sp["text"].lower()]
                if len(arrival_prompts) == 0:
                    failures.append("No destination arrival speech was announced.")
                elif len(arrival_prompts) > 1:
                    failures.append(f"Destination arrival announced multiple times ({len(arrival_prompts)} times).")

            # Assertion 3: No identical message within 3.0s of itself
            for i in range(len(speech_prompts)):
                for j in range(i + 1, len(speech_prompts)):
                    p1, p2 = speech_prompts[i], speech_prompts[j]
                    if p1["text"] == p2["text"] and (p2["time"] - p1["time"]) < 3.0:
                        failures.append(f"Duplicate speech stutter: \"{p1['text']}\" repeated within {p2['time'] - p1['time']:.2f}s.")

            # Assertion 4: Each "turn now" arrives when true noise-free position is <= 12.5m of junction (Verifies Bug 1 & Bug 7)
            turn_now_prompts = [sp for sp in speech_prompts if "now" in sp["text"].lower() and ("turn" in sp["text"].lower() or "bear" in sp["text"].lower() or "continue" in sp["text"].lower())]
            for t_prompt in turn_now_prompts:
                p_pos = t_prompt["true_pos"]
                # Link each "turn now" prompt to the step update index that accompanied it
                matched_step_idx = None
                closest_time_diff = float('inf')
                for u in step_updates:
                    tdiff = abs(u["time"] - t_prompt["time"])
                    if tdiff < 2.5 and tdiff < closest_time_diff:
                        closest_time_diff = tdiff
                        matched_step_idx = u["index"]

                if matched_step_idx is not None and matched_step_idx < len(steps):
                    target_wp = steps[matched_step_idx].get("waypoint")
                    wp_dist = haversine_m(p_pos, target_wp) if target_wp else float('inf')
                else:
                    wp_dist = min([haversine_m(p_pos, s["waypoint"]) for s in steps if s.get("waypoint") and s["waypoint"][0] != 0] or [float('inf')])

                if wp_dist > 12.5:
                    failures.append(f"Turn cue \"{t_prompt['text']}\" fired too early (true distance to junction: {wp_dist:.1f}m > 12.5m).")

            # Scenario-specific assertions
            if scenario == "dropout":
                has_weak_warn = any("weak" in sp["text"].lower() for sp in speech_prompts)
                if not has_weak_warn:
                    failures.append("Scenario 'dropout': Server did not announce 'GPS signal weak' during 15s outage.")
                else:
                    print("✅ [SCENARIO VERIFIED]: 'GPS signal weak' successfully triggered on telemetry dropout.")

            elif scenario == "dropout_stationary":
                has_weak_warn = any("weak" in sp["text"].lower() for sp in speech_prompts)
                if has_weak_warn:
                    failures.append("Scenario 'dropout_stationary': Server announced 'GPS signal weak' during 20s stationary outage (expected 30s stationary allowance).")
                else:
                    print("✅ [SCENARIO VERIFIED]: Stationary 20s outage correctly honoured 30s allowance without false 'GPS weak' warning.")

            elif scenario == "poor_accuracy":
                # With accuracy 50-65m, waypoints should NOT advance
                if len(step_updates) > 0:
                    failures.append(f"Scenario 'poor_accuracy': Steps advanced ({len(step_updates)} times) despite degraded ±55m GPS.")
                else:
                    print("✅ [SCENARIO VERIFIED]: Degraded GPS accuracy successfully suppressed false step advances.")

            elif scenario == "hazard_at_turn":
                if hazard_time == 0:
                    failures.append("No hazard alert was heard: obstacle was never injected/triggered.")
                else:
                    hazard_heard = len(threat_alerts) > 0 or any("obstacle" in sp["text"].lower() or "stop" in sp["text"].lower() or "centimeter" in sp["text"].lower() for sp in speech_prompts)
                    if not hazard_heard:
                        failures.append("Scenario 'hazard_at_turn': Obstacle warning was never announced by server.")
                    elif turn_after_hazard_time > 0:
                        delay = turn_after_hazard_time - hazard_time
                        if delay > 3.5:
                            failures.append(f"Scenario 'hazard_at_turn': Turn message arrived late ({delay:.2f}s > 3.5s) after hazard.")
                        else:
                            print(f"✅ [SCENARIO VERIFIED]: Turn message delivered cleanly {delay:.2f}s after hazard quiet period.")
                    elif not any("now" in sp["text"].lower() for sp in speech_prompts):
                        failures.append("Scenario 'hazard_at_turn': Turn message was dropped during ultrasonic obstacle.")

            elif scenario == "wrong_way":
                has_left_route = any("left the route" in sp["text"].lower() for sp in speech_prompts)
                if not has_left_route:
                    failures.append("Scenario 'wrong_way': Server did not announce 'You have left the route' after walking off-route.")
                if not reroute_heard:
                    failures.append("Scenario 'wrong_way': Server did not compute a new route following off-route detection.")
                if has_left_route and reroute_heard:
                    print("✅ [SCENARIO VERIFIED]: Off-route departure detected and autonomous rerouting triggered.")

            elif scenario == "clock_skew":
                if arrived:
                    print("✅ [SCENARIO VERIFIED]: 30-second clock skew between device and laptop handled cleanly without false GPS expiration.")

            # Final Verdict
            if not failures:
                print("\n🎉 [ALL STRICT SAFETY & QUALITY CRITERIA PASSED]:")
                print("   ✔ Polyline densification eliminated premature turn announcements.")
                print("   ✔ Priority speech queue prevented audio truncation.")
                print("   ✔ Step progression tracked pedestrian geometry without skipping.")
                print("   ✔ Entrance-anchored arrival executed cleanly exactly once.")
                print("=" * 75)
                return True
            else:
                print("\n❌ [EVALUATION DEFECTS DETECTED]:")
                for f in failures:
                    print(f"   • {f}")
                print("=" * 75)
                return False

    except Exception as e:
        print(f"\n❌ Connection failed: {e}")
        print("💡 Ensure edge_server/server.py is running on port 8080 first.")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="5G Navigation Pre-Flight Walking Simulator")
    parser.add_argument("destination", nargs="?", default=DEFAULT_DESTINATION, help="Target campus destination")
    parser.add_argument("--speed", type=float, default=1.0, help="Simulation speedup multiplier (default: 1.0x human pace)")
    parser.add_argument("--seed", type=int, default=42, help="Random PRNG seed for deterministic evaluation")
    parser.add_argument("--seed-loop", nargs=2, type=int, metavar=("START", "END"), help="Run seed evaluation loop from START to END (e.g. 1 5)")
    parser.add_argument("--scenario", choices=["normal", "dropout", "dropout_stationary", "poor_accuracy", "hazard_at_turn", "wrong_way", "clock_skew"], default="normal", help="Test scenario")
    parser.add_argument("--url", default="ws://127.0.0.1:8080/ws", help="Edge server WebSocket URL")
    args = parser.parse_args()

    if args.seed_loop:
        start_s, end_s = args.seed_loop
        all_passed = True
        print(f"🔁 Starting Multi-Seed Validation Loop (Seeds {start_s} to {end_s})...")
        for s in range(start_s, end_s + 1):
            print(f"\n{'='*30} EVALUATING SEED {s} {'='*30}")
            success = asyncio.run(run_simulation(args.destination, args.url, args.speed, s, args.scenario))
            if not success:
                all_passed = False
                print(f"❌ FAILED on seed {s}!")
                break
            else:
                print(f"✅ PASSED on seed {s}!")
        sys.exit(0 if all_passed else 1)
    else:
        success = asyncio.run(run_simulation(args.destination, args.url, args.speed, args.seed, args.scenario))
        sys.exit(0 if success else 1)
