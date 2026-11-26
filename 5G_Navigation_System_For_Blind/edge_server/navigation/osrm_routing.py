import re
import sys
import math
import bisect
import requests
try:
    from .campus_snapping import snap_to_campus_landmark, get_search_terms_for_landmark
except ImportError:
    from campus_snapping import snap_to_campus_landmark, get_search_terms_for_landmark

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Spatial distance ceiling: Strictly reject any destination farther than 35 km
MAX_SEARCH_RADIUS_KM = 35.0

# Server failure tracker to avoid blocking on repeatedly rejected servers
_REJECT_COUNT = {}

import os
import json

# Verified IIT Guwahati Campus Offline Landmarks Registry
# Enables instant 0 ms pin-drop coordinate lookup from campus_landmarks.json
def _load_campus_coordinates():
    landmarks_file = os.path.join(os.path.dirname(__file__), "campus_landmarks.json")
    coords_dict = {}
    if os.path.exists(landmarks_file):
        try:
            with open(landmarks_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data:
                coords = item.get("entrance") or item.get("coordinates")
                if coords:
                    coords_tuple = (float(coords[0]), float(coords[1]))
                    coords_dict[item["canonical"].lower()] = coords_tuple
                    for alias in item.get("aliases", []):
                        coords_dict[alias.lower()] = coords_tuple
        except Exception as e:
            print(f"⚠️ Notice: Could not parse campus_landmarks.json: {e}")
    return coords_dict

IITG_CAMPUS_COORDINATES = _load_campus_coordinates()

def haversine_distance_km(coord1, coord2):
    """
    Calculates great-circle distance between two GPS coordinates in kilometers.
    """
    if not coord1 or not coord2:
        return 0.0
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    R = 6371.0  # Earth's radius in kilometers
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2.0)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2.0)**2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c

def query_overpass_poi(location_name, near_coords, radius=35000):
    """
    Intelligently queries OpenStreetMap Overpass API for named features
    (hostels, buildings, departments, amenities) matching location_name or its campus aliases
    within `radius` meters of the user's live GPS coordinates.
    Zero hardcoding: dynamically queries live OSM spatial database.
    """
    try:
        lat, lon = near_coords
        terms = get_search_terms_for_landmark(location_name)
        patterns = []
        for t in terms:
            clean_t = re.sub(r'[^a-zA-Z0-9\s]', '', t).strip()
            if len(clean_t) >= 2:
                patterns.append(re.escape(clean_t))
        if not patterns:
            return None
        regex_pattern = "|".join(dict.fromkeys(patterns))
        
        overpass_url = "https://overpass-api.de/api/interpreter"
        query = f"""
        [out:json][timeout:8];
        (
          node["name"~"({regex_pattern})",i](around:{radius},{lat},{lon});
          way["name"~"({regex_pattern})",i](around:{radius},{lat},{lon});
          relation["name"~"({regex_pattern})",i](around:{radius},{lat},{lon});
        );
        out center 5;
        """
        
        response = requests.post(overpass_url, data={'data': query}, timeout=8)
        if response.status_code == 200:
            data = response.json()
            elements = data.get('elements', [])
            if elements:
                elem = elements[0]
                if 'lat' in elem and 'lon' in elem:
                    return float(elem['lat']), float(elem['lon'])
                elif 'center' in elem:
                    return float(elem['center']['lat']), float(elem['center']['lon'])
    except Exception as e:
        print(f"⚠️ Overpass spatial lookup error: {e}")
    return None

def geocode_location(location_name, near_coords=None, max_radius_km=MAX_SEARCH_RADIUS_KM):
    """
    Converts a text destination into GPS coordinates.
    Strict Spatial Ceiling (30-35 km threshold):
    - Dynamically bounds the search to a 30-35 km box around the user's live position.
    - Inspects every candidate: strictly rejects any match farther than 35 km!
    - Prevents jumping to other cities/states (e.g. Delhi, Bangladesh, Cyprus).
    """
    # 0. Instant Offline Campus Landmarks Match (0 ms, zero network lag, immune to OSM rate limits)
    norm_name = re.sub(r'[^a-z0-9\s]', '', location_name.lower()).strip()
    search_terms = get_search_terms_for_landmark(location_name)
    all_check_terms = [norm_name] + [re.sub(r'[^a-z0-9\s]', '', t.lower()).strip() for t in search_terms]
    for term in all_check_terms:
        if term in IITG_CAMPUS_COORDINATES:
            campus_pin = IITG_CAMPUS_COORDINATES[term]
            print(f"📍 [CAMPUS PIN MATCH]: Resolved '{location_name}' directly to IIT Guwahati coordinates {campus_pin}")
            return campus_pin

    url = "https://nominatim.openstreetmap.org/search"
    headers = {
        'User-Agent': '5G_Wearable_Navigation_Project_Dev/1.0'
    }
    
    # 1. Attempt with strict local bounding box (35 km radius)
    if near_coords:
        lat, lon = near_coords
        d_deg = (max_radius_km / 111.0) * 1.15
        
        # Test query variants using canonical name and campus aliases
        search_terms = get_search_terms_for_landmark(location_name)
        query_variants = []
        for term in search_terms:
            query_variants.append(term)
            query_variants.append(f"{term}, IIT Guwahati")
            query_variants.append(f"{term}, Guwahati")
            
        for q in query_variants:
            params = {
                'q': q,
                'format': 'json',
                'limit': 5,
                'addressdetails': 1,
                'viewbox': f"{lon - d_deg},{lat + d_deg},{lon + d_deg},{lat - d_deg}",
                'bounded': 1  # Strictly bounded to local 35 km area
            }
            try:
                response = requests.get(url, params=params, headers=headers, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    for item in data:
                        cand_coords = (float(item['lat']), float(item['lon']))
                        dist = haversine_distance_km(near_coords, cand_coords)
                        if dist <= max_radius_km:
                            matched_name = item.get('display_name', q)[:45]
                            print(f"📍 Found local match within {dist:.2f} km: {cand_coords} ('{matched_name}')")
                            return cand_coords
                        else:
                            print(f"⚠️ Rejected far candidate {cand_coords} ({dist:.1f} km > {max_radius_km} km ceiling)")
            except Exception:
                pass

        # 2. Query local OpenStreetMap Overpass features strictly within 35 km
        print(f"🔍 Searching local OpenStreetMap neighborhood (<{max_radius_km} km) for '{location_name}'...")
        overpass_coords = query_overpass_poi(location_name, near_coords, radius=int(max_radius_km * 1000))
        if overpass_coords:
            dist = haversine_distance_km(near_coords, overpass_coords)
            if dist <= max_radius_km:
                print(f"📍 Overpass found local feature within {dist:.2f} km: {overpass_coords}")
                return overpass_coords
            else:
                print(f"⚠️ Rejected Overpass candidate: {dist:.1f} km > {max_radius_km} km ceiling")

    # 3. Fallback: Broader search, BUT enforce strict 35 km validation on any result
    try:
        params = {
            'q': location_name,
            'format': 'json',
            'limit': 10,
            'addressdetails': 1,
            'bounded': 0
        }
        response = requests.get(url, params=params, headers=headers, timeout=6)
        if response.status_code == 200:
            data = response.json()
            for item in data:
                cand_coords = (float(item['lat']), float(item['lon']))
                if near_coords:
                    dist = haversine_distance_km(near_coords, cand_coords)
                    if dist <= max_radius_km:
                        print(f"📍 Found candidate within {dist:.2f} km: {cand_coords}")
                        return cand_coords
                    else:
                        cand_name = item.get('display_name', 'Unknown')[:40]
                        print(f"⚠️ Rejected candidate '{cand_name}': {dist:.1f} km away (exceeds {max_radius_km} km ceiling)")
                else:
                    return cand_coords
    except Exception:
        pass

    print(f"❌ No matching destination found within {max_radius_km} km of current location.")
    return None

def _load_nav_routing_config():
    cfg_file = os.path.join(os.path.dirname(__file__), "nav_config.json")
    cfg = {
        "route_max_detour_factor": 3.0,
        "route_detour_allow_m": 800.0,
        "route_max_snap_m": 80.0,
        "route_max_start_snap_m": 250.0,
        "route_prefer_foot_pct": 0.15
    }
    if os.path.exists(cfg_file):
        try:
            with open(cfg_file, "r", encoding="utf-8") as f:
                cfg.update(json.load(f))
        except Exception:
            pass
    return cfg

def _bearing_deg(a, b):
    """Compass direction (0–360°) when walking from point a to point b. a, b = [lat, lon]."""
    lat1, lat2 = math.radians(a[0]), math.radians(b[0])
    dlon = math.radians(b[1] - a[1])
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def _add_bend_steps(route_points, steps, cfg):
    """Adds 'The path curves left/right' steps where the path bends but OSRM gave no instruction."""
    if not cfg.get("bend_steps_enabled", True) or len(route_points) < 3 or not steps:
        return steps
    min_deg = float(cfg.get("bend_min_deg", 35.0))            # how sharp a bend must be
    win_m   = float(cfg.get("bend_window_m", 20.0))           # heading 20 m before vs 20 m after
    near_m  = float(cfg.get("bend_skip_near_step_m", 25.0))   # skip if a real instruction is this close

    # 1) Distance along the route at every point (metres from the start)
    along = [0.0]
    for p, q in zip(route_points, route_points[1:]):
        along.append(along[-1] + haversine_distance_km(p, q) * 1000.0)
    total = along[-1]

    # 2) Where each existing OSRM step sits along the route (searched in order)
    step_along, start = [], 0
    for s in steps:
        k = min(range(start, len(route_points)),
                key=lambda k: haversine_distance_km(s["waypoint"], route_points[k]))
        step_along.append(along[k])
        start = k

    # 3) Find bends: compare direction before and after each point
    cands = []                                                 # (distance along, point index, angle)
    for i in range(1, len(route_points) - 1):
        a = along[i]
        if a < win_m or a > total - win_m:                     # ignore start and end
            continue
        j = min(max(bisect.bisect_right(along, a - win_m) - 1, 0), i - 1)       # point ~20 m behind
        k = max(min(bisect.bisect_left(along, a + win_m), len(along) - 1), i + 1)  # point ~20 m ahead
        delta = (_bearing_deg(route_points[i], route_points[k])
                 - _bearing_deg(route_points[j], route_points[i]) + 540.0) % 360.0 - 180.0  # + right, − left
        if abs(delta) >= min_deg:
            if cands and a - cands[-1][0] <= win_m:            # same bend → keep the sharpest point
                if abs(delta) > abs(cands[-1][2]):
                    cands[-1] = (a, i, delta)
            else:
                cands.append((a, i, delta))

    # 4) Build new steps, skipping bends that already have a real instruction nearby
    added = []
    for a, i, delta in cands:
        if any(abs(a - sa) <= near_m for sa in step_along):
            continue
        side = "left" if delta < 0 else "right"
        added.append((a, {
            "instruction": f"The path curves {side}. Follow the curve.",
            "distance_m": 0,
            "street": "",
            "waypoint": [float(route_points[i][0]), float(route_points[i][1])],
            "maneuver": "bend",
            "modifier": ("slight " if abs(delta) < 60 else "") + side,
        }))
    if not added:
        return steps
    print(f"🧭 [ROUTING]: added {len(added)} curve instruction(s)")

    # 5) Merge in route order, renumber, and recompute 'distance to next instruction'
    merged = sorted(list(zip(step_along, steps)) + added, key=lambda x: x[0])
    out = []
    for n, (a, s) in enumerate(merged):
        s = dict(s)
        nxt = merged[n + 1][0] if n + 1 < len(merged) else a
        s["distance_m"] = int(round(nxt - a))
        s["step_number"] = n + 1
        out.append(s)
    return out


def get_walking_route_details(start_coords, end_coords, include_metadata=False):
    """
    Fetches walking directions from OSRM including turn-by-turn steps,
    total distance, and the full route polyline geometry for interactive mapping.
    If include_metadata=True, also returns structured step objects with waypoints.
    """
    start_lat, start_lon = start_coords
    end_lat, end_lon = end_coords
    
    params = {
        'steps': 'true',
        'overview': 'full',
        'geometries': 'geojson'
    }
    
    headers = {'User-Agent': '5G_Wearable_Navigation_Project_Dev/1.0'}
    ROUTING_BASES = [
        ("foot_server", "https://routing.openstreetmap.de/routed-foot/route/v1/foot"),
        ("osrm_demo",   "http://router.project-osrm.org/route/v1/foot"),
    ]
    cfg = _load_nav_routing_config()
    max_factor   = float(cfg.get("route_max_detour_factor", 3.0))     # reject routes longer than 3x the straight-line distance
    detour_allow = float(cfg.get("route_detour_allow_m", 800.0))      # ...plus this allowance (metres) for very short trips
    max_snap_m   = float(cfg.get("route_max_snap_m", 80.0))          # reject if the server moved the start or end further than this
    max_start_snap_m = float(cfg.get("route_max_start_snap_m", 250.0))  # approach phase guides the user to the path start
    prefer_pct   = float(cfg.get("route_prefer_foot_pct", 0.15))       # prefer the walking server if it's at most 15% longer

    straight_m = haversine_distance_km(start_coords, end_coords) * 1000.0

    def _fetch(name_base):
        name, base = name_base
        url = f"{base}/{start_lon},{start_lat};{end_lon},{end_lat}"
        timeout = 4 if name == "foot_server" else 8
        try:
            d = requests.get(url, params=params, headers=headers, timeout=timeout).json()
            if d.get('code') != 'Ok' or not d.get('routes'):
                return None                                   # no counting for errors
            dist = float(d['routes'][0].get('distance', 0))
            wps = d.get('waypoints') or [{}, {}]
            start_snap = float(wps[0].get('distance', 0) or 0)
            end_snap = float(wps[-1].get('distance', 0) or 0)
            ok = (dist <= max(max_factor * straight_m, straight_m + detour_allow)
                  and start_snap <= max_start_snap_m and end_snap <= max_snap_m)
            if ok:
                _REJECT_COUNT[name] = 0                       # good route resets the count
            elif name == "foot_server":
                _REJECT_COUNT[name] = _REJECT_COUNT.get(name, 0) + 1
            print(f"🧭 [ROUTING CANDIDATE] {name}: {dist:.0f} m | start moved {start_snap:.0f} m | "
                  f"end moved {end_snap:.0f} m{'' if ok else '  ❌ rejected'}")
            return {"name": name, "data": d, "dist": dist, "ok": ok}
        except Exception as e:
            print(f"🧭 [ROUTING CANDIDATE] {name}: unavailable ({e})")
            return None                                   # no counting for network errors

    try:
        from concurrent.futures import ThreadPoolExecutor
        active_bases = [nb for nb in ROUTING_BASES
                        if not (nb[0] == "foot_server" and _REJECT_COUNT.get("foot_server", 0) >= 3)]
        with ThreadPoolExecutor(max_workers=len(active_bases)) as pool:
            candidates = [c for c in pool.map(_fetch, active_bases) if c and c["ok"]]

        if not candidates:
            print(f"❌ [ROUTING]: No sensible walking route (straight-line {straight_m:.0f} m).")
            return ([], 0, [], []) if include_metadata else ([], 0, [])

        shortest = min(candidates, key=lambda c: c["dist"])
        foot = next((c for c in candidates if c["name"] == "foot_server"), None)
        chosen = foot if (foot and foot["dist"] <= shortest["dist"] * (1 + prefer_pct)) else shortest
        print(f"✅ [ROUTING CHOSEN]: {chosen['name']} ({chosen['dist']:.0f} m)")
        data = chosen["data"]

        route = data['routes'][0]
        total_distance = int(route.get('distance', 0))
        
        # Extract GeoJSON coordinates: OSRM gives [lon, lat], convert to [lat, lon] for Leaflet
        raw_coords = route.get('geometry', {}).get('coordinates', [])
        route_points = [[coord[1], coord[0]] for coord in raw_coords]

        # Extract turn-by-turn steps
        steps = route['legs'][0]['steps']
        instructions = []
        structured_steps = []
        for step in steps:
            maneuver = step['maneuver']['type']
            modifier = step['maneuver'].get('modifier', '')
            street_name = (step.get('name') or '').strip() or 'the walkway'
            distance = int(step.get('distance', 0))
            loc = step['maneuver'].get('location', [0, 0])
            waypoint = [float(loc[1]), float(loc[0])]  # [lat, lon]
            
            if distance == 0 and maneuver != 'arrive':
                continue
                
            if maneuver == 'depart':
                instruction = f"Head out on {street_name} for {distance} meters." if distance > 0 else f"Head out on {street_name}."
            elif maneuver == 'arrive':
                instruction = "You have arrived at your destination."
            elif modifier == 'straight':
                instruction = f"Continue straight on {street_name} for {distance} meters."
            elif modifier:
                instruction = f"Turn {modifier.replace('-', ' ')} onto {street_name}, and walk for {distance} meters."
            else:
                instruction = f"Continue on {street_name} for {distance} meters."
            
            instructions.append(instruction)
            structured_steps.append({
                "step_number": len(instructions),
                "instruction": instruction,
                "distance_m": distance,
                "waypoint": waypoint,
                "maneuver": maneuver,
                "modifier": modifier,
                "street": street_name
            })
            
        try:
            structured_steps = _add_bend_steps(route_points, structured_steps, cfg)
            instructions = [s["instruction"] for s in structured_steps]
        except Exception as bend_err:
            print(f"⚠️ [CURVE STEPS SKIPPED]: {bend_err}")

        if include_metadata:
            return instructions, total_distance, route_points, structured_steps
        return instructions, total_distance, route_points

    except Exception as e:
        print(f"Error fetching route from OSRM: {e}")
        return ([], 0, [], []) if include_metadata else ([], 0, [])

def get_detailed_route(start_coords, destination_name):
    """
    Master function: Computes the complete navigation package:
    start & end GPS, total distance, spoken steps, and mapping route points.
    """
    # Snap destination to canonical campus landmark if needed
    snapped_name, conf, was_corrected = snap_to_campus_landmark(destination_name)
    if was_corrected:
        print(f"🎯 [CAMPUS ROUTER SNAPPED]: '{destination_name}' -> '{snapped_name}' ({conf}% match)")
        destination_name = snapped_name

    print(f"Finding coordinates for: '{destination_name}'...")
    end_coords = geocode_location(destination_name, near_coords=start_coords)
    
    if not end_coords:
        return {
            "success": False,
            "destination_name": destination_name,
            "start_coords": start_coords,
            "end_coords": None,
            "total_distance_m": 0,
            "steps": ["Sorry, I could not find a matching location nearby."],
            "route_points": []
        }
        
    # Enforce strict 35 km spatial proximity ceiling on walking route
    dist_km = haversine_distance_km(start_coords, end_coords)
    if dist_km > MAX_SEARCH_RADIUS_KM:
        print(f"❌ Destination {end_coords} is {dist_km:.1f} km away (exceeds {MAX_SEARCH_RADIUS_KM} km ceiling). Rejected!")
        return {
            "success": False,
            "destination_name": destination_name,
            "start_coords": start_coords,
            "end_coords": None,
            "total_distance_m": 0,
            "steps": [f"Sorry, that location is too far away ({int(dist_km)} km)."],
            "route_points": []
        }
        
    print(f"Destination found at: {end_coords} ({dist_km:.2f} km away). Fetching walking route...")
    
    start_lat, start_lon = start_coords
    end_lat, end_lon = end_coords
    map_url = f"https://www.google.com/maps/dir/?api=1&origin={start_lat},{start_lon}&destination={end_lat},{end_lon}&travelmode=walking"
    print(f"\n🌍 Visual Map Link:\n{map_url}\n")
    
    steps, total_dist, route_points, structured_steps = get_walking_route_details(start_coords, end_coords, include_metadata=True)
    
    if not steps:
        return {
            "success": False,
            "destination_name": destination_name,
            "start_coords": start_coords,
            "end_coords": end_coords,
            "total_distance_m": 0,
            "steps": ["Sorry, I could not calculate a walking route to that location."],
            "structured_steps": [],
            "route_points": []
        }
        
    return {
        "success": True,
        "destination_name": destination_name,
        "start_coords": start_coords,
        "end_coords": end_coords,
        "dest_coords": end_coords,
        "total_distance_m": total_dist,
        "steps": steps,
        "structured_steps": structured_steps,
        "route_points": route_points,
        "map_url": map_url
    }

def get_navigation_instructions(start_coords, destination_name):
    """
    Backward-compatible wrapper returning only the list of step instructions.
    """
    res = get_detailed_route(start_coords, destination_name)
    return res["steps"]

if __name__ == "__main__":
    from location_service import get_live_location
    live_start = get_live_location()
    
    try:
        user_dest = input(f"\nEnter a destination to test from {live_start} (e.g. 'Library', 'Hostel'): ").strip()
    except Exception:
        user_dest = ""
        
    if not user_dest:
        user_dest = "Central Library"
        
    result = get_detailed_route(live_start, user_dest)
    print(f"\n--- Walking Route to '{result['destination_name']}' ({result['total_distance_m']}m) ---")
    for i, step in enumerate(result['steps']):
        print(f"Step {i+1}: {step}")
