import sys
import os
import requests

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Ensure parent and edge server imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
server_dir = os.path.join(project_root, 'edge_server')
sys.path.append(server_dir)
sys.path.append(os.path.join(server_dir, 'navigation'))

from osrm_routing import geocode_location, get_walking_route_details, haversine_distance_km
from campus_snapping import snap_to_campus_landmark, get_search_terms_for_landmark

def test_destination_routing(destination_query, user_coords=(26.18705, 91.69160)):
    print("=" * 65)
    print(f"🧭 DESTINATION GEOCODING & ROUTING DIAGNOSTIC")
    print("=" * 65)
    print(f"📍 User Live Location  : {user_coords}")
    print(f"🎯 Target Destination  : '{destination_query}'")
    
    # 1. Test campus snapping
    snapped, conf, corrected = snap_to_campus_landmark(destination_query)
    print(f"⚡ Campus Fuzzy Snapped : '{destination_query}' -> '{snapped}' (Confidence: {conf}%)")
    
    # 2. Inspect all search aliases
    search_terms = get_search_terms_for_landmark(snapped)
    print(f"🔍 Search Terms Used   : {search_terms}")
    print("-" * 65)
    
    # 3. Geocode with local spatial ceiling
    print("⏳ Querying OpenStreetMap (Nominatim & Overpass)...")
    dest_coords = geocode_location(snapped, near_coords=user_coords, max_radius_km=35.0)
    
    if not dest_coords:
        print("\n❌ FAILED to geocode destination within 35 km!")
        print("Plausible cause: OpenStreetMap does not index this specific phrasing in Guwahati.")
        return
        
    dist_km = haversine_distance_km(user_coords, dest_coords)
    dist_m = dist_km * 1000.0
    print("\n" + "=" * 65)
    print(f"✅ DESTINATION FOUND LOCALLY!")
    print("=" * 65)
    print(f"📍 Destination Coords  : {dest_coords}")
    print(f"📏 Straight-line Dist  : {dist_m:.1f} meters ({dist_km:.2f} km)")
    
    # 4. Test OSRM Walking Route
    print("\n🚶 Testing OSRM Walking Route...")
    steps, total_dist_m, route_points = get_walking_route_details(user_coords, dest_coords)
    
    if steps and len(steps) > 0:
        print(f"🛣️  Walking Distance   : {total_dist_m} meters")
        print(f"🧭 Total Turn Steps    : {len(steps)}")
        print("\n📋 Turn-by-Turn Steps:")
        for idx, step in enumerate(steps[:5], 1):
            print(f"   {idx}. {step}")
        if len(steps) > 5:
            print(f"   ... and {len(steps) - 5} more steps.")
    else:
        print("⚠️ OSRM could not find walking path between these coordinates.")

if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "New SAC"
    test_destination_routing(query)
