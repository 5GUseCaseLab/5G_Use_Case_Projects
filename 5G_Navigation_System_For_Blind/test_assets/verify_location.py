import sys
import os
import time
import socket
import webbrowser
import math

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

from udp_network_handler import UDPNetworkHandler
from campus_snapping import snap_to_campus_landmark

def haversine_distance_km(coord1, coord2):
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def main():
    print("=" * 65)
    print("       5G WEARABLE SYSTEM - LIVE GPS COORDINATE VERIFIER       ")
    print("=" * 65)
    
    server_ip = UDPNetworkHandler.get_local_ip()
    print(f"📡 Listening on UDP Port 5006 for iPhone packets (Host: {server_ip})...")
    print("⏳ Sampling incoming GPS packets for 3 seconds...")
    
    handler = UDPNetworkHandler(host="0.0.0.0", telemetry_port=5006)
    handler.start()
    
    active_gps = None
    start_time = time.time()
    
    try:
        while time.time() - start_time < 5.0:
            gps = handler.get_latest_gps()
            if gps:
                active_gps = gps
                break
            time.sleep(0.2)
    finally:
        handler.stop()
        
    print("\n" + "=" * 65)
    if active_gps:
        lat, lon = active_gps
        gmaps_url = f"https://www.google.com/maps?q={lat:.6f},{lon:.6f}"
        osm_url = f"https://www.openstreetmap.org/?mlat={lat:.6f}&mlon={lon:.6f}#map=18/{lat:.6f}/{lon:.6f}"
        
        print("🛰️  STATUS: LIVE SATELLITE GPS CONFIRMED!")
        print("=" * 65)
        print(f"📍 Latitude       : {lat:.6f}")
        print(f"📍 Longitude      : {lon:.6f}")
        print(f"🗺️  Google Maps URL: {gmaps_url}")
        print(f"🌐 OpenStreetMap  : {osm_url}")
        print("=" * 65)
        
        # Campus landmark distance benchmarks
        campus_benchmarks = {
            "Student Activity Centre (New SAC)": (26.1887, 91.6915),
            "Brahmaputra Hostel": (26.1915, 91.6998),
            "Lecture Hall Complex (LHC)": (26.1873, 91.6938),
            "Central Library": (26.1877, 91.6922),
            "Food Court": (26.1895, 91.6920)
        }
        
        print("\n📏 Distance from your detected position to campus landmarks:")
        for name, coords in campus_benchmarks.items():
            dist_m = haversine_distance_km((lat, lon), coords) * 1000.0
            print(f"   • {name:35s}: {dist_m:6.1f} meters")
            
        print("\n🌐 Opening your exact detected location in your browser...")
        try:
            webbrowser.open(gmaps_url)
        except Exception:
            pass
            
    else:
        print("⚠️  STATUS: NO LIVE PACKETS DETECTED IN LAST 5 SECONDS")
        print("=" * 65)
        print("Troubleshooting:")
        print(f"1. Check that 'NMEA Send Location' is transmitting to: {server_ip}:5006")
        print("2. Confirm protocol is set to UDP.")
        print("3. Ensure iPhone and laptop are on the same Wi-Fi network.")
        print("=" * 65)

if __name__ == "__main__":
    main()
