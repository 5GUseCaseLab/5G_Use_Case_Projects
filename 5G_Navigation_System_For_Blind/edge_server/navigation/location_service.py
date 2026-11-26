import json
import os
import sys
import urllib.request

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

CONFIG_FILE = os.path.join(os.path.dirname(__file__), "location_config.json")

def get_live_location():
    """
    Intelligently retrieves the user's starting location:
    1. First tries dynamic IP/Wi-Fi geolocation over the network.
    2. Falls back to location_config.json if offline or network fails.
    3. Final fallback is IIT Guwahati campus reference center.
    """
    # Default fallback
    fallback_coords = (26.1885, 91.6917)
    
    enable_ip_lookup = True
    # Load custom config if exists
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                cfg = json.load(f)
                lat = cfg.get("fallback_latitude", fallback_coords[0])
                lon = cfg.get("fallback_longitude", fallback_coords[1])
                fallback_coords = (float(lat), float(lon))
                enable_ip_lookup = cfg.get("enable_ip_lookup", False)
        except Exception:
            pass

    # Try live network geolocation only if enabled (prevents jumping across river to Fancy Bazaar)
    if enable_ip_lookup:
        try:
            url = "http://ip-api.com/json/?fields=status,message,lat,lon,city,regionName,country"
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                if data.get("status") == "success":
                    lat = float(data["lat"])
                    lon = float(data["lon"])
                    city = data.get("city", "Local Area")
                    print(f"📍 [LIVE GEOLOCATION DETECTED]: ({lat:.4f}, {lon:.4f}) in {city}")
                    return (lat, lon)
        except Exception:
            pass

        # Try secondary provider
        try:
            url = "https://ipapi.co/json/"
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                if "latitude" in data and "longitude" in data:
                    lat = float(data["latitude"])
                    lon = float(data["longitude"])
                    city = data.get("city", "Local Area")
                    print(f"📍 [LIVE GEOLOCATION DETECTED]: ({lat:.4f}, {lon:.4f}) in {city}")
                    return (lat, lon)
        except Exception:
            pass

    print(f"📍 [CONFIGURED LOCATION USED]: {fallback_coords} (from location_config.json)")
    return fallback_coords

if __name__ == "__main__":
    coords = get_live_location()
    print("Resolved location:", coords)
