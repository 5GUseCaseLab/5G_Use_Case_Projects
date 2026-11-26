import requests
import time
from osrm_routing import get_navigation_instructions
from speech_to_text import get_voice_command

def get_live_laptop_location():
    """
    Detects the laptop's rough live location using its public IP address.
    This acts as a temporary stand-in for the Arduino GPS module.
    """
    print("🌍 Detecting live location via IP...")
    try:
        response = requests.get('https://ipinfo.io/json')
        data = response.json()
        
        lat, lon = data['loc'].split(',')
        print(f"📍 Location found: {data.get('city', 'Unknown')}, {data.get('region', 'Unknown')} ({lat}, {lon})")
        return (float(lat), float(lon))
        
    except Exception as e:
        print("❌ Could not detect live location. Defaulting to NYC fallback.")
        return (40.748817, -73.985428)

def run_local_test():
    print("=========================================")
    print("   5G NAVIGATION - LOCAL HARDWARE TEST   ")
    print("=========================================")
    
    # 1. Get Live Location (Mocking the Arduino GPS)
    print("\n[Step 1: Fetching GPS]")
    current_gps = get_live_laptop_location()
    
    # 2. THE EARS: Listen to the laptop mic for a destination
    print("\n[Step 2: Audio Input]")
    destination_text = get_voice_command()
    
    if not destination_text:
        print("\n⚠️ Navigation cancelled: No destination heard.")
        return

    # 3. THE BRAIN: Send text + GPS to OSRM
    print("\n[Step 3: OSRM Routing]")
    route_steps = get_navigation_instructions(current_gps, destination_text)
    
    # 4. THE MOUTH: Print the instructions (TTS audio module will come next)
    print("\n=========================================")
    print("           NAVIGATION STEPS              ")
    print("=========================================")
    for step in route_steps:
        print(f"➡️ {step}")
        time.sleep(0.5) # Small pause to simulate reading

if __name__ == "__main__":
    run_local_test()
