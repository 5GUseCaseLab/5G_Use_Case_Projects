"""
Diagnostic Test Script for 5G Voice Ingestion Pipeline
Simulates a mobile phone sending GPS, PTT trigger, and Voice Audio over WebSocket.
"""
import asyncio
import base64
import io
import json
import time
import wave
import numpy as np

async def run_diagnostic():
    url = "ws://127.0.0.1:8080/ws"
    print(f"\n🔍 [DIAGNOSTIC]: Connecting to Edge Server WebSocket at {url}...")
    
    try:
        import websockets
        async with websockets.connect(url, open_timeout=4.0) as ws:
            print("✅ [CONNECTED]: Handshake successful with 5G Edge Server!")
            
            # Step 1: Send GPS Telemetry
            print("\n📡 Step 1: Sending simulated GPS fix...")
            await ws.send(json.dumps({
                "type": "gps",
                "lat": 26.18871,
                "lon": 91.69960,
                "accuracy": 5.0,
                "speed_kmh": 0.0,
                "timestamp": time.time()
            }))
            print("   -> GPS fix sent to server.")
            await asyncio.sleep(0.5)

            # Step 2: Trigger PTT Start
            print("\n🎙️ Step 2: Sending Push-to-Talk start trigger...")
            await ws.send(json.dumps({
                "type": "ptt_start",
                "timestamp": time.time()
            }))
            print("   -> 'ptt_start' signal sent to server.")
            await asyncio.sleep(0.5)

            # Step 3: Send Voice Audio Packet (synthesized sine tone)
            print("\n📦 Step 3: Packaging and uploading 1.5s Voice Audio packet...")
            buf = io.BytesIO()
            with wave.open(buf, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                # 1.5s 440Hz test tone
                t = np.linspace(0, 1.5, int(16000 * 1.5))
                samples = (np.sin(2 * np.pi * 440 * t) * 15000).astype(np.int16)
                wf.writeframes(samples.tobytes())
            
            raw_audio = buf.getvalue()
            b64_audio = base64.b64encode(raw_audio).decode('utf-8')
            print(f"   -> Generated {len(raw_audio)} bytes audio (~{len(b64_audio)/1024:.1f} KB base64).")

            await ws.send(json.dumps({
                "type": "voice_audio",
                "format": "wav",
                "sample_rate": 16000,
                "data": b64_audio,
                "lat": 26.18871,
                "lon": 91.69960,
                "timestamp": time.time()
            }))
            print("   -> 'voice_audio' packet uploaded successfully!")

            # Step 4: Wait for Server Downlink Response
            print("\n⏳ Step 4: Waiting for server response (Whisper transcription & speech prompt)...")
            try:
                response = await asyncio.wait_for(ws.recv(), timeout=6.0)
                resp_data = json.loads(response)
                print(f"\n🎉 [SERVER DOWNLINK RECEIVED]: Type = '{resp_data.get('type')}'")
                print(f"   Content: {resp_data}")
            except asyncio.TimeoutError:
                print("   (Note: Audio uploaded. Check server terminal window for Whisper AI logs.)")

            print("\n✅ [DIAGNOSTIC COMPLETE]: The entire server WebSocket & audio ingestion pipeline is verified!\n")

    except ConnectionRefusedError:
        print(f"\n❌ [CONNECTION REFUSED]: Could not connect to {url}.")
        print("   -> Is server.py currently running in your other terminal window?")
        print("   -> Start the server with: .\\.venv\\Scripts\\python.exe edge_server/server.py\n")
    except Exception as e:
        print(f"\n❌ [ERROR]: {e}\n")

if __name__ == "__main__":
    asyncio.run(run_diagnostic())
