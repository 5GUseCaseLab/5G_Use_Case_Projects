"""Run in the UNO Q terminal (no installs needed for step 1):
       python3 check_link.py 10.56.6.16 8765
   Start  python server.py  on the laptop first."""
import socket, sys, time

host = sys.argv[1] if len(sys.argv) > 1 else "10.56.6.16"
port = int(sys.argv[2]) if len(sys.argv) > 2 else 8765
print(f"1) TCP connect to {host}:{port} (5 tries)")
ok = 0
for i in range(5):
    t0 = time.time()
    try:
        socket.create_connection((host, port), timeout=6).close()
        print(f"   try {i+1}: OK  {1000*(time.time()-t0):.0f} ms"); ok += 1
    except Exception as e:
        print(f"   try {i+1}: FAIL {type(e).__name__}: {e}")
    time.sleep(0.5)
if not ok:
    print("\nRESULT: cannot reach the server. Check: server.py running? firewall rule for TCP 8765? "
          "ZeroTier IP correct? both devices authorised in my.zerotier.com?")
    sys.exit(1)
try:
    import json
    from websockets.sync.client import connect
except ImportError:
    print("\nRESULT: port is reachable. (websockets not installed here, so skipping step 2 - that is fine.)")
    sys.exit(0)
print("2) WebSocket + ping")
ws = connect(f"ws://{host}:{port}", open_timeout=8, compression=None)
for i in range(5):
    t0 = time.time(); ws.send(json.dumps({"ping": i})); ws.recv(timeout=5)
    print(f"   ping {i+1}: {1000*(time.time()-t0):.0f} ms")
print("\nRESULT: server reachable and answering.")
