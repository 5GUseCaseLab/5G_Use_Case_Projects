import os
from pathlib import Path

# 1. Set the path to the Tier 0 ONNX model (must be set BEFORE importing unoq_service)
os.environ.setdefault("TIER0_MODEL", str(Path(__file__).parent / "tier0.onnx"))

# 2. Set the Tier 1 Cloud URL
os.environ.setdefault("TIER1_URL", "https://your-cloudflare-url.trycloudflare.com") 

import uvicorn
from unoq_service import app

if __name__ == "__main__":
    print("Starting Tier 0 Edge AI Service...")
    # This binds the server to 0.0.0.0 so your laptop can access it over Wi-Fi
    uvicorn.run(app, host="0.0.0.0", port=8000)