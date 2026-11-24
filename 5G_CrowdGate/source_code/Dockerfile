# Optional: run CrowdGate in Docker on the MEC (CPU only).
#   docker build -t crowdgate .
#   docker run -d --name crowdgate --network host --restart unless-stopped \
#          -e CAMERA_URL="rtsp://USER:PASSWORD@192.168.128.10:554/avstream/channel=1/stream=0.sdp" \
#          -v $PWD:/app crowdgate \
#          python crowdgate_mec.py --model yolo11s.pt --openvino --config config.json
# --network host serves the dashboard and /status on port 5000 of the host.
FROM python:3.10-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt openvino
COPY . .
CMD ["python", "crowdgate_mec.py", "--model", "yolo11s.pt", "--openvino"]
