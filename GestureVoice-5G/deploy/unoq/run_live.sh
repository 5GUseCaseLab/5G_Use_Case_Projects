#!/bin/bash
# Runs on the Arduino UNO Q. Streams the CCTV camera to the MEC server whenever
# both are reachable, and restarts after any drop. Predictions go to ~/live.log.
#
# Edit these for your network:
SERVER=51.0.0.229:9009
CAMERA_IP=192.168.128.10
CAMERA=rtsp://$CAMERA_IP:554/h264/ch1/sub/av_stream
VIDEO_PORT=8081
REPO="$HOME/5G_model_MEC_HOST"
PYTHON="$REPO/.venv311/bin/python"
LOG=$HOME/live.log

cd "$REPO" || exit 1
while true; do
    if ! ping -c 1 -W 2 "${SERVER%:*}" >/dev/null 2>&1; then
        echo "$(date +%T) waiting for MEC ${SERVER%:*}" | tee -a "$LOG"; sleep 10; continue
    fi
    if ! ping -c 1 -W 2 "$CAMERA_IP" >/dev/null 2>&1; then
        echo "$(date +%T) waiting for camera $CAMERA_IP" | tee -a "$LOG"; sleep 10; continue
    fi
    echo "$(date +%T) starting stream to $SERVER" | tee -a "$LOG"
    SIGNBRIDGE_CAMERA="$CAMERA" "$PYTHON" -u -m signbridge.cli.run_host \
        --server "$SERVER" --video-port "$VIDEO_PORT" --no-preview --no-speech 2>&1 \
        | grep -a --line-buffered -v -E "^[IWE]0000|^INFO|^WARNING: All log" \
        | while IFS= read -r line; do echo "$(date +%T) $line"; done | tee -a "$LOG"
    echo "$(date +%T) stream ended; retrying in 5 s" | tee -a "$LOG"
    sleep 5
done
