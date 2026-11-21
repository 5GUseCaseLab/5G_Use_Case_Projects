# Test footage and cameras

## DriveCache frames (or any frame folder)
DriveCache keeps its PNGs in Git LFS. Clone, then pull only the folders you want (each 1000-frame folder is ~2.7 GB):

    git clone https://github.com/AdityaGupta2409/DriveCache && cd DriveCache
    git lfs install
    git lfs pull --include="Road Scenario 2/YOLO inference of frames 20 min apart/**"

Convert (laptop, needs only opencv; uses ffmpeg if installed, or `pip install imageio-ffmpeg`):

    python tools/frames_to_video.py "DriveCache/Road Scenario 2/YOLO inference of frames 20 min apart" -o rs2_20min.mp4
    python tools/frames_to_video.py --all DriveCache --out videos       # every frame folder -> one mp4 each

Defaults: 640 px wide (the board downsizes to 640 anyway, and 1080p H.264 is too heavy for the UNO Q to decode in
real time), 30 fps (DriveCache capture rate), H.264. Frame numbers come from the file names: small gaps are
filled by holding the previous frame so time stays real; big gaps split the output into _part1, _part2 ...
Pointer files (LFS not pulled) are detected and the exact `git lfs pull` command is printed.
Then upload the mp4 from the web page, or copy it to the board and use "Play path".
Note: the DriveCache frames already have YOLO boxes drawn on them; that does not stop detection, but it is visible.

## Phone as a live IP camera
1. Phone and UNO Q on the same Wi-Fi (or the same Tailscale network). Install an IP-camera app (e.g. "IP Webcam" on
   Android; iOS apps such as "IP Camera Lite") and start its server. Landscape, rear camera, 640x480 or 1280x720, ~15-30 fps.
2. Open the stream URL in the phone/laptop browser first to confirm it works. IP Webcam: `http://<phone-ip>:8080/video`
   (MJPEG). RTSP apps: `rtsp://<phone-ip>:8554/...`.
3. In the board's web page, paste that URL into "video path already on the board" and press **Play path**
   (or start the app with `SOURCE=file VIDEO_PATH=http://<phone-ip>:8080/video`).
Streams are read continuously, only the newest frame is kept (no pacing, no loop), and a dropped connection reconnects
automatically. Mount the phone rigidly, facing forward, and calibrate it ([calibration.md](calibration.md)): its field of view is not 68 deg
unless it is the Moto's main camera; try ~65-70 deg for a rear camera at 16:9, but IP-camera apps often crop - check the overlay.
