# RUN THIS ON YOUR LAPTOP!
import cv2
from flask import Flask, Response
import socket

app = Flask(__name__)

# Try DSHOW first; if blank, fall back to default driver
CAM_INDEX = 0  # Try 0, 1, or 2 based on the scanner above

camera = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
if not camera.isOpened():
    camera = cv2.VideoCapture(CAM_INDEX)

# Set resolution without problematic zoom properties
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
camera.set(cv2.CAP_PROP_FPS, 30)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success or frame is None:
            continue
        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    hostname = socket.gethostname()
    laptop_ip = socket.gethostbyname(hostname)
    print("\n" + "="*60)
    print("🎥 LOGITECH CAMERA STREAM IS RUNNING!")
    print(f"VIDEO_SOURCE = \"http://{laptop_ip}:5000/video\"")
    print("="*60 + "\n")
    app.run(host='0.0.0.0', port=5000)
