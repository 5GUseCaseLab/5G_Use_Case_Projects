# RUN THIS ON YOUR LAPTOP!
import cv2
from flask import Flask, Response
import socket

app = Flask(__name__)
camera = cv2.VideoCapture(0) # 0 is your laptop's built-in webcam
camera.set(cv2.CAP_PROP_FPS, 30)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success: break
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    hostname = socket.gethostname()
    laptop_ip = socket.gethostbyname(hostname)
    print("\n" + "="*60)
    print("🎥 LAPTOP CAMERA IS LIVE OVER USB/WIFI!")
    print(f"When you paste the code into Arduino Lab, use this link:")
    print(f"VIDEO_SOURCE = \"http://{laptop_ip}:5000/video\"")
    print("="*60 + "\n")
    # 0.0.0.0 makes the camera available over the USB-C cable network
    app.run(host='0.0.0.0', port=5000)