import cv2
from flask import Flask, Response

app = Flask(__name__)

# EXACT RTSP URL from your working LPD script
rtsp_url = "rtsp://admin:admin123@192.168.128.10:554/avstream/channel=1/stream=1.sdp"
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("Error: Could not open RTSP stream initially.")

def generate_frames():
    global cap
    frame_count = 0
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("[ERROR] Camera dropped frame. Reconnecting...")
            cap.release()
            cv2.waitKey(1000)
            cap = cv2.VideoCapture(rtsp_url)
            continue
            
        # Terminal debugger: Prints a success message every 30 frames
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"[SUCCESS] Actively streaming frames to Arduino... (Count: {frame_count})")
            
        # Compress frame for the Arduino
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# The visual webpage for your laptop browser
@app.route('/')
def index():
    return '''
    <html>
        <head><title>5G MEC Camera Bridge</title></head>
        <body style="background-color:#121212; color:#00ff88; text-align:center; font-family:monospace;">
            <h2>5G SPARSH CAMERA STREAM (MEC SERVER)</h2>
            <img src="/video_feed" style="border: 2px solid #333; border-radius: 8px; max-width: 90%;">
        </body>
    </html>
    '''

# The raw data endpoint for the Arduino
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🎥 5G MEC CAMERA BRIDGE IS LIVE!")
    print("Test in your browser here : http://192.168.116.30:6000/")
    print("Put this exact link into your Arduino UNO Q script:")
    print("VIDEO_SOURCE = \"http://192.168.116.30:6000/video_feed\"")
    print("="*60 + "\n")
    
    # Switched to Port 6000 to match the Docker container's exposed rules
    app.run(host='0.0.0.0', port=6000, debug=False)