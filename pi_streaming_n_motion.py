"""
==============================================================
Raspberry Pi Camera Streamer & Robot Controller (Flask Server)
==============================================================

This script runs on a **Raspberry Pi** and serves two main purposes:

1. CAMERA STREAMING:
   - Captures video frames from the Pi camera using OpenCV.
   - Streams the video in MJPEG format over HTTP via the `/video_feed` route.
   - This stream is used by a remote PC to perform object detection and tracking.

2. ROBOT CONTROL RECEIVER:
   - Exposes a `/command` route that accepts POST requests containing:
     - `vx`: Left-right movement velocity
     - `vy`: Forward-backward movement velocity
     - `omega`: Angular (rotation) velocity
   - Calls the `move_robot(vx, vy, omega)` function with these values.
   - You can replace `move_robot()` with actual motor control logic (e.g., GPIO, PWM).

Expected Workflow:
- A PC processes the video stream and sends movement commands to this server.
- This script then interprets the commands and moves the robot accordingly.

Make sure this server is running on port 6000 (or whatever is configured on the PC side).

Author: CHUI Ho Yin
"""

from flask import Flask, Response, request, jsonify
import cv2
import threading

# Initialize Flask app
app = Flask(__name__)

# Initialize camera and threading lock
camera = cv2.VideoCapture(0)  # Use the Pi Camera or USB webcam
frame_lock = threading.Lock()
latest_frame = None  # This will hold the most recent frame

# Background thread that continuously captures frames from the camera
def capture_frames():
    global latest_frame
    while True:
        ret, frame = camera.read()
        if not ret:
            continue
        with frame_lock:
            latest_frame = frame  # Store the latest frame safely

# MJPEG streaming endpoint for the PC to access the live video feed
@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            with frame_lock:
                if latest_frame is None:
                    continue
                # Encode frame to JPEG
                ret, buffer = cv2.imencode('.jpg', latest_frame)
                if not ret:
                    continue
                frame_bytes = buffer.tobytes()
            # Yield frame in MJPEG format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Endpoint to receive motion commands from the PC
@app.route('/command', methods=['POST'])
def command():
    data = request.get_json()

    # Extract motion commands sent from the PC
    vx = data.get("vx", 0)       # X-axis velocity
    vy = data.get("vy", 0)       # Y-axis velocity (not used in this demo)
    omega = data.get("omega", 0) # Angular velocity (rotation)

    print(f"[COMMAND RECEIVED] vx: {vx}, vy: {vy}, omega: {omega}")

    # Pass these values to the robot control function
    move_robot(vx, vy, omega)

    return jsonify({"status": "ok"})

# Replace this stub with actual GPIO/motor control logic
def move_robot(vx, vy, omega):
    """
    Function to control the robot's movement.
    Replace this with real motor control code using GPIO or motor driver libraries.
    """
    print(f"[MOVING ROBOT] vx: {vx:.2f}, vy: {vy:.2f}, omega: {omega:.2f}")
    # Example:
    # motor_controller.set_velocity(vx, vy, omega)

# Start camera capture thread and Flask app
if __name__ == '__main__':
    threading.Thread(target=capture_frames, daemon=True).start()
    app.run(host='0.0.0.0', port=6000)