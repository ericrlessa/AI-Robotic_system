"""
=========================================================
PC-Side Object Tracking and Robot Command Sender (Flask)
=========================================================

This Python script runs on a **laptop/PC** and performs the following:

1. Connects to a Raspberry Pi's live camera stream over HTTP.
2. Uses a YOLO + DeepSORT model to detect and track objects in the video.
3. Allows a user (via the web interface) to select an object to track.
4. Based on the tracked object’s position in the frame, it calculates motion commands:
   - `vx`: left-right velocity
   - `vy`: forward-backward velocity
   - `omega`: angular rotation
5. Sends these motion commands to the Raspberry Pi over HTTP to control a robot.
6. Also streams back the annotated video feed (bounding boxes, IDs, etc.) via Flask.

Required Setup:
- Raspberry Pi should be running a camera stream and receiving commands on port 6000.
- `camera_functions.py` must include the YOLO/DeepSORT implementation.
- Frontend (`index_test_yolo_tracking.html`) handles object selection and display.

Author: CHUI Ho Yin
"""
# --- Import necessary libraries ---
import cv2  # For image processing
from flask import Flask, Response, render_template, request, jsonify  # Flask web server
import threading  # For background processing
import socket  # To get IP address
import requests  # For sending HTTP requests
import numpy as np  # For image decoding and processing
from camera_functions import yolo_results, yolo_ds_draw, yolo_ds_model_initalize, yolo_ds_update  # Custom functions for detection and tracking

# --- Raspberry Pi IP configuration ---
# This is the IP and port where the Raspberry Pi's camera server and robot control API is running.
PI_IP = "http://192.168.1.100:6000"  # <-- Change this to match your Raspberry Pi setup

# --- Initialize Flask web app ---
app = Flask(__name__)

# --- Global variables for frame handling and tracking state ---
frame_lock = threading.Lock()
latest_frame = None
annotated_frame = None
last_detections = []
frame_count = 0
tracking_defined = False
target_id = None
target_class_label = None
tracking_action = ""

# --- Function to update current tracking action (for frontend display) ---
def update_tracking_action(action):
    global tracking_action
    tracking_action = action

# --- Load YOLO and DeepSORT tracking model ---
model, class_names, tracker = yolo_ds_model_initalize(model_name="yolo11n.pt")

# --- Movement calculation logic based on object position ---
def tracking_move(id, class_label, objects_centers, latest_frame):
    """
    Determine how the robot should move based on the object's position in the frame.
    Returns: (vx, vy, omega)
    """
    frame_height, frame_width, _ = latest_frame.shape

    # Define horizontal thresholds (left/center/right)
    left_threshold = frame_width // 3
    right_threshold = 2 * frame_width // 3

    # Find the object center based on ID
    object_center = None
    for obj in objects_centers:
        if obj[0] == str(id):
            object_center = obj
            break

    if object_center is None:
        print(f"Error: Object with ID {id} not found.")
        return 0, 0, 0

    # Get object center coordinates
    center_x, center_y = object_center[2], object_center[3]

    # Motion control variables
    vx = 0       # Left-right movement
    vy = 0       # Forward-backward movement
    omega = 0    # Rotation

    # Decide rotation based on horizontal object position
    if center_x < left_threshold:
        omega = 0.3  # Rotate left
        action = "Turn Left"
    elif center_x > right_threshold:
        omega = -0.3  # Rotate right
        action = "Turn Right"
    else:
        omega = 0
        action = "no action"

    update_tracking_action(action)
    return vx, vy, omega

# --- Background thread: Capture and process frames from Raspberry Pi ---
def capture_and_process():
    global latest_frame, annotated_frame, last_detections, frame_count, tracking_defined, target_id, target_class_label

    # Connect to video stream served by Raspberry Pi
    stream_url = f"{PI_IP}/video_feed"
    stream = requests.get(stream_url, stream=True)

    byte_data = b""
    for chunk in stream.iter_content(chunk_size=1024):
        byte_data += chunk
        a = byte_data.find(b'\xff\xd8')  # Start of JPEG
        b = byte_data.find(b'\xff\xd9')  # End of JPEG

        if a != -1 and b != -1:
            jpg = byte_data[a:b+2]
            byte_data = byte_data[b+2:]
            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue

        frame_count += 1

        with frame_lock:
            latest_frame = frame.copy()

            if frame_count % 2 == 0:
                # Run YOLO detection every 2 frames (performance balance)
                results, detections = yolo_results(model, latest_frame)
                last_detections = yolo_ds_update(latest_frame, detections, tracker)

            # Draw detection/tracking info on frame
            annotated_frame, objects_centers = yolo_ds_draw(latest_frame, last_detections, class_names)

            # If user has selected a target, calculate movement and send to Pi
            if tracking_defined:
                vx, vy, omega = tracking_move(target_id, target_class_label, objects_centers, frame)
                try:
                    # Send motion command to Raspberry Pi
                    requests.post(f"{PI_IP}/command", json={
                        "vx": vx,
                        "vy": vy,
                        "omega": omega
                    }, timeout=0.2)
                except Exception as e:
                    print(f"[WARN] Could not send motion data to Pi: {e}")

# --- Utility: Get local IP address for displaying on web UI ---
def get_ip_address():
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    return ip_address

# --- Route: Web interface main page ---
@app.route('/')
def index():
    ip_address = get_ip_address()
    return render_template('index_test_yolo_tracking.html', ip_address=ip_address)

# --- Route: MJPEG stream for browser ---
@app.route('/video_feed')
def video_feed():
    def generate():
        global annotated_frame
        while True:
            with frame_lock:
                if annotated_frame is None:
                    continue
                ret, buffer = cv2.imencode('.jpg', annotated_frame)
                if not ret:
                    continue
                frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- Route: Receive selected object ID/class from the frontend ---
@app.route('/select_object', methods=['POST'])
def select_object():
    global tracking_defined, target_id, target_class_label
    data = request.get_json()
    target_id = int(data['id'])  # The ID of the object selected by the user
    target_class_label = data['class_label']
    tracking_defined = True
    return jsonify({"status": "success", "message": f"Tracking {target_class_label} with ID {target_id}."})

# --- Route: Used by frontend to poll the current tracking action ---
@app.route('/get_tracking_action', methods=['GET'])
def get_tracking_action():
    global tracking_action
    return jsonify({"action": tracking_action})

# --- Start the Flask app and background processing thread ---
if __name__ == '__main__':
    threading.Thread(target=capture_and_process, daemon=True).start()
    app.run(host='0.0.0.0', port=4000, threaded=True)
