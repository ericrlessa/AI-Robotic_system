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

import sys
import os
import cv2
from flask import Flask, Response, render_template, request, jsonify
import threading
import socket
import requests
import numpy as np
from camera_functions import yolo_results, yolo_ds_draw, yolo_ds_model_initalize, yolo_ds_update

# Add the gesture/src folder to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../gesture/src')))

from robot_controller import ObjectTrackingRobotController

class WebApp:
    def __init__(self, pi_ip="http://192.168.2.104:5000", host='0.0.0.0', port=4000):
        # Initialize Flask app
        self.app = Flask(__name__)
    
        self.PI_IP = pi_ip
        # Configuration
        self.host = host
        self.port = port
        
        self.robot_controller = ObjectTrackingRobotController(pi_ip)
        
        # Setup routes
        self._setup_routes()
    
    def _setup_routes(self):
        """Configure all Flask routes"""
        self.app.route('/')(self.index)
        self.app.route('/video_feed')(self.video_feed)
        self.app.route('/select_object', methods=['POST'])(self.select_object)
        self.app.route('/get_tracking_action', methods=['GET'])(self.get_tracking_action)
    
    def get_ip_address(self):
        """Utility: Get local IP address for displaying on web UI"""
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        return ip_address

    def index(self):
        """Route: Web interface main page"""
        ip_address = self.get_ip_address()
        return render_template('index_test_yolo_tracking.html', ip_address=ip_address)

    def video_feed(self):
        """Route: MJPEG stream for browser"""
        def generate():
            while True:
                with self.robot_controller.frame_lock:
                    if self.robot_controller.annotated_frame is None:
                        continue
                    ret, buffer = cv2.imencode('.jpg', self.robot_controller.annotated_frame)
                    if not ret:
                        continue
                    frame_bytes = buffer.tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

        return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

    def select_object(self):
        """Route: Receive selected object ID/class from the frontend"""
        data = request.get_json()
        self.robot_controller.target_id = int(data['id'])  # The ID of the object selected by the user
        self.robot_controller.target_class_label = data['class_label']
        self.robot_controller.tracking_defined = True
        return jsonify({
            "status": "success",
            "message": f"Tracking {self.robot_controller.target_class_label} with ID {self.robot_controller.target_id}."
        })

    def get_tracking_action(self):
        """Route: Used by frontend to poll the current tracking action"""
        return jsonify({"action": self.robot_controller.tracking_action})

    def capture_and_process(self):
        """Background thread: Capture and process frames from Raspberry Pi"""
        # Connect to video stream served by Raspberry Pi
        stream_url = f"{self.PI_IP}/video_feed"
        stream = requests.get(stream_url, stream=True)

        byte_data = b""
        frame = None  # Initialize frame variable
        
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

                self.robot_controller.process_frame(frame)

    def run(self):
        """Start the Flask app and background processing thread"""
        threading.Thread(target=self.capture_and_process, daemon=True).start()
        self.app.run(host=self.host, port=self.port, threaded=True)

if __name__ == '__main__':
    controller = WebApp()
    controller.run()