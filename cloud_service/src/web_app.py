import requests
import cv2
from flask import Flask, Response, render_template, request, jsonify
from threading import Thread

class WebApp:
    def __init__(self, robot_controller, pi_ip="http://192.168.2.104:5000", host='0.0.0.0', port=5000):
        # Initialize Flask app
        self.app = Flask(__name__)
    
        self.PI_IP = pi_ip
        # Configuration
        self.host = host
        self.port = port

        self.public_ip = None
        
        self.robot_controller = robot_controller
        
        # Setup routes
        self._setup_routes()
    
    def _setup_routes(self):
        """Configure all Flask routes"""
        self.app.route('/')(self.index)
        self.app.route('/video_feed')(self.video_feed)
        self.app.route('/select_object', methods=['POST'])(self.select_object)
        self.app.route('/get_tracking_action', methods=['GET'])(self.get_tracking_action)
    
    def get_ip_address(self):
        try:
            if self.public_ip is None:
                # Try IMDSv2 first
                token = requests.put(
                    'http://169.254.169.254/latest/api/token',
                    headers={'X-aws-ec2-metadata-token-ttl-seconds': '21600'},
                    timeout=2
                ).text
                headers = {'X-aws-ec2-metadata-token': token}
                public_ip = requests.get(
                    'http://169.254.169.254/latest/meta-data/public-ipv4',
                    headers=headers,
                    timeout=2
                ).text
                    
            return public_ip
        except:
            return "localhost"
    
    def index(self):
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

    def run(self):
        """Start the Flask app and background processing thread"""
        Thread(target=self.app.run, kwargs={
            'host': self.host,
            'port': self.port,
            'threaded': True,
            'use_reloader': False  # Important!
        }).start()