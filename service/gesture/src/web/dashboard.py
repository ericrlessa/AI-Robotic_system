import threading
import queue
from flask import Flask, render_template, Response, request, jsonify, redirect, url_for
import cv2
import time
import logging
from core.hand_gesture_connection import HandGestureServiceConnection

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Dashboard:
    def __init__(self):
        self.app = Flask(__name__)
        self.frame = None
        self.ip_address = None
        self.cloud = False
        self.handGestureService = None

        self.app.route("/")(self.index)
        self.app.route("/video_feed")(self.video_feed)
        self.app.route("/video_page")(self.video_page)
        self.app.route("/start", methods=["POST"])(self.start)

        @self.app.after_request
        def add_header(response):
            response.cache_control.no_store = True
            return response

    def index(self):
        if(self.handGestureService is not None and self.handGestureService.check_connection()):
            return redirect(url_for("video_page"))

        return render_template("index.html")
    
    def video_page(self):
        return render_template("video-feed.html")

    def video_feed(self):
        def generate():
            while True:
                if self.frame is not None:
                    try:
                        _, jpeg = cv2.imencode('.jpg', self.frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                    except cv2.error as e:
                        logging.error(f"OpenCV error in video_feed: {e}")
                    except Exception as e:
                        logging.exception(f"Error encoding frame: {e}")
                time.sleep(0.03)

        return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

    def update_frame(self, frame):
        self.frame = frame  # Directly update the frame

    def run(self):
        threading.Thread(target=self._run_app, daemon=True).start()

    def _run_app(self):
        self.app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)

    def start(self):
        try:
            ip_address = request.form.get('ipAddress')
            mode = request.form.get('mode')

            if (mode == "cloud"):
                self.cloud = True

            self.ip_address = ip_address

            self.handGestureService = HandGestureServiceConnection(self, cloud=self.cloud, raspberry_pi_ip=self.ip_address)
            threading.Thread(target=self.handGestureService.run, daemon=True).start()

            return redirect(url_for("video_page"))
        except Exception as e:
            logging.exception(f"Error in apply_config: {e}")
            return jsonify({"status": "error", "message": "Failed to apply configuration."})
