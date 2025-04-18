import sys
import os
import cv2
import threading
import requests
import numpy as np
import time
from .camera_functions import yolo_results, yolo_ds_draw, yolo_ds_model_initalize, yolo_ds_update

# Add the gesture/src folder to sys.path
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../gesture/src')))
from gesture.src.core.hand_gesture import HandGestureService


class ObjectTrackingRobotController:
    def __init__(self, pi_ip=None):
        # Frame handling and tracking state
        self.frame_lock = threading.Lock()
        self.latest_frame = None
        self.annotated_frame = None
        self.last_detections = []
        self.frame_count = 0
        self.tracking_defined = False
        self.target_id = None
        self.target_class_label = None
        self.tracking_action = ""

        self.PI_IP = pi_ip
        
        # Initialize models
        self.model, self.class_names, self.tracker = yolo_ds_model_initalize(
            model_name="../../../models/yolov8n.pt"
        )
        self.hand_gesture_service = HandGestureService()
        
        # Frame counters
        self.frame_count_move_robot = 0
        self.frame_count_yolo = 0
        self.frame_count_gesture = 0
        
        # Timing for FPS calculation
        self.prev_time = time.time()
        self.prev_fps = 0
        
    
    def update_tracking_action(self, action):
        """Update current tracking action (for frontend display)"""
        self.tracking_action = action
    
    def tracking_move(self, object_center, frame, ltrb):
        """
        Determine how the robot should move based on the object's position in the frame.
        Returns: (vx, vy, omega)
        """
        frame_height, frame_width, _ = frame.shape

        # Define horizontal thresholds (left/center/right)
        left_threshold = frame_width // 3
        right_threshold = 2 * frame_width // 3

        # Get object center coordinates
        center_x, center_y = object_center[2], object_center[3]

        # Motion control variables
        vx = 0       # Left-right movement
        vy = 0       # Forward-backward movement
        omega = 0    # Rotation

        frame_area = frame_height * frame_width
        
        l, t, r, b = ltrb
        area_bbox = (r - l) * (b - t)

        speed = 0.25
        if area_bbox < 0.2 * frame_area or b - t <= (0.50*frame_height):
            vy = speed
        elif area_bbox > 0.6 * frame_area or b - t >= (0.85*frame_height): 
            vy = -speed
        
        # Decide rotation based on horizontal object position
        if center_x < left_threshold:
            omega = speed  # Rotate left
            action = "Turn Left"
        elif center_x > right_threshold:
            omega = -speed  # Rotate right
            action = "Turn Right"
        else:
            omega = 0
            action = "no action"

        self.update_tracking_action(action)
        return vx, vy, omega

    def send_gesture_command(self, gesture_name):
        """Send gesture command to Raspberry Pi"""
        try:
            requests.post(f"{self.PI_IP}/command_gesture", json={
                "gesture": gesture_name
            }, timeout=0.2)
        except Exception as e:
            print(f"[WARN] Could not send gesture data to Pi: {e}")

    def send_tracking_command(self, vx, vy, omega):
        try:
            if omega != 0 or vy != 0 or vx != 0:
                # Send motion command to Raspberry Pi
                requests.post(f"{self.PI_IP}/command", json={
                    "vx": vx,
                    "vy": vy,
                    "omega": omega
                }, timeout=0.2)
        except Exception as e:
            print(f"[WARN] Could not send motion data to Pi: {e}")

    def process_frame(self, frame):
        self.frame_count_move_robot += 1
        self.frame_count_yolo += 1
        self.frame_count_gesture += 1
        
        with self.frame_lock:
            self.latest_frame = frame.copy()
            self.annotated_frame = self.latest_frame

            self.process_object_detection(frame)
            self.process_hand_gesture()

    def process_object_detection(self, frame):
        # Process YOLO every 5 frames
        if self.frame_count_yolo >= 5:
            self.frame_count_yolo = 0
            
            results, detections = yolo_results(self.model, self.latest_frame)
            self.last_detections = yolo_ds_update(self.latest_frame, detections, self.tracker)

            self.annotated_frame, objects_centers, ltrb = yolo_ds_draw(
                self.latest_frame, self.last_detections, self.class_names, tracking_confirmed=True
            )
            
            # If user has selected a target, calculate movement and send to Pi
            if self.tracking_defined and self.frame_count_move_robot >= 25:
                self.frame_count_move_robot = 0

                # Find the object center based on ID
                object_center = None
                for obj in objects_centers:
                    if obj[0] == str(self.target_id):
                        object_center = obj
                        break

                if object_center is not None:
                    vx, vy, omega = self.tracking_move(object_center, frame, ltrb)
                    self.send_tracking_command(vx, vy, omega)
                else:
                    self.tracking_defined = False

    def process_hand_gesture(self):
        # Process hand gestures if not tracking an object
        if not self.tracking_defined:
            result = self.hand_gesture_service.hands.process(self.annotated_frame)
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    gesture_name = self.hand_gesture_service.process_gesture(hand_landmarks.landmark)
                    if("Fire" == gesture_name):
                        gesture_name = "Up"

                    if self.frame_count_gesture >= 10:
                        self.frame_count_gesture = 0
                        self.send_gesture_command(gesture_name)
                    self.hand_gesture_service.draw_landmarks(self.annotated_frame, hand_landmarks)

                    brect = self.hand_gesture_service.calc_bounding_rect(self.annotated_frame, hand_landmarks)
                    cv2.rectangle(self.annotated_frame, (brect[0], brect[1]), (brect[2], brect[3]), (0, 255, 0), 2)
                    cv2.putText(
                        self.annotated_frame, f'Gesture: {gesture_name}', (180, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA
                    )
            
            # Calculate and display FPS
            fps, self.prev_time = self.hand_gesture_service.calculate_fps(self.prev_time, self.prev_fps)
            self.prev_fps = fps
            cv2.putText(
                self.latest_frame, f'FPS: {int(fps)}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA
            )

    def run(self):
        """Start the Flask app and background processing thread"""
        threading.Thread(target=self.capture_and_process, daemon=True).start()
        self.app.run(host=self.host, port=self.port, threaded=True)
