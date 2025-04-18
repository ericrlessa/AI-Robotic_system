import os
import sys
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import time
from .tcp_connection_client import connect_to_server, send_data, check_connection

#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

#from cloud.aws.pubsub_aws_iot import publish, get_connection

class HandGestureService:
    def __init__(self, model_path=None,                 
                 width=960, height=540):
        if model_path is None:
            # Dynamically resolve path to: /models/hand_gesture_model.tflite
            self.model_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), '../../../../models/hand_gesture_model.tflite')
            )
        else:
            self.model_path = model_path
        self.width = width
        self.height = height
        self.gesture_names = ["Up", "Down", "Left", "Right", "Left Up", "Left Down", "Right Down", "Right Up", "Fire"]

        self.mp_hands, self.hands, self.mpDraw, self.handLmsStyle, self.handConStyle = self.init_mediapipe_hands()
        self.interpreter, self.input_details, self.output_details = self.init_tflite_model()

    def init_mediapipe_hands(self):
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
        mpDraw = mp.solutions.drawing_utils
        handLmsStyle = mpDraw.DrawingSpec(color=(0, 255, 0), thickness=2)
        handConStyle = mpDraw.DrawingSpec(color=(255, 255, 255), thickness=2)
        return mp_hands, hands, mpDraw, handLmsStyle, handConStyle

    def init_tflite_model(self):
        interpreter = tf.lite.Interpreter(model_path=self.model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        return interpreter, input_details, output_details

    def normalize_landmarks(self, landmarks):
        base_x, base_y = landmarks[0].x, landmarks[0].y
        normalized = np.array([[lm.x - base_x, lm.y - base_y] for lm in landmarks])
        return normalized.flatten()

    def process_gesture(self, landmarks):
        normalized_landmarks = self.normalize_landmarks(landmarks)
        input_data = np.array(normalized_landmarks, dtype=np.float32).reshape(self.input_details[0]['shape'])
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        predicted_class = np.argmax(output_data)
        return self.gesture_names[predicted_class] if predicted_class is not None else "Stop"

    def calc_bounding_rect(self, image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]
        landmark_array = np.empty((0, 2), int)
        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            landmark_point = [np.array((landmark_x, landmark_y))]
            landmark_array = np.append(landmark_array, landmark_point, axis=0)
        x, y, w, h = cv2.boundingRect(landmark_array)
        return [x, y, x + w, y + h]

    def calculate_fps(self, prev_time, prev_fps):
        current_time = time.time()
        if(current_time - prev_time > 0):
            fps = 0.9 * prev_fps + 0.1 * (1 / (current_time - prev_time))
            return fps, current_time
        else:
            return 0, current_time

    def draw_landmarks(self, frame, hand_landmarks):
        self.mpDraw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
            self.handLmsStyle, self.handConStyle)

