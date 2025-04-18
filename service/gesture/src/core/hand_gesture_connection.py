import os
import sys
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import time
from core.tcp_connection_client import connect_to_server, send_data, check_connection
from core.hand_gesture import HandGestureService


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

#from cloud.aws.pubsub_aws_iot import publish, get_connection

class HandGestureServiceConnection:
    def __init__(self, dashboard, cloud=False, raspberry_pi_ip='192.168.2.104',
                 port=12000):
        
        self.hand_gesture = HandGestureService()

        self.raspberry_pi_ip = raspberry_pi_ip
        self.port = port

        self.cloud = cloud

        #if(self.cloud):
            #self.connection = get_connection("pi_sender")
        #else: 
        self.connection = connect_to_server(self.raspberry_pi_ip, self.port)

        self.dashboard = dashboard
        self.cloud = cloud

    def check_connection(self):
        if(self.cloud):
            return True
        else:
            return check_connection(self.connection)
           

    def close_connection(self):
        if(self.cloud):
            disconnect_future = self.connection.disconnect()
            disconnect_future.result()
        else:
            self.connection.close()

    def send_gesture_command(self, gesture_name):
        # #if(self.cloud):
        #     publish(self.connection, message=gesture_name.encode())
        # else:
        send_data(self.connection, gesture_name.encode())

    def run(self):
        cap = cv2.VideoCapture(0)
        prev_time = time.time()
        prev_fps = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (self.hand_gesture.width, self.hand_gesture.height))
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            result = self.hand_gesture.hands.process(rgb_frame)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    gesture_name = self.hand_gesture.process_gesture(hand_landmarks.landmark)
                    self.send_gesture_command(gesture_name)
                    self.hand_gesture.draw_landmarks(frame, hand_landmarks)

                    brect = self.hand_gesture.calc_bounding_rect(frame, hand_landmarks)
                    cv2.rectangle(frame, (brect[0], brect[1]), (brect[2], brect[3]), (0, 255, 0), 2)
                    cv2.putText(frame, f'Gesture: {gesture_name}', (180, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (255, 0, 0), 2, cv2.LINE_AA)

            fps, prev_time = self.hand_gesture.calculate_fps(prev_time, prev_fps)
            prev_fps = fps
            cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                        cv2.LINE_AA)
            self.dashboard.update_frame(frame)
            #cv2.imshow('Hand Gesture Recognition', frame)

            if cv2.waitKey(1) & 0xFF == 27:
                self.close_connection()
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    service = HandGestureService()
    service.run()
