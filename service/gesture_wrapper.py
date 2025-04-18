# gesture_wrapper.py

import threading
from object_detection.src.robot_controller import ObjectTrackingRobotController
from object_tracking_wrapper import capture_frame_n_process


def handle_gesture(robot_controller, next_prompt, pi_ip):
    print("[GESTURE] Activating gesture recognition...")

    def run_gesture():
        capture_frame_n_process(search=False, robot_controller=robot_controller, pi_ip=pi_ip)

    threading.Thread(target=run_gesture, daemon=True).start()
    return {"status": "gesture_mode_started", "next_prompt": next_prompt}
