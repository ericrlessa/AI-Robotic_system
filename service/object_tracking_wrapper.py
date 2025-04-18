# object_tracking_wrapper.py
import sys
import cv2
import os
import threading
from object_detection.src.camera_functions import yolo_ds_model_initalize, yolo_results, yolo_ds_update, yolo_ds_draw
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'object_detection/src')))
from object_detection.src.robot_controller import ObjectTrackingRobotController
import requests
import numpy as np

def capture_frame_n_process(search=False, robot_controller=None, pi_ip = ""):
        """Background thread: Capture and process frames from Raspberry Pi"""
        # Connect to video stream served by Raspberry Pi
        stream_url = f"{pi_ip}/video_feed"
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
                if search:
                    return frame
                if search == False:
                    robot_controller.process_frame(frame)

def handle_object_tracking(function, params, robot_controller, next_prompt, pi_ip):
    if function == "object_search":
        desc = params.get("target_description")
        frame = capture_frame_n_process(search=True, robot_controller=robot_controller, pi_ip=pi_ip)
        image = frame.copy()
        #lower the resolution
        image = cv2.resize(image, (640, 480))
        model, class_names, tracker = yolo_ds_model_initalize()
        results, detections = yolo_results(model, image)
        last_detections = yolo_ds_update(image, detections, tracker)
        annotated, objects_centers = yolo_ds_draw(image, last_detections, class_names, tracking_confirmed=False)
        ret, buffer = cv2.imencode('.jpg', annotated)
        image_bytes = buffer.tobytes()
        print(f"[TRACKING] Running object search: {desc}")
        # You could return dummy data or call into your frame/yolo system
        return {"status": "object_search_started", "description": desc, "result_image": image_bytes, "next_prompt": next_prompt}

    if function == "object_tracking":
        obj_id = params.get("id")
        class_label = params.get("class")
        print(f"[TRACKING] Starting object tracking on ID {obj_id} ({class_label})")
        robot_controller.target_id = int(obj_id)  # The ID of the object selected by the user
        robot_controller.target_class_label = class_label
        robot_controller.tracking_defined = True
        capture_frame_n_process(search=False, robot_controller=robot_controller, pi_ip=pi_ip)
        threading.Thread(target=capture_frame_n_process, daemon=True).start()
        # Insert command here to start real-time tracking
        return {"status": "tracking_started", "id": obj_id, "class": class_label, "next_prompt": next_prompt}

    return {"status": "error", "message": "Unknown tracking command", "next_prompt": next_prompt}
