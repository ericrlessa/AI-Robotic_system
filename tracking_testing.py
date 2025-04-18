import cv2
from flask import Flask, Response, render_template, request, jsonify
import threading
import socket
from camera_functions import yolo_results, yolo_ds_draw, yolo_ds_model_initalize, yolo_ds_update
from motor_control import tracking_move
# def tracking_move(id, class_label, objects_centers, latest_frame):
#     """
#     Function to track the object based on its ID and class label.
#     Args:
#         id (int): The ID of the object.
#         class_label (str): The class label of the object.
#         center_x (int): The x-coordinate of the object's center.
#         center_y (int): The y-coordinate of the object's center.
#         frame_shape (tuple): The shape of the frame (height, width).
#     """
#         # Get the width and height of the frame
#     frame_height, frame_width, _ = latest_frame.shape

#     # Define movement thresholds
#     left_threshold = frame_width // 3
#     right_threshold = 2 * frame_width // 3
#     up_threshold = frame_height // 3
#     down_threshold = 2 * frame_height // 3
    
#     # Get the object's center coordinates
#     object_center = None
#     for obj in objects_centers:
#         if obj[0] == str(id):
#             object_center = obj
#             break
    
#     # If the object with the selected ID is not found, return
#     if object_center is None:
#         print(f"Error: Object with ID {id} not found.")
#         return

#     # Get the center coordinates of the selected object
#     center_x, center_y = object_center[2], object_center[3]
#     # Movement variables
#     # Movement variables
#     action = ""

#     # Determine movement based on car's position
#     if center_x < left_threshold:
#         action = "Turn Left"
#     elif center_x > right_threshold:
#         action = "Turn Right"
#     else:
#         action = "no action"
#     update_tracking_action(action)

def update_tracking_action(action):
    """
    Update the tracking action in the Flask backend, sending it to the frontend.
    """
    global tracking_action
    tracking_action = action

# Initialize Flask app
app = Flask(__name__)

model, class_names, tracker = yolo_ds_model_initalize(model_name="yolo11n.pt")

frame_lock = threading.Lock()
latest_frame = None
annotated_frame = None
last_detections = []
frame_count = 0
tracking_defined = False
target_id = None
target_class_label = None
tracking_action = ""

def capture_and_process():
    global latest_frame, annotated_frame, last_detections, frame_count, tracking_defined, target_id, target_class_label
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_count += 1

        with frame_lock:
            latest_frame = frame.copy()

            if frame_count % 2 == 0:
                # Run YOLO detection every 2nd frame
                results, detections = yolo_results(model, latest_frame)
                last_detections = yolo_ds_update(latest_frame, detections, tracker)

            # Now unpack all three returned values from yolo_ds_draw
            annotated_frame, objects_centers = yolo_ds_draw(latest_frame, last_detections, class_names)

            # Move the motors based on the detected objects
            if tracking_defined:
                action = tracking_move(target_id, target_class_label, objects_centers, frame)
                update_tracking_action(action)


def get_ip_address():
    hostname = socket.gethostname()  # Get the hostname of the server
    ip_address = socket.gethostbyname(hostname)  # Resolve the IP address of the hostname
    return ip_address

@app.route('/')
def index():
    ip_address = get_ip_address()
    return render_template('index_test_yolo_tracking.html', ip_address=ip_address)

# Function to generate video frames for Flask streaming
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

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Handle object selection via front-end input
@app.route('/select_object', methods=['POST'])
def select_object():
    global tracking_defined, target_id, target_class_label
    data = request.get_json()
    target_id = int(data['id'])
    target_class_label = data['class_label']
    tracking_defined = True
    return jsonify({"status": "success", "message": f"Tracking {target_class_label} with ID {target_id}."})

@app.route('/get_tracking_action', methods=['GET'])
def get_tracking_action():
    global tracking_action
    return jsonify({"action": tracking_action})

if __name__ == '__main__':
    threading.Thread(target=capture_and_process, daemon=True).start()
    app.run(host='0.0.0.0', port=4000, threaded=True)