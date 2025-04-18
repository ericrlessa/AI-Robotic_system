import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


def yolo_ds_model_initalize(model_name = "../../../models/yolo11n.pt",     
                            nms_max_overlap=1.0,
                            max_iou_distance=0.5,
                            max_age=30,
                            n_init=10,
                            nn_budget=100000):
    """
    Initialize the YOLO model and DeepSort tracker.
    Args:
        model_name (str): The name of the YOLO model.
        max_dist (float): Maximum distance for matching.
        min_confidence (float): Minimum confidence for detections.
        nms_max_overlap (float): Non-maximum suppression overlap threshold.
        max_iou_distance (float): Maximum IOU distance for matching.
        max_age (int): Maximum age for tracking.
        n_init (int): Number of initial detections before tracking starts.
        nn_budget (int): Budget for nearest neighbor matching.
    Returns:
        model (YOLO): The loaded YOLO model.
        class_names (list): The class names of the model.
        tracker (DeepSort): The initialized DeepSort tracker.
    """
    # Load the YOLO model
    model = YOLO(model_name)
    # get the classes
    class_names = model.names
    # Initialize the DeepSort tracker
    tracker = DeepSort(max_age=max_age,
                        n_init=n_init,
                        nn_budget=nn_budget,
                        nms_max_overlap=nms_max_overlap,
                        max_iou_distance=max_iou_distance)
    
    return model, class_names, tracker

def yolo_results(model, frame):
    """
    Perform object detection using the YOLO model.
    Args:
        model (YOLO): The loaded YOLO model.
        frame (numpy.ndarray): The input frame.
    Returns:
        results (list): The detection results.
        detections (list): The detected bounding boxes, confidences, and class IDs.
    """
    # Perform object detection
    results = model.track(frame, conf=0.8, persist=True, verbose=False)
    detections = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))
    return results, detections

def yolo_ds_update(frame, detections, tracker):
    """
    Update the DeepSort tracker with the new detections.
    Args:
        frame (numpy.ndarray): The input frame.
        detections (list): The detection results.
        tracker (DeepSort): The DeepSort tracker.
    Returns:
        last_detections (list): The updated detections.
    """
    # Update the tracker with the new detections
    last_detections = tracker.update_tracks(detections, frame=frame)
    return last_detections

def yolo_ds_draw(frame, last_detections, class_names, tracking_confirmed=True):
    """
    Draw the detection results on the frame.
    Args:
        frame (numpy.ndarray): The input frame.
        last_detections (list): The updated detections.
        class_names (list): The class names of the model.
        tracking_confirmed (bool): Whether to filter by tracking confirmation. (video = True, image = False)
    """
    annotated = frame.copy()
    objects_centers = []

    for track in last_detections:
        if tracking_confirmed:
            if not track.is_confirmed():
                continue
        track_id = track.track_id
        l, t, r, b = map(int, track.to_ltrb())
        ltrb = (l, t, r, b)
        class_id = track.get_det_class() if hasattr(track, "get_det_class") else None
        class_label = class_names.get(class_id, "object")
        label = f"{class_label} ID:{track_id}"
        print(f"Track ID: {track_id}, Class: {class_label}, Bounding Box: ({l}, {t}, {r}, {b})")

        # Draw bounding box and label
        cv2.rectangle(annotated, (l, t), (r, b), (0, 255, 0), 2)
        cv2.putText(annotated, label, (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        center_x = (l + r) // 2
        center_y = (t + b) // 2
        cv2.circle(annotated, (center_x, center_y), 5, (0, 0, 255), -1)

        # Add the object data for front-end selection
        objects_centers.append((track_id, class_label, center_x, center_y))

    return annotated, objects_centers, ltrb

def lmm_find_object(prompt, last_detections):
    """
    Find the object in the last detections based on the prompt.
    Args:
        prompt (str): The prompt to search for.
        last_detections (img): The updated detections (yolo+deepsort result).
    Returns:
        found_objects_id (list): The IDs of the found objects.
    """

    return found_objects_id

