import os
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from typing import Tuple, List, Optional


def yolo_ds_model_initialize(
    model_name: Optional[str] = None,
    nms_max_overlap: Optional[float] = None,
    max_iou_distance: Optional[float] = None,
    max_age: Optional[int] = None,
    n_init: Optional[int] = None,
    nn_budget: Optional[int] = None
) -> Tuple[YOLO, List[str], DeepSort]:
    """
    Initialize YOLO model and DeepSort tracker with env var fallback.
    
    Args:
        All parameters support env var overrides (see .env.example).
        Defaults are used if neither arg nor env var is provided.
    
    Returns:
        model: Loaded YOLO model
        class_names: List of class names
        tracker: Configured DeepSort tracker
    """
    # Load from environment variables if args are None
    config = {
        "model_name": model_name or os.getenv("YOLO_MODEL_PATH", "models/yolo11n.pt"),
        "nms_max_overlap": float(nms_max_overlap or os.getenv("DEEPSORT_NMS_OVERLAP", 0.6)),
        "max_iou_distance": float(max_iou_distance or os.getenv("DEEPSORT_MAX_IOU_DISTANCE", 0.7)),
        "max_age": int(max_age or os.getenv("DEEPSORT_MAX_AGE", 35)),
        "n_init": int(n_init or os.getenv("DEEPSORT_N_INIT", 7)),
        "nn_budget": int(nn_budget or os.getenv("DEEPSORT_NN_BUDGET", 200))
    }

    # Validate paths exist
    if not os.path.exists(config["model_name"]):
        raise FileNotFoundError(f"Model file not found: {config['model_name']}")

    # Initialize models
    model = YOLO(config["model_name"])
    tracker = DeepSort(
        max_age=config["max_age"],
        n_init=config["n_init"],
        nn_budget=config["nn_budget"],
        nms_max_overlap=config["nms_max_overlap"],
        max_iou_distance=config["max_iou_distance"]
    )
    
    return model, model.names, tracker

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
    results = model.track(frame, conf=0.7, persist=True, verbose=False)
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

def yolo_ds_draw(frame, last_detections, class_names):
    """
    Draw the detection results on the frame.
    Args:
        frame (numpy.ndarray): The input frame.
        last_detections (list): The updated detections.
        class_names (list): The class names of the model.
    Returns:
        annotated_frame (numpy.ndarray): The annotated frame.
        objects_centers (list): List of detected objects with their ID, class, and bounding box.
    """
    annotated = frame.copy()
    objects_centers = []
    ltrb = None

    for track in last_detections:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        l, t, r, b = map(int, track.to_ltrb())
        ltrb = (l, t, r, b)
        class_id = track.get_det_class() if hasattr(track, "get_det_class") else None
        class_label = class_names.get(class_id, "object")
        label = f"{class_label} ID:{track_id}"

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

