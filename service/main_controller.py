# main_controller.py

from flask import Flask, request, jsonify, render_template, Response
from lmm_mcp.src.chat_memory import ChatMemory
from object_tracking_wrapper import handle_object_tracking
from gesture_wrapper import handle_gesture
import base64, requests, os
from werkzeug.utils import secure_filename
import whisper
import json
import socket
import cv2
import sys
import numpy as np
import re
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'object_detection/src')))
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '/gesture/src')))

from object_detection.src.robot_controller import ObjectTrackingRobotController
import threading

pi_ip="http://192.168.2.104:5000"
host='0.0.0.0'
port=4000
robot_controller = ObjectTrackingRobotController(pi_ip)
next_prompt = None
result_imgage = None

# --- Flask app setup ---
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# --- LMM memory ---
initial_prompt = """
You are an intelligent assistant embedded in a robot car. You receive voice commands from the user and sensor inputs from the robot’s vision system (YOLO + DeepSORT). Your job is to understand user intent and return a valid JSON command to control robot behavior.

---

You must respond with **valid, raw JSON ONLY** — with **no markdown**, no backticks, no extra commentary. The JSON must always match this schema:

{
  "function": "function_name" | null,
  "parameters": { ... },
  "response": "What the robot should say to the user",
  "next_prompt": "Prompt for the next system action, or null"
}

---

### Supported Inputs:
- Natural language commands (e.g., "follow the man in red")
- Detection results from YOLO + DeepSORT (object IDs, bounding boxes, classes)
- Images or cropped detections
- Short video clips (e.g., for re-identifying lost objects)
- System messages like: "object lost", "tracking failure", "gesture detected", etc.

---

### Primary Functions:
- object_search: Search for a described object or class
- object_tracking: Track a known object by ID or visual match
- gesture_recognition: Activate and interpret gesture input

---

### Output Rules:

- Use `"function": "object_search"` if the user describes an object **without providing detection data or ID**
- Use `"function": "object_tracking"` only when:
  1. Detection results have already been received and a specific ID can be matched, OR
  2. The user explicitly references an ID or object from a prior step
- Use `"function": "gesture_recognition"` if the user switches to gesture control
- Use `"function": null` for small talk or when no function needs to be run
- Always provide a helpful `"response"` for the user
- Use `"next_prompt"` to request follow-up input from the system (e.g., detection data)
- Never guess an ID. Only use `"id"` and `"class"` when you're confident detection data is available
- Do not include triple backticks (` ``` `), markdown headers, or comments. Output must be **raw JSON only**

---

### Good Examples:

User: "Follow the man in the red jacket."
→
{
  "function": "object_search",
  "parameters": {
    "target_description": "man in red jacket"
  },
  "response": "I'll look for the man in the red jacket.",
  "next_prompt": "Detection data received. Match the description to an object and return function, id, class, and next_prompt: null."
}

User: "Detection data received. Match the description to an object and return function, id, class, and next_prompt: null."
→
{
  "function": "object_tracking",
  "parameters": {
    "id": 4,
    "class": "person"
  },
  "response": "Now tracking the person in red with ID 4.",
  "next_prompt": null
}

User: "Switch to gesture mode."
→
{
  "function": "gesture_recognition",
  "parameters": {},
  "response": "Gesture control activated.",
  "next_prompt": null
}

User: "Thanks!"
→
{
  "function": null,
  "parameters": {},
  "response": "You're welcome!",
  "next_prompt": null
}

User: "System Event: object lost – occlusion detected."
→
{
  "function": null,
  "parameters": {},
  "response": "The target is lost due to occlusion. Would you like me to search again or switch to gesture control?",
  "next_prompt": null
}

"""
memory = ChatMemory(system_prompt=initial_prompt)

# --- Audio transcription ---
def transcribe_audio(file_path):
    model = whisper.load_model("base")
    result = model.transcribe(file_path)
    return result['text']

# --- Image encoding ---
def encode_image(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# --- Call LLM via Ollama ---
def call_ollama(messages, image=None):
    payload = {
        "model": "llava:13b",
        "messages": messages,
        "stream": False
    }
    print(payload)
    res = requests.post("http://localhost:11434/api/chat", json=payload)
    return res.json()["message"]["content"]

# --- Function Dispatcher ---
def dispatch_action(json_data):
    print("dispatch step")
    function = json_data.get("function")
    print("loaded function from json")
    params = json_data.get("parameters", {})
    print("loaded params from json")
    next_prompt = json_data.get("next_prompt", None)

    if function == "object_tracking" or function == "object_search":
        print("in obj")
        return handle_object_tracking(function, params, robot_controller, next_prompt, pi_ip)

    elif function == "gesture_recognition":
        print("in gesture")
        return handle_gesture(robot_controller, next_prompt, pi_ip)

    return {"status": "no_action", "message": json_data.get("response", "No function selected.")}

def clean_lmm_output(response_text):
    try:
        print("text clearning")
        # Remove triple backticks and 'json' label
        cleaned = (
            response_text.replace("```json", "")
                        .replace("```", "")
                        .replace("\\", "")   # <== REMOVE ALL BACKSLASHES
                        .strip()
        )
        
        parsed = json.loads(cleaned)
        print("text clearned")
        return parsed

    except Exception as e:
        print("[ERROR] Failed to parse LMM response:", e)
        return {
            "function": None,
            "parameters": {},
            "response": response_text,
            "next_prompt": None
        }

# --- Utility functions ---
def get_ip_address():
        """Utility: Get local IP address for displaying on web UI"""
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        return ip_address


# --- Routes ---
@app.route("/")
def index():
    """Route: Web interface main page"""
    ip_address = get_ip_address()
    return render_template("index.html", ip_address=ip_address)


@app.route("/chat", methods=["POST"])
def chat():
    global next_prompt, result_imgage
    print(next_prompt)
    try:
        user_text = request.form.get("message")
        image_file = request.files.get("image")
        audio_file = request.files.get("audio")

        if audio_file:
            audio_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(audio_file.filename))
            audio_file.save(audio_path)
            user_text = transcribe_audio(audio_path)

        if not user_text:
            return jsonify({"error": "No valid text or audio input."}), 400
        
        encoded_image = None
        
        if next_prompt:
            user_text = next_prompt
            next_prompt = None

        # Encode image if provided
        #encoded_image = None
        if image_file:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(image_file.filename))
            image_file.save(image_path)
            encoded_image = encode_image(image_path)
            memory.add_user(user_text, encoded_image)
        else:
            if result_imgage:
                encoded_image = result_imgage
                result_imgage = None
            memory.add_user(user_text, encoded_image) #if there is no image upload, or pass on from previous chat, will be none
        # Call the LMM
        response_text = call_ollama(memory.get_messages(), image=encoded_image)
        memory.add_assistant(response_text)
        print(response_text)

        response_json = clean_lmm_output(response_text)

        # Dispatch the action
        action_result = dispatch_action(response_json)
        next_prompt = response_json.get("next_prompt", None)
        print(next_prompt)
        if next_prompt == "null":
            next_prompt = None
        result_imgage = action_result.get("result_image", None)
        return jsonify({
            "lmm_response": response_json,
            "action_result": action_result["status"]
        })

    except Exception as e:
        print("Error in /chat:", str(e))
        return jsonify({"error": str(e)}), 500
    
@app.route("/video_feed")
def video_feed():
    """Route: MJPEG stream for browser"""
    def generate():
        while True:
            with robot_controller.frame_lock:
                if robot_controller.annotated_frame is None:
                    continue
                ret, buffer = cv2.imencode('.jpg', robot_controller.annotated_frame)
                if not ret:
                    continue
                frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=4000, threaded=True)