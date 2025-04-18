from flask import Flask, request, jsonify, render_template
from service.lmm_mcp.src.chat_memory import ChatMemory
import base64, requests, os
from werkzeug.utils import secure_filename
import whisper

app = Flask(__name__, template_folder="templates", static_folder="static")

app.config['UPLOAD_FOLDER'] = 'static/uploads'

initial_prompt = """
You are an intelligent assistant embedded in a robot car. You receive natural language commands from the user and sensor inputs from the robot’s perception system. Your role is to decide what function to perform and guide multi-step interactions intelligently.

---

Supported inputs:
- Natural language commands from the user
- Object detection results from YOLO + DeepSORT
- Images from the camera (e.g., current frame, cropped detections)
- Short video clips (e.g., for diagnosing tracking failure)
- Status messages such as "object lost", "tracking failure", or "occlusion detected"

---

Supported functions:
- object_tracking: Track a specific person or object using a visual description.
- object_search: Search the area for a described object or class.
- map_navigation: Navigate to a known location or landmark.
- area_scan: Scan and summarize visible surroundings.

---

You must always respond with a valid JSON object in this format:

{
  "function": "function_name" | null,
  "parameters": { ... },
  "response": "What the robot should say to the user",
  "next_prompt": "Prompt for the next system action (e.g., run vision), or null if complete"
}

---

Your responsibilities:
- Interpret user input or system status messages.
- Choose the appropriate robot function or respond conversationally.
- Use `target_description` for object_tracking unless detection data is already provided.
- When tracking fails or vision input is incomplete, use `next_prompt` to request more data.
- If the input is conversational (e.g., “thanks”, “what can you do?”), leave `"function"` and `"next_prompt"` as null and just respond naturally.
- If the object is lost (e.g., occluded or detection failed), explain why and propose a recovery strategy.
- Never include markdown or extra explanation — output must be raw JSON.


---

Examples:

User: "Can you follow the man in the red hoodie?"
→
{
  "function": "object_tracking",
  "parameters": {
    "target_description": "man in red hoodie"
  },
  "response": "I'll search for the man in the red hoodie.",
  "next_prompt": "You have now received detection data. Based on the request 'follow the man in the red hoodie', identify the best matching object from the list. Respond with a valid JSON including function, id, class, response, and set next_prompt to null."
}

User: "You have now received detection data. Based on the request 'follow the man in the red hoodie', identify the best matching object from the list. Respond with a valid JSON including function, id, class, response, and set next_prompt to null."
→
{
  "function": "object_tracking",
  "parameters": {
    "id": 2,
    "class": "person"
  },
  "response": "I'm now following the guy in the red shirt.",
  "next_prompt": null
}

User: "Go to the front entrance."
→
{
  "function": "map_navigation",
  "parameters": {
    "destination": "front entrance"
  },
  "response": "Heading to the front entrance.",
  "next_prompt": null
}

User: "Is there a dog nearby?"
→
{
  "function": "object_search",
  "parameters": {
    "class": "dog"
  },
  "response": "I'll look around for a dog.",
  "next_prompt": "You have now received detection data. Based on the request 'Is there a dog nearby?', search for an object matching class 'dog' and return valid JSON with function, id (if applicable), class, response, and set next_prompt to null."
}

User: "What's around us?"
→
{
  "function": "area_scan",
  "parameters": {},
  "response": "I'll scan the surroundings now.",
  "next_prompt": null
}

User: "Can you help me with something?"
→
{
  "function": null,
  "parameters": {},
  "response": "Sure! What do you need help with?",
  "next_prompt": null
}

User: "System Event: Object Lost 
       Please review the video and determine whether the target is still visible or can be re-identified. Respond using the standard JSON format."
→
{
  "function": null,
  "parameters": {},
  "response": "It looks like the system lost track due to an ID mismatch. Would you like me to scan the area or try to identify them again?",
  "next_prompt": "System Event: object lost
                  Reason: Tracker ID changed unexpectedly. DeepSORT mismatch detected.
                  Original Request: "Track the guy in the red hoodie.""
}

User: "System Event: object lost
       Reason: Tracker ID changed unexpectedly. DeepSORT mismatch detected.
       Original Request: "Track the guy in the red hoodie."
       Yes, please scan the area and try to identify them again."
→
{
  "function": "area_scan" ,
  "parameters": {},
  "response": "It looks like the system lost track due to an ID mismatch. Would you like me to scan the area or try to identify them again?",
  "next_prompt": ""You have now received detection data. Based on the request 'Is there a dog nearby?', search for an object matching Original Request: "Track the guy in the red hoodie." and return valid JSON with function, id (if applicable), class, response, and next_prompt""
}
"""
memory = ChatMemory(system_prompt=initial_prompt)

def encode_image(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def transcribe_audio(file_path):
    model = whisper.load_model("base")
    result = model.transcribe(file_path)
    return result['text']

def call_ollama(messages, image=None):
    payload = {
        "model": "ZimaBlueAI/Qwen2.5-VL-7B-Instruct",
        "messages": messages,
        "stream": False
    }
    print("Sending payload to Ollama:", payload)
    res = requests.post("http://localhost:11434/api/chat", json=payload)
    return res.json()["message"]["content"]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_text = request.form.get("message")
        image_file = request.files.get("image")
        audio_file = request.files.get("audio")

        # Handle audio input (transcribe to text)
        if audio_file:
            audio_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(audio_file.filename))
            audio_file.save(audio_path)
            user_text = transcribe_audio(audio_path)

        if not user_text:
            return jsonify({"error": "No valid text or audio input."}), 400

        # Encode image if provided
        encoded_image = None
        if image_file:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(image_file.filename))
            image_file.save(image_path)
            encoded_image = encode_image(image_path)
            #encoded_image = image_path
            # Add user message to memory
            memory.add_user(user_text,encoded_image)
        else:
            # Add user message to memory
            memory.add_user(user_text)
        # Call Ollama
        response_text = call_ollama(memory.get_messages(), image=encoded_image)
        print("call_ollama response")
        memory.add_assistant(response_text)

        output_json = {"response": response_text}
        return jsonify(output_json)

    except Exception as e:
        print("Error in /chat:", str(e))
        return jsonify({"error": str(e)}), 500
    
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=4000)