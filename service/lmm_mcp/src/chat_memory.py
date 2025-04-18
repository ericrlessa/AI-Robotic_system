# chat_memory.py

class ChatMemory:
    def __init__(self, system_prompt, max_tokens=20000000):
        self.system_prompt = {"role": "user", "content": system_prompt}
        self.history = [self.system_prompt]
        self.max_tokens = max_tokens

    def add_user(self, content, image=None):
        message = {"role": "user", "content": content}
        if image:
            message["images"] = [image]
        self.history.append(message)

    def add_assistant(self, content):
        self.history.append({"role": "assistant", "content": content})

    def trim(self):
        # Keep system prompt, remove oldest turns if too long
        while self.token_count() > self.max_tokens and len(self.history) > 3:
            self.history.pop(1)  # Remove second item (first user/assistant turn)

    def token_count(self):
        return sum(len(m["content"].split()) for m in self.history)

    def get_messages(self):
        self.trim()
        return self.history
