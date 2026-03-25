
import requests
from llama_cpp import Llama
from ai_assistant.memory import SemanticMemory
import os

class AIModel:
    def __init__(self, api_key: str = None, model_path: str = "./your-model.gguf", memory: SemanticMemory = None):
        self.api_key = api_key
        self.model_path = model_path
        self.memory = memory
        self._llm = None  # Singleton instance for offline model

    def _load_offline_model(self):
        """Loads the GGUF model once and keeps it in memory."""
        if self._llm is None:
            if os.path.exists(self.model_path):
                print(f"--- Loading GGUF model from {self.model_path}... ---")
                self._llm = Llama(model_path=self.model_path, n_ctx=2048)
            else:
                print(f"--- GGUF model not found at {self.model_path} ---")
                return None
        return self._llm

    def get_offline_response(self, prompt: str, memory_context: str = "") -> str:
        """Gets a response from the local GGUF model."""
        llm = self._load_offline_model()
        if llm is None:
            return "Offline model not available."

        full_prompt = (
            "You are a helpful and intelligent AI assistant. Answer based on the context provided.\n\n"
            f"--- Context ---\n{memory_context}\n\n"
            f"--- User's Question ---\n{prompt}\n\n"
            "AI Assistant:"
        )

        try:
            output = llm(full_prompt, max_tokens=250, echo=False)
            return output['choices'][0]['text'].strip()
        except Exception as e:
            return f"Error in offline mode: {e}"

    def get_online_response(self, prompt: str, memory_context: str = "", search_results: str = "") -> str:
        """Gets a response from an online service (defaulting to DeepSeek or OpenAI-compatible)."""
        if not self.api_key:
            return "Online mode requires an API key."

        full_prompt = (
            "You are a professional and capable AI assistant. Answer using the context and search results.\n\n"
            f"--- Context (Memory) ---\n{memory_context}\n\n"
            f"--- Search Results ---\n{search_results}\n\n"
            f"--- User's Question ---\n{prompt}\n\n"
            "AI Assistant:"
        )

        api_url = "https://api.deepseek.com/v1/chat/completions" # Can be switched to OpenAI if preferred
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "deepseek-chat", # Example model
            "messages": [{"role": "user", "content": full_prompt}]
        }
        try:
            response = requests.post(api_url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content'].strip()
        except requests.exceptions.RequestException as e:
            return f"Error connecting to online service: {e}"
        except (KeyError, IndexError) as e:
            return f"Error parsing API response: {e}"

    def get_ai_response(self, prompt: str, online: bool = False, search_results: str = "") -> str:
        """Main method to get AI response, deciding between modes and using memory."""
        memory_context = ""
        if self.memory:
            relevant_memories = self.memory.search_memories(prompt, top_k=3)
            memory_context = "\n".join(relevant_memories)

        if online and self.api_key:
            return self.get_online_response(prompt, memory_context, search_results)
        else:
            return self.get_offline_response(prompt, memory_context)
