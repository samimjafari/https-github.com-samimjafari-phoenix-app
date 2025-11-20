
import requests
from llama_cpp import Llama
from googlesearch import search
import json
import os

class AI:
    def __init__(self, api_key: str, model_path: str = "./your-model.gguf", memory_file: str = "memory.json"):
        self.api_key = api_key
        self.model_path = model_path
        self.memory_file = memory_file
        self.memory = self.load_memory()

    def load_memory(self) -> list:
        """Loads memory from a JSON file."""
        if os.path.exists(self.memory_file):
            with open(self.memory_file, 'r') as f:
                return json.load(f)
        return []

    def save_memory(self):
        """Saves memory to a JSON file."""
        with open(self.memory_file, 'w') as f:
            json.dump(self.memory, f, indent=4)

    def add_memory(self, text: str):
        """Adds a new piece of information to the memory."""
        self.memory.append(text)
        self.save_memory()
        print(f"--- Added to memory: '{text}' ---")

    def get_memory_context(self) -> str:
        """Returns the stored memories as a string."""
        if not self.memory:
            return "No memories available."
        return "\n".join(self.memory)

    def search_web(self, query: str) -> str:
        """
        Performs a web search and returns the top results.
        """
        print("--- Searching the web ---")
        try:
            search_results = [str(result) for result in search(query, num_results=3)]
            return "\n".join(search_results)
        except Exception as e:
            return f"Error during web search: {e}"

    def get_offline_response(self, prompt: str) -> str:
        """
        Gets a response from a local GGUF model using llama-cpp-python.
        """
        print("--- Using OFFLINE mode ---")

        memory_context = self.get_memory_context()
        augmented_prompt = (
            "Based on your memory, please answer the user's question.\n\n"
            f"--- Your Memory ---\n{memory_context}\n\n"
            f"--- User's Question ---\n{prompt}"
        )

        try:
            llm = Llama(model_path=self.model_path, n_ctx=2048)
            output = llm(augmented_prompt, max_tokens=150, echo=False)
            return output['choices'][0]['text'].strip()
        except Exception as e:
            return f"Error in offline mode: {e}"

    def get_online_response(self, prompt: str) -> str:
        """
        Gets a response from an online AI service, augmented with web search results and memory.
        """
        print("--- Using ONLINE mode ---")

        search_results = self.search_web(prompt)
        memory_context = self.get_memory_context()

        augmented_prompt = (
            "Based on the following web search results and your memory, please answer the user's question.\n\n"
            f"--- Your Memory ---\n{memory_context}\n\n"
            f"--- Search Results ---\n{search_results}\n\n"
            f"--- User's Question ---\n{prompt}"
        )

        api_url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": augmented_prompt}]
        }
        try:
            response = requests.post(api_url, headers=headers, json=data)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content'].strip()
        except requests.exceptions.RequestException as e:
            return f"Error connecting to online service: {e}"
        except (KeyError, IndexError) as e:
            return f"Error parsing API response: {e}"

    def is_connected_to_internet(self) -> bool:
        """
        Checks for an active internet connection.
        """
        try:
            requests.get("http://www.google.com", timeout=5)
            return True
        except requests.ConnectionError:
            return False

    def get_ai_response(self, prompt: str) -> str:
        """
        Decides whether to use online or offline mode based on internet connectivity.
        """
        if self.is_connected_to_internet():
            return self.get_online_response(prompt)
        else:
            return self.get_offline_response(prompt)
