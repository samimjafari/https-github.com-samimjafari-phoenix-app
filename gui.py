
import tkinter as tk
from tkinter import scrolledtext, simpledialog
from ai_library.core import AI
import os

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Library")

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        self.ai = AI(api_key=api_key)

        # --- UI Elements ---
        self.prompt_label = tk.Label(root, text="Enter your prompt:")
        self.prompt_label.pack()

        self.prompt_entry = tk.Entry(root, width=80)
        self.prompt_entry.pack()

        self.send_button = tk.Button(root, text="Send", command=self.send_prompt)
        self.send_button.pack()

        self.response_area = scrolledtext.ScrolledText(root, width=80, height=20)
        self.response_area.pack()

        self.add_memory_button = tk.Button(root, text="Add to Memory", command=self.add_to_memory)
        self.add_memory_button.pack()

    def send_prompt(self):
        prompt = self.prompt_entry.get()
        if prompt:
            self.response_area.insert(tk.END, "User: " + prompt + "\n")
            self.prompt_entry.delete(0, tk.END)

            # Get AI response
            response = self.ai.get_ai_response(prompt)
            self.response_area.insert(tk.END, "AI: " + response + "\n\n")

    def add_to_memory(self):
        memory_text = simpledialog.askstring("Add to Memory", "Enter a fact or note to remember:")
        if memory_text:
            self.ai.add_memory(memory_text)
            self.response_area.insert(tk.END, f"--- Added to memory: '{memory_text}' ---\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
