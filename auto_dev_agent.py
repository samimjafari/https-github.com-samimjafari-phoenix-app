import os
import sys
import subprocess
import time
# Try to import from the project structure
try:
    from ai_assistant.models import AIModel
    from ai_assistant.memory import SemanticMemory
except ImportError:
    # Fallback for direct execution if paths are not set
    sys.path.append(os.getcwd())
    from ai_assistant.models import AIModel
    from ai_assistant.memory import SemanticMemory

# Constants for our autonomous agent
LOG_FILE = "error_log.txt"
REPO_ROOT = os.getcwd()

class AutoDevAgent:
    """An autonomous AI agent that detects errors, generates patches, and applies them."""

    def __init__(self, api_key: str = None, model_path: str = "./your-model.gguf"):
        # Use a consistent key for internal memory
        self.memory = SemanticMemory(db_path="autodev_memories.db", encryption_key=b"12345678901234567890123456789012")
        self.model = AIModel(api_key=api_key, model_path=model_path, memory=self.memory)

    def run_command(self, command: list) -> str:
        """Executes a system command and returns stdout and stderr combined."""
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            return result.stdout
        except subprocess.CalledProcessError as e:
            return f"Error executing command {command}:\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}"

    def get_file_content(self, filepath: str) -> str:
        """Reads and returns the content of a file."""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return f.read()
        return f"File {filepath} not found."

    def write_file_content(self, filepath: str, content: str):
        """Writes the provided content to a file."""
        with open(filepath, 'w') as f:
            f.write(content)

    def analyze_error_and_patch(self, error_message: str, file_path: str):
        """Uses the AI model to analyze a given error and generate a code patch."""
        file_content = self.get_file_content(file_path)

        prompt = (
            f"You are a professional software engineer. Fix the following error in the file '{file_path}'.\n\n"
            f"--- ERROR MESSAGE ---\n{error_message}\n\n"
            f"--- ORIGINAL FILE CONTENT ---\n{file_content}\n\n"
            "Respond with ONLY the fixed code. Do not include any explanations, markdown markers, or other text."
        )

        # Get AI response
        patch = self.model.get_ai_response(prompt, online=False)

        # Apply patch if it looks reasonable (not empty or error)
        if patch and "Error" not in patch and len(patch) > 10:
            print(f"Applying patch to {file_path}...")
            self.write_file_content(file_path, patch)

            # Commit changes to git
            self.run_command(["git", "add", file_path])
            self.run_command(["git", "commit", "-m", f"Auto-patch: fixed error in {file_path}"])
            print("Patch applied and committed successfully.")
        else:
            print("AI could not generate a valid patch.")

    def monitor_and_evolve(self):
        """Main loop: monitors for errors and attempts to self-correct."""
        print("--- 🔄 Starting Autonomous Development Loop... ---")
        while True:
            # For demonstration, we check for errors in a log file.
            # In production, this would be hooked into CI/CD or runtime error handlers.
            if os.path.exists(LOG_FILE):
                with open(LOG_FILE, 'r') as f:
                    error_data = f.read().splitlines()

                if error_data:
                    # Example format: "path/to/file.py: Error message text"
                    for line in error_data:
                        if ":" in line:
                            parts = line.split(":", 1)
                            file_path = parts[0].strip()
                            error_msg = parts[1].strip()

                            if os.path.exists(file_path):
                                print(f"Detected error in {file_path}: {error_msg}")
                                self.analyze_error_and_patch(error_msg, file_path)

                    # Clear log file after processing
                    open(LOG_FILE, 'w').close()

            # Sleep before next check
            time.sleep(30)

if __name__ == "__main__":
    # Initialize and run agent
    agent = AutoDevAgent()
    agent.monitor_and_evolve()
