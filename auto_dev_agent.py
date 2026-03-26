import os
import sys
import subprocess
import time
import re

# Try to import from the project structure
try:
    from ai_assistant.models import AIModel
    from ai_assistant.memory import SemanticMemory
    from ai_assistant.utils import derive_key
except ImportError:
    # Fallback for direct execution if paths are not set
    sys.path.append(os.getcwd())
    from ai_assistant.models import AIModel
    from ai_assistant.memory import SemanticMemory
    from ai_assistant.utils import derive_key

# Constants for our autonomous agent
LOG_FILE = "error_log.txt"
REPO_ROOT = os.getcwd()

class AutoDevAgent:
    """An autonomous AI agent that detects errors, generates patches, and applies them."""

    def __init__(self, api_key: str = None, model_path: str = "./your-model.gguf", password: str = "default_secure_password"):
        # Derive a 32-byte key for internal memory encryption
        # In a real environment, the password should be provided via environment variable
        env_password = os.getenv("AUTODEV_PASSWORD", password)
        encryption_key = derive_key(env_password)

        self.memory = SemanticMemory(db_path="autodev_memories.db", encryption_key=encryption_key)
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

    def validate_patch(self, filepath: str, patch: str) -> bool:
        """Performs basic validation of a patch to prevent corruption."""
        if not patch or len(patch) < 10:
            return False

        # If the file is a Python script, try to compile the patch
        if filepath.endswith(".py"):
            try:
                compile(patch, filepath, 'exec')
                return True
            except SyntaxError as e:
                print(f"Syntactic validation failed for patch to {filepath}: {e}")
                return False

        # Generic validation for other file types
        return True

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
            "Your task is to provide the full corrected code for the file. "
            "Respond with ONLY the code. Do not include any explanations, markdown markers, or other text."
        )

        # Get AI response
        patch = self.model.get_ai_response(prompt, online=False)

        # Clean the patch if it contains common AI artifacts (like markdown code blocks)
        if patch.startswith("```python"):
            patch = re.sub(r"```python\n(.*?)\n```", r"\1", patch, flags=re.DOTALL)
        elif patch.startswith("```"):
            patch = re.sub(r"```\n(.*?)\n```", r"\1", patch, flags=re.DOTALL)

        # Apply patch if it passes validation
        if self.validate_patch(file_path, patch):
            print(f"Applying patch to {file_path}...")
            # Create a backup before overwriting
            self.write_file_content(file_path + ".bak", file_content)
            self.write_file_content(file_path, patch)

            # Commit changes to git
            try:
                self.run_command(["git", "add", file_path])
                self.run_command(["git", "commit", "-m", f"Auto-patch: fixed error in {file_path} via AutoDevAgent"])
                print("Patch applied and committed successfully.")
            except Exception as e:
                print(f"Failed to commit patch: {e}")
        else:
            print(f"AI generated patch for {file_path} failed validation.")

    def monitor_and_evolve(self):
        """Main loop: monitors for errors and attempts to self-correct."""
        print("--- 🔄 Starting Autonomous Development Loop... ---")
        while True:
            # Check for errors in a log file.
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
