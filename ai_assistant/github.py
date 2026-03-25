
import os
import subprocess
from github import Github
import sys

class GitHubIntegration:
    def __init__(self, token: str = None):
        self.github_api = Github(token) if token else None

    def generate_ssh_key(self, email: str = "assistant@ai.local"):
        """Generates an SSH key pair internally and guides the user."""
        ssh_dir = os.path.expanduser("~/.ssh")
        if not os.path.exists(ssh_dir):
            os.makedirs(ssh_dir, mode=0o700)

        key_path = os.path.join(ssh_dir, "id_rsa_assistant")
        if not os.path.exists(key_path):
            print(f"--- Generating new SSH key pair at {key_path}... ---")
            try:
                subprocess.run(
                    ["ssh-keygen", "-t", "rsa", "-b", "4096", "-C", email, "-N", "", "-f", key_path],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                print(f"--- SSH key generated successfully. ---")
            except Exception as e:
                return f"Error generating SSH key: {e}"

        pub_key_path = key_path + ".pub"
        try:
            with open(pub_key_path, "r") as f:
                pub_key = f.read()
            return f"Success! Your public key is:\n\n{pub_key}\n\n--- Copy and add this key to your GitHub account: Settings -> SSH and GPG keys ---"
        except Exception as e:
            return f"Error reading public key: {e}"

    def analyze_repository(self, repo_name: str) -> str:
        """Reads and analyzes a public GitHub repository."""
        g = self.github_api if self.github_api else Github()
        try:
            repo = g.get_repo(repo_name)
            readme = repo.get_readme().decoded_content.decode()
            contents = repo.get_contents("")
            file_structure = "\n".join([f"- {c.name} ({c.type})" for c in contents])

            summary = (
                f"Repository: {repo.full_name}\n"
                f"Description: {repo.description}\n"
                f"Stars: {repo.stargazers_count}\n"
                f"--- File Structure ---\n{file_structure}\n"
                f"--- README Summary ---\n{readme[:500]}..."
            )
            return summary
        except Exception as e:
            return f"Error analyzing repository: {e}"
