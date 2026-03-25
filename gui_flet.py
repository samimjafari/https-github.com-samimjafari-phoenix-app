
import flet as ft
import os
import sys

# Add current dir to path for imports
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from ai_assistant.utils import derive_key, execute_command, is_dangerous
from ai_assistant.memory import SemanticMemory
from ai_assistant.models import AIModel
from ai_assistant.search import SearchEngine
from ai_assistant.github import GitHubIntegration

def main(page: ft.Page):
    page.title = "Glowing-Guacamole AI"
    page.theme_mode = ft.ThemeMode.DARK
    page.padding = 10
    page.window_width = 800
    page.window_height = 900

    # Application state
    state = {
        "memory": None,
        "model": None,
        "search_engine": None,
        "github": None,
        "api_key": os.getenv("DEEPSEEK_API_KEY")
    }

    # --- UI Layout Elements ---
    chat_list = ft.ListView(expand=True, spacing=10, padding=10, auto_scroll=True)
    user_input = ft.TextField(
        hint_text="Ask a question or type a command...",
        expand=True,
        border_radius=20,
        on_submit=lambda e: send_message(e)
    )

    def append_message(msg: str, is_user: bool = False, is_code: bool = False):
        if is_user:
            chat_list.controls.append(
                ft.Container(
                    content=ft.Text(msg, color="white", weight="bold"),
                    bgcolor="#2c3e50",
                    padding=10,
                    border_radius=10,
                    alignment=ft.alignment.center_right
                )
            )
        else:
            if is_code:
                chat_list.controls.append(
                    ft.Container(
                        content=ft.Text(msg, font_family="monospace", size=12),
                        bgcolor="#1a1a1a",
                        padding=10,
                        border_radius=5,
                        border=ft.border.all(1, "#333333")
                    )
                )
            else:
                chat_list.controls.append(
                    ft.Container(
                        content=ft.Markdown(msg, selectable=True, extension_set=ft.MarkdownExtensionSet.GITHUB_WEB),
                        bgcolor="#34495e",
                        padding=10,
                        border_radius=10,
                        alignment=ft.alignment.center_left
                    )
                )
        page.update()

    def send_message(e):
        prompt = user_input.value.strip()
        if not prompt:
            return

        user_input.value = ""
        append_message(prompt, is_user=True)
        page.update()

        # Command handling
        if prompt.startswith("!"):
            cmd = prompt[1:]
            if is_dangerous(cmd):
                append_message(f"Command '{cmd}' is dangerous and blocked for safety.", is_user=False)
            else:
                append_message(f"Executing: {cmd}...", is_user=False)
                output = execute_command(cmd)
                append_message(output, is_user=False, is_code=True)
        elif prompt.startswith("/search "):
            query = prompt[8:]
            append_message(f"Searching web for: {query}...", is_user=False)
            search_results = state["search_engine"].search_web(query)
            response = state["model"].get_ai_response(query, online=True, search_results=search_results)
            append_message(response)
            state["memory"].add_memory(f"User Search: {query}\nAI: {response}")
        elif prompt.startswith("/github "):
            repo = prompt[8:]
            append_message(f"Analyzing GitHub repo: {repo}...", is_user=False)
            response = state["github"].analyze_repository(repo)
            append_message(response)
        elif prompt == "/ssh-gen":
            response = state["github"].generate_ssh_key()
            append_message(response)
        elif prompt == "/help":
            help_text = """
**Available Commands:**
- `!cmd` : Execute a system command (e.g. `!ls`)
- `/search [query]` : Online search with AI analysis
- `/github [user/repo]` : Analyze a GitHub repository
- `/ssh-gen` : Generate a secure SSH key for GitHub
- `/help` : Show this message
- Regular text to chat with AI (offline/online)
            """
            append_message(help_text)
        else:
            # Standard AI interaction
            response = state["model"].get_ai_response(prompt, online=(state["api_key"] is not None))
            append_message(response)
            state["memory"].add_memory(f"User: {prompt}\nAI: {response}")

    # --- Login/Auth View ---
    def unlock_app(e):
        password = password_field.value
        if not password:
            password_field.error_text = "Master password is required."
            page.update()
            return

        # Handle salt
        salt_file = ".memory.salt"
        if os.path.exists(salt_file):
            with open(salt_file, "rb") as f:
                salt = f.read()
        else:
            salt = os.urandom(16)
            with open(salt_file, "wb") as f:
                f.write(salt)

        try:
            key = derive_key(password, salt)
            # Initialize core modules
            state["memory"] = SemanticMemory("assistant_memory.db", key)
            state["memory"].search_memories("test", top_k=1) # Verify key

            state["model"] = AIModel(api_key=state["api_key"], memory=state["memory"])
            state["search_engine"] = SearchEngine(api_key=state["api_key"])
            state["github"] = GitHubIntegration(token=os.getenv("GITHUB_TOKEN"))

            # Transition to Chat View
            page.clean()
            page.add(
                ft.Column([
                    ft.Text("Glowing-Guacamole AI Assistant", size=24, weight="bold", color="green"),
                    ft.Divider(),
                    chat_list,
                    ft.Row([user_input, ft.IconButton(icon=ft.icons.Icons.SEND, on_click=send_message)],
                           alignment=ft.MainAxisAlignment.CENTER)
                ], expand=True)
            )
            append_message("Welcome! System is online and encrypted. How can I help you today?")
            page.update()
        except Exception as ex:
            password_field.error_text = f"Initialization failed: {ex}"
            page.update()

    password_field = ft.TextField(
        label="Enter Master Password",
        password=True,
        can_reveal_password=True,
        width=300,
        on_submit=unlock_app
    )

    login_screen = ft.Container(
        content=ft.Column([
            ft.Icon(ft.icons.Icons.LOCK, size=64, color="blue"),
            ft.Text("Encrypted Access", size=28, weight="bold"),
            ft.Text("Your memories are protected by AES-256-GCM encryption.", size=14, color="grey"),
            password_field,
            ft.ElevatedButton("Unlock Assistant", on_click=unlock_app, icon=ft.icons.Icons.KEY),
        ], alignment=ft.MainAxisAlignment.CENTER, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
        expand=True
    )

    page.add(login_screen)

if __name__ == "__main__":
    ft.app(target=main)
