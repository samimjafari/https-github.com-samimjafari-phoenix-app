
import sys
from getpass import getpass
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
import os

# Set dummy password for salt generation during installation if needed
# but let's just make sure the script can run.

try:
    from ai_assistant.utils import check_and_install_dependencies, derive_key, execute_command, is_dangerous
    from ai_assistant.memory import SemanticMemory
    from ai_assistant.models import AIModel
    from ai_assistant.search import SearchEngine
    from ai_assistant.github import GitHubIntegration
except ImportError:
    # If not installed as a package, add current dir to path
    sys.path.append(os.path.abspath(os.path.dirname(__file__)))
    from ai_assistant.utils import check_and_install_dependencies, derive_key, execute_command, is_dangerous
    from ai_assistant.memory import SemanticMemory
    from ai_assistant.models import AIModel
    from ai_assistant.search import SearchEngine
    from ai_assistant.github import GitHubIntegration

console = Console()

def main():
    # Only try auto-install if run manually
    if len(sys.argv) > 1 and sys.argv[1] == "--install":
        check_and_install_dependencies()
        return

    console.print(Panel("[bold green]Welcome to the Glowing-Guacamole AI Assistant (CLI)[/bold green]"))

    password = getpass("Enter your master password for database encryption: ")

    # Use a hidden file for the salt to make it persistent across restarts
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
        memory = SemanticMemory("assistant_memory.db", key)
        # Verify password/key by attempting to search (even empty)
        memory.search_memories("test", top_k=1)
    except Exception as e:
        console.print("[red]Error initializing memory (incorrect password?): [/red]", e)
        sys.exit(1)

    api_key = os.getenv("DEEPSEEK_API_KEY")
    # Initialize components
    model = AIModel(api_key=api_key, memory=memory)
    search_engine = SearchEngine(api_key=api_key)
    github = GitHubIntegration(token=os.getenv("GITHUB_TOKEN"))

    console.print("[blue]System Ready. Type 'exit' to quit or '/help' for options.[/blue]")

    while True:
        try:
            user_input = console.input("[bold yellow]>>> [/bold yellow]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[blue]Goodbye![/blue]")
            break

        if user_input.lower() in ["exit", "quit"]:
            break

        if not user_input:
            continue

        # Handle system commands
        if user_input.startswith("!"):
            cmd = user_input[1:]
            if is_dangerous(cmd):
                confirm = console.input(f"[red]Warning: '{cmd}' might be dangerous. Execute? (y/N): [/red]").lower()
                if confirm != 'y':
                    continue
            console.print(f"[dim]Executing: {cmd}...[/dim]")
            output = execute_command(cmd)
            console.print(Panel(output, title="Command Output"))
            continue

        # Handle slash commands
        if user_input.startswith("/search "):
            query = user_input[8:]
            console.print(f"[dim]Searching web...[/dim]")
            search_results = search_engine.search_web(query)
            response = model.get_ai_response(query, online=True, search_results=search_results)
        elif user_input.startswith("/github "):
            repo = user_input[8:]
            console.print(f"[dim]Analyzing GitHub repository...[/dim]")
            response = github.analyze_repository(repo)
        elif user_input == "/ssh-gen":
            response = github.generate_ssh_key()
        elif user_input == "/help":
            response = """
**Commands:**
- `!cmd` : Execute system command (e.g. `!ls`)
- `/search [query]` : Search web and get AI analysis
- `/github [repo]` : Analyze a public GitHub repo (e.g. `/github samimjafari/glowing-guacamole`)
- `/ssh-gen` : Generate SSH key for GitHub
- `/help` : Show this help message
- `exit` : Quit
            """
        else:
            # Regular AI interaction
            response = model.get_ai_response(user_input, online=(api_key is not None))

        # Show response
        console.print(Panel(Markdown(response), title="AI Assistant"))

        # Add interaction to semantic memory
        memory.add_memory(f"User: {user_input}\nAssistant: {response}")

if __name__ == "__main__":
    main()
