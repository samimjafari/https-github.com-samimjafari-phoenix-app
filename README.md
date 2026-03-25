# Glowing-Guacamole AI Assistant 🥑🚀

A professional, offline-first AI assistant with **semantic memory**, **AES-256-GCM encryption**, **web search**, and **GitHub integration**. Designed for Desktop, Android (Termux/Native), and Kali Linux.

## ✨ Key Features

- **🔒 High Security:** All your memories, notes, and conversations are stored in a local SQLite database encrypted with **AES-256-GCM**. Access is controlled by your master password.
- **🧠 Semantic Memory:** Uses `sentence-transformers` for concept-based search. The assistant understands the context of your previous interactions.
- **🤖 Dual Mode:**
  - **Offline:** Run GGUF models (LLaMA, Mistral, Qwen) locally using `llama-cpp-python`.
  - **Online:** Connect to DeepSeek or OpenAI-compatible APIs for high-performance reasoning.
- **🌐 Web Search:** Integrated search (DeepSeek API or custom scraper) for real-time information.
- **🐙 GitHub Integration:** Automatically generates SSH keys and analyzes public repositories.
- **💻 Multi-Platform:**
  - **GUI:** Modern interface built with **Flet** (Windows, Linux, Android).
  - **CLI:** Professional terminal interface for **Kali Linux** and **Termux**.
- **⚡ System Commands:** Safely execute shell commands directly from the assistant.

## 🛠 Installation

### 1. Requirements
- Python 3.8+
- C++ Compiler (for `llama-cpp-python` if no binary wheel is available)

### 2. Setup
Clone the repository and install dependencies:
```bash
git clone https://github.com/samimjafari/glowing-guacamole.git
cd glowing-guacamole
pip install -e .
```

### 3. Termux / Kali Linux Installation
For **Termux** or **Kali Linux**, you can use the provided setup script:
```bash
chmod +x setup_termux.sh
./setup_termux.sh
```

### 4. Model Setup (Offline Mode)
1. Download a GGUF model (e.g., [TinyLlama](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF)).
2. Rename it to `your-model.gguf` and place it in the project root.

## 🚀 How to Run

### Graphical User Interface (GUI)
```bash
python gui_flet.py
```

### Command Line Interface (CLI)
```bash
python cli.py
```

## 📖 Commands
- `!ls` : Execute system command (e.g., `ls`, `pwd`).
- `/search [query]` : Search the web and get AI analysis.
- `/github [user/repo]` : Analyze a GitHub repository.
- `/ssh-gen` : Generate a secure SSH key pair.
- `/help` : Show all available options.

## 🔒 Security Note
When you run the app for the first time, you will be asked for a **Master Password**. This password is used to derive a unique encryption key. **If you lose this password, your stored memories will be unrecoverable.**

---
*Created with ❤️ for the open-source community.*
