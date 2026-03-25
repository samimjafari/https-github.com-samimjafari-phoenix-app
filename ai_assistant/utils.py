
import os
import subprocess
import sys
import base64
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

def check_and_install_dependencies():
    """
    Checks if required dependencies are installed and attempts to install them if missing.
    """
    required_pkgs = ["requests", "llama_cpp", "sentence_transformers", "numpy", "rich", "flet", "github", "bs4"]
    missing = False
    for pkg in required_pkgs:
        try:
            __import__(pkg)
        except ImportError:
            missing = True
            break

    if missing:
        print(f"--- Attempting to install missing dependencies... ---")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            print("--- Dependencies installed successfully. Please restart the application. ---")
            sys.exit(0)
        except subprocess.CalledProcessError as err:
            print(f"--- Automatic installation failed: {err} ---")
            print("--- Please try installing dependencies manually: pip install -r requirements.txt ---")
            print("--- If llama-cpp-python or sentence-transformers fails, ensure you have a C++ compiler and enough disk space. ---")
            sys.exit(1)

def derive_key(password: str, salt: bytes) -> bytes:
    """
    Derives a 32-byte (256-bit) key from a password and salt using PBKDF2.
    """
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
        backend=default_backend()
    )
    return kdf.derive(password.encode())

def encrypt_data(data: str, key: bytes) -> str:
    """
    Encrypts a string using AES-256-GCM.
    Returns a base64-encoded string containing (nonce + ciphertext + tag).
    """
    aesgcm = AESGCM(key)
    nonce = os.urandom(12)
    ciphertext = aesgcm.encrypt(nonce, data.encode(), None)
    return base64.b64encode(nonce + ciphertext).decode()

def decrypt_data(encrypted_data_b64: str, key: bytes) -> str:
    """
    Decrypts a base64-encoded string using AES-256-GCM.
    """
    aesgcm = AESGCM(key)
    try:
        data = base64.b64decode(encrypted_data_b64)
        nonce = data[:12]
        ciphertext = data[12:]
        return aesgcm.decrypt(nonce, ciphertext, None).decode()
    except Exception as e:
        return f"Decryption failed: {e}"

def execute_command(command: str) -> str:
    """
    Executes a system command safely.
    """
    try:
        # Using shell=True carefully. This is intended for Kali/Termux users.
        result = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT, text=True)
        return result
    except subprocess.CalledProcessError as e:
        return f"Error executing command: {e.output}"

def is_dangerous(command: str) -> bool:
    """
    Checks if a command is potentially dangerous.
    """
    dangerous_keywords = ["rm ", "sudo ", "mkfs", "dd ", "> /dev/", "chmod ", "chown ", "apt install", "pip install", "rmdir"]
    return any(kw in command for kw in dangerous_keywords)
