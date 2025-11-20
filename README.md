# AI Library with Online and Offline Modes

This project provides a Python script that can switch between online and offline modes for an AI-powered library. It uses `llama-cpp-python` for offline inference with GGUF models and `requests` to connect to an online AI service.

## Setup

### 1. Install Dependencies
Make sure you have Python installed. Then, install the required libraries using pip:
```bash
pip install requests llama-cpp-python
```

### 2. Download a GGUF Model
This script requires a GGUF model for offline mode. Due to size constraints, you need to download it manually.

1.  **Download the model:** You can download a model from Hugging Face. For example, to download the TinyLlama model, you can use this link:
    [https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf)

2.  **Place the model in the project directory:** Rename the downloaded file to `your-model.gguf` and place it in the same directory as the `main.py` script.

### 3. Configure the Online Mode
For online mode, you need to provide an API key and, if necessary, update the API endpoint.

*   **API Key:** Open `main.py` and replace `"YOUR_SECRET_API_KEY"` with your actual API key.
    ```python
    my_api_key = "YOUR_SECRET_API_KEY"  # <-- IMPORTANT: Replace with your actual API key
    ```
*   **API Endpoint:** The script is pre-configured to use the OpenAI API. If you are using a different service, you will need to update the `api_url` in the `get_online_response` function in `main.py`.

## How to Run
Once you've completed the setup, you can run the script with the following command:
```bash
python main.py
```

The script will automatically detect if you have an internet connection and choose the appropriate mode.
