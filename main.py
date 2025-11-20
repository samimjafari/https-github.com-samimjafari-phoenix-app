
import requests
from llama_cpp import Llama

# ------------------- OFFLINE MODE -------------------
def get_offline_response(prompt: str) -> str:
    """
    Gets a response from a local GGUF model using llama-cpp-python.
    """
    print("--- Using OFFLINE mode ---")
    try:
        # Load the GGUF model (replace with your model's path)
        # Make sure you have llama-cpp-python installed: pip install llama-cpp-python
        llm = Llama(model_path="./your-model.gguf", n_ctx=2048)

        # Get a response from the model
        output = llm(prompt, max_tokens=150, echo=False)

        return output['choices'][0]['text'].strip()

    except Exception as e:
        return f"Error in offline mode: {e}"


# ------------------- ONLINE MODE -------------------
def get_online_response(prompt: str, api_key: str) -> str:
    """
    Gets a response from an online AI service using an API key.
    """
    print("--- Using ONLINE mode ---")
    # URL of the online API.
    # This example uses the OpenAI API. Replace with the API endpoint for your provider.
    api_url = "https://api.openai.com/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "gpt-4",  # Or any other model the API supports
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        response = requests.post(api_url, headers=headers, json=data)
        response.raise_for_status()  # Raise an exception for bad status codes

        # Parse the JSON response to get the content
        # This part might need to be adjusted based on the API's response structure
        return response.json()['choices'][0]['message']['content'].strip()

    except requests.exceptions.RequestException as e:
        return f"Error connecting to online service: {e}"
    except (KeyError, IndexError) as e:
        return f"Error parsing API response: {e}"


# ------------------- CONNECTION CHECK -------------------
def is_connected_to_internet() -> bool:
    """
    Checks for an active internet connection.
    """
    try:
        # Try to connect to a reliable server
        requests.get("http://www.google.com", timeout=5)
        return True
    except requests.ConnectionError:
        return False


# ------------------- MAIN LOGIC -------------------
def get_ai_response(prompt: str, api_key: str) -> str:
    """
    Decides whether to use online or offline mode based on internet connectivity.
    """
    if is_connected_to_internet():
        return get_online_response(prompt, api_key)
    else:
        return get_offline_response(prompt)


# ------------------- EXAMPLE USAGE -------------------
if __name__ == "__main__":
    # Your secret API key for online mode
    my_api_key = "YOUR_SECRET_API_KEY"  # <-- IMPORTANT: Replace with your actual API key

    # The prompt you want to send to the AI
    user_prompt = "Hello, can you tell me about the history of artificial intelligence?"

    # Get the response (the function will automatically choose the mode)
    response = get_ai_response(user_prompt, my_api_key)

    # Print the result
    print("\nAI Response:")
    print(response)
