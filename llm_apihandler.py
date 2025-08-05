import os
import requests
import json
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")
API_BASE_URL = "https://openrouter.ai/api/v1"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def get_llm_response(prompt: str, model_name: str = "meta-llama/llama-3-8b-instruct") -> str:
    """
    Sends a prompt to a specified model via the OpenRouter API and gets a single response.

    Args:
        prompt: The full prompt to send to the model.
        model_name: The identifier for the model on OpenRouter (e.g., "meta-llama/llama-3-8b-instruct").

    Returns:
        The text content of the model's response. Returns an error message on failure.
    """
    body = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        response = requests.post(f"{API_BASE_URL}/chat/completions", headers=HEADERS, data=json.dumps(body))
        response.raise_for_status()  # Raises an exception for bad status codes (4xx or 5xx)
        
        response_json = response.json()
        return response_json['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        print(f"An API error occurred: {e}")
        return f"Error: Could not get response from API. Details: {e}"

def generate_samples(prompt: str, model_name: str, num_samples: int = 5) -> list[str]:
    """
    Generates multiple, diverse samples for a given prompt.
    This is required for SelfCheckGPT's consistency checks.

    Args:
        prompt: The prompt to send.
        model_name: The model to use.
        num_samples: The number of diverse samples to generate.

    Returns:
        A list of generated text samples.
    """
    # Note: OpenRouter doesn't support generating multiple sequences in one call
    # like Hugging Face's `num_return_sequences`. We must call it multiple times.
    # We use temperature to encourage diversity in the responses.
    body = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7, # A value > 0 encourages diverse, creative sampling
    }
    
    samples = []
    for _ in range(num_samples):
        try:
            response = requests.post(f"{API_BASE_URL}/chat/completions", headers=HEADERS, data=json.dumps(body))
            response.raise_for_status()
            response_json = response.json()
            samples.append(response_json['choices'][0]['message']['content'])
        except requests.exceptions.RequestException as e:
            print(f"API error during sampling: {e}")
            # Add a placeholder or skip if a sample fails
            samples.append("Error during sample generation.")
            
    return samples
