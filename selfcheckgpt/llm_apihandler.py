import os
import requests
import json
from dotenv import load_dotenv
import time
import concurrent.futures
from threading import Lock

# Load environment variables from a .env file
load_dotenv()


API_KEY = os.getenv("OPENROUTER_API_KEY")
if not API_KEY:
    print("[ERROR] OPENROUTER_API_KEY not found in environment. API requests will fail.")
else:
    print("[DEBUG] OPENROUTER_API_KEY loaded.")
API_BASE_URL = "https://openrouter.ai/api/v1"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Add rate limiting
request_lock = Lock()
last_request_time = 0
MIN_REQUEST_INTERVAL = 0.1  # 100ms between requests

def generate_samples(prompt: str, model_name: str, num_samples: int = 20, seed: int = 42) -> list[str]:
    print(f"[DEBUG] Generating {num_samples} samples for model '{model_name}' with seed {seed}.")
    """
    Generates multiple, diverse samples for a given prompt with rate limiting.
    This is required for SelfCheckGPT's consistency checks.

    Args:
        prompt: The prompt to send.
        model_name: The model to use.
        num_samples: The number of diverse samples to generate.
        seed: An integer seed for reproducibility.

    Returns:
        A list of generated text samples.
    """
    
    def make_single_request(sample_seed: int):
        global last_request_time
        body = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 150,
            "top_p": 0.9,
            "seed": sample_seed, # Use the unique seed for this request
        }

        with request_lock:
            # Rate limiting
            elapsed = time.time() - last_request_time
            if elapsed < MIN_REQUEST_INTERVAL:
                time.sleep(MIN_REQUEST_INTERVAL - elapsed)
            last_request_time = time.time()

        try:
            response = requests.post(f"{API_BASE_URL}/chat/completions", headers=HEADERS, data=json.dumps(body))
            if response.status_code == 429:  # Rate limited
                print(f"[WARNING] Rate limited by API. Sleeping 1s. Status: {response.status_code}")
                time.sleep(1)
                return None
            if response.status_code != 200:
                print(f"[ERROR] API returned status {response.status_code}: {response.text}")
                return None
            response.raise_for_status()
            response_json = response.json()
            content = response_json['choices'][0]['message']['content'].strip()
            print(f"[DEBUG] Sample generated (seed={sample_seed}): {content[:60]}...")
            return content
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] RequestException: {e}")
            return None
    
    # Use ThreadPoolExecutor for parallel requests (but with rate limiting)
    samples = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Create a list of unique seeds for each sample
        seeds = [seed + i for i in range(num_samples)]
        futures = [executor.submit(make_single_request, s) for s in seeds]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result and len(result.split()) > 10:
                samples.append(result)
            elif result:
                print(f"[DEBUG] Sample too short (seeded): {result}")
            else:
                print(f"[DEBUG] No result for one sample.")
    print(f"[DEBUG] Total valid samples generated: {len(samples)}")
    return samples
