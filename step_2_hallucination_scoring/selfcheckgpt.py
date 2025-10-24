# from selfcheckgpt.modeling_selfcheck import SelfCheckNLI
#import llm_apihandler 

# LLM API HANDLER FOR OPENROUTER.AI WITH RATE LIMITING
# import os
# import requests
# import json
# from dotenv import load_dotenv
# import time
# import concurrent.futures
# from threading import Lock

# # Load environment variables from a .env file
# load_dotenv()


# API_KEY = os.getenv("OPENROUTER_API_KEY")
# API_BASE_URL = "https://openrouter.ai/api/v1"
# HEADERS = {
#     "Authorization": f"Bearer {API_KEY}",
#     "Content-Type": "application/json"
# }

# # Add rate limiting
# request_lock = Lock()
# last_request_time = 0
# MIN_REQUEST_INTERVAL = 0.1  # 100ms between requests

# def generate_samples(prompt: str, model_name: str, num_samples: int = 20, seed: int = 42) -> list[str]:
#     """
#     Generates multiple, diverse samples for a given prompt with rate limiting.
#     This is required for SelfCheckGPT's consistency checks.

#     Args:
#         prompt: The prompt to send.
#         model_name: The model to use.
#         num_samples: The number of diverse samples to generate.
#         seed: An integer seed for reproducibility.

#     Returns:
#         A list of generated text samples.
#     """
    
#     def make_single_request(sample_seed: int):
#         global last_request_time
#         body = {
#             "model": model_name,
#             "messages": [{"role": "user", "content": prompt}],
#             "temperature": 0.7,
#             "max_tokens": 150,
#             "top_p": 0.9,
#             "seed": sample_seed, # Use the unique seed for this request
#         }

#         with request_lock:
#             # Rate limiting
#             elapsed = time.time() - last_request_time
#             if elapsed < MIN_REQUEST_INTERVAL:
#                 time.sleep(MIN_REQUEST_INTERVAL - elapsed)
#             last_request_time = time.time()

#         # try:
#             response = requests.post(f"{API_BASE_URL}/chat/completions", headers=HEADERS, data=json.dumps(body))
#             if response.status_code == 429:  # Rate limited
#                 time.sleep(1)
#                 return None
#             if response.status_code != 200:
#                 return None
#             response.raise_for_status()
#             response_json = response.json()
#             content = response_json['choices'][0]['message']['content'].strip()
#             return content
    
#     # Use ThreadPoolExecutor for parallel requests (but with rate limiting)
#     samples = []
#     with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
#         # Create a list of unique seeds for each sample
#         seeds = [seed + i for i in range(num_samples)]
#         futures = [executor.submit(make_single_request, s) for s in seeds]
#         for future in concurrent.futures.as_completed(futures):
#             result = future.result()
#             if result and len(result.split()) > 10:
#                 samples.append(result)
#     return samples

        # # 1. Generate self-consistency samples
        # self.samples = self._generate_and_filter_samples(prompt, model_name, seed)
        # if not self.samples:
        #     return {}

        # """
        # Initializes the scorer and the underlying SelfCheckGPT NLI model.
        # """
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # # self.selfcheck_nli = SelfCheckNLI(device=self.device)
        # self.samples = []

        # # 5. Run the NLI check for sentences/chunks
        # contradiction_probabilities = self.selfcheck_nli.predict(
        #     sentences=sentences_to_check,
        #     sampled_passages=self.samples
        # )

        # Map each sentence/chunk to its score, handling None from predict()
        # sentence_level_scores = {}
        # for sentence, contradiction_probability in zip(sentences_to_check, contradiction_probabilities):
        #     if contradiction_probability is not None:
        #         sentence_level_scores[sentence] = max(0.0, min(1.0, float(contradiction_probability)))
        # 4. Compute cosine similarity for each sentence/chunk against gold_text

    # def _generate_and_filter_samples(self, prompt: str, model_name: str, seed: int, min_samples: int = 10) -> List[str]:
    #     """
    #     Generate and filter consistency samples for hallucination detection.
        
    #     Args:
    #         prompt: The full prompt (including context) that generated the answer.
    #         model_name: The name of the model to use for generating consistency samples.
    #         seed: An integer seed for reproducibility of samples.
    #         min_samples: Minimum number of samples required (default: 10).
            
    #     Returns:
    #         A list of filtered and deduplicated samples, or empty list if insufficient samples.
    #     """
    #     samples = []
        
    #     try:
    #         # Generate 10 samples from the same model for consistency
    #         samples = llm_apihandler.generate_samples(prompt, model_name, num_samples=10, seed=seed)
    #         # Simple filter for very short samples and deduplicate
    #         samples = [s for s in samples if len(s.split()) > 10]
    #         samples = deduplicate_samples(samples)
    #     except Exception:
    #         return []
                
    #     return samples

    # def get_sample_level_hallucination_scores(self, prompt: str, model_name: str, seed: int) -> List[Any]:
    #     """
    #     Scores each sample against the others for hallucination using self-consistency checks.
    #     Returns:
    #         A list of hallucination scores, one for each sample.
    #         e.g., [0.1, 0.9, 0.5, ...]
    #     """
    #     # generate samples if not already generated
    #     if not self.samples:
    #         self.samples = self._generate_and_filter_samples(prompt, model_name, seed)
        

    #     sample_level_hallucination_scores = []
    #     for i, sample in enumerate(self.samples):
    #         other_samples = self.samples[:i] + self.samples[i+1:]
    #         evaluating_sample = sample

    #         # Get NLI probability for contradiction for the evaluating sample
    #         eval_sample_contradiction_probability = self.selfcheck_nli.predict(
    #             sentences=evaluating_sample,
    #             sampled_passages=other_samples
    #         )
            
    #         # Filter out None values from the scores
    #         # valid_contradiction_probabilities = [s for s in eval_sample_contradiction_probability if s is not None]
            
    #         # Aggregate for the sample by taking the max sentence score
    #         # sample_score = max(valid_contradiction_probabilities) if valid_contradiction_probabilities else 0.0
    #         sample_level_hallucination_scores.append(eval_sample_contradiction_probability)
            
    #     return {
    #         'sample_level_hallucination_scores': sample_level_hallucination_scores
    #     }

    # def calculate_95th_percentile(self, scores_list):
    #     """Calculate 95th percentile for a list of scores, filtering out None values."""
        
    #     if not scores_list:
    #         return 0.0
        
    #     # Filter out None values and convert to numeric
    #     valid_scores = [score for score in scores_list if score is not None]
    #     if valid_scores:
    #         return np.percentile(valid_scores, 95)
    #     else:
    #         return 0.0

# FOR CALCULATING HALLUCINATION SAMPLE WISE 

        # model_response_uncertainty = scorer.aggregate_confidence_scores(sample_level_hallu_scores)['conf_agg_mean']

        # sample_level_95th_percentile = scorer.calculate_95th_percentile(
        #     sample_level_hallu_scores.get('sample_level_sem_similarity_scores', [])
        # )

        #record.hallucination_label = whole_answer_hallu_label
        # store hallucination_score as a probability-like value where higher means more likely hallucination

        # record.model_response_uncertainty = model_response_uncertainty
        # record.confidence = 1 - model_response_uncertainty

        # record.sample_level_hallu_scores = sample_level_hallu_scores['sample_level_sem_similarity_scores']
        # record.sample_level_95th_percentile = sample_level_95th_percentile