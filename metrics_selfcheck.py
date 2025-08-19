import torch
import spacy
from typing import Any, List
from data_requirements import QARecord, deduplicate_samples 
import numpy as np

from selfcheckgpt.modeling_selfcheck import SelfCheckNLI
import llm_apihandler 


class HallucinationScorer:
    def __init__(self):
        """
        Initializes the scorer and the underlying SelfCheckGPT NLI model.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.selfcheck_nli = SelfCheckNLI(device=self.device)
        self.samples = []
        
        # Initialize spacy for sentence splitting (tokenization)
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            from spacy.cli import download
            download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

    def _generate_and_filter_samples(self, prompt: str, model_name: str, seed: int, min_samples: int = 20) -> List[str]:
        """
        Generate and filter consistency samples for hallucination detection.
        
        Args:
            prompt: The full prompt (including context) that generated the answer.
            model_name: The name of the model to use for generating consistency samples.
            seed: An integer seed for reproducibility of samples.
            min_samples: Minimum number of samples required (default: 10).
            
        Returns:
            A list of filtered and deduplicated samples, or empty list if insufficient samples.
        """
        samples = []
        
        try:
            # Generate 20 samples from the same model for consistency
            samples = llm_apihandler.generate_samples(prompt, model_name, num_samples=20, seed=seed)
            # Simple filter for very short samples and deduplicate
            samples = [s for s in samples if len(s.split()) > 10]
            samples = deduplicate_samples(samples)
        except Exception:
            return []
                
        # Check if we have enough samples
        if len(samples) < min_samples:
            return []
            
        return samples

    def get_sentence_level_hallucination_scores(self, original_answer: str, prompt: str, model_name: str, seed: int) -> dict:
        """
        This function scores the answer for hallucination at the sentence level.
        Args:
            original_answer: The single, generated answer to be checked.
            prompt: The full prompt (including context) that generated the answer.
            model_name: The name of the model to use for generating consistency samples.
            seed: An integer seed for reproducibility of samples.

        Returns:
            A dictionary containing sentence-level hallucination scores.
            e.g., {
                'sentence_level_hallucination_scores': {'sentence 1': 0.1, 'sentence 2': 0.9},
            }
        """
        # 1. Generate self-consistency samples
        self.samples = self._generate_and_filter_samples(prompt, model_name, seed)
        
        if not self.samples:
            return {}

        # 2. Simple sentence splitting
        doc = self.nlp(original_answer)
        sentences_to_check = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 5]
        
        if not sentences_to_check:
            return {}
        
        # 3. Run the NLI check for sentences level
        # SelfcheckGPT NLI only has entailment and contradiction probs, no neutral probs
        # samples are considered as answers (hypothesis), so any sentence (premise) not consistent
        #  with the answer is a contradiction/ hallucination
        contradiction_probabilities = self.selfcheck_nli.predict(
            sentences=sentences_to_check,
            sampled_passages=samples
        )
        
        # Map each sentence to its score, handling None from predict()
        sentence_level_scores = {}
        for sentence, contradiction_probability in zip(sentences_to_check, contradiction_probabilities):
            if contradiction_probability is not None:
                sentence_level_scores[sentence] = max(0.0, min(1.0, float(contradiction_probability)))
        
        if not sentence_level_scores:
            return {}

        return {
            'sentence_level_hallucination_scores': sentence_level_scores
        }

    def get_sample_level_hallucination_scores(self, prompt: str, model_name: str, seed: int) -> List[Any]:
        """
        Scores each sample against the others for hallucination using self-consistency checks.
        Returns:
            A list of hallucination scores, one for each sample.
            e.g., [0.1, 0.9, 0.5, ...]
        """
        # generate samples if not already generated
        if not self.samples:
            self.samples = self._generate_and_filter_samples(prompt, model_name, seed)
        

        sample_level_hallucination_scores = []
        for i, sample in enumerate(self.samples):
            other_samples = self.samples[:i] + self.samples[i+1:]
            evaluating_sample = sample

            # Get NLI probability for contradiction for the evaluating sample
            eval_sample_contradiction_probability = self.selfcheck_nli.predict(
                sentences=evaluating_sample,
                sampled_passages=other_samples
            )
            
            # Filter out None values from the scores
            # valid_contradiction_probabilities = [s for s in eval_sample_contradiction_probability if s is not None]
            
            # Aggregate for the sample by taking the max sentence score
            # sample_score = max(valid_contradiction_probabilities) if valid_contradiction_probabilities else 0.0
            sample_level_hallucination_scores.append(eval_sample_contradiction_probability)
            
        return {
            'sample_level_hallucination_scores': sample_level_hallucination_scores
        }

    def confidence_to_binary(self, confidence_dict, threshold=0.5):
        """Convert confidence scores to binary labels using standard threshold."""
        return {k: 1 if v >= threshold else 0 for k, v in confidence_dict.items()}

    def aggregate_confidence_scores(self, confidence_dict):
        """Aggregate sentence-level confidence scores into scalar values."""
        if not confidence_dict:
            return {'conf_agg_max': 0.0, 'conf_agg_mean': 0.0}
        
        scores = list(confidence_dict.values())
        return {
            'conf_agg_max': max(scores),
            'conf_agg_mean': sum(scores) / len(scores)
        }

    def aggregate_binary_scores(self, binary_dict):
        """Aggregate binary scores using majority vote."""
        if not binary_dict:
            return {'bin_majority': 0}
        
        scores = list(binary_dict.values())
        majority_vote = 1 if sum(scores) > len(scores) / 2 else 0  # Ties → 0 (explicit)
        return {'bin_majority': majority_vote}

    def calculate_95th_percentile(self, scores_list):
        """Calculate 95th percentile for a list of scores, filtering out None values."""
        
        if not scores_list:
            return 0.0
        
        # Filter out None values and convert to numeric
        valid_scores = [score for score in scores_list if score is not None]
        if valid_scores:
            return np.percentile(valid_scores, 95)
        else:
            return 0.0