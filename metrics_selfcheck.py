import torch
import spacy
from dataclasses import dataclass, field
from typing import Dict, Any, List
import re
import string
import numpy as np

from selfcheckgpt.modeling_selfcheck import SelfCheckNLI
import llm_apihandler 

@dataclass
class QARecord:
    """A class to hold all data for a single QA pair, with sentence-level scores."""
    qa_id: str
    context: str
    prompt: str
    answer: str = ""
    # Confidence is now a dictionary to hold scores for each sentence
    confidence: Dict[str, float] = None
    # Store the full NLI probability distributions for each sentence
    nli_probabilities: Dict[str, Any] = None
    # Binary confidence scores (0/1) with default threshold 0.5
    binary_confidence: Dict[str, int] = None
    # -- Fields for sentence-level aggregation --
    conf_agg_max: float = 0.0
    conf_agg_mean: float = 0.0
    bin_majority: int = 0
    # -- Fields for sample-level aggregation --
    per_sample_scores: List[float] = field(default_factory=list)
    risk_score_mean: float = 0.0
    risk_score_p95: float = 0.0
    self_consistency_vote: int = 0

def normalize_text(text: str) -> str:
    """Lowercases, removes punctuation, and collapses whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def deduplicate_samples(samples: list[str]) -> list[str]:
    """Deduplicates samples based on normalized text."""
    seen = set()
    unique_samples = []
    for s in samples:
        normalized_s = normalize_text(s)
        if normalized_s not in seen:
            seen.add(normalized_s)
            unique_samples.append(s)
    return unique_samples

class HallucinationScorer:
    def __init__(self):
        """
        Initializes the scorer and the underlying SelfCheckGPT NLI model.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.selfcheck_nli = SelfCheckNLI(device=self.device)
        
        # Initialize spacy for sentence splitting
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            from spacy.cli import download
            download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

    def score_answer(self, original_answer: str, prompt: str, model_name: str, seed: int) -> dict:
        """
        Args:
            original_answer: The single, generated answer to be checked.
            prompt: The full prompt (including context) that generated the answer.
            model_name: The name of the model to use for generating consistency samples.
            seed: An integer seed for reproducibility of samples.

        Returns:
            A dictionary containing sentence-level hallucination scores and NLI probabilities.
            e.g., {
                'hallucination_scores': {'sentence 1': 0.1, 'sentence 2': 0.9},
                'nli_probabilities': {'sentence 1': [0.8, 0.1, 0.1], ...}
            }
        """
        # 1. Generate self-consistency samples
        samples = []
        
        try:
            # Generate 20 samples from the same model for consistency
            samples = llm_apihandler.generate_samples(prompt, model_name, num_samples=20, seed=seed)
            # Simple filter for very short samples and deduplicate
            samples = [s for s in samples if len(s.split()) > 10]
            samples = deduplicate_samples(samples)
        except Exception:
            return {}
                
        # Check if we have enough samples
        if len(samples) < 10:
            return {}

        # 2. Simple sentence splitting
        doc = self.nlp(original_answer)
        sentences_to_check = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 5]
        
        if not sentences_to_check:
            return {}
        
        # 3. Run the NLI check, getting the hallucination scores directly
        hallucination_scores = self.selfcheck_nli.predict(
            sentences=sentences_to_check,
            sampled_passages=samples
        )
        
        # Map each sentence to its score, handling None from predict()
        sentence_scores = {}
        for sentence, score in zip(sentences_to_check, hallucination_scores):
            if score is not None:
                sentence_scores[sentence] = max(0.0, min(1.0, float(score)))
        
        if not sentence_scores:
            return {}

        # Note: We can no longer return nli_probabilities as the library doesn't provide them
        return {
            'hallucination_scores': sentence_scores
        }

    def score_samples_self_consistency(self, samples: List[str]) -> List[float]:
        """
        Scores each sample against the others for self-consistency.
        Returns:
            A list of hallucination scores, one for each sample.
        """
        if not samples or len(samples) < 2:
            return []

        per_sample_scores = []
        for i, sample in enumerate(samples):
            other_samples = samples[:i] + samples[i+1:]
            
            doc = self.nlp(sample)
            sentences_to_check = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 5]
            
            if not sentences_to_check:
                per_sample_scores.append(0.0)
                continue

            # Get NLI probabilities for [entailment, neutral, contradiction]
            hallucination_scores = self.selfcheck_nli.predict(
                sentences=sentences_to_check,
                sampled_passages=other_samples
            )
            
            # Filter out None values from the scores
            valid_scores = [s for s in hallucination_scores if s is not None]
            
            # Aggregate for the sample by taking the max sentence score
            sample_score = max(valid_scores) if valid_scores else 0.0
            per_sample_scores.append(sample_score)
            
        return per_sample_scores

def confidence_to_binary(confidence_dict, threshold=0.5):
    """Convert confidence scores to binary labels using standard threshold."""
    return {k: 1 if v >= threshold else 0 for k, v in confidence_dict.items()}

def aggregate_confidence_scores(confidence_dict):
    """Aggregate sentence-level confidence scores into scalar values."""
    if not confidence_dict:
        return {'conf_agg_max': 0.0, 'conf_agg_mean': 0.0}
    
    scores = list(confidence_dict.values())
    return {
        'conf_agg_max': max(scores),
        'conf_agg_mean': sum(scores) / len(scores)
    }

def aggregate_binary_scores(binary_dict):
    """Aggregate binary scores using majority vote."""
    if not binary_dict:
        return {'bin_majority': 0}
    
    scores = list(binary_dict.values())
    majority_vote = 1 if sum(scores) > len(scores) / 2 else 0  # Ties → 0 (explicit)
    return {'bin_majority': majority_vote}
    return {'bin_majority': majority_vote}
    """Aggregate binary scores using majority vote."""
    if not binary_dict:
        return {'bin_majority': 0}
    
    scores = list(binary_dict.values())
    majority_vote = 1 if sum(scores) > len(scores) / 2 else 0  # Ties → 0 (explicit)
    return {'bin_majority': majority_vote}
    return {'bin_majority': majority_vote}
