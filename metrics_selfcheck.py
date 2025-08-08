import torch
import spacy
from dataclasses import dataclass
from typing import Dict

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

class HallucinationScorer:
    """
    A class to handle hallucination scoring using ONLY the NLI variant of SelfCheckGPT.
    """
    def __init__(self):
        """
        Initializes the scorer and the underlying SelfCheckGPT NLI model.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.selfcheck_nli = SelfCheckNLI(device=self.device)
        
        # Initialize spacy for sentence splitting
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading spacy model 'en_core_web_sm'...")
            from spacy.cli import download
            download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

        print("HallucinationScorer (NLI only) initialized.")

    def score_answer(self, original_answer: str, prompt: str, model_name: str) -> dict:
        """
        Args:
            original_answer: The single, generated answer to be checked.
            prompt: The full prompt (including context) that generated the answer.
            model_name: The name of the model to use for generating consistency samples.

        Returns:
            A dictionary mapping each sentence to its NLI contradiction score.
            e.g., {'sentence 1': 0.1, 'sentence 2': 0.9}
        """
        # 1. Generate the self-consistency samples
        print(f"Generating samples for NLI scoring...")
        samples = llm_apihandler.generate_samples(prompt, model_name, num_samples=3)

        # 2. Split the original answer into sentences
        sentences_to_check = [sent.text.strip() for sent in self.nlp(original_answer).sents]
        if not sentences_to_check:
            # Return a neutral score if the answer is empty
            return {}

        # 3. Run the NLI check
        print("Scoring with NLI...")
        #Get a score for each sentence
        scores = self.selfcheck_nli.predict(
            sentences=sentences_to_check,
            sampled_passages=samples
        )
        #Map each sentence to its score
        sentence_scores = {sentence: float(score) for sentence, score in zip(sentences_to_check, scores)}
        print("-" * 20)
        return sentence_scores
