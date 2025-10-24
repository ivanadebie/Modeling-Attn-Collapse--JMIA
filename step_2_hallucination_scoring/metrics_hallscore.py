import torch
import spacy
from typing import Any, List
from data_requirements import QARecord, deduplicate_samples 
import numpy as np

class HallucinationScorer:
    def compute_semantic_similarity(self, chunk: str, gold_text: str, model_name: str = None) -> float:
        """
        Compute semantic similarity between chunk and gold_text using spaCy's Doc.similarity.
        Returns a float in [0, 1].
        """
        chunk_doc = self.nlp(chunk)
        gold_doc = self.nlp(gold_text)
        sim = chunk_doc.similarity(gold_doc)
        # Clamp to [0, 1] for consistency
        return max(0.0, min(1.0, float(sim)))
    def __init__(self):
        
        # Initialize spacy for sentence splitting (tokenization)
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            from spacy.cli import download
            download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

    def get_sentence_level_sem_similarity_scores(self, original_answer: str, gold_text: str, model_name: str, seed: int) -> dict:
        """
        This function scores the answer for hallucination at the sentence level, defaulting to sentence steps, and falling back to ~64-token chunks for long/noisy sentences.
        Long sentences are those with word count > 80% of the average word count for all sentences in the answer.
        """

        # 1. Sentence splitting

        doc = self.nlp(original_answer)
        sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 5]
        if not sentences:
            return {}

        # 2. Calculate average word count
        word_counts = [len(s.split()) for s in sentences]
        avg_word_count = np.mean(word_counts) if word_counts else 0
        threshold = 0.8 * avg_word_count

        # 3. For long sentences, split into ~64-token chunks
        sentences_to_check = []
        for s, wc in zip(sentences, word_counts):
            if wc > threshold:
                words = s.split()
                for i in range(0, len(words), 64):
                    chunk = ' '.join(words[i:i+64])
                    if len(chunk.strip()) > 5:
                        sentences_to_check.append(chunk)
            else:
                sentences_to_check.append(s)



        sentence_level_scores = {}
        for chunk in sentences_to_check:
            score = self.compute_semantic_similarity(chunk, gold_text, model_name=model_name)
            sentence_level_scores[chunk] = score
        return {
            'sentence_level_sem_similarity_scores': sentence_level_scores
        }

    def confidence_to_binary(self, confidence_dict, threshold=0.5):
        """Convert confidence scores to binary labels using standard threshold."""
        return {k: 1 if v >= threshold else 0 for k, v in confidence_dict.items()}

    def aggregate_confidence_scores(self, confidence_dict):
        """Aggregate sentence-level confidence scores into scalar values."""
        scores_dict = confidence_dict.get('sentence_level_sem_similarity_scores', {})
        scores = list(scores_dict.values()) if isinstance(scores_dict, dict) else []
        if not scores:
            return {'conf_agg_max': 0.0, 'conf_agg_mean': 0.0}
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

