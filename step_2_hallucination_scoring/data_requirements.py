from dataclasses import dataclass, field
from typing import Dict, Any, List
import re
import string

@dataclass
class QARecord:
    """A class to hold all data for a single QA pair, with sentence-level scores."""
    question_id: str
    model_prompt: str
    config_idx: int = 0
    domain: str = ""
    config: str = ""
    question: str = ""
    evidence_position: float = 0.0
    distractors: str = ""
    row_index: int = 0
    model_answer: str = ""
    # confidence: Dict[str, float] = None
    # model_response_uncertainty: float = 0.0
    # hallucination_label: int = 0
    hallucination_score: float = 0.0
    sentence_level_hallu_scores: Dict[str, float] = field(default_factory=dict)
    # sample_level_hallu_scores: Dict[str, float] = field(default_factory=dict)
    # sample_level_95th_percentile: float = 0.0
    is_gold_binary: int = 0
    evidence_position: float = 0.0

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
    print(unique_samples)
    return unique_samples
  