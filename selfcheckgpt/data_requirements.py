from dataclasses import dataclass, field
from typing import Dict, Any, List
import re
import string

@dataclass
class QARecord:
    """A class to hold all data for a single QA pair, with sentence-level scores."""
    qa_id: str
    context: str
    prompt: str
    question: str = ""
    gold_text: str = ""
    distractor_1_text: str = ""
    distractor_2_text: str = ""
    answer: str = ""
    # confidence: Dict[str, float] = None
    # model_response_uncertainty: float = 0.0
    # hallucination_label: int = 0
    hallucination_score: float = 0.0
    sentence_level_hallu_scores: Dict[str, float] = field(default_factory=dict)
    # sample_level_hallu_scores: Dict[str, float] = field(default_factory=dict)
    # sample_level_95th_percentile: float = 0.0
    distractor_density_class: str = ""
    interference_type: str = ""
    evidence_position: str = ""
    is_gold_binary: int = 0
    gold_text_chunk: str = ""

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
  