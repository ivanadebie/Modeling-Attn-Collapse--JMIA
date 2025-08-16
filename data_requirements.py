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
