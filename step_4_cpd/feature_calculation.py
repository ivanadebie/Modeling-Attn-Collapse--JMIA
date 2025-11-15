"""Feature calculation utilities for the hallucination change-point pipeline."""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Set, Tuple

import numpy as np
import pandas as pd

from step_4_cpd.config_utils import add_config_columns, derive_config_features


from step_4_cpd.generate_metrics_v1 import (
    canon,
    jaccard,
    ngram_set,
    preprocess_tokens,
    tokenize,
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return int(default)
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def calculate_features(
    df: pd.DataFrame,
    sequence_column: str = "sequence_id",
    chunk_column: str = "chunk_index",
    similarity_threshold: float = 0.7,
) -> List[Dict[str, Any]]:
    """Compute per-chunk scalar features, derived config metrics, and hallucination labels."""
    if df.empty:
        return []

    df = df.copy()

    if "config" in df.columns:
        add_config_columns(df)
    else:
        df["config"] = ""
        add_config_columns(df)

    if sequence_column in df.columns:
        grouped = df.groupby(sequence_column, sort=False)
    else:
        df["__sequence_id__"] = 0
        sequence_column = "__sequence_id__"
        grouped = df.groupby(sequence_column, sort=False)

    features: List[Dict[str, Any]] = []

    for sequence_id, group in grouped:
        if chunk_column in group.columns:
            group = group.sort_values(by=chunk_column)
        else:
            group[chunk_column] = range(len(group))
        group = group.reset_index(drop=True)

        # Per-chunk features
        for chunk_idx, row_chunk in enumerate(group.itertuples()):
            sentence_text = str(row_chunk.sentence_text)
            
            # Placeholder for attention-based features
            attention_entropy = 0.0
            effective_rank = 0.0
            self_attention_ratio = 0.0
            head_similarity_ratio = 0.0
            cross_attention_mass_to_gold = 0.0
            distractor_attention_max = 0.0

            # Simplified hallucination proxy for this chunk
            similarity_score = _safe_float(row_chunk.cosine_similarity, 0.0)
            similarity_flag = similarity_score < similarity_threshold
            lack_of_evidence_flag = not _safe_int(row_chunk.is_gold_binary, 0)
            is_hallucination = similarity_flag or lack_of_evidence_flag
            hallu_signal = 1.0 if is_hallucination else 0.0

            feature_row = {
                "sequence_id": sequence_id,
                "chunk_index": chunk_idx,
                "sentence_text": sentence_text,
                "is_hallucination": is_hallucination,
                "hallu_signal": hallu_signal,
                "similarity_flag": similarity_flag,
                "lack_of_evidence_flag": lack_of_evidence_flag,
                "attention_entropy": attention_entropy,
                "effective_rank": effective_rank,
                "self_attention_ratio": self_attention_ratio,
                "head_similarity_ratio": head_similarity_ratio,
                "cross_attention_mass_to_gold": cross_attention_mass_to_gold,
                "distractor_attention_max": distractor_attention_max,
            }
            
            # Add config-derived features
            config_features = derive_config_features(str(row_chunk.config))
            feature_row.update(config_features)
            
            features.append(feature_row)

    return features


def calculate_features_from_records(
    records: Iterable[Dict[str, Any]],
    similarity_threshold: float = 0.7,
) -> List[Dict[str, Any]]:
    """Public helper accepting an iterable of dicts instead of a DataFrame."""
    return calculate_features(pd.DataFrame(list(records)), similarity_threshold=similarity_threshold)


def append_config_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Expose decoded config columns for downstream analysis tables/plots."""
    add_config_columns(df)
    return df
