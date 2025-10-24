"""Feature calculation utilities for the hallucination change-point pipeline."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd

from step_4_cpd.config_utils import add_config_columns, derive_config_features


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
            group = group.sort_values(chunk_column)
        else:
            group = group.sort_index()
            group[chunk_column] = np.arange(len(group))
        group = group.reset_index(drop=True)

        total_steps = len(group)
        denom = max(total_steps - 1, 1)

        gold_series = None
        for candidate in ["is_gold_binary", "chunk_gold_match", "chunk_level_gold_matches"]:
            if candidate in group.columns:
                gold_series = group[candidate]
                break
        if gold_series is None:
            gold_series = pd.Series([0] * total_steps)
        gold_series = gold_series.fillna(0).astype(int)
        gold_indices = [idx for idx, flag in enumerate(gold_series) if flag == 1]

        if "hallu_signal" in group.columns:
            hallu_signal = group["hallu_signal"].fillna(0).astype(int).tolist()
        else:
            similarity_series = None
            for candidate in ["semantic_similarity", "sentence_hallu_score", "hallucination_score"]:
                if candidate in group.columns:
                    similarity_series = group[candidate]
                    break
            if similarity_series is None:
                similarity_series = pd.Series([1.0] * total_steps)
            similarity_series = similarity_series.fillna(0.0).astype(float)
            lack_of_evidence = (1 - gold_series).clip(lower=0)
            hallu_signal = (
                (similarity_series < similarity_threshold) & (lack_of_evidence == 1)
            ).astype(int).tolist()

        onset_idx = next((idx for idx, flag in enumerate(hallu_signal) if flag), None)

        prev_effective_rank = 0.0
        prev_cross_mass = 0.0
        prev_distractor_max = 0.0

        for step_idx, row in group.iterrows():
            record: Dict[str, Any] = {
                "sequence_id": sequence_id,
                "chunk_index": _safe_int(row.get(chunk_column, step_idx)),
                "sentence_text": row.get("sentence_text", ""),
                "config": row.get("config", ""),
                "question_id": row.get("question_id", ""),
                "domain": row.get("domain", ""),
                "config_idx": row.get("config_idx", ""),
                "row_index": row.get("row_index", ""),
            }

            record.update(derive_config_features(row.get("config", "")))

            record["normalized_chunk_index"] = step_idx / denom if total_steps > 1 else 0.0
            record["is_gold_binary"] = int(gold_series.iloc[step_idx])

            similarity_value = _safe_float(
                row.get(
                    "semantic_similarity",
                    row.get("sentence_hallu_score", row.get("hallucination_score", 1.0)),
                )
            )
            record["semantic_similarity"] = similarity_value
            record["semantic_distance"] = 1.0 - similarity_value
            record["hallucination_score"] = record["semantic_distance"]
            record["similarity_flag"] = _safe_int(
                row.get("similarity_flag", int(similarity_value < similarity_threshold))
            )
            record["lack_of_evidence_flag"] = _safe_int(
                row.get("lack_of_evidence_flag", int(1 - record["is_gold_binary"]))
            )
            record["hallu_signal"] = int(hallu_signal[step_idx])
            record["hallucination_label"] = 1 if onset_idx is not None and step_idx == onset_idx else 0

            record["alignment_score"] = _safe_float(row.get("model_alignment_score", np.nan))
            record["evidence_overlap"] = _safe_float(row.get("evidence_overlap", np.nan))
            record["interference_score"] = _safe_float(row.get("interference_score", np.nan))

            if gold_indices:
                distance = min(abs(step_idx - gold_idx) for gold_idx in gold_indices)
                record["evidence_distance_normalized"] = distance / denom if total_steps > 1 else 0.0
            else:
                record["evidence_distance_normalized"] = 1.0

            record["attention_entropy"] = _safe_float(row.get("attention_entropy", np.nan))
            record["self_attention_ratio"] = _safe_float(row.get("self_attention_ratio", np.nan))
            record["effective_rank"] = _safe_float(row.get("effective_rank", np.nan))
            record["head_similarity_ratio"] = _safe_float(row.get("head_similarity_ratio", np.nan))
            record["cross_attention_mass_to_gold"] = _safe_float(
                row.get("cross_attention_mass_to_gold", np.nan)
            )
            record["distractor_attention_max"] = _safe_float(row.get("distractor_attention_max", np.nan))

            record["effective_rank_delta"] = record["effective_rank"] - prev_effective_rank
            record["cross_attention_mass_delta"] = (
                record["cross_attention_mass_to_gold"] - prev_cross_mass
            )
            record["distractor_attention_max_delta"] = (
                record["distractor_attention_max"] - prev_distractor_max
            )

            prev_effective_rank = record["effective_rank"]
            prev_cross_mass = record["cross_attention_mass_to_gold"]
            prev_distractor_max = record["distractor_attention_max"]

            features.append(record)

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
