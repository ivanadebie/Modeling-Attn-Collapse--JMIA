"""
Feature engineering for CPD experiments.

This module provides functions to compute derived features from
sentence-level attention metrics, including EWMA smoothing, delta
features, and CUSUM statistics.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional

from ..utils import (
    compute_ewma,
    compute_rolling_variance,
    compute_delta,
    compute_cusum
)


def add_sentence_index(df: pd.DataFrame,
                       seq_col: str = "sequence_id",
                       chunk_col: str = "chunk_index",
                       sent_in_chunk_col: str = "sentence_in_chunk") -> pd.DataFrame:
    """
    Add sentence_index column (cumulative index within sequence).
    
    Args:
        df: Input DataFrame
        seq_col: Sequence ID column
        chunk_col: Chunk index column
        sent_in_chunk_col: Sentence within chunk column (optional)
        
    Returns:
        DataFrame with sentence_index column
    """
    df = df.copy()
    
    # Sort by sequence and chunk (and sentence within chunk if available)
    sort_cols = [seq_col, chunk_col]
    if sent_in_chunk_col in df.columns:
        sort_cols.append(sent_in_chunk_col)
    
    df = df.sort_values(sort_cols).reset_index(drop=True)
    
    # Create cumulative sentence index within each sequence
    df["sentence_index"] = df.groupby(seq_col).cumcount()
    
    return df


def add_ewma_features(df: pd.DataFrame,
                      feature_cols: List[str],
                      seq_col: str = "sequence_id",
                      alpha: float = 0.3,
                      suffix: str = "_ewma") -> pd.DataFrame:
    """
    Add EWMA-smoothed versions of features.
    
    Args:
        df: Input DataFrame (must be sorted by sequence and time)
        feature_cols: Columns to smooth
        seq_col: Sequence ID column
        alpha: EWMA smoothing parameter
        suffix: Suffix for new column names
        
    Returns:
        DataFrame with EWMA features added
    """
    df = df.copy()
    
    for col in feature_cols:
        if col not in df.columns:
            continue
        
        new_col = f"{col}{suffix}"
        df[new_col] = df.groupby(seq_col)[col].transform(
            lambda x: x.ewm(alpha=alpha, adjust=False).mean()
        )
    
    return df


def add_delta_features(df: pd.DataFrame,
                       feature_cols: List[str],
                       seq_col: str = "sequence_id",
                       prefix: str = "delta_") -> pd.DataFrame:
    """
    Add delta (first difference) features.
    
    Args:
        df: Input DataFrame (must be sorted by sequence and time)
        feature_cols: Columns to differentiate
        seq_col: Sequence ID column
        prefix: Prefix for new column names
        
    Returns:
        DataFrame with delta features added
    """
    df = df.copy()
    
    for col in feature_cols:
        if col not in df.columns:
            continue
        
        new_col = f"{prefix}{col}"
        df[new_col] = df.groupby(seq_col)[col].diff().fillna(0.0)
    
    return df


def add_cusum_features(df: pd.DataFrame,
                       feature_cols: List[str],
                       seq_col: str = "sequence_id",
                       baseline: str = "median",
                       suffix: str = "_cusum") -> pd.DataFrame:
    """
    Add CUSUM features (cumulative sum relative to baseline).
    
    Args:
        df: Input DataFrame (must be sorted by sequence and time)
        feature_cols: Columns to compute CUSUM for
        seq_col: Sequence ID column
        baseline: "median" or "mean"
        suffix: Suffix for new column names
        
    Returns:
        DataFrame with CUSUM features added
    """
    df = df.copy()
    
    for col in feature_cols:
        if col not in df.columns:
            continue
        
        new_col = f"{col}{suffix}"
        
        def compute_cusum_group(x):
            values = x.values.astype(float)
            ref = np.nanmedian(values) if baseline == "median" else np.nanmean(values)
            return np.cumsum(np.nan_to_num(values - ref, nan=0.0))
        
        df[new_col] = df.groupby(seq_col)[col].transform(compute_cusum_group)
    
    return df


def add_variance_features(df: pd.DataFrame,
                          feature_cols: List[str],
                          seq_col: str = "sequence_id",
                          window: int = 5,
                          suffix: str = "_variance") -> pd.DataFrame:
    """
    Add rolling variance features.
    
    Args:
        df: Input DataFrame (must be sorted by sequence and time)
        feature_cols: Columns to compute variance for
        seq_col: Sequence ID column
        window: Rolling window size
        suffix: Suffix for new column names
        
    Returns:
        DataFrame with variance features added
    """
    df = df.copy()
    
    for col in feature_cols:
        if col not in df.columns:
            continue
        
        new_col = f"{col}{suffix}"
        df[new_col] = df.groupby(seq_col)[col].transform(
            lambda x: x.rolling(window=window, min_periods=1).var().fillna(0.0)
        )
    
    return df


def add_reallocation_rate(df: pd.DataFrame,
                          gold_attn_col: str,
                          seq_col: str = "sequence_id",
                          eps: float = 1e-8) -> pd.DataFrame:
    """
    Add attention reallocation rate feature.
    
    Rate = |Δgold_attn| / (|gold_attn| + eps)
    
    Args:
        df: Input DataFrame
        gold_attn_col: Gold attention column
        seq_col: Sequence ID column
        eps: Small constant for stability
        
    Returns:
        DataFrame with reallocation rate added
    """
    df = df.copy()
    
    if gold_attn_col not in df.columns:
        return df
    
    # Compute delta
    delta_gold = df.groupby(seq_col)[gold_attn_col].diff().fillna(0.0)
    gold_vals = df[gold_attn_col].abs()
    
    df["attn_reallocation_rate"] = delta_gold.abs() / (gold_vals + eps)
    
    return df


def add_gold_attention_drop(df: pd.DataFrame,
                            gold_attn_col: str,
                            seq_col: str = "sequence_id") -> pd.DataFrame:
    """
    Add gold attention drop feature (max(0, -Δgold_attn)).
    
    Args:
        df: Input DataFrame
        gold_attn_col: Gold attention column
        seq_col: Sequence ID column
        
    Returns:
        DataFrame with gold_attn_drop added
    """
    df = df.copy()
    
    if gold_attn_col not in df.columns:
        return df
    
    delta_gold = df.groupby(seq_col)[gold_attn_col].diff().fillna(0.0)
    df["gold_attn_drop"] = np.maximum(0, -delta_gold)
    
    return df


def engineer_all_features(df: pd.DataFrame,
                          seq_col: str = "sequence_id",
                          time_col: str = "sentence_index",
                          ewma_alpha: float = 0.3,
                          variance_window: int = 5) -> pd.DataFrame:
    """
    Apply all feature engineering steps.
    
    Args:
        df: Input DataFrame with sentence-level attention metrics
        seq_col: Sequence ID column
        time_col: Time/index column
        ewma_alpha: EWMA smoothing parameter
        variance_window: Rolling variance window
        
    Returns:
        DataFrame with all engineered features
    """
    # Ensure sorted
    df = df.sort_values([seq_col, time_col]).reset_index(drop=True)
    
    # Base attention features
    base_features = [
        "sent_avg_entropy",
        "sent_distractor_attention_max",
        "sent_cross_attention_to_gold"
    ]
    
    # Filter to existing columns
    base_features = [c for c in base_features if c in df.columns]
    
    # 1. EWMA smoothing
    df = add_ewma_features(df, base_features, seq_col, ewma_alpha)
    
    # 2. Delta features (on EWMA)
    ewma_features = [f"{c}_ewma" for c in base_features]
    df = add_delta_features(df, ewma_features, seq_col)
    
    # 3. Variance features
    df = add_variance_features(df, ["sent_avg_entropy_ewma"], seq_col, variance_window)
    
    # Rename for compatibility
    if "sent_avg_entropy_ewma_variance" in df.columns:
        df["entropy_variance"] = df["sent_avg_entropy_ewma_variance"]
    
    # 4. CUSUM features
    df = add_cusum_features(df, ["sent_avg_entropy_ewma"], seq_col, "median")
    
    # Rename for compatibility
    if "sent_avg_entropy_ewma_cusum" in df.columns:
        df["entropy_cusum"] = df["sent_avg_entropy_ewma_cusum"]
    
    # 5. Gold attention specific features
    gold_col = "sent_cross_attention_to_gold"
    if gold_col in df.columns:
        df = add_reallocation_rate(df, gold_col, seq_col)
        df = add_gold_attention_drop(df, gold_col, seq_col)
        df = add_cusum_features(df, [gold_col], seq_col, "median")
    
    # 6. Create standard delta features
    if "delta_sent_avg_entropy_ewma" in df.columns:
        df["delta_entropy"] = df["delta_sent_avg_entropy_ewma"]
    if "delta_sent_distractor_attention_max_ewma" in df.columns:
        df["delta_distractor_max"] = df["delta_sent_distractor_attention_max_ewma"]
    
    return df


def extract_window_features(df: pd.DataFrame,
                            cp_index: int,
                            feature_cols: List[str],
                            window: int = 5) -> Dict[str, float]:
    """
    Extract window-based features around a candidate change point.
    
    For each feature, computes: mean, std, min, max, slope
    
    Args:
        df: DataFrame for a single sequence (sorted by time)
        cp_index: Index of candidate change point
        feature_cols: Columns to extract features from
        window: Window size (cp_index - window to cp_index + window + 1)
        
    Returns:
        Dict with window features
    """
    n = len(df)
    lo = max(0, cp_index - window)
    hi = min(n, cp_index + window + 1)
    
    window_df = df.iloc[lo:hi]
    
    features = {}
    
    for col in feature_cols:
        if col not in df.columns:
            continue
        
        values = window_df[col].values.astype(float)
        values = np.nan_to_num(values, nan=0.0)
        
        if len(values) == 0:
            features[f"{col}_win_mean"] = 0.0
            features[f"{col}_win_std"] = 0.0
            features[f"{col}_win_min"] = 0.0
            features[f"{col}_win_max"] = 0.0
            features[f"{col}_win_slope"] = 0.0
            continue
        
        features[f"{col}_win_mean"] = float(np.mean(values))
        features[f"{col}_win_std"] = float(np.std(values))
        features[f"{col}_win_min"] = float(np.min(values))
        features[f"{col}_win_max"] = float(np.max(values))
        
        # Slope via linear regression
        if len(values) > 1:
            x = np.arange(len(values))
            slope = np.polyfit(x, values, 1)[0]
            features[f"{col}_win_slope"] = float(slope)
        else:
            features[f"{col}_win_slope"] = 0.0
    
    return features


def extract_jump_features(df: pd.DataFrame,
                          cp_index: int,
                          entropy_col: str = "sent_avg_entropy_ewma",
                          gold_col: str = "sent_cross_attention_to_gold_ewma") -> Dict[str, float]:
    """
    Extract jump features at change point.
    
    Args:
        df: DataFrame for a single sequence (sorted by time)
        cp_index: Index of candidate change point
        entropy_col: Entropy column
        gold_col: Gold attention column
        
    Returns:
        Dict with jump features
    """
    features = {}
    
    if cp_index <= 0 or cp_index >= len(df):
        return {"entropy_jump": 0.0, "gold_jump": 0.0}
    
    if entropy_col in df.columns:
        val_curr = df.iloc[cp_index][entropy_col]
        val_prev = df.iloc[cp_index - 1][entropy_col]
        features["entropy_jump"] = float(val_curr - val_prev)
    else:
        features["entropy_jump"] = 0.0
    
    if gold_col in df.columns:
        val_curr = df.iloc[cp_index][gold_col]
        val_prev = df.iloc[cp_index - 1][gold_col]
        features["gold_jump"] = float(val_curr - val_prev)
    else:
        features["gold_jump"] = 0.0
    
    return features
