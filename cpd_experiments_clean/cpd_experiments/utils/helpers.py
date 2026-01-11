"""
Utility functions for CPD experiments.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any, Union
import json
import ast
import gc
import torch
from scipy.stats import median_abs_deviation


# =============================================================================
# DATA PARSING UTILITIES
# =============================================================================

def parse_list_column(val: Any) -> List:
    """
    Parse a column value that might be a JSON string, literal string, or already a list.
    
    Args:
        val: Value to parse
        
    Returns:
        Parsed list or empty list if parsing fails
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return []
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        val = val.strip()
        if not val:
            return []
        # Try JSON first
        try:
            parsed = json.loads(val)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass
        # Try literal_eval
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, list):
                return parsed
        except (ValueError, SyntaxError):
            pass
        # Return as single-element list
        return [val]
    return []


def safe_to_numeric(x: Any, default: float = 0.0) -> float:
    """Safely convert value to numeric."""
    try:
        result = pd.to_numeric(x, errors='coerce')
        return default if pd.isna(result) else float(result)
    except:
        return default


# =============================================================================
# SEQUENCE UTILITIES
# =============================================================================

def get_onset_index(labels: np.ndarray) -> Optional[int]:
    """
    Find first onset index where label transitions from 0 to positive.
    
    Args:
        labels: Array of hallucination labels
        
    Returns:
        Index of first onset or None if no onset
    """
    labels = np.asarray(labels)
    labels = np.nan_to_num(labels, nan=0.0).astype(int)
    
    for i in range(1, len(labels)):
        if labels[i-1] == 0 and labels[i] > 0:
            return i
    return None


def build_onset_map(df: pd.DataFrame, 
                    seq_col: str = "sequence_id",
                    time_col: str = "sentence_index", 
                    label_col: str = "hallu_label") -> Dict[str, Optional[int]]:
    """
    Build mapping from sequence_id to onset index.
    
    Args:
        df: DataFrame with sequences
        seq_col: Sequence ID column
        time_col: Time/index column
        label_col: Label column
        
    Returns:
        Dict mapping sequence_id to onset index (or None)
    """
    onset_map = {}
    for sid, g in df.groupby(seq_col):
        g = g.sort_values(time_col)
        labels = g[label_col].values
        onset_map[sid] = get_onset_index(labels)
    return onset_map


def has_onset(labels: np.ndarray) -> bool:
    """Check if sequence has an onset (0 → positive transition)."""
    return get_onset_index(labels) is not None


# =============================================================================
# FEATURE ENGINEERING UTILITIES  
# =============================================================================

def compute_ewma(series: pd.Series, alpha: float = 0.3) -> pd.Series:
    """Compute Exponentially Weighted Moving Average."""
    return series.ewm(alpha=alpha, adjust=False).mean()


def compute_rolling_variance(series: pd.Series, window: int = 5) -> pd.Series:
    """Compute rolling variance."""
    return series.rolling(window=window, min_periods=1).var().fillna(0.0)


def compute_delta(series: pd.Series) -> pd.Series:
    """Compute first difference (delta)."""
    return series.diff().fillna(0.0)


def compute_cusum(series: pd.Series, baseline: str = "median") -> pd.Series:
    """
    Compute CUSUM (Cumulative Sum) relative to baseline.
    
    Args:
        series: Input series
        baseline: "median" or "mean"
        
    Returns:
        CUSUM series
    """
    values = series.values.astype(float)
    if baseline == "median":
        ref = np.nanmedian(values)
    else:
        ref = np.nanmean(values)
    return pd.Series(np.cumsum(values - ref), index=series.index)


def robust_zscore(X: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """
    Compute robust z-score using median and MAD.
    
    Args:
        X: Input array (n_samples, n_features) or (n_samples,)
        eps: Small constant for numerical stability
        
    Returns:
        Robust z-scored array
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    med = np.nanmedian(X, axis=0)
    mad = np.nanmedian(np.abs(X - med), axis=0)
    
    # Scale MAD to approximate std (for normal distribution)
    denom = 1.4826 * mad
    denom = np.where(denom < eps, 1.0, denom)
    
    Z = (X - med) / denom
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
    
    return Z.squeeze() if Z.shape[1] == 1 else Z


# =============================================================================
# EVALUATION UTILITIES
# =============================================================================

def evaluate_first_cp(true_onset: Optional[int], 
                      pred_cps: List[int], 
                      tolerance: int = 2) -> Dict[str, Any]:
    """
    Evaluate first predicted change point against true onset.
    
    Args:
        true_onset: True onset index (or None if no onset)
        pred_cps: List of predicted change points
        tolerance: Tolerance in sentences for matching
        
    Returns:
        Dict with TP, FP, FN, and TTD (time-to-detection)
    """
    pred_cp = min(pred_cps) if pred_cps else None
    
    if true_onset is None:
        # Negative sequence (no onset)
        if pred_cp is None:
            return {"TP": 0, "FP": 0, "FN": 0, "TTD": None}
        else:
            return {"TP": 0, "FP": 1, "FN": 0, "TTD": None}
    else:
        # Positive sequence (has onset)
        if pred_cp is None:
            return {"TP": 0, "FP": 0, "FN": 1, "TTD": None}
        elif abs(pred_cp - true_onset) <= tolerance:
            return {"TP": 1, "FP": 0, "FN": 0, "TTD": pred_cp - true_onset}
        else:
            return {"TP": 0, "FP": 1, "FN": 1, "TTD": pred_cp - true_onset}


def compute_prf1(TP: int, FP: int, FN: int) -> Tuple[float, float, float]:
    """Compute Precision, Recall, F1."""
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def compute_ttd_stats(ttds: List[float]) -> Dict[str, float]:
    """Compute TTD statistics."""
    if not ttds:
        return {
            "TTD_n": 0,
            "TTD_mean": np.nan,
            "TTD_median": np.nan,
            "TTD_std": np.nan,
            "TTD_min": np.nan,
            "TTD_max": np.nan,
            "TTD_p10": np.nan,
            "TTD_p90": np.nan
        }
    
    arr = np.array(ttds)
    return {
        "TTD_n": len(arr),
        "TTD_mean": float(np.mean(arr)),
        "TTD_median": float(np.median(arr)),
        "TTD_std": float(np.std(arr)),
        "TTD_min": float(np.min(arr)),
        "TTD_max": float(np.max(arr)),
        "TTD_p10": float(np.percentile(arr, 10)),
        "TTD_p90": float(np.percentile(arr, 90))
    }


# =============================================================================
# MEMORY UTILITIES
# =============================================================================

def clear_gpu_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def get_gpu_memory_info() -> Dict[str, float]:
    """Get GPU memory usage info."""
    if not torch.cuda.is_available():
        return {"available": False}
    
    return {
        "available": True,
        "allocated_gb": torch.cuda.memory_allocated() / 1e9,
        "reserved_gb": torch.cuda.memory_reserved() / 1e9,
        "max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9
    }


# =============================================================================
# COLUMN RESOLUTION UTILITIES
# =============================================================================

def pick_first_existing(df: pd.DataFrame, 
                        candidates: List[str], 
                        required: bool = True,
                        default: Optional[str] = None) -> Optional[str]:
    """
    Pick first existing column from candidates list.
    
    Args:
        df: DataFrame to check
        candidates: List of candidate column names
        required: If True, raise error if none found
        default: Default value if not required and none found
        
    Returns:
        First existing column name or default
    """
    for col in candidates:
        if col in df.columns:
            return col
    
    if required:
        raise KeyError(f"None of these columns exist: {candidates}")
    return default


def ensure_columns(df: pd.DataFrame, 
                   required_cols: List[str], 
                   name: str = "df") -> None:
    """
    Ensure required columns exist in DataFrame.
    
    Args:
        df: DataFrame to check
        required_cols: List of required column names
        name: Name for error message
        
    Raises:
        KeyError if any required columns are missing
    """
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"{name} missing required columns: {missing}")
