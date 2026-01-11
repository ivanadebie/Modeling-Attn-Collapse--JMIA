"""
Evaluation metrics and utilities for CPD experiments.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from sklearn.metrics import (
    precision_recall_curve, 
    average_precision_score,
    precision_recall_fscore_support
)
import matplotlib.pyplot as plt
from pathlib import Path

from .helpers import compute_prf1, compute_ttd_stats, evaluate_first_cp


# =============================================================================
# PR CURVE UTILITIES
# =============================================================================

def compute_pr_curve(y_true: np.ndarray, 
                     y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Compute precision-recall curve and average precision.
    
    Args:
        y_true: True binary labels
        y_score: Predicted scores/probabilities
        
    Returns:
        Tuple of (precision, recall, thresholds, average_precision)
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    return precision, recall, thresholds, ap


def find_best_threshold(precision: np.ndarray, 
                        recall: np.ndarray, 
                        thresholds: np.ndarray,
                        min_precision: Optional[float] = None) -> Dict[str, float]:
    """
    Find best threshold based on F1 score.
    
    Args:
        precision: Precision values
        recall: Recall values
        thresholds: Threshold values
        min_precision: Minimum precision constraint (optional)
        
    Returns:
        Dict with best threshold and corresponding metrics
    """
    # Compute F1 for each threshold
    f1_scores = []
    for i in range(len(thresholds)):
        p, r = precision[i+1], recall[i+1]  # precision_recall_curve convention
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        f1_scores.append(f1)
    
    if not f1_scores:
        return {"threshold": None, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    # Apply precision constraint if specified
    if min_precision is not None:
        valid_mask = precision[1:len(thresholds)+1] >= min_precision
        if not np.any(valid_mask):
            # No thresholds meet constraint, return best F1 overall
            best_idx = int(np.argmax(f1_scores))
            return {
                "threshold": float(thresholds[best_idx]),
                "precision": float(precision[best_idx+1]),
                "recall": float(recall[best_idx+1]),
                "f1": float(f1_scores[best_idx]),
                "note": f"No threshold meets precision >= {min_precision}"
            }
        
        # Find best F1 among valid thresholds
        f1_masked = np.where(valid_mask, f1_scores, -1)
        best_idx = int(np.argmax(f1_masked))
    else:
        best_idx = int(np.argmax(f1_scores))
    
    return {
        "threshold": float(thresholds[best_idx]),
        "precision": float(precision[best_idx+1]),
        "recall": float(recall[best_idx+1]),
        "f1": float(f1_scores[best_idx])
    }


def plot_pr_curve(precision: np.ndarray,
                  recall: np.ndarray,
                  ap: float,
                  title: str = "Precision-Recall Curve",
                  save_path: Optional[Path] = None) -> None:
    """
    Plot precision-recall curve.
    
    Args:
        precision: Precision values
        recall: Recall values
        ap: Average precision
        title: Plot title
        save_path: Path to save figure (optional)
    """
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, lw=2)
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title(f"{title}\nAP = {ap:.3f}", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1.05])
    plt.ylim([0, 1.05])
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# =============================================================================
# SEQUENCE-LEVEL EVALUATION
# =============================================================================

def evaluate_sequence_level(candidates_df: pd.DataFrame,
                            onset_map: Dict[str, Optional[int]],
                            proba_col: str = "proba_true",
                            seq_col: str = "sequence_id",
                            cp_col: str = "cp",
                            mode: str = "earliest") -> Tuple[np.ndarray, np.ndarray]:
    """
    Build sequence-level scores and labels for PR evaluation.
    
    Args:
        candidates_df: DataFrame with candidates and probabilities
        onset_map: Mapping from sequence_id to onset index
        proba_col: Probability column name
        seq_col: Sequence ID column name
        cp_col: Change point column name
        mode: "earliest" (use earliest candidate) or "max" (use max probability)
        
    Returns:
        Tuple of (y_true, y_score) arrays at sequence level
    """
    all_seqs = sorted(onset_map.keys())
    
    # Build score per sequence
    seq_scores = {sid: 0.0 for sid in all_seqs}
    
    for sid, g in candidates_df.groupby(seq_col):
        if sid not in seq_scores:
            continue
        g = g.sort_values(cp_col)
        
        if mode == "earliest":
            seq_scores[sid] = float(g.iloc[0][proba_col])
        else:  # max
            seq_scores[sid] = float(g[proba_col].max())
    
    y_true = np.array([1 if onset_map.get(sid) is not None else 0 for sid in all_seqs])
    y_score = np.array([seq_scores[sid] for sid in all_seqs])
    
    return y_true, y_score


def evaluate_threshold_sweep(candidates_df: pd.DataFrame,
                             onset_map: Dict[str, Optional[int]],
                             thresholds: List[float],
                             proba_col: str = "proba_true",
                             seq_col: str = "sequence_id",
                             cp_col: str = "cp",
                             tolerance: int = 2) -> pd.DataFrame:
    """
    Evaluate performance across threshold sweep.
    
    Args:
        candidates_df: DataFrame with candidates and probabilities
        onset_map: Mapping from sequence_id to onset index
        thresholds: List of thresholds to evaluate
        proba_col: Probability column name
        seq_col: Sequence ID column name  
        cp_col: Change point column name
        tolerance: Tolerance for matching
        
    Returns:
        DataFrame with metrics per threshold
    """
    results = []
    
    # Pre-group candidates by sequence
    by_seq = {sid: g.sort_values(cp_col) for sid, g in candidates_df.groupby(seq_col)}
    
    for tau in thresholds:
        TP = FP = FN = 0
        ttds = []
        detected_onset_seqs = 0
        fp_seqs_no_onset = 0
        n_onset_seqs = sum(1 for v in onset_map.values() if v is not None)
        
        for sid, onset in onset_map.items():
            # Get candidates above threshold for this sequence
            if sid in by_seq:
                g = by_seq[sid]
                hits = g[g[proba_col] >= tau]
            else:
                hits = pd.DataFrame()
            
            if len(hits) == 0:
                # No prediction
                if onset is not None:
                    FN += 1
                continue
            
            # Use earliest candidate above threshold
            first_cp = int(hits.iloc[0][cp_col])
            
            if onset is not None:
                detected_onset_seqs += 1
                ttd = first_cp - onset
                
                if abs(ttd) <= tolerance:
                    TP += 1
                    ttds.append(ttd)
                else:
                    FP += 1
                    FN += 1
                    ttds.append(ttd)
            else:
                FP += 1
                fp_seqs_no_onset += 1
        
        P, R, F1 = compute_prf1(TP, FP, FN)
        ttd_stats = compute_ttd_stats(ttds)
        
        results.append({
            "threshold": tau,
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "Precision": P,
            "Recall": R,
            "F1": F1,
            "onset_seqs": n_onset_seqs,
            "detected_onset_seqs": detected_onset_seqs,
            "detect_rate": detected_onset_seqs / n_onset_seqs if n_onset_seqs > 0 else 0,
            "fp_seqs_no_onset": fp_seqs_no_onset,
            **ttd_stats
        })
    
    return pd.DataFrame(results)


# =============================================================================
# CPD-SPECIFIC EVALUATION
# =============================================================================

def evaluate_cpd_sweep(df: pd.DataFrame,
                       feature_cols: List[str],
                       onset_map: Dict[str, Optional[int]],
                       penalties: List[float],
                       cpd_func,  # Callable that takes (signal, penalty) and returns changepoints
                       seq_col: str = "sequence_id",
                       time_col: str = "sentence_index",
                       tolerance: int = 2) -> pd.DataFrame:
    """
    Evaluate CPD algorithm across penalty sweep.
    
    Args:
        df: DataFrame with features
        feature_cols: List of feature column names
        onset_map: Mapping from sequence_id to onset index
        penalties: List of penalties to evaluate
        cpd_func: CPD function that takes (signal, penalty) and returns changepoints
        seq_col: Sequence ID column
        time_col: Time/index column
        tolerance: Tolerance for matching
        
    Returns:
        DataFrame with metrics per penalty
    """
    from .helpers import robust_zscore
    
    results = []
    
    for pen in penalties:
        TP = FP = FN = 0
        ttds = []
        
        for sid, g in df.groupby(seq_col):
            g = g.sort_values(time_col)
            X = g[feature_cols].values.astype(float)
            X = np.nan_to_num(X, nan=0.0)
            
            # Apply robust z-score
            Xz = robust_zscore(X)
            
            # Run CPD
            try:
                cps = cpd_func(Xz, pen)
            except Exception:
                cps = []
            
            # Evaluate
            true_onset = onset_map.get(sid)
            eval_result = evaluate_first_cp(true_onset, cps, tolerance)
            
            TP += eval_result["TP"]
            FP += eval_result["FP"]
            FN += eval_result["FN"]
            if eval_result["TTD"] is not None:
                ttds.append(eval_result["TTD"])
        
        P, R, F1 = compute_prf1(TP, FP, FN)
        ttd_stats = compute_ttd_stats(ttds)
        
        results.append({
            "penalty": pen,
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "Precision": P,
            "Recall": R,
            "F1": F1,
            **ttd_stats
        })
    
    return pd.DataFrame(results)


# =============================================================================
# SUMMARY UTILITIES
# =============================================================================

def create_experiment_summary(results: Dict[str, pd.DataFrame],
                              metric: str = "F1") -> pd.DataFrame:
    """
    Create summary table from multiple experiment results.
    
    Args:
        results: Dict mapping experiment name to results DataFrame
        metric: Metric to use for selecting best row
        
    Returns:
        Summary DataFrame with best result per experiment
    """
    summary_rows = []
    
    for exp_name, df in results.items():
        if len(df) == 0:
            continue
        
        # Get best row by metric
        best_idx = df[metric].idxmax()
        best_row = df.loc[best_idx].to_dict()
        best_row["experiment"] = exp_name
        summary_rows.append(best_row)
    
    summary = pd.DataFrame(summary_rows)
    
    # Reorder columns
    first_cols = ["experiment"]
    other_cols = [c for c in summary.columns if c not in first_cols]
    summary = summary[first_cols + other_cols]
    
    return summary.sort_values(metric, ascending=False).reset_index(drop=True)
