"""
E8: Feature Ablation Analysis

Performs drop-one-group ablation to identify most important feature groups
for the Stage 2 classifier in E7 two-stage detection.

Key findings:
- raw_ewma_base features most critical (largest AP drop when removed)
- attn_level features may introduce noise (AP improves when dropped)
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from tqdm import tqdm

from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import average_precision_score

from ..utils import (
    build_onset_map,
    compute_pr_curve,
    find_best_threshold
)


# =============================================================================
# FEATURE GROUP DEFINITIONS
# =============================================================================

def define_feature_groups(feature_cols: List[str]) -> Dict[str, List[str]]:
    """
    Define feature groups based on column name patterns.
    
    Args:
        feature_cols: List of all feature column names
        
    Returns:
        Dict mapping group name to list of columns
    """
    def cols_matching(substrs: List[str]) -> List[str]:
        out = []
        for c in feature_cols:
            c_low = c.lower()
            if any(s in c_low for s in substrs):
                out.append(c)
        return sorted(list(set(out)))
    
    groups = {
        # Raw/EWMA base window stats
        "raw_ewma_base": cols_matching(["_ewma_win_"]),
        
        # Delta features
        "deltas": cols_matching(["delta_"]) + cols_matching(["drop_raw"]),
        
        # Gold attention drop
        "gold_drop": cols_matching(["gold_attn_drop"]),
        
        # Attention level features
        "attn_level": cols_matching(["entropy_variance", "attn_realloc"]),
        
        # Semantic onset features
        "onset_semantic": cols_matching(["delta_overlap", "delta_interf"]),
        
        # Jump features
        "jumps": cols_matching(["_jump"]),
        
        # CUSUM features
        "cusum": cols_matching(["cusum"]),
    }
    
    # Deduplicate (keep first group priority)
    seen = set()
    for k in groups:
        clean = []
        for c in groups[k]:
            if c not in seen:
                clean.append(c)
                seen.add(c)
        groups[k] = clean
    
    # Remove empty groups
    groups = {k: v for k, v in groups.items() if len(v) > 0}
    
    return groups


# =============================================================================
# ABLATION UTILITIES
# =============================================================================

def cv_predict_proba(X: np.ndarray, 
                     y: np.ndarray,
                     groups: Optional[np.ndarray] = None,
                     n_splits: int = 5, 
                     seed: int = 42) -> np.ndarray:
    """
    Out-of-fold predicted probabilities.
    
    Args:
        X: Feature matrix
        y: Labels
        groups: Group labels for GroupKFold (optional)
        n_splits: Number of CV folds
        seed: Random seed
        
    Returns:
        OOF probability predictions
    """
    proba = np.zeros(len(y), dtype=float)
    
    pipeline = Pipeline([
        ("scaler", RobustScaler()),
        ("clf", GradientBoostingClassifier(random_state=seed))
    ])
    
    if groups is not None:
        kfold = GroupKFold(n_splits=n_splits)
        splits = kfold.split(X, y, groups)
    else:
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = kfold.split(X, y)
    
    for train_idx, test_idx in splits:
        pipeline.fit(X[train_idx], y[train_idx])
        proba[test_idx] = pipeline.predict_proba(X[test_idx])[:, 1]
    
    return proba


def sequence_scores_from_candidates(cand_df: pd.DataFrame,
                                    all_seq_ids: List[str],
                                    proba_col: str = "proba_oof",
                                    seq_col: str = "sequence_id",
                                    cp_col: str = "cp",
                                    mode: str = "earliest") -> Dict[str, float]:
    """
    Build sequence-level scores from candidate probabilities.
    
    Args:
        cand_df: Candidate DataFrame with probabilities
        all_seq_ids: All sequence IDs
        proba_col: Probability column
        seq_col: Sequence ID column
        cp_col: Change point column
        mode: "earliest" or "max"
        
    Returns:
        Dict mapping sequence_id to score
    """
    scores = {sid: 0.0 for sid in all_seq_ids}
    
    for sid, g in cand_df.groupby(seq_col):
        if sid not in scores:
            continue
        g = g.sort_values(cp_col)
        
        if mode == "earliest":
            scores[sid] = float(g.iloc[0][proba_col])
        else:
            scores[sid] = float(g[proba_col].max())
    
    return scores


def best_f1_and_ap(y_true: np.ndarray, 
                   y_score: np.ndarray) -> Dict[str, float]:
    """
    Compute best F1 and average precision.
    
    Args:
        y_true: True binary labels
        y_score: Predicted scores
        
    Returns:
        Dict with AP, best_F1, best_threshold, best_P, best_R
    """
    precision, recall, thresholds = compute_pr_curve(y_true, y_score)[:3]
    ap = average_precision_score(y_true, y_score)
    
    # Compute F1 at each threshold
    f1s = []
    for i in range(len(thresholds)):
        p, r = precision[i+1], recall[i+1]
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        f1s.append(f1)
    
    if not f1s:
        return {"AP": ap, "best_F1": 0.0, "best_threshold": 0.5, "best_P": 0.0, "best_R": 0.0}
    
    best_idx = int(np.argmax(f1s))
    
    return {
        "AP": float(ap),
        "best_F1": float(f1s[best_idx]),
        "best_threshold": float(thresholds[best_idx]),
        "best_P": float(precision[best_idx + 1]),
        "best_R": float(recall[best_idx + 1])
    }


# =============================================================================
# ABLATION RUNNER
# =============================================================================

def run_ablation(cand_df: pd.DataFrame,
                 df_main: pd.DataFrame,
                 seq_col: str = "sequence_id",
                 label_col: str = "hallu_label",
                 time_col: str = "sentence_index",
                 y_col: str = "is_true_cp",
                 n_splits: int = 5,
                 seed: int = 42) -> pd.DataFrame:
    """
    Run drop-one-group ablation analysis.
    
    Args:
        cand_df: Candidate DataFrame with features
        df_main: Main DataFrame for onset map
        seq_col: Sequence ID column
        label_col: Label column
        time_col: Time column
        y_col: Candidate label column
        n_splits: CV folds
        seed: Random seed
        
    Returns:
        Ablation results DataFrame
    """
    # Build onset map
    onset_map = build_onset_map(df_main, seq_col, time_col, label_col)
    all_seq_ids = sorted(onset_map.keys())
    y_true_seq = np.array([1 if onset_map.get(sid) is not None else 0 for sid in all_seq_ids])
    
    # Get feature columns
    non_features = {seq_col, "cp", y_col, "proba_true", "proba_oof"}
    feat_cols_all = [c for c in cand_df.columns if c not in non_features]
    
    print(f"Total feature columns: {len(feat_cols_all)}")
    
    # Define feature groups
    groups = define_feature_groups(feat_cols_all)
    
    print("\nFeature groups:")
    for name, cols in groups.items():
        print(f"  {name}: {len(cols)} columns")
    
    # Build ablation settings
    ablation_settings = [("BASELINE_all_features", [])]
    for gname, gcols in groups.items():
        if len(gcols) > 0:
            ablation_settings.append((f"DROP_{gname}", gcols))
    
    # Run ablations
    results = []
    
    for setting_name, drop_cols in tqdm(ablation_settings, desc="Running ablations"):
        keep_cols = [c for c in feat_cols_all if c not in set(drop_cols)]
        
        if len(keep_cols) == 0:
            print(f"⚠️ {setting_name}: no features left, skipping")
            continue
        
        # Prepare data
        X = cand_df[keep_cols].astype(float).fillna(0.0).values
        y = cand_df[y_col].astype(int).values
        groups_arr = cand_df[seq_col].values
        
        # OOF prediction
        proba_oof = cv_predict_proba(X, y, groups_arr, n_splits, seed)
        
        cand_tmp = cand_df[[seq_col, "cp", y_col]].copy()
        cand_tmp["proba_oof"] = proba_oof
        
        # Sequence-level scores
        scores_earliest = sequence_scores_from_candidates(cand_tmp, all_seq_ids, "proba_oof", seq_col, "cp", "earliest")
        scores_max = sequence_scores_from_candidates(cand_tmp, all_seq_ids, "proba_oof", seq_col, "cp", "max")
        
        y_score_earliest = np.array([scores_earliest[sid] for sid in all_seq_ids])
        y_score_max = np.array([scores_max[sid] for sid in all_seq_ids])
        
        # Compute metrics
        m_earliest = best_f1_and_ap(y_true_seq, y_score_earliest)
        m_max = best_f1_and_ap(y_true_seq, y_score_max)
        
        results.append({
            "setting": setting_name,
            "dropped_n_cols": len(drop_cols),
            "kept_n_cols": len(keep_cols),
            "AP_earliest": m_earliest["AP"],
            "bestF1_earliest": m_earliest["best_F1"],
            "thr_earliest": m_earliest["best_threshold"],
            "P_earliest": m_earliest["best_P"],
            "R_earliest": m_earliest["best_R"],
            "AP_max": m_max["AP"],
            "bestF1_max": m_max["best_F1"],
            "thr_max": m_max["best_threshold"],
            "P_max": m_max["best_P"],
            "R_max": m_max["best_R"],
        })
    
    return pd.DataFrame(results).sort_values("bestF1_earliest", ascending=False)


def analyze_ablation_results(results_df: pd.DataFrame) -> Dict[str, any]:
    """
    Analyze ablation results to identify important feature groups.
    
    Args:
        results_df: Ablation results DataFrame
        
    Returns:
        Analysis dict
    """
    baseline = results_df[results_df["setting"] == "BASELINE_all_features"].iloc[0]
    baseline_ap = baseline["AP_earliest"]
    baseline_f1 = baseline["bestF1_earliest"]
    
    analysis = {
        "baseline_AP": baseline_ap,
        "baseline_F1": baseline_f1,
        "feature_importance": []
    }
    
    for _, row in results_df.iterrows():
        if row["setting"] == "BASELINE_all_features":
            continue
        
        ap_delta = row["AP_earliest"] - baseline_ap
        f1_delta = row["bestF1_earliest"] - baseline_f1
        
        # Negative delta = removing hurts performance = feature is important
        importance = -ap_delta
        
        analysis["feature_importance"].append({
            "group": row["setting"].replace("DROP_", ""),
            "n_features": row["dropped_n_cols"],
            "AP_delta": ap_delta,
            "F1_delta": f1_delta,
            "importance_score": importance
        })
    
    # Sort by importance (highest first)
    analysis["feature_importance"].sort(key=lambda x: x["importance_score"], reverse=True)
    
    return analysis


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run_e8_ablation(cand_df: pd.DataFrame,
                    df_main: pd.DataFrame,
                    seq_col: str = "sequence_id",
                    label_col: str = "hallu_label",
                    time_col: str = "sentence_index",
                    output_dir: Optional[Path] = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Run E8 feature ablation analysis.
    
    Args:
        cand_df: Candidate DataFrame with features
        df_main: Main DataFrame for onset map
        seq_col: Sequence ID column
        label_col: Label column
        time_col: Time column
        output_dir: Output directory
        
    Returns:
        Tuple of (results_df, analysis_dict)
    """
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("E8: FEATURE ABLATION ANALYSIS")
    print("="*60)
    
    # Run ablation
    results_df = run_ablation(cand_df, df_main, seq_col, label_col, time_col)
    
    # Analyze results
    analysis = analyze_ablation_results(results_df)
    
    # Print results
    print("\n" + "="*60)
    print("ABLATION RESULTS")
    print("="*60)
    print(f"Baseline AP: {analysis['baseline_AP']:.4f}")
    print(f"Baseline F1: {analysis['baseline_F1']:.4f}")
    
    print("\nFeature Group Importance (sorted by impact):")
    print("-" * 50)
    for item in analysis["feature_importance"]:
        direction = "↓" if item["AP_delta"] < 0 else "↑"
        print(f"  {item['group']:20s}: AP {direction} {abs(item['AP_delta']):.4f} "
              f"(n_features={item['n_features']})")
    
    # Save results
    if output_dir:
        results_df.to_csv(output_dir / "E8_ablation_summary.csv", index=False)
        
        # Save analysis
        import json
        with open(output_dir / "E8_analysis.json", "w") as f:
            json.dump(analysis, f, indent=2)
        
        print(f"\nResults saved to {output_dir}")
    
    return results_df, analysis


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run E8 Feature Ablation")
    parser.add_argument("--candidates", required=True, help="Path to candidate features CSV")
    parser.add_argument("--data", required=True, help="Path to main data CSV")
    parser.add_argument("--output", default="./output/E8", help="Output directory")
    
    args = parser.parse_args()
    
    print(f"Loading candidates from {args.candidates}")
    cand_df = pd.read_csv(args.candidates)
    
    print(f"Loading main data from {args.data}")
    df_main = pd.read_csv(args.data)
    
    run_e8_ablation(cand_df, df_main, output_dir=Path(args.output))


if __name__ == "__main__":
    main()
