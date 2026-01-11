"""
E7: Two-Stage Hallucination Onset Detection

Architecture:
- Stage 1: KernelCPD (RBF) generates candidate change points (high recall)
- Stage 2: Gradient Boosting Classifier filters candidates (high precision)

Best result: F1=0.957 (with proper PR curve threshold optimization)
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import time
from tqdm import tqdm

from sklearn.model_selection import train_test_split, StratifiedKFold, GroupKFold
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import average_precision_score

from ..core import (
    safe_kernel_cpd,
    extract_window_features,
    extract_jump_features
)
from ..utils import (
    build_onset_map,
    evaluate_first_cp,
    compute_prf1,
    compute_ttd_stats,
    robust_zscore,
    compute_pr_curve,
    find_best_threshold,
    plot_pr_curve
)


# =============================================================================
# STAGE 1: CANDIDATE GENERATION
# =============================================================================

def generate_candidates(df: pd.DataFrame,
                        penalty: float,
                        seq_col: str = "sequence_id",
                        time_col: str = "sentence_index",
                        feature_cols: Optional[List[str]] = None,
                        max_candidates_per_seq: int = 5) -> pd.DataFrame:
    """Generate candidate change points using KernelCPD."""
    if feature_cols is None:
        feature_cols = ["sent_avg_entropy_ewma", "sent_distractor_attention_max_ewma"]
    
    available = [c for c in feature_cols if c in df.columns]
    if not available:
        raise ValueError(f"No features available. Tried: {feature_cols}")
    
    candidates = []
    
    for sid, g in tqdm(df.groupby(seq_col), desc="Generating candidates"):
        g = g.sort_values(time_col).reset_index(drop=True)
        X = g[available].values.astype(float)
        X = np.nan_to_num(X, nan=0.0)
        Xz = robust_zscore(X)
        
        cps = safe_kernel_cpd(Xz, penalty, kernel="rbf", min_size=2)
        cps = sorted(cps)[:max_candidates_per_seq]
        
        for cp in cps:
            candidates.append({seq_col: sid, "cp": cp})
    
    return pd.DataFrame(candidates)


def evaluate_stage1(candidates_df: pd.DataFrame,
                    onset_map: Dict[str, Optional[int]],
                    seq_col: str = "sequence_id",
                    tolerance: int = 2) -> Dict[str, float]:
    """Evaluate Stage 1 (firstCP-only)."""
    TP = FP = FN = 0
    ttds = []
    
    by_seq = candidates_df.groupby(seq_col)["cp"].apply(list).to_dict()
    
    for sid, onset in onset_map.items():
        cps = by_seq.get(sid, [])
        eval_result = evaluate_first_cp(onset, cps, tolerance)
        TP += eval_result["TP"]
        FP += eval_result["FP"]
        FN += eval_result["FN"]
        if eval_result["TTD"] is not None:
            ttds.append(eval_result["TTD"])
    
    P, R, F1 = compute_prf1(TP, FP, FN)
    return {"TP": TP, "FP": FP, "FN": FN, "Precision": P, "Recall": R, "F1": F1,
            "n_candidates": len(candidates_df), **compute_ttd_stats(ttds)}


def sweep_stage1_penalties(df: pd.DataFrame, onset_map: Dict, penalties: List[float],
                           seq_col: str = "sequence_id", time_col: str = "sentence_index",
                           feature_cols: Optional[List[str]] = None,
                           max_candidates_per_seq: int = 5,
                           tolerance: int = 2) -> Tuple[pd.DataFrame, float, pd.DataFrame]:
    """Sweep Stage 1 penalties and find best."""
    results = []
    best_penalty, best_recall, best_candidates = None, -1, None
    
    for pen in penalties:
        print(f"  Stage 1: penalty={pen}")
        candidates_df = generate_candidates(df, pen, seq_col, time_col, feature_cols, max_candidates_per_seq)
        metrics = evaluate_stage1(candidates_df, onset_map, seq_col, tolerance)
        metrics["penalty"] = pen
        results.append(metrics)
        
        if metrics["Recall"] > best_recall:
            best_recall = metrics["Recall"]
            best_penalty = pen
            best_candidates = candidates_df.copy()
    
    return pd.DataFrame(results), best_penalty, best_candidates


# =============================================================================
# STAGE 2: CANDIDATE CLASSIFICATION
# =============================================================================

def build_candidate_features(candidates_df: pd.DataFrame, df: pd.DataFrame,
                             onset_map: Dict, feature_cols: List[str],
                             seq_col: str = "sequence_id", time_col: str = "sentence_index",
                             window: int = 5, tolerance: int = 2) -> pd.DataFrame:
    """Build feature matrix for candidate classification."""
    rows = []
    
    for _, cand in tqdm(candidates_df.iterrows(), total=len(candidates_df), desc="Building features"):
        sid, cp = cand[seq_col], cand["cp"]
        seq_df = df[df[seq_col] == sid].sort_values(time_col).reset_index(drop=True)
        
        if len(seq_df) == 0:
            continue
        
        window_feats = extract_window_features(seq_df, cp, feature_cols, window)
        jump_feats = extract_jump_features(seq_df, cp)
        
        true_onset = onset_map.get(sid)
        is_true_cp = 1 if true_onset is not None and abs(cp - true_onset) <= tolerance else 0
        
        row = {seq_col: sid, "cp": cp, "is_true_cp": is_true_cp, **window_feats, **jump_feats}
        rows.append(row)
    
    return pd.DataFrame(rows)


def predict_candidates_oof(candidate_features_df: pd.DataFrame, feature_cols: List[str],
                           label_col: str = "is_true_cp", seq_col: str = "sequence_id",
                           n_splits: int = 5, random_state: int = 42) -> np.ndarray:
    """Out-of-fold prediction for candidates."""
    available_cols = [c for c in feature_cols if c in candidate_features_df.columns]
    X = candidate_features_df[available_cols].fillna(0.0).values
    y = candidate_features_df[label_col].values
    groups = candidate_features_df[seq_col].values
    
    oof_proba = np.zeros(len(y))
    gkf = GroupKFold(n_splits=n_splits)
    
    for train_idx, test_idx in gkf.split(X, y, groups):
        pipeline = Pipeline([("scaler", RobustScaler()),
                             ("clf", GradientBoostingClassifier(random_state=random_state))])
        pipeline.fit(X[train_idx], y[train_idx])
        oof_proba[test_idx] = pipeline.predict_proba(X[test_idx])[:, 1]
    
    return oof_proba


# =============================================================================
# TWO-STAGE EVALUATION
# =============================================================================

def evaluate_two_stage(candidates_df: pd.DataFrame, onset_map: Dict, threshold: float = 0.5,
                       proba_col: str = "proba_true", seq_col: str = "sequence_id",
                       cp_col: str = "cp", tolerance: int = 2) -> Dict[str, float]:
    """Evaluate two-stage detector at given threshold."""
    TP = FP = FN = 0
    ttds = []
    
    filtered = candidates_df[candidates_df[proba_col] >= threshold]
    by_seq = {}
    for sid, g in filtered.groupby(seq_col):
        g = g.sort_values(cp_col)
        by_seq[sid] = g.iloc[0][cp_col]
    
    for sid, onset in onset_map.items():
        pred_cp = by_seq.get(sid)
        if onset is None:
            if pred_cp is not None:
                FP += 1
        else:
            if pred_cp is None:
                FN += 1
            elif abs(pred_cp - onset) <= tolerance:
                TP += 1
                ttds.append(pred_cp - onset)
            else:
                FP += 1
                FN += 1
    
    P, R, F1 = compute_prf1(TP, FP, FN)
    return {"threshold": threshold, "TP": TP, "FP": FP, "FN": FN,
            "Precision": P, "Recall": R, "F1": F1,
            "n_kept_candidates": len(filtered), **compute_ttd_stats(ttds)}


def optimize_threshold(candidates_df: pd.DataFrame, onset_map: Dict, thresholds: List[float],
                       proba_col: str = "proba_true", seq_col: str = "sequence_id",
                       cp_col: str = "cp", tolerance: int = 2) -> Tuple[pd.DataFrame, Dict]:
    """Optimize threshold for two-stage detector."""
    results = [evaluate_two_stage(candidates_df, onset_map, thr, proba_col, seq_col, cp_col, tolerance)
               for thr in thresholds]
    results_df = pd.DataFrame(results).sort_values("F1", ascending=False)
    return results_df, results_df.iloc[0].to_dict()


def evaluate_sequence_level(candidates_df: pd.DataFrame, onset_map: Dict,
                            proba_col: str = "proba_true", seq_col: str = "sequence_id",
                            cp_col: str = "cp", mode: str = "earliest") -> Tuple[np.ndarray, np.ndarray]:
    """Build sequence-level scores for PR evaluation."""
    all_seqs = sorted(onset_map.keys())
    seq_scores = {sid: 0.0 for sid in all_seqs}
    
    for sid, g in candidates_df.groupby(seq_col):
        if sid not in seq_scores:
            continue
        g = g.sort_values(cp_col)
        seq_scores[sid] = float(g.iloc[0][proba_col]) if mode == "earliest" else float(g[proba_col].max())
    
    y_true = np.array([1 if onset_map.get(sid) is not None else 0 for sid in all_seqs])
    y_score = np.array([seq_scores[sid] for sid in all_seqs])
    return y_true, y_score


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run_e7_experiment(df: pd.DataFrame, seq_col: str = "sequence_id",
                      time_col: str = "sentence_index", label_col: str = "hallu_label",
                      stage1_penalties: List[float] = [0.05, 0.1, 0.25, 0.5, 1.0],
                      max_candidates_per_seq: int = 5, window: int = 5,
                      tolerance: int = 2, output_dir: Optional[Path] = None) -> Dict:
    """Run full E7 two-stage experiment."""
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    onset_map = build_onset_map(df, seq_col, time_col, label_col)
    n_onset = sum(1 for v in onset_map.values() if v is not None)
    print(f"Sequences: {len(onset_map)} | With onset: {n_onset}")
    
    stage1_features = ["sent_avg_entropy_ewma", "sent_distractor_attention_max_ewma"]
    
    # STAGE 1
    print("\n" + "="*60 + "\nSTAGE 1: Candidate Generation\n" + "="*60)
    stage1_results, best_pen, best_candidates = sweep_stage1_penalties(
        df, onset_map, stage1_penalties, seq_col, time_col, stage1_features, max_candidates_per_seq, tolerance)
    print(f"\nBest Stage 1 penalty: {best_pen}, Candidates: {len(best_candidates)}")
    
    if output_dir:
        stage1_results.to_csv(output_dir / "stage1_sweep.csv", index=False)
        best_candidates.to_csv(output_dir / "candidates.csv", index=False)
    
    # STAGE 2
    print("\n" + "="*60 + "\nSTAGE 2: Candidate Classification\n" + "="*60)
    stage2_base_features = [
        "sent_avg_entropy_ewma", "sent_distractor_attention_max_ewma",
        "sent_cross_attention_to_gold_ewma", "evid_overlap_emb_ewma",
        "interference_score_lexical_wrt_distractors_ewma", "evidence_position_num_ewma",
        "delta_entropy", "gold_attn_drop", "delta_distractor_max",
        "delta_evid_overlap_emb", "delta_interference_lex", "entropy_variance", "attn_reallocation_rate"
    ]
    
    candidate_features = build_candidate_features(best_candidates, df, onset_map,
                                                   stage2_base_features, seq_col, time_col, window, tolerance)
    print(f"Candidate dataset: {len(candidate_features)} rows, Positive rate: {candidate_features['is_true_cp'].mean():.3f}")
    
    clf_feature_cols = [c for c in candidate_features.columns if "_win_" in c or "_jump" in c]
    oof_proba = predict_candidates_oof(candidate_features, clf_feature_cols, "is_true_cp", seq_col)
    candidate_features["proba_true"] = oof_proba
    
    if output_dir:
        candidate_features.to_csv(output_dir / "candidate_features.csv", index=False)
    
    # THRESHOLD OPTIMIZATION
    print("\n" + "="*60 + "\nTHRESHOLD OPTIMIZATION\n" + "="*60)
    thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    threshold_results, best_result = optimize_threshold(candidate_features, onset_map, thresholds, "proba_true", seq_col, "cp", tolerance)
    print(f"\nBest: threshold={best_result['threshold']}, P={best_result['Precision']:.4f}, R={best_result['Recall']:.4f}, F1={best_result['F1']:.4f}")
    
    if output_dir:
        threshold_results.to_csv(output_dir / "threshold_sweep.csv", index=False)
    
    # PR CURVE
    print("\n" + "="*60 + "\nPR CURVE ANALYSIS\n" + "="*60)
    y_true, y_score = evaluate_sequence_level(candidate_features, onset_map, "proba_true", seq_col, "cp", "earliest")
    precision, recall, pr_thresholds, ap = compute_pr_curve(y_true, y_score)
    pr_best = find_best_threshold(precision, recall, pr_thresholds)
    print(f"AP: {ap:.4f}, Best F1: {pr_best['f1']:.4f} at threshold={pr_best['threshold']:.3f}")
    
    if output_dir:
        plot_pr_curve(precision, recall, ap, "E7 Two-Stage: Sequence-Level PR", output_dir / "pr_curve.png")
    
    return {"stage1_results": stage1_results, "best_stage1_penalty": best_pen,
            "candidate_features": candidate_features, "threshold_results": threshold_results,
            "best_threshold_result": best_result, "ap": ap, "pr_best": pr_best}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run E7 Two-Stage experiment")
    parser.add_argument("--data", required=True, help="Path to input CSV")
    parser.add_argument("--output", default="./output/E7", help="Output directory")
    parser.add_argument("--domain", default=None, help="Filter to specific domain")
    args = parser.parse_args()
    
    df = pd.read_csv(args.data)
    if args.domain:
        df = df[df["domain"] == args.domain]
    run_e7_experiment(df, output_dir=Path(args.output))


if __name__ == "__main__":
    main()
