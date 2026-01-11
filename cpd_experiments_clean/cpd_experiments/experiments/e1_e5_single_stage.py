"""
E1-E5: Single-Stage CPD Experiments

These experiments test various CPD configurations for direct onset detection:
- E1: Penalty sweep with PELT L2 on base features
- E2: Feature differencing (delta features)
- E2.5: Attention level change features (entropy variance, reallocation rate)
- E3: Robust cost using KernelCPD with RBF kernel
- E4: Onset-specific features (delta overlap, delta interference)
- E5: CUSUM features

Best result: E3 with F1=0.618
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import time

from ..core import CPDRunner, safe_pelt, safe_kernel_cpd
from ..utils import (
    build_onset_map,
    evaluate_first_cp,
    compute_prf1,
    compute_ttd_stats,
    robust_zscore,
    ensure_columns
)


# =============================================================================
# EXPERIMENT DEFINITIONS
# =============================================================================

EXPERIMENTS = {
    "E1": {
        "name": "Penalty Sweep (PELT L2)",
        "description": "Base features with PELT L2 cost, penalty sweep",
        "algorithm": "pelt",
        "model": "l2",
        "features": [
            "evidence_position_num_ewma",
            "interference_score_lexical_wrt_distractors_ewma"
        ]
    },
    "E2": {
        "name": "Feature Differencing",
        "description": "Delta features for detecting changes",
        "algorithm": "pelt",
        "model": "l2",
        "features": [
            "delta_entropy",
            "gold_attn_drop",
            "delta_distractor_max"
        ]
    },
    "E2.5": {
        "name": "Attention Level Change",
        "description": "Entropy variance and reallocation rate",
        "algorithm": "pelt",
        "model": "l2",
        "features": [
            "entropy_variance",
            "attn_reallocation_rate"
        ]
    },
    "E3": {
        "name": "Robust Cost (KernelCPD RBF)",
        "description": "Base features with KernelCPD RBF kernel",
        "algorithm": "kernel",
        "model": "rbf",
        "features": [
            "evidence_position_num_ewma",
            "interference_score_lexical_wrt_distractors_ewma"
        ]
    },
    "E4": {
        "name": "Onset Features",
        "description": "Delta evidence overlap and interference",
        "algorithm": "pelt",
        "model": "l2",
        "features": [
            "delta_evid_overlap_emb",
            "delta_interference_lex"
        ]
    },
    "E5": {
        "name": "CUSUM Features",
        "description": "Cumulative sum features",
        "algorithm": "pelt",
        "model": "l2",
        "features": [
            "gold_cusum",
            "entropy_cusum",
            "overlap_cusum"
        ]
    }
}


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def run_single_experiment(df: pd.DataFrame,
                          exp_config: Dict,
                          penalties: List[float],
                          seq_col: str = "sequence_id",
                          time_col: str = "sentence_index",
                          label_col: str = "hallu_label",
                          tolerance: int = 2,
                          min_size: int = 2) -> pd.DataFrame:
    """
    Run a single CPD experiment with penalty sweep.
    
    Args:
        df: Input DataFrame with features
        exp_config: Experiment configuration dict
        penalties: List of penalties to sweep
        seq_col: Sequence ID column
        time_col: Time/index column
        label_col: Label column
        tolerance: Tolerance for matching
        min_size: Minimum segment size for CPD
        
    Returns:
        DataFrame with results per penalty
    """
    # Validate features exist
    features = exp_config["features"]
    available_features = [f for f in features if f in df.columns]
    
    if not available_features:
        print(f"Warning: No features available for {exp_config['name']}")
        return pd.DataFrame()
    
    # Build onset map
    onset_map = build_onset_map(df, seq_col, time_col, label_col)
    
    # Create CPD runner
    runner = CPDRunner(
        algorithm=exp_config["algorithm"],
        model=exp_config["model"],
        min_size=min_size
    )
    
    results = []
    
    for penalty in penalties:
        TP = FP = FN = 0
        ttds = []
        t0 = time.time()
        
        for sid, g in df.groupby(seq_col):
            g = g.sort_values(time_col)
            
            # Extract and normalize features
            X = g[available_features].values.astype(float)
            X = np.nan_to_num(X, nan=0.0)
            Xz = robust_zscore(X)
            
            # Run CPD
            cps = runner.detect(Xz, penalty)
            
            # Evaluate
            true_onset = onset_map.get(sid)
            eval_result = evaluate_first_cp(true_onset, cps, tolerance)
            
            TP += eval_result["TP"]
            FP += eval_result["FP"]
            FN += eval_result["FN"]
            if eval_result["TTD"] is not None:
                ttds.append(eval_result["TTD"])
        
        runtime = time.time() - t0
        P, R, F1 = compute_prf1(TP, FP, FN)
        ttd_stats = compute_ttd_stats(ttds)
        
        results.append({
            "penalty": penalty,
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "Precision": P,
            "Recall": R,
            "F1": F1,
            "runtime_sec": runtime,
            **ttd_stats
        })
    
    return pd.DataFrame(results)


def run_all_experiments(df: pd.DataFrame,
                        penalties: List[float] = [0.05, 0.1, 0.25, 0.5, 1.0],
                        seq_col: str = "sequence_id",
                        time_col: str = "sentence_index",
                        label_col: str = "hallu_label",
                        tolerance: int = 2,
                        output_dir: Optional[Path] = None) -> Dict[str, pd.DataFrame]:
    """
    Run all E1-E5 experiments.
    
    Args:
        df: Input DataFrame with features
        penalties: List of penalties to sweep
        seq_col: Sequence ID column
        time_col: Time/index column
        label_col: Label column
        tolerance: Tolerance for matching
        output_dir: Directory to save results
        
    Returns:
        Dict mapping experiment ID to results DataFrame
    """
    results = {}
    
    # Ensure output directory exists
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    for exp_id, exp_config in EXPERIMENTS.items():
        print(f"\n{'='*60}")
        print(f"Running {exp_id}: {exp_config['name']}")
        print(f"Description: {exp_config['description']}")
        print(f"Features: {exp_config['features']}")
        print(f"{'='*60}")
        
        exp_results = run_single_experiment(
            df, exp_config, penalties,
            seq_col, time_col, label_col, tolerance
        )
        
        if len(exp_results) > 0:
            # Sort by F1
            exp_results = exp_results.sort_values("F1", ascending=False)
            
            # Print best result
            best = exp_results.iloc[0]
            print(f"\nBest result:")
            print(f"  Penalty: {best['penalty']}")
            print(f"  F1: {best['F1']:.4f} (P={best['Precision']:.4f}, R={best['Recall']:.4f})")
            
            # Save if output_dir specified
            if output_dir:
                exp_results.to_csv(output_dir / f"{exp_id}_results.csv", index=False)
        
        results[exp_id] = exp_results
    
    # Create summary
    summary = create_summary(results)
    print("\n" + "="*60)
    print("SUMMARY (Best F1 per experiment)")
    print("="*60)
    print(summary.to_string(index=False))
    
    if output_dir:
        summary.to_csv(output_dir / "E1_E5_summary.csv", index=False)
    
    return results


def create_summary(results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Create summary table from experiment results.
    
    Args:
        results: Dict mapping experiment ID to results DataFrame
        
    Returns:
        Summary DataFrame
    """
    summary_rows = []
    
    for exp_id, df in results.items():
        if len(df) == 0:
            continue
        
        best = df.iloc[0]  # Already sorted by F1
        
        summary_rows.append({
            "ExpID": exp_id,
            "Name": EXPERIMENTS[exp_id]["name"],
            "best_penalty": best["penalty"],
            "Precision": best["Precision"],
            "Recall": best["Recall"],
            "F1": best["F1"],
            "TTD_mean": best.get("TTD_mean", np.nan),
            "TTD_median": best.get("TTD_median", np.nan)
        })
    
    return pd.DataFrame(summary_rows).sort_values("F1", ascending=False)


# =============================================================================
# CLI
# =============================================================================

def main():
    """Main entry point for E1-E5 experiments."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run E1-E5 CPD experiments")
    parser.add_argument("--data", required=True, help="Path to input CSV")
    parser.add_argument("--output", default="./output/E1_E5", help="Output directory")
    parser.add_argument("--domain", default=None, help="Filter to specific domain")
    parser.add_argument("--penalties", nargs="+", type=float, 
                        default=[0.05, 0.1, 0.25, 0.5, 1.0],
                        help="Penalty values to sweep")
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.data}")
    df = pd.read_csv(args.data)
    
    if args.domain:
        df = df[df["domain"] == args.domain]
        print(f"Filtered to domain {args.domain}: {len(df)} rows")
    
    # Run experiments
    results = run_all_experiments(
        df,
        penalties=args.penalties,
        output_dir=Path(args.output)
    )
    
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
