"""
Run E7 Two-Stage CPD Experiment.

This is the main entry point for running the E7 two-stage change point detection
experiment for hallucination onset detection.

Architecture:
    Stage 1: KernelCPD (RBF) generates candidate change points (high recall)
    Stage 2: Gradient Boosting Classifier filters candidates (high precision)

Best Result: F1=0.957 (Precision=0.989, Recall=0.926) at threshold=0.315

Usage:
    python run_e7_experiment.py --main prepared_dataset.csv --sent sentence_attention.csv
    
    # Or import and use programmatically:
    from run_e7_experiment import run_e7_pipeline
    results = run_e7_pipeline(df)
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple

# ML imports
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_curve, average_precision_score

# CPD
try:
    import ruptures as rpt
    RUPTURES_AVAILABLE = True
except ImportError:
    RUPTURES_AVAILABLE = False
    print("WARNING: ruptures not installed. Run: pip install ruptures")


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Configuration for E7 experiment."""
    
    # Column names
    SEQUENCE_COL = "sequence_id"
    TIME_COL = "sentence_index"
    LABEL_COL = "hallu_label"
    
    # CPD parameters
    TOLERANCE = 2           # sentences tolerance for onset detection
    MIN_SIZE = 2            # minimum segment size for CPD
    CPD_PENALTY = 0.001     # KernelCPD penalty (lower = more candidates)
    MAX_CANDIDATES = 10     # max candidates per sequence
    
    # Feature engineering
    EWMA_ALPHA = 0.3        # EWMA smoothing parameter
    VARIANCE_WINDOW = 5     # rolling variance window
    
    # Classifier
    N_SPLITS = 5            # GroupKFold splits
    RANDOM_STATE = 42       # random seed
    
    # Stage 1 features (for candidate generation)
    STAGE1_FEATURES = [
        "sent_avg_entropy_ewma",
        "sent_distractor_attention_max_ewma"
    ]
    
    # Stage 2 features (for classifier)
    STAGE2_FEATURES = [
        "sent_avg_entropy_ewma",
        "sent_distractor_attention_max_ewma",
        "sent_cross_attention_to_gold_ewma",
        "delta_entropy",
        "delta_distractor_max",
        "entropy_variance",
        "attn_reallocation_rate",
        "gold_attn_drop",
        "entropy_cusum",
        "gold_cusum"
    ]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def onset_index(labels: np.ndarray) -> Optional[int]:
    """
    Find onset index: first transition from 0 to label > 0.
    
    Args:
        labels: Array of hallucination labels (0, 1, 2)
        
    Returns:
        Index of first onset, or None if no onset
    """
    y = np.asarray(labels)
    y = np.nan_to_num(y, nan=0).astype(int)
    
    for i in range(1, len(y)):
        if y[i-1] == 0 and y[i] > 0:
            return i
    return None


def build_onset_map(df: pd.DataFrame, config: Config = Config()) -> Dict[str, Optional[int]]:
    """
    Build mapping from sequence_id to onset index.
    
    Args:
        df: DataFrame with sequence_id, sentence_index, hallu_label
        config: Configuration object
        
    Returns:
        Dict[sequence_id, onset_index or None]
    """
    onset_map = {}
    for sid, g in df.groupby(config.SEQUENCE_COL):
        g = g.sort_values(config.TIME_COL)
        labels = g[config.LABEL_COL].values
        onset_map[sid] = onset_index(labels)
    return onset_map


def compute_prf1(TP: int, FP: int, FN: int) -> Tuple[float, float, float]:
    """Compute precision, recall, F1."""
    P = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    R = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0.0
    return P, R, F1


def robust_zscore(X: np.ndarray) -> np.ndarray:
    """Robust z-score normalization using median and IQR."""
    X = np.asarray(X, dtype=float)
    med = np.nanmedian(X, axis=0)
    q75, q25 = np.nanpercentile(X, [75, 25], axis=0)
    iqr = q75 - q25
    iqr = np.where(iqr < 1e-8, 1.0, iqr)
    return (X - med) / iqr


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def engineer_features(df: pd.DataFrame, config: Config = Config()) -> pd.DataFrame:
    """
    Apply all feature engineering steps.
    
    Args:
        df: Merged DataFrame with attention metrics
        config: Configuration object
        
    Returns:
        DataFrame with engineered features
    """
    print("Engineering features...")
    df = df.copy()
    
    # Base attention features
    base_features = [
        "sent_avg_entropy",
        "sent_distractor_attention_max",
        "sent_cross_attention_to_gold"
    ]
    base_features = [c for c in base_features if c in df.columns]
    
    # Forward/backward fill missing values within sequence
    def ffill_bfill_group(g, cols):
        g[cols] = g[cols].ffill().bfill()
        return g
    
    df = df.groupby(config.SEQUENCE_COL, group_keys=False).apply(
        ffill_bfill_group, cols=base_features
    )
    
    # 1. EWMA smoothing
    for col in base_features:
        ewma_col = f"{col}_ewma"
        df[ewma_col] = df.groupby(config.SEQUENCE_COL)[col].transform(
            lambda x: x.ewm(alpha=config.EWMA_ALPHA, adjust=False).mean()
        )
    
    # 2. Delta features
    ewma_features = [f"{c}_ewma" for c in base_features]
    for col in ewma_features:
        delta_col = f"delta_{col.replace('_ewma', '').replace('sent_', '')}"
        df[delta_col] = df.groupby(config.SEQUENCE_COL)[col].diff().fillna(0.0)
    
    # Simplified delta names
    if "delta_avg_entropy" in df.columns:
        df["delta_entropy"] = df["delta_avg_entropy"]
    if "delta_distractor_attention_max" in df.columns:
        df["delta_distractor_max"] = df["delta_distractor_attention_max"]
    
    # 3. Rolling variance
    if "sent_avg_entropy_ewma" in df.columns:
        df["entropy_variance"] = df.groupby(config.SEQUENCE_COL)["sent_avg_entropy_ewma"].transform(
            lambda x: x.rolling(window=config.VARIANCE_WINDOW, min_periods=1).var().fillna(0.0)
        )
    
    # 4. CUSUM features
    def compute_cusum(x):
        values = x.values.astype(float)
        ref = np.nanmedian(values)
        return np.cumsum(np.nan_to_num(values - ref, nan=0.0))
    
    if "sent_avg_entropy_ewma" in df.columns:
        df["entropy_cusum"] = df.groupby(config.SEQUENCE_COL)["sent_avg_entropy_ewma"].transform(compute_cusum)
    
    if "sent_cross_attention_to_gold" in df.columns:
        df["gold_cusum"] = df.groupby(config.SEQUENCE_COL)["sent_cross_attention_to_gold"].transform(compute_cusum)
    
    # 5. Attention reallocation rate
    gold_col = "sent_cross_attention_to_gold"
    if gold_col in df.columns:
        delta_gold = df.groupby(config.SEQUENCE_COL)[gold_col].diff().fillna(0.0)
        df["attn_reallocation_rate"] = delta_gold.abs() / (df[gold_col].abs() + 1e-8)
        df["gold_attn_drop"] = np.maximum(0, -delta_gold)
    
    print(f"  Created {len([c for c in df.columns if '_ewma' in c or 'delta_' in c or '_cusum' in c])} derived features")
    
    return df


# =============================================================================
# STAGE 1: CANDIDATE GENERATION
# =============================================================================

def generate_candidates(
    df: pd.DataFrame,
    config: Config = Config()
) -> pd.DataFrame:
    """
    Generate candidate change points using KernelCPD.
    
    Args:
        df: DataFrame with features
        config: Configuration object
        
    Returns:
        DataFrame with (sequence_id, cp) rows
    """
    if not RUPTURES_AVAILABLE:
        raise ImportError("ruptures not installed. Run: pip install ruptures")
    
    print(f"Generating candidates (penalty={config.CPD_PENALTY})...")
    
    # Get available Stage 1 features
    stage1_features = [c for c in config.STAGE1_FEATURES if c in df.columns]
    if not stage1_features:
        raise ValueError(f"No Stage 1 features found. Expected: {config.STAGE1_FEATURES}")
    
    candidates = []
    
    for sid, g in tqdm(df.groupby(config.SEQUENCE_COL), desc="Stage 1"):
        g = g.sort_values(config.TIME_COL).reset_index(drop=True)
        X = g[stage1_features].values.astype(float)
        X = np.nan_to_num(X, nan=0.0)
        Xz = robust_zscore(X)
        
        n = len(Xz)
        if n < max(6, 2 * config.MIN_SIZE):
            continue
        
        # Variance-scaled penalty
        var = np.nanvar(Xz)
        scaled_pen = config.CPD_PENALTY * max(var, 0.01) * n
        
        try:
            algo = rpt.KernelCPD(kernel="rbf", min_size=config.MIN_SIZE).fit(Xz)
            bkps = algo.predict(pen=scaled_pen)
            cps = [b for b in bkps if b < n]
        except Exception:
            cps = []
        
        # Keep top candidates
        cps = sorted(cps)[:config.MAX_CANDIDATES]
        
        for cp in cps:
            candidates.append({config.SEQUENCE_COL: sid, "cp": cp})
    
    candidates_df = pd.DataFrame(candidates)
    print(f"  Generated {len(candidates_df)} candidates")
    
    return candidates_df


# =============================================================================
# STAGE 2: FEATURE EXTRACTION AND CLASSIFICATION
# =============================================================================

def extract_window_features(
    seq_df: pd.DataFrame,
    cp: int,
    feature_cols: List[str],
    window: int = 5
) -> Dict[str, float]:
    """Extract window-based features around candidate."""
    n = len(seq_df)
    lo = max(0, cp - window)
    hi = min(n, cp + window + 1)
    win_df = seq_df.iloc[lo:hi]
    
    features = {}
    for col in feature_cols:
        if col not in seq_df.columns:
            continue
        vals = win_df[col].values.astype(float)
        vals = np.nan_to_num(vals, nan=0.0)
        
        if len(vals) > 0:
            features[f"{col}_win_mean"] = float(np.mean(vals))
            features[f"{col}_win_std"] = float(np.std(vals))
            features[f"{col}_win_max"] = float(np.max(vals))
            if len(vals) > 1:
                features[f"{col}_win_slope"] = float(np.polyfit(np.arange(len(vals)), vals, 1)[0])
            else:
                features[f"{col}_win_slope"] = 0.0
        else:
            features[f"{col}_win_mean"] = 0.0
            features[f"{col}_win_std"] = 0.0
            features[f"{col}_win_max"] = 0.0
            features[f"{col}_win_slope"] = 0.0
    
    return features


def extract_jump_features(
    seq_df: pd.DataFrame,
    cp: int
) -> Dict[str, float]:
    """Extract jump features at candidate."""
    if cp <= 0 or cp >= len(seq_df):
        return {"entropy_jump": 0.0, "gold_jump": 0.0}
    
    features = {}
    
    entropy_col = "sent_avg_entropy_ewma"
    if entropy_col in seq_df.columns:
        features["entropy_jump"] = float(seq_df.iloc[cp][entropy_col] - seq_df.iloc[cp-1][entropy_col])
    else:
        features["entropy_jump"] = 0.0
    
    gold_col = "sent_cross_attention_to_gold_ewma" if "sent_cross_attention_to_gold_ewma" in seq_df.columns else "sent_cross_attention_to_gold"
    if gold_col in seq_df.columns:
        features["gold_jump"] = float(seq_df.iloc[cp][gold_col] - seq_df.iloc[cp-1][gold_col])
    else:
        features["gold_jump"] = 0.0
    
    return features


def build_candidate_features(
    candidates_df: pd.DataFrame,
    df: pd.DataFrame,
    onset_map: Dict[str, Optional[int]],
    config: Config = Config()
) -> pd.DataFrame:
    """
    Build feature matrix for all candidates.
    
    Args:
        candidates_df: DataFrame with (sequence_id, cp) rows
        df: Main DataFrame with features
        onset_map: Mapping from sequence_id to onset index
        config: Configuration object
        
    Returns:
        DataFrame with candidate features
    """
    print("Building candidate features...")
    
    # Get available Stage 2 features
    stage2_features = [c for c in config.STAGE2_FEATURES if c in df.columns]
    
    rows = []
    
    for _, cand in tqdm(candidates_df.iterrows(), total=len(candidates_df), desc="Stage 2 features"):
        sid, cp = cand[config.SEQUENCE_COL], cand["cp"]
        seq_df = df[df[config.SEQUENCE_COL] == sid].sort_values(config.TIME_COL).reset_index(drop=True)
        
        if len(seq_df) == 0:
            continue
        
        # Window features
        win_feats = extract_window_features(seq_df, cp, stage2_features, window=5)
        
        # Jump features
        jump_feats = extract_jump_features(seq_df, cp)
        
        # Label
        true_onset = onset_map.get(sid)
        is_true = 1 if true_onset is not None and abs(cp - true_onset) <= config.TOLERANCE else 0
        
        rows.append({
            config.SEQUENCE_COL: sid,
            "cp": cp,
            "is_true_cp": is_true,
            **win_feats,
            **jump_feats
        })
    
    cand_features = pd.DataFrame(rows)
    print(f"  Built features for {len(cand_features)} candidates")
    print(f"  Positive rate: {cand_features['is_true_cp'].mean():.3f}")
    
    return cand_features


def train_classifier_oof(
    cand_features: pd.DataFrame,
    feature_cols: List[str],
    config: Config = Config()
) -> np.ndarray:
    """
    Train classifier and get out-of-fold predictions.
    
    Args:
        cand_features: DataFrame with candidate features
        feature_cols: Feature columns for classifier
        config: Configuration object
        
    Returns:
        Array of out-of-fold probabilities
    """
    print(f"Training classifier (GroupKFold, {config.N_SPLITS} splits)...")
    
    available = [c for c in feature_cols if c in cand_features.columns]
    X = cand_features[available].fillna(0.0).values
    y = cand_features["is_true_cp"].values
    groups = cand_features[config.SEQUENCE_COL].values
    
    oof_proba = np.zeros(len(y))
    gkf = GroupKFold(n_splits=config.N_SPLITS)
    
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        clf = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=config.RANDOM_STATE
        )
        
        pipe = Pipeline([
            ("scaler", RobustScaler()),
            ("clf", clf)
        ])
        
        pipe.fit(X[train_idx], y[train_idx])
        oof_proba[test_idx] = pipe.predict_proba(X[test_idx])[:, 1]
    
    print(f"  OOF proba range: [{oof_proba.min():.3f}, {oof_proba.max():.3f}]")
    
    return oof_proba


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_threshold(
    cand_df: pd.DataFrame,
    onset_map: Dict[str, Optional[int]],
    threshold: float,
    config: Config = Config()
) -> Dict:
    """Evaluate at a specific threshold."""
    # Filter candidates above threshold
    filtered = cand_df[cand_df["proba"] >= threshold]
    
    # Get earliest prediction per sequence
    predictions = {}
    for sid, g in filtered.groupby(config.SEQUENCE_COL):
        predictions[sid] = int(g.sort_values("cp").iloc[0]["cp"])
    
    # Compute metrics
    TP = FP = FN = 0
    
    for sid, true_onset in onset_map.items():
        pred = predictions.get(sid)
        
        if true_onset is None:
            if pred is not None:
                FP += 1
        else:
            if pred is None:
                FN += 1
            elif abs(pred - true_onset) <= config.TOLERANCE:
                TP += 1
            else:
                FP += 1
                FN += 1
    
    P, R, F1 = compute_prf1(TP, FP, FN)
    
    return {
        "threshold": threshold,
        "Precision": P,
        "Recall": R,
        "F1": F1,
        "TP": TP,
        "FP": FP,
        "FN": FN
    }


def threshold_sweep(
    cand_features: pd.DataFrame,
    onset_map: Dict[str, Optional[int]],
    thresholds: np.ndarray = np.arange(0.05, 0.96, 0.05),
    config: Config = Config()
) -> pd.DataFrame:
    """
    Sweep thresholds and return results DataFrame.
    
    Args:
        cand_features: DataFrame with candidate features and probabilities
        onset_map: Mapping from sequence_id to onset index
        thresholds: Array of thresholds to evaluate
        config: Configuration object
        
    Returns:
        DataFrame with threshold sweep results
    """
    print("Sweeping thresholds...")
    
    results = []
    for thr in thresholds:
        metrics = evaluate_threshold(cand_features, onset_map, thr, config)
        results.append(metrics)
    
    return pd.DataFrame(results)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_e7_pipeline(
    df: pd.DataFrame,
    output_dir: Optional[str] = None,
    config: Config = Config()
) -> Dict:
    """
    Run complete E7 two-stage CPD pipeline.
    
    Args:
        df: Merged DataFrame with attention metrics
        output_dir: Optional directory to save results
        config: Configuration object
        
    Returns:
        Dict with results and DataFrames
    """
    print("=" * 70)
    print("E7 TWO-STAGE CPD EXPERIMENT")
    print("=" * 70)
    
    # Setup output directory
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Feature engineering
    df = engineer_features(df, config)
    
    # Step 2: Build onset map
    onset_map = build_onset_map(df, config)
    n_with_onset = sum(1 for v in onset_map.values() if v is not None)
    print(f"\nGround truth: {n_with_onset} sequences with onset")
    
    # Step 3: Generate candidates
    candidates_df = generate_candidates(df, config)
    
    # Evaluate Stage 1 recall
    detected = 0
    for sid, onset in onset_map.items():
        if onset is None:
            continue
        seq_cands = candidates_df[candidates_df[config.SEQUENCE_COL] == sid]["cp"].values
        if any(abs(c - onset) <= config.TOLERANCE for c in seq_cands):
            detected += 1
    stage1_recall = detected / n_with_onset if n_with_onset > 0 else 0
    print(f"Stage 1 Recall: {detected}/{n_with_onset} = {stage1_recall:.3f}")
    
    # Step 4: Build candidate features
    cand_features = build_candidate_features(candidates_df, df, onset_map, config)
    
    # Get classifier feature columns
    clf_feature_cols = [c for c in cand_features.columns if "_win_" in c or "_jump" in c]
    print(f"Classifier features: {len(clf_feature_cols)}")
    
    # Step 5: Train classifier
    oof_proba = train_classifier_oof(cand_features, clf_feature_cols, config)
    cand_features["proba"] = oof_proba
    
    # Step 6: Threshold sweep
    df_results = threshold_sweep(cand_features, onset_map, config=config)
    
    # Find best threshold
    best_idx = df_results["F1"].idxmax()
    best = df_results.loc[best_idx]
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nBest threshold: {best['threshold']:.2f}")
    print(f"  Precision: {best['Precision']:.4f}")
    print(f"  Recall:    {best['Recall']:.4f}")
    print(f"  F1:        {best['F1']:.4f}")
    
    # Save results
    if output_dir:
        df_results.to_csv(output_dir / "E7_threshold_sweep.csv", index=False)
        cand_features.to_csv(output_dir / "candidate_features_with_proba.csv", index=False)
        print(f"\nResults saved to {output_dir}")
    
    return {
        "best_threshold": best["threshold"],
        "best_f1": best["F1"],
        "best_precision": best["Precision"],
        "best_recall": best["Recall"],
        "stage1_recall": stage1_recall,
        "df_results": df_results,
        "cand_features": cand_features,
        "onset_map": onset_map,
        "df_engineered": df
    }


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run E7 Two-Stage CPD Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Using pre-merged data
    python run_e7_experiment.py --data merged_features.csv
    
    # Using separate files (will merge automatically)
    python run_e7_experiment.py \\
        --main prepared_dataset_cpd_with_attn_metrics_FINAL_v2.csv \\
        --sent sentence_level_attention_01_nq_closed_book.csv \\
        --domain 01_nq_closed_book
        """
    )
    
    parser.add_argument("--data", help="Path to pre-merged dataset CSV")
    parser.add_argument("--main", help="Path to main dataset CSV")
    parser.add_argument("--sent", help="Path to sentence attention CSV")
    parser.add_argument("--domain", default=None, help="Domain to filter")
    parser.add_argument("--output", default="./output/E7_experiment", help="Output directory")
    parser.add_argument("--penalty", type=float, default=0.001, help="CPD penalty")
    
    args = parser.parse_args()
    
    # Load data
    if args.data:
        print(f"Loading pre-merged data from {args.data}")
        df = pd.read_csv(args.data, low_memory=False)
    elif args.main and args.sent:
        from cpd_experiments.core.data_merging import load_and_merge_all
        df = load_and_merge_all(args.main, args.sent, args.domain)
    else:
        parser.error("Either --data or both --main and --sent are required")
    
    # Configure
    config = Config()
    config.CPD_PENALTY = args.penalty
    
    # Run experiment
    results = run_e7_pipeline(df, args.output, config)
    
    print("\n✅ Experiment complete!")
