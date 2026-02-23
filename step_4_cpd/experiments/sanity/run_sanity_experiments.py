"""
Sanity Check Experiments for CPD
=================================
Test individual features on a fixed configuration to validate the CPD pipeline.

Fixed Configuration:
- Domain: domain1 (closed_book_qa)
- Distractor Density: Medium
- Interference Type: nonsensical

Features to Test (one at a time):
1. hallu_score (chunk-level hallucination score)
2. distractor_density_chunk
3. evidence_overlap_ratio (or evid_overlap_emb)
4. interference_score_lexical_wrt_distractors
5. evidence_position
6. att_avg_entropy
7. cross_attention_mass_to_gold
8. distractor_attention_max
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import ruptures as rpt
from scipy.ndimage import uniform_filter1d
from sklearn.preprocessing import RobustScaler

# Add parent directory to path for imports
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))


def parse_config(config_str: str) -> Dict[str, str]:
    """
    Parse config string into components.
    Expected format: "density-interference_code-evidence_pos"
    Example: "Medium-N-Beg" where N=nonsensical
    
    Interference codes: N, T, P (need to determine mapping)
    """
    parts = config_str.split('-')
    if len(parts) >= 3:
        return {
            'density': parts[0],
            'interference': parts[1],
            'evidence_pos': parts[2]
        }
    return {}


def load_and_filter_data(
    data_file: str,
    domain: str = "01_nq_closed_book",
    density: str = "Medium",
    interference: str = "N"
) -> pd.DataFrame:
    """
    Load data and filter for the fixed configuration.
    
    Note: The domain format is "01_nq_closed_book" (not "domain1")
    and interference codes are single letters (N, T, P)
    """
    print(f"Loading data from: {data_file}")
    df = pd.read_csv(data_file)
    print(f"Loaded: {len(df):,} rows x {len(df.columns)} columns")
    
    # Parse config column if it exists
    if 'config' in df.columns:
        config_parsed = df['config'].apply(parse_config)
        df['density_level'] = config_parsed.apply(lambda x: x.get('density', ''))
        df['interference_type'] = config_parsed.apply(lambda x: x.get('interference', ''))
        df['evidence_pos_category'] = config_parsed.apply(lambda x: x.get('evidence_pos', ''))
    
    # Filter for fixed configuration
    mask = pd.Series([True] * len(df))
    
    if 'domain' in df.columns:
        mask &= (df['domain'] == domain)
        print(f"Filtered by domain={domain}: {mask.sum():,} rows")
    
    if 'density_level' in df.columns:
        mask &= (df['density_level'] == density)
        print(f"Filtered by density={density}: {mask.sum():,} rows")
    
    if 'interference_type' in df.columns:
        mask &= (df['interference_type'] == interference)
        print(f"Filtered by interference={interference}: {mask.sum():,} rows")
    
    filtered_df = df[mask].copy()
    print(f"\nFinal filtered dataset: {len(filtered_df):,} rows")
    
    if len(filtered_df) == 0:
        print("\nWARNING: No data found for this configuration!")
        print(f"Available domains: {df['domain'].unique() if 'domain' in df.columns else 'N/A'}")
        print(f"Available densities: {df['density_level'].unique() if 'density_level' in df.columns else 'N/A'}")
        print(f"Available interference: {df['interference_type'].unique() if 'interference_type' in df.columns else 'N/A'}")
    
    return filtered_df


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    """Normalize signal using RobustScaler."""
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)
    
    # Check for constant signal
    signal_var = np.var(signal)
    if signal_var < 1e-15:
        print(f"  [WARN] Signal is constant (variance={signal_var:.2e})")
        return signal.flatten()
    
    scaler = RobustScaler()
    normalized = scaler.fit_transform(signal)
    return normalized.flatten()


def smooth_signal(signal: np.ndarray, window: int = 3) -> np.ndarray:
    """Smooth signal using uniform filter."""
    return uniform_filter1d(signal, size=window, mode='nearest')


def apply_ewma(signal: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """Apply Exponential Weighted Moving Average smoothing."""
    return pd.Series(signal).ewm(alpha=alpha, adjust=False).mean().values


def run_cpd_on_feature(
    df: pd.DataFrame,
    feature_name: str,
    min_size: int = 2,
    penalty: float = 0.3
) -> Dict:
    """
    Run PELT CPD on a single feature across all sequences.
    
    Returns summary statistics across all sequences.
    Note: No ground truth evaluation - just testing if features produce detections.
    """
    print(f"\n{'='*60}")
    print(f"Testing Feature: {feature_name}")
    print(f"{'='*60}")
    
    if feature_name not in df.columns:
        print(f"  [ERROR] Column '{feature_name}' not found in data")
        return {
            'feature': feature_name,
            'status': 'missing_column',
            'num_sequences': 0
        }
    
    # Group by sequence (question_id + config_idx)
    if 'question_id' not in df.columns:
        print(f"  [ERROR] 'question_id' column not found")
        return {
            'feature': feature_name,
            'status': 'missing_question_id',
            'num_sequences': 0
        }
    
    group_cols = ['question_id']
    if 'config_idx' in df.columns:
        group_cols.append('config_idx')
    
    # Print feature statistics
    print(f"\n  Feature Statistics:")
    print(f"    Min: {df[feature_name].min():.4f}")
    print(f"    Max: {df[feature_name].max():.4f}")
    print(f"    Mean: {df[feature_name].mean():.4f}")
    print(f"    Std: {df[feature_name].std():.4f}")
    print(f"    Variance: {df[feature_name].var():.4e}")
    print(f"    Non-null: {df[feature_name].notna().sum()}/{len(df)}")
       
    results = []
    
    for sequence_id, seq_df in df.groupby(group_cols):
        # Sort by chunk_index
        if 'chunk_index' in seq_df.columns:
            seq_df = seq_df.sort_values('chunk_index').reset_index(drop=True)
        
        # Extract feature signal
        signal = seq_df[[feature_name]].copy()
        
        # Handle NaN with forward-fill
        signal = signal.fillna(method='ffill').fillna(method='bfill').values.flatten()
        
        if len(signal) < min_size:
            continue
        
        # Normalize, apply EWMA smoothing, then smooth further
        signal = normalize_signal(signal)
        signal = apply_ewma(signal, alpha=0.3)
        signal = smooth_signal(signal, window=3)
        
        # Reshape for PELT
        signal = signal.reshape(-1, 1)
        
        # Run PELT
        try:
            algo = rpt.Pelt(model="l2", min_size=min_size).fit(signal)
            bkps = algo.predict(pen=penalty)
            detected_cps = [cp for cp in bkps if cp < len(signal)]
        except Exception as e:
            print(f"  [WARN] PELT failed for sequence {sequence_id}: {e}")
            continue
        
        results.append({
            'sequence_id': str(sequence_id),
            'num_chunks': len(signal),
            'num_detected': len(detected_cps),
            'detected_cps': str(detected_cps)
        })
    
    # Aggregate results
    if not results:
        print(f"  [ERROR] No sequences processed")
        return {
            'feature': feature_name,
            'status': 'no_sequences',
            'num_sequences': 0
        }
    
    results_df = pd.DataFrame(results)
    
    summary = {
        'feature': feature_name,
        'status': 'success',
        'num_sequences': len(results_df),
        'avg_chunks_per_seq': results_df['num_chunks'].mean(),
        'avg_detections': results_df['num_detected'].mean(),
        'min_detections': results_df['num_detected'].min(),
        'max_detections': results_df['num_detected'].max(),
        'sequences_with_detections': (results_df['num_detected'] > 0).sum(),
        'pct_with_detections': (results_df['num_detected'] > 0).sum() / len(results_df) * 100
    }
    
    # Print summary
    print(f"\n  Sequences Processed: {summary['num_sequences']}")
    print(f"  Avg Chunks/Sequence: {summary['avg_chunks_per_seq']:.1f}")
    print(f"  Avg Detections: {summary['avg_detections']:.2f}")
    print(f"  Detection Range: {summary['min_detections']}-{summary['max_detections']}")
    print(f"  Sequences w/ Detections: {summary['sequences_with_detections']}/{summary['num_sequences']} ({summary['pct_with_detections']:.1f}%)")
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Run sanity check CPD experiments on individual features"
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default=r"c:\Users\ivana\OneDrive\Documents\Modeling-Attn-Collapse--JMIA\step_4_cpd\dataset_prep_code\prepared_dataset_cpd_with_attn_metrics_FINAL.csv",
        help="Path to prepared dataset CSV"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../output/sanity",
        help="Output directory for results"
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="01_nq_closed_book",
        help="Domain to test (default: 01_nq_closed_book)"
    )
    parser.add_argument(
        "--density",
        type=str,
        default="Medium",
        help="Distractor density level (default: Medium)"
    )
    parser.add_argument(
        "--interference",
        type=str,
        default="N",
        help="Interference type code: N/T/P (default: N for nonsensical)"
    )
    parser.add_argument(
        "--min_size",
        type=int,
        default=2,
        help="Minimum segment size for PELT (default: 2 for short sequences)"
    )
    parser.add_argument(
        "--penalty",
        type=float,
        default=0.3,
        help="Penalty parameter for PELT (default: 0.3 for higher sensitivity)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("SANITY CHECK EXPERIMENTS - Individual Feature CPD")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Domain: {args.domain}")
    print(f"  Density: {args.density}")
    print(f"  Interference: {args.interference}")
    print(f"  Min Size: {args.min_size}")
    print(f"  Penalty: {args.penalty}")
    
    # Load and filter data
    df = load_and_filter_data(
        args.data_file,
        domain=args.domain,
        density=args.density,
        interference=args.interference
    )
    
    if len(df) == 0:
        print("\n[ERROR] No data available for this configuration. Exiting.")
        return
    
    # Define features to test
    features_to_test = [
        'distractor_density_chunk',
        'evidence_overlap_ratio',  # Can also try 'evid_overlap_emb'
        'interference_score_lexical_wrt_distractors',
        'evidence_position',
        'att_avg_entropy',
        'cross_attention_mass_to_gold',
        'distractor_attention_max'
    ]
    
    # Run CPD on each feature
    all_results = []
    for feature in features_to_test:
        result = run_cpd_on_feature(
            df,
            feature,
            min_size=args.min_size,
            penalty=args.penalty
        )
        all_results.append(result)
    
    # Save results
    results_df = pd.DataFrame(all_results)
    output_file = output_path / "sanity_check_results.csv"
    results_df.to_csv(output_file, index=False)
    
    print(f"\n{'='*60}")
    print("SUMMARY - All Features")
    print(f"{'='*60}\n")
    print(results_df.to_string(index=False))
    
    print(f"\n[SUCCESS] Results saved to: {output_file}")
    
    # Identify most responsive features
    successful = results_df[results_df['status'] == 'success'].copy()
    if len(successful) > 0:
        successful = successful.sort_values('pct_with_detections', ascending=False)
        print(f"\nMOST RESPONSIVE FEATURES (by % sequences with detections):")
        for idx, row in successful.head(5).iterrows():
            print(f"  {row['feature']}: {row['pct_with_detections']:.1f}% ({row['avg_detections']:.2f} avg detections)")


if __name__ == "__main__":
    main()
