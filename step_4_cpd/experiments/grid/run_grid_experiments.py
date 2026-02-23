"""
Grid Search Experiments for CPD
================================
Systematic evaluation across all configurations.

This script implements the nested loop logic:
for dataset in datasets:
  for evidence_pos in evidence_pos_levels:
    for density in density_levels:
      for interference in interference_levels:
        key = (dataset, evidence_pos, density, interference)
        signal = build_signal_for_key(key)
        bkps = algo.predict(pen=pen)
        summary[key] = summarize_cpds(signal, bkps)
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from itertools import product

import numpy as np
import pandas as pd
import ruptures as rpt
from scipy.ndimage import uniform_filter1d
from sklearn.preprocessing import RobustScaler

# Add parent directory to path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))


def parse_config(config_str: str) -> Dict[str, str]:
    """Parse config string into components."""
    parts = config_str.split('-')
    if len(parts) >= 3:
        return {
            'density': parts[0],
            'interference': parts[1],
            'evidence_pos': parts[2]
        }
    return {}


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    """Normalize signal using RobustScaler."""
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)
    
    signal_var = np.var(signal)
    if signal_var < 1e-15:
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


def run_cpd_on_config(
    df: pd.DataFrame,
    feature_name: str,
    config_key: Tuple,
    min_size: int = 5,
    penalty: float = 1.0
) -> Dict:
    """
    Run CPD on all sequences for a specific configuration.
    
    Args:
        df: Full dataset
        feature_name: Feature to use as signal
        config_key: (dataset, evidence_pos, density, interference)
        min_size: Minimum segment size for PELT
        penalty: Penalty parameter for PELT
    
    Returns:
        Dictionary with detection statistics for this config
    """
    dataset, evidence_pos, density, interference = config_key
    
    # Filter for this config
    mask = (df['domain'] == dataset)
    
    if 'density_level' in df.columns:
        mask &= (df['density_level'] == density)
    if 'evidence_pos_category' in df.columns:
        mask &= (df['evidence_pos_category'] == evidence_pos)
    if 'interference_type' in df.columns:
        mask &= (df['interference_type'] == interference)
    
    filtered_df = df[mask].copy()
    
    if len(filtered_df) == 0:
        return {
            'config_key': str(config_key),
            'dataset': dataset,
            'evidence_pos': evidence_pos,
            'density': density,
            'interference': interference,
            'status': 'no_data',
            'num_sequences': 0
        }
    
    # Group by sequence
    group_cols = ['question_id']
    if 'config_idx' in filtered_df.columns:
        group_cols.append('config_idx')
    
    detections_list = []
    
    for sequence_id, seq_df in filtered_df.groupby(group_cols):
        # Sort by chunk_index
        if 'chunk_index' in seq_df.columns:
            seq_df = seq_df.sort_values('chunk_index').reset_index(drop=True)
        
        # Extract signal
        signal = seq_df[[feature_name]].copy()
        
        # Handle NaN with forward-fill
        signal = signal.fillna(method='ffill').fillna(method='bfill').values.flatten()
        
        if len(signal) < min_size:
            continue
        
        # Normalize, apply EWMA smoothing, then smooth further
        signal = normalize_signal(signal)
        signal = apply_ewma(signal, alpha=0.3)
        signal = smooth_signal(signal, window=3)
        signal = signal.reshape(-1, 1)
        
        # Verify signal shape before PELT
        assert signal.ndim == 2, f"Signal must be 2D, got shape {signal.shape}"
        assert signal.shape[1] >= 1, f"Signal must have at least 1 feature, got {signal.shape[1]}"
        num_chunks, num_features = signal.shape
        
        # Run PELT
        try:
            algo = rpt.Pelt(model="l2", min_size=min_size).fit(signal)
            bkps = algo.predict(pen=penalty)
            detected_cps = [cp for cp in bkps if cp < len(signal)]
        except Exception as e:
            continue
        
        detections_list.append(len(detected_cps))
    
    # Aggregate
    if not detections_list:
        return {
            'config_key': str(config_key),
            'dataset': dataset,
            'evidence_pos': evidence_pos,
            'density': density,
            'interference': interference,
            'status': 'no_sequences',
            'num_sequences': 0
        }
    
    detections_array = np.array(detections_list)
    
    return {
        'config_key': str(config_key),
        'dataset': dataset,
        'evidence_pos': evidence_pos,
        'density': density,
        'interference': interference,
        'status': 'success',
        'num_sequences': len(detections_list),
        'avg_detections': detections_array.mean(),
        'std_detections': detections_array.std(),
        'min_detections': detections_array.min(),
        'max_detections': detections_array.max(),
        'pct_with_detections': (detections_array > 0).sum() / len(detections_array) * 100
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run grid search CPD experiments across all configurations"
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default="../dataset_prep_code/prepared_dataset_cpd_with_attn_metrics_FINAL.csv",
        help="Path to prepared dataset CSV"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../output/grid",
        help="Output directory for results"
    )
    parser.add_argument(
        "--feature",
        type=str,
        default="hallu_score",
        help="Feature to use as signal (default: hallu_score)"
    )
    parser.add_argument(
        "--min_size",
        type=int,
        default=5,
        help="Minimum segment size for PELT (default: 5)"
    )
    parser.add_argument(
        "--penalty",
        type=float,
        default=1.0,
        help="Penalty parameter for PELT (default: 1.0)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("GRID SEARCH EXPERIMENTS - CPD Across All Configurations")
    print("="*70)
    print(f"\nParameters:")
    print(f"  Feature: {args.feature}")
    print(f"  Min Size: {args.min_size}")
    print(f"  Penalty: {args.penalty}")
    
    # Load data
    print(f"\nLoading data from: {args.data_file}")
    df = pd.read_csv(args.data_file)
    print(f"Loaded: {len(df):,} rows x {len(df.columns)} columns")
    
    # Parse config column
    if 'config' in df.columns:
        config_parsed = df['config'].apply(parse_config)
        df['density_level'] = config_parsed.apply(lambda x: x.get('density', ''))
        df['interference_type'] = config_parsed.apply(lambda x: x.get('interference', ''))
        df['evidence_pos_category'] = config_parsed.apply(lambda x: x.get('evidence_pos', ''))
    
    # Extract unique levels
    datasets = sorted(df['domain'].unique().tolist())
    evidence_pos_levels = sorted(df['evidence_pos_category'].unique().tolist())
    density_levels = sorted(df['density_level'].unique().tolist())
    interference_levels = sorted(df['interference_type'].unique().tolist())
    
    print(f"\nConfiguration Levels Found:")
    print(f"  Datasets: {len(datasets)} - {datasets}")
    print(f"  Evidence Positions: {len(evidence_pos_levels)} - {evidence_pos_levels}")
    print(f"  Densities: {len(density_levels)} - {density_levels}")
    print(f"  Interference Types: {len(interference_levels)} - {interference_levels}")
    
    total_configs = len(datasets) * len(evidence_pos_levels) * len(density_levels) * len(interference_levels)
    print(f"\nTotal Configurations: {total_configs}")
    
    # Run grid search
    print("\n" + "="*70)
    print("Running CPD for all configurations...")
    print("="*70)
    
    all_results = []
    config_count = 0
    
    for dataset, evidence_pos, density, interference in product(
        datasets, evidence_pos_levels, density_levels, interference_levels
    ):
        config_key = (dataset, evidence_pos, density, interference)
        config_count += 1
        
        if config_count % 10 == 0 or config_count == 1:
            print(f"\n[{config_count}/{total_configs}] Processing: {dataset} | {evidence_pos} | {density} | {interference}")
        
        result = run_cpd_on_config(
            df,
            args.feature,
            config_key,
            min_size=args.min_size,
            penalty=args.penalty
        )
        all_results.append(result)
    
    # Save results
    results_df = pd.DataFrame(all_results)
    results_file = output_path / "grid_search_results.csv"
    results_df.to_csv(results_file, index=False)
    
    print(f"\n{'='*70}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"Configurations with data: {(results_df['status'] == 'success').sum()}/{len(results_df)}")
    print(f"Configurations with sequences: {(results_df['num_sequences'] > 0).sum()}/{len(results_df)}")
    
    # Show statistics
    successful = results_df[results_df['status'] == 'success']
    if len(successful) > 0:
        print(f"\nDetection Statistics (across successful configs):")
        print(f"  Avg Detections: {successful['avg_detections'].mean():.2f}")
        print(f"  Avg % with Detections: {successful['pct_with_detections'].mean():.1f}%")
        print(f"  Min % with Detections: {successful['pct_with_detections'].min():.1f}%")
        print(f"  Max % with Detections: {successful['pct_with_detections'].max():.1f}%")
        
        # Top configs by detection rate
        print(f"\nTop 10 Configurations by Detection Rate:")
        top_configs = successful.nlargest(10, 'pct_with_detections')
        for idx, row in top_configs.iterrows():
            print(f"  {row['dataset']:25s} | {row['evidence_pos']:5s} | {row['density']:7s} | {row['interference']:3s} | {row['pct_with_detections']:5.1f}%")
        
        # Bottom configs
        print(f"\nBottom 10 Configurations by Detection Rate:")
        bottom_configs = successful.nsmallest(10, 'pct_with_detections')
        for idx, row in bottom_configs.iterrows():
            print(f"  {row['dataset']:25s} | {row['evidence_pos']:5s} | {row['density']:7s} | {row['interference']:3s} | {row['pct_with_detections']:5.1f}%")
    
    print(f"\n[SUCCESS] Results saved to: {results_file}")


if __name__ == "__main__":
    main()
