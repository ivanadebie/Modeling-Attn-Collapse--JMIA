import os
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ruptures as rpt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from scipy.ndimage import uniform_filter1d

from cpd_code.analysis_utils import UNKNOWN_VALUE


def get_all_sequence_files(data_dir: str, config: str = "") -> List[str]:
    """Get all sequence files from the data directory or single file."""
    data_path = Path(data_dir)
    if data_path.is_file():
        return [str(data_path)]
    
    # Find all .csv data files
    sequence_files = list(data_path.rglob("*.csv"))
    print(f"Found {len(sequence_files)} .csv files in {data_dir}")
    return [str(f) for f in sequence_files]


def load_features_from_csv(file_paths: List[str]) -> pd.DataFrame:
    """
    Load features from a list of .csv files and combine them into a single DataFrame.
    """
    all_features = []
    for file_path in file_paths:
        try:
            df = pd.read_csv(file_path)
            # Don't overwrite domain if it already exists in the CSV
            if 'domain' not in df.columns:
                # Extract domain from filename only if not present
                filename = Path(file_path).stem
                domain = filename.split('_')[-3] + '_' + filename.split('_')[-2] + '_' + filename.split('_')[-1] if len(filename.split('_')) >= 7 else UNKNOWN_VALUE
                df['domain'] = domain
            all_features.append(df)
        except Exception as e:
            print(f"Could not load {file_path}: {e}")
    
    if not all_features:
        return pd.DataFrame()
        
    return pd.concat(all_features, ignore_index=True)


def plot_metric_vs_feature(results_df: pd.DataFrame, output_dir: str):
    """Plot relationships between metrics and features."""
    output_path = Path(output_dir)
    
    # This is a placeholder, as the dummy data doesn't have rich features.
    # In a real scenario, you would plot change-points against input features.
    if "avg_change_point" in results_df.columns and "total_sequences" in results_df.columns:
        print("Skipping metric vs. feature plot: not enough feature columns in placeholder data.")
    else:
        print("Skipping metric vs. feature plot: required columns not found.")


def plot_advanced_visualizations(features_df: pd.DataFrame, output_dir: str):
    """
    Generate a comprehensive set of visualizations including per-layer line graphs
    for entropy metrics only.
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    print(f"Generating advanced visualizations in: {output_path}")

    # Define metrics for line plots only (no heatmaps needed)
    line_plot_metrics = [
        "avg_entropy_per_layer"
    ]
    domains = features_df['domain'].unique()

    # Per-Layer Line Graphs for entropy
    for metric_prefix in line_plot_metrics:
        layer_cols = sorted([c for c in features_df.columns if c.startswith(metric_prefix)])
        if not layer_cols:
            print(f"Skipping line plot for '{metric_prefix}': No data columns found.")
            continue

        # Melt the DataFrame to long format for plotting
        id_vars = ['domain', 'is_hallucination']
        value_vars = layer_cols
        long_df = features_df.melt(id_vars=id_vars, value_vars=value_vars, var_name='layer', value_name=metric_prefix)
        long_df['layer'] = long_df['layer'].str.extract(r'(\d+)').astype(int)

        g = sns.FacetGrid(long_df, col='domain', col_wrap=4, hue='is_hallucination', sharey=False)
        g.map(sns.lineplot, 'layer', metric_prefix, alpha=0.8)
        g.add_legend()
        g.set_titles("{col_name}")
        g.set_axis_labels("Layer", metric_prefix.replace('_', ' ').title())
        plt.suptitle(f"Average {metric_prefix.replace('_', ' ')} per Layer", y=1.02)
        
        plot_filename = output_path / f"{metric_prefix}_per_layer.png"
        plt.savefig(plot_filename, bbox_inches='tight', dpi=100)
        plt.close()
        print(f"Saved line plot: {plot_filename}")


def run_experiments(
    data_dir: str, output_dir: str, config: str = "", overwrite: bool = False
) -> None:
    """
    Run change point detection experiments on all sequences with different feature sets.
    """
    print(f"Running CPD experiments on data from: {data_dir}")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    sequence_files = get_all_sequence_files(data_dir, config)
    if not sequence_files:
        print("No sequence files found for CPD.")
        return

    # Get all features from first file
    features_df = load_features_from_csv([sequence_files[0]])
    if features_df.empty:
        print("No features in first file")
        return

    all_features = features_df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Exclude ID/config/label columns that shouldn't be features
    exclude_cols = ['true_change_point', 'question_id', 'chunk_index', 'config_idx', 'row_index', 
                    'hallucination_label', 'hallucination_score', 'hallu_label', 'model_temperature', 
                    'model_top_p', 'gold_span_start', 'gold_span_end',
                    # Exclude features with 100% NaN or zero variance across sequences
                    'evidence_overlap_ratio', 'distractor_density']
    all_features = [c for c in all_features if c not in exclude_cols]

    # Define scalar features - use ONLY features with good variance (< 10% zero-variance sequences)
    # Good features: normalized_chunk_index, evid_overlap_emb, evidence_position, 
    #                distractor_density_chunk, interference_score, total_response_tokens, 
    #                interference_token_hits, hallu_score
    scalar_features = [c for c in all_features if any(k in c.lower() for k in 
                        ['evidence_position', 'normalized_chunk_index', 
                         'interference_score_lexical', 'evid_overlap_emb', 
                         'total_response_tokens', 'interference_token_hits', 
                         'distractor_density_chunk'])]
    
    attention_features = [c for c in all_features if any(k in c.lower() for k in 
                          ['att_avg_entropy', 'attention_entropy', 'cross_attention_mass', 
                           'distractor_attention_max'])]
    
    # combined includes hallu_score and all good scalar features
    combined_features = scalar_features + attention_features + ['hallu_score']
    combined_features = [c for c in combined_features if c in all_features or c == 'hallu_score']
    
    feature_sets = {
        'scalars': scalar_features,
        'attention': attention_features,
        'combined': combined_features
    }

    for set_name, feats in feature_sets.items():
        print(f"Running CPD with feature set: {set_name}")
        run_cpd_on_features(data_dir, output_dir, config, overwrite, feats, set_name)
    
    # After all feature sets are processed, create aggregated summary
    print("\n" + "="*80)
    aggregate_all_results(output_dir)
    print("\n" + "="*80)
    aggregate_all_results(output_dir)


def aggregate_results(output_dir: str, config: str = ""):
    """Aggregate results from all sequences."""
    print(f"Aggregating results from: {output_dir}")
    
    results_file = Path(output_dir) / "cpd_results.csv"
    if results_file.exists():
        df = pd.read_csv(results_file)
        print(f"Loaded {len(df)} CPD results")
        
        # Create aggregated summary
        summary = {
            "total_sequences": len(df),
            "avg_change_point": df["change_point"].mean(),
            "avg_confidence": df["confidence"].mean()
        }
        
        summary_file = Path(output_dir) / "aggregated_summary.csv"
        pd.DataFrame([summary]).to_csv(summary_file, index=False)
        print(f"Saved aggregated summary to: {summary_file}")
    else:
        print("No CPD results file found to aggregate")


def plot_results(aggregated_df: pd.DataFrame, output_dir: str):
    """Plot key metrics from the aggregated CPD results."""
    output_path = Path(output_dir)
    
    if "avg_f1" in aggregated_df.columns:
        plt.figure(figsize=(10, 6))
        # Since it's single row, plot bars
        metrics = ['avg_precision', 'avg_recall', 'avg_f1', 'avg_time_to_detection']
        values = [aggregated_df[m].iloc[0] for m in metrics if m in aggregated_df.columns]
        labels = [m.replace('avg_', '').replace('_', ' ').title() for m in metrics if m in aggregated_df.columns]
        plt.bar(labels, values)
        plt.title("Average CPD Metrics")
        plt.ylabel("Value")
        plt.savefig(output_path / "cpd_metrics.png")
        plt.close()
        print(f"Saved CPD metrics plot to {output_path}")
    
    if "avg_num_change_points" in aggregated_df.columns:
        plt.figure(figsize=(10, 6))
        plt.bar(['Avg Num Change Points'], [aggregated_df["avg_num_change_points"].iloc[0]])
        plt.title("Average Number of Change Points")
        plt.ylabel("Number")
        plt.savefig(output_path / "num_change_points.png")
        plt.close()
        print(f"Saved num change points plot to {output_path}")


def normalize_features(df: pd.DataFrame, method: str = 'robust') -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if method == 'zscore':
        scaler = StandardScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        return df
    # Handle NaN values before scaling
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    # Remove TRULY constant features (zero or near-zero variance)
    # Use a smaller threshold - features with variance > 1e-15 are considered variable
    # This catches truly constant features (all identical values) but keeps low-variance features
    variances = df[numeric_cols].var()
    variance_threshold = 1e-15  # Very small threshold - only remove exact constants
    non_constant_cols = variances[variances > variance_threshold].index.tolist()
    constant_cols = variances[variances <= variance_threshold].index.tolist()
    
    # Only scale non-constant features
    if len(non_constant_cols) > 0:
        df[non_constant_cols] = scaler.fit_transform(df[non_constant_cols])
        
        # After scaling, check for features that became constant (IQR=0 for RobustScaler)
        # These can happen when all values are identical even if numerically different
        post_scale_variances = df[non_constant_cols].var()
        truly_constant = post_scale_variances[post_scale_variances == 0].index.tolist()
        if truly_constant:
            constant_cols.extend(truly_constant)
            non_constant_cols = [c for c in non_constant_cols if c not in truly_constant]
    
    # Drop constant columns as they provide no information
    if len(constant_cols) > 0:
        df = df.drop(columns=constant_cols)
    
    return df


def smooth_features(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if col not in df.columns:
            continue
        # Fill NaN before smoothing
        df[col] = df[col].fillna(0)
        df[col] = uniform_filter1d(df[col].values, size=window, mode='nearest')
    return df


def apply_pca(df: pd.DataFrame, n_components: int = 10) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    # Fill NaN values before PCA
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    # Check if we have any features at all
    if len(numeric_cols) == 0:
        return df
    
    # Remove constant features before PCA (they have no variance)
    # Use same threshold as normalize_features for consistency
    variances = df[numeric_cols].var()
    variance_threshold = 1e-15  # Only remove exact constants
    non_constant_cols = variances[variances > variance_threshold].index.tolist()
    
    if len(non_constant_cols) == 0:
        # All features are constant, drop them
        df = df.drop(columns=numeric_cols)
        return df
    
    # Adjust n_components to not exceed available dimensions
    n_samples, n_features = df[non_constant_cols].shape
    n_components_actual = min(n_components, n_samples, n_features)
    
    if n_components_actual < 1:
        return df
    
    pca = PCA(n_components=n_components_actual)
    pca_data = pca.fit_transform(df[non_constant_cols])
    pca_cols = [f'pca_{i}' for i in range(n_components_actual)]
    pca_df = pd.DataFrame(pca_data, columns=pca_cols, index=df.index)
    df = pd.concat([df.drop(columns=numeric_cols), pca_df], axis=1)
    return df


def compute_cpd_metrics(detected_cps: List[int], true_cp: Optional[int], tolerance: int = 1) -> Optional[Dict[str, float]]:
    if true_cp is None:
        return None
    # Check if any detected is within tolerance
    within = [d for d in detected_cps if abs(d - true_cp) <= tolerance]
    tp = 1 if within else 0
    fp = len(detected_cps) - tp
    fn = 1 - tp
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    time_to_detection = min(abs(d - true_cp) for d in detected_cps) if detected_cps else float('inf')
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'time_to_detection': time_to_detection
    }



def run_cpd_on_features(data_dir: str, output_dir: str, config: str, overwrite: bool, feature_cols: List[str], suffix: str) -> None:
    output_path = Path(output_dir)
    results_file = output_path / f"cpd_results_{suffix}.csv"
    if results_file.exists() and not overwrite:
        print(f"CPD results already exist at {results_file}. Use --overwrite to regenerate.")
        return

    sequence_files = get_all_sequence_files(data_dir, config)
    cpd_results: List[Dict] = []
    
    # Load the entire dataset
    for file_path in sequence_files:
        full_df = load_features_from_csv([file_path])
        if full_df.empty:
            continue
        
        # Group by domain, question_id AND config_idx to separate sequences
        if 'question_id' not in full_df.columns:
            print(f"Warning: 'question_id' column not found in {file_path}. Treating entire file as one sequence.")
            grouped = [(Path(file_path).stem, full_df)]
        elif 'domain' in full_df.columns and 'config_idx' in full_df.columns:
            # Group by domain, question_id, and config_idx for unique sequences
            grouped = full_df.groupby(['domain', 'question_id', 'config_idx'])
        elif 'config_idx' in full_df.columns:
            # Group by question_id and config_idx
            grouped = full_df.groupby(['question_id', 'config_idx'])
        else:
            # Fallback to just question_id
            grouped = full_df.groupby('question_id')
        
        # Process each sequence
        for sequence_id, features_df in grouped:
            if features_df.empty:
                continue
            
            # Sort by chunk_index (or sentence_index) if available
            if 'chunk_index' in features_df.columns:
                features_df = features_df.sort_values('chunk_index').reset_index(drop=True)
            elif 'sentence_index' in features_df.columns:
                features_df = features_df.sort_values('sentence_index').reset_index(drop=True)
            
            # Store label columns before processing
            label_cols = ['hallucination_label', 'hallu_label', 'true_change_point', 'chunk_index']
            stored_labels = {}
            for col in label_cols:
                if col in features_df.columns:
                    stored_labels[col] = features_df[col].copy()

            # Normalize (this may drop constant features)
            features_df = normalize_features(features_df, method='robust')

            # Smooth
            features_df = smooth_features(features_df, window=3)

            # Select features - check what remains after normalization
            # First try to use the requested feature_cols, but also accept any numeric columns
            available_feats = [c for c in feature_cols if c in features_df.columns and pd.api.types.is_numeric_dtype(features_df[c])]
            
            # If none of the requested features remain, use any available numeric features
            if not available_feats:
                numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
                # Exclude ID/metadata columns
                exclude = ['question_id', 'chunk_index', 'config_idx', 'row_index', 'hallucination_label', 
                          'hallu_label', 'true_change_point']
                available_feats = [c for c in numeric_cols if c not in exclude]
            
            if not available_feats:
                print(f"No available features for sequence {sequence_id}")
                continue

            # Extract feature data for CPD
            feature_data = features_df[available_feats].copy()
            
            # PCA if many features
            if len(available_feats) > 10:
                pca_result = apply_pca(feature_data, n_components=10)
                
                # Check if PCA created components
                pca_feats = [c for c in pca_result.columns if c.startswith('pca_')]
                
                if pca_feats:
                    # PCA succeeded
                    feature_data = pca_result
                    available_feats = pca_feats
                else:
                    # PCA failed, keep using original features
                    print(f"PCA produced no components for sequence {sequence_id}, using original {len(available_feats)} features")

            signal = feature_data[available_feats].values
            signal = np.nan_to_num(signal, nan=0.0).astype(float)
            if signal.shape[0] < 2:
                continue

            # Find true change point from stored labels
            true_cp = None
            hallu_col = None
            for col in ['hallucination_label', 'hallu_label']:
                if col in stored_labels:
                    hallu_col = col
                    break
            
            if hallu_col:
                # Find first index where label == 1
                hallu_series = stored_labels[hallu_col]
                hallu_indices = hallu_series[hallu_series == 1].index
                if len(hallu_indices) > 0:
                    true_cp = hallu_indices[0]

            # Run CPD
            # Use L2 cost model (works better than RBF) with lower penalty for short sequences
            # Penalty = 1.0 works well for detecting hallucination boundaries
            algo = rpt.Pelt(model="l2").fit(signal)
            penalty = 1.0  # Lower penalty to increase sensitivity for short sequences
            change_points = algo.predict(pen=penalty)
            detected_cps = [c for c in change_points if c < len(signal)]

            # Compute metrics
            metrics = compute_cpd_metrics(detected_cps, true_cp, tolerance=1)

            result = {
                "feature_set": suffix,
                "sequence_id": str(sequence_id),
                "change_points": str(detected_cps),
                "num_change_points": len(detected_cps),
                "true_change_point": true_cp,
            }
            if metrics:
                result.update(metrics)
            cpd_results.append(result)

    if not cpd_results:
        print("No CPD results were generated.")
        return

    results_df = pd.DataFrame(cpd_results)
    
    # Append to consolidated file (or create if doesn't exist)
    if results_file.exists() and not overwrite:
        existing_df = pd.read_csv(results_file)
        # Remove existing results for this feature set
        existing_df = existing_df[existing_df['feature_set'] != suffix]
        results_df = pd.concat([existing_df, results_df], ignore_index=True)
    
    results_df.to_csv(results_file, index=False)
    print(f"Saved CPD results for {len(cpd_results)} sequences ({suffix}) to: {results_file}")


def aggregate_all_results(output_dir: str):
    """Create aggregated summary from all feature set results files."""
    output_path = Path(output_dir)
    
    # Look for individual results files
    result_files = {
        'scalars': output_path / "cpd_results_scalars.csv",
        'attention': output_path / "cpd_results_attention.csv",
        'combined': output_path / "cpd_results_combined.csv"
    }
    
    # Load all available results
    all_dfs = []
    for feature_set, file_path in result_files.items():
        if file_path.exists():
            df = pd.read_csv(file_path)
            if 'feature_set' not in df.columns:
                df['feature_set'] = feature_set
            all_dfs.append(df)
            print(f"Loaded {len(df)} results from {file_path.name}")
    
    if not all_dfs:
        print("No CPD results files found to aggregate")
        return
    
    # Combine all results
    df = pd.concat(all_dfs, ignore_index=True)
    print(f"Total: {len(df)} CPD results across {len(all_dfs)} feature sets")

    # Group by feature set and calculate metrics
    summary_rows = []
    for feature_set in df['feature_set'].unique():
        subset = df[df['feature_set'] == feature_set]
        
        summary = {
            "Feature Set": feature_set.capitalize(),
            "Total Sequences": len(subset),
            "Sequences with Ground Truth": subset['true_change_point'].notna().sum(),
            "Avg Change Points Detected": subset["num_change_points"].mean(),
            "Sequences with Detections": (subset["num_change_points"] > 0).sum(),
        }
        
        # Add performance metrics if available
        if "precision" in subset.columns and subset["precision"].notna().sum() > 0:
            summary["Avg Precision"] = subset["precision"].mean()
            summary["Avg Recall"] = subset["recall"].mean()
            summary["Avg F1 Score"] = subset["f1"].mean()
            summary["Avg Time to Detection"] = subset[subset["time_to_detection"] != float('inf')]["time_to_detection"].mean()
        else:
            summary["Avg Precision"] = 0.0
            summary["Avg Recall"] = 0.0
            summary["Avg F1 Score"] = 0.0
            summary["Avg Time to Detection"] = float('inf')
            
        summary_rows.append(summary)
    
    summary_df = pd.DataFrame(summary_rows)
    summary_file = output_path / "aggregated_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    
    print(f"\nAggregated Summary:")
    print(summary_df.to_string(index=False))
    print(f"\nSaved aggregated summary to: {summary_file}")