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

from step_4_cpd.analysis_utils import UNKNOWN_VALUE


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
            # Extract domain from filename, e.g., answer_hallu_scored_01_nq_closed_book.csv -> nq_closed_book
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
    and faceted heatmaps for various metrics.
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    print(f"Generating advanced visualizations in: {output_path}")

    # Define metrics for line plots and heatmaps
    line_plot_metrics = [
        "avg_eff_rank_per_layer", "avg_entropy_per_layer",
        "avg_head_similarity_per_layer", "avg_self_attn_per_layer"
    ]
    heatmap_metrics = [
        "effective_rank", "entropy", "head_similarity", "attention_ratio"
    ]
    domains = features_df['domain'].unique()

    # 1. Per-Layer Line Graphs
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
        plt.savefig(plot_filename)
        plt.close()
        print(f"Saved line plot: {plot_filename}")

    # 2. Faceted Heatmaps
    for metric in heatmap_metrics:
        for domain in domains:
            domain_df = features_df[features_df['domain'] == domain]
            
            # Create a FacetGrid for Hallucination vs. Non-Hallucination
            g = sns.FacetGrid(domain_df, col='is_hallucination', height=6)
            
            # Define a helper function to draw the heatmap
            def draw_heatmap(data, **kwargs):
                # Pivot data to create a matrix for the heatmap
                # Assuming 'layer' and 'head' columns exist for heatmap plotting
                # This part is complex and may need adjustment based on actual data structure
                # For now, we'll assume a placeholder structure or skip if not possible
                if 'layer' in data.columns and 'head' in data.columns:
                    pivot_data = data.pivot_table(index='layer', columns='head', values=metric, aggfunc='mean')
                    sns.heatmap(pivot_data, **kwargs)
                else:
                    print(f"Skipping heatmap for '{metric}' in domain '{domain}': 'layer' or 'head' columns not found.")

            g.map_dataframe(draw_heatmap, cmap='viridis')
            g.set_titles(f"{domain} - {{col_name}}")
            g.set_axis_labels("Head", "Layer")
            plt.suptitle(f"Heatmap of {metric.replace('_', ' ').title()} for {domain}", y=1.02)

            plot_filename = output_path / f"{domain}_{metric}_heatmap.png"
            plt.savefig(plot_filename)
            plt.close()
            print(f"Saved heatmap: {plot_filename}")


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
    if 'true_change_point' in all_features:
        all_features.remove('true_change_point')

    # Define groups based on names
    scalar_features = [c for c in all_features if any(k in c.lower() for k in ['binary', 'position', 'density', 'score', 'ratio', 'index', 'overlap', 'alignment'])]
    attention_features = [c for c in all_features if any(k in c.lower() for k in ['entropy', 'rank', 'similarity', 'attention', 'mass', 'max', 'self', 'head', 'layer'])]
    feature_sets = {
        'scalars': scalar_features,
        'attention': attention_features,
        'combined': all_features
    }

    for set_name, feats in feature_sets.items():
        print(f"Running CPD with feature set: {set_name}")
        run_cpd_on_features(data_dir, output_dir, config, overwrite, feats, set_name)


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
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df


def smooth_features(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = uniform_filter1d(df[col].values, size=window, mode='nearest')
    return df


def apply_pca(df: pd.DataFrame, n_components: int = 10) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(df[numeric_cols])
    pca_cols = [f'pca_{i}' for i in range(n_components)]
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
    for file_path in sequence_files:
        features_df = load_features_from_csv([file_path])
        if features_df.empty:
            continue

        # Normalize
        features_df = normalize_features(features_df, method='robust')

        # Smooth
        features_df = smooth_features(features_df, window=3)

        # Select features
        available_feats = [c for c in feature_cols if c in features_df.columns and pd.api.types.is_numeric_dtype(features_df[c])]
        if not available_feats:
            print(f"No available features for {file_path}")
            continue

        # PCA if many
        if len(available_feats) > 10:
            features_df = apply_pca(features_df, n_components=10)
            available_feats = [f'pca_{i}' for i in range(10)]

        signal = features_df[available_feats].values
        signal = np.nan_to_num(signal, nan=0.0).astype(float)
        if signal.shape[0] < 2:
            continue

        # Load true_cp
        true_cp = None
        if 'true_change_point' in features_df.columns:
            true_cp = features_df['true_change_point'].iloc[0]
            if pd.isna(true_cp):
                true_cp = None
            else:
                true_cp = int(true_cp)

        # Run CPD
        algo = rpt.Pelt(model="rbf").fit(signal)
        penalty = np.log(len(signal))
        change_points = algo.predict(pen=penalty)
        detected_cps = [c for c in change_points if c < len(signal)]

        # Compute metrics
        metrics = compute_cpd_metrics(detected_cps, true_cp, tolerance=1)

        sequence_id = Path(file_path).stem
        result = {
            "sequence_id": sequence_id,
            "change_points": str(detected_cps),  # since list
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
    results_df.to_csv(results_file, index=False)
    print(f"Saved CPD results for {len(results_df)} sequences to: {results_file}")

    # Aggregate
    aggregate_results_for_set(output_dir, suffix)


def aggregate_results_for_set(output_dir: str, suffix: str):
    results_file = Path(output_dir) / f"cpd_results_{suffix}.csv"
    if not results_file.exists():
        print("No CPD results file found to aggregate")
        return

    df = pd.read_csv(results_file)
    print(f"Loaded {len(df)} CPD results for {suffix}")

    # Create aggregated summary
    summary = {
        "total_sequences": len(df),
        "avg_num_change_points": df["num_change_points"].mean(),
    }
    if "precision" in df.columns:
        summary["avg_precision"] = df["precision"].mean()
        summary["avg_recall"] = df["recall"].mean()
        summary["avg_f1"] = df["f1"].mean()
        summary["avg_time_to_detection"] = df["time_to_detection"].mean()

    summary_file = Path(output_dir) / f"aggregated_summary_{suffix}.csv"
    pd.DataFrame([summary]).to_csv(summary_file, index=False)
    print(f"Saved aggregated summary to: {summary_file}")