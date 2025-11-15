import os
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ruptures as rpt
import seaborn as sns


def get_all_sequence_files(data_dir: str, config: str = "") -> List[str]:
    """Get all sequence files from the data directory."""
    data_path = Path(data_dir)
    if not data_path.exists():
        return []
    
    # Find all .npz data files
    sequence_files = list(data_path.rglob("*.npz"))
    print(f"Found {len(sequence_files)} .npz files in {data_dir}")
    return [str(f) for f in sequence_files]


def load_features_from_npz(file_paths: List[str]) -> pd.DataFrame:
    """
    Load features from a list of .npz files and combine them into a single DataFrame.
    This function is designed to be robust against inconsistencies in the .npz files,
    such as multi-dimensional arrays that should be 1D.
    """
    all_features = []
    for file_path in file_paths:
        try:
            with np.load(file_path, allow_pickle=True) as data:
                record_data = {}
                # Ensure all arrays are 1-dimensional for DataFrame compatibility
                for key in data.keys():
                    arr = data[key]
                    if hasattr(arr, 'ndim') and arr.ndim > 1:
                        arr = arr.flatten()
                    record_data[key] = arr

                # Before creating a DataFrame, check for consistent lengths
                # This prevents errors if arrays have different numbers of elements
                lengths = [len(v) for v in record_data.values() if hasattr(v, '__len__')]
                if len(set(lengths)) > 1:
                    print(f"Warning: Skipping {file_path} due to inconsistent array lengths.")
                    continue
                
                all_features.append(pd.DataFrame(record_data))

        except Exception as e:
            print(f"Could not load or process {file_path}: {e}")
    
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


def run_cpd_on_all(
    data_dir: str, output_dir: str, config: str = "", overwrite: bool = False
) -> None:
    """
    Run change point detection on all sequences using the Pelt algorithm.
    """
    print(f"Running CPD on data from: {data_dir}")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results_file = output_path / "cpd_results.csv"
    if results_file.exists() and not overwrite:
        print(f"CPD results already exist at {results_file}. Use --overwrite to regenerate.")
        return

    sequence_files = get_all_sequence_files(data_dir, config)
    if not sequence_files:
        print("No sequence files found for CPD.")
        return

    print(f"Found {len(sequence_files)} sequence files for CPD analysis.")
    cpd_results: List[Dict[str, Any]] = []

    for file_path in sequence_files:
        features_df = load_features_from_npz([file_path])
        if features_df.empty or "hallu_signal" not in features_df.columns:
            print(f"Warning: Skipping {file_path}, no valid data or 'hallu_signal' column.")
            continue

        signal = features_df["hallu_signal"].values
        if len(signal) < 2:
            print(f"Warning: Skipping {file_path}, signal too short for CPD.")
            continue
        
        # Use Pelt algorithm for change point detection
        # The "l2" model assumes piecewise constant signal with Gaussian noise
        algo = rpt.Pelt(model="l2").fit(signal)
        
        # A common penalty value is log(n), where n is the number of samples
        # This penalty balances goodness of fit with the number of change points
        penalty = np.log(len(signal))
        change_points = algo.predict(pen=penalty)

        sequence_id = Path(file_path).stem
        cpd_results.append(
            {
                "sequence_id": sequence_id,
                # change_points includes the end of the last segment, which is len(signal)
                # We typically report the indices where the change occurs
                "change_points": [c for c in change_points if c < len(signal)],
                "num_change_points": len(change_points) - 1,
            }
        )

    if not cpd_results:
        print("No CPD results were generated.")
        return

    results_df = pd.DataFrame(cpd_results)
    results_df.to_csv(results_file, index=False)
    print(f"Saved CPD results for {len(results_df)} sequences to: {results_file}")


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
    
    plt.figure(figsize=(10, 6))
    sns.histplot(data=aggregated_df, x="avg_change_point", kde=True)
    plt.title("Distribution of Detected Change-Points")
    plt.xlabel("Average Change-Point Index")
    plt.ylabel("Frequency")
    plt.savefig(output_path / "change_point_distribution.png")
    plt.close()
    print(f"Saved change-point distribution plot to {output_path}")

    plt.figure(figsize=(10, 6))
    sns.histplot(data=aggregated_df, x="avg_confidence", kde=True)
    plt.title("Distribution of CPD Confidence Scores")
    plt.xlabel("Average Confidence")
    plt.ylabel("Frequency")
    plt.savefig(output_path / "confidence_distribution.png")
    plt.close()
    print(f"Saved confidence distribution plot to {output_path}")