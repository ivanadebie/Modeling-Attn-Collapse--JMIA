import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def load_data(filepath):
    """
    Loads data and handles potential parsing errors.
    """
    print(f"--- Loading data from {filepath} ---")
    try:
        # low_memory=False helps with mixed types
        df = pd.read_csv(filepath, low_memory=False)
        
        # Clean column names
        df.columns = df.columns.str.strip()
        print(f"Data loaded. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)

def get_strong_pairs(corr_matrix, threshold=0.8):
    """
    Extracts pairs of features with correlation above the threshold.
    """
    strong_pairs = []
    columns = corr_matrix.columns
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            val = corr_matrix.iloc[i, j]
            if abs(val) >= threshold:
                strong_pairs.append((columns[i], columns[j], val))
    
    # Sort by absolute correlation strength (descending)
    strong_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    return strong_pairs

def analyze_dataset(df_subset, dataset_name, output_dir):
    """
    Performs aggregation and correlation analysis for a single dataset group.
    """
    print(f"\nAnalyzing Group: {dataset_name}")
    
    # 1. Clean Data: Aggregate duplicates for the same configuration
    # We assume 'sequence_id' and 'chunk_index' define a unique row.
    # Adjust these columns if your unique identifier is different (e.g., just 'config')
    group_cols = ['config', 'sequence_id', 'chunk_index', 'question_id']
    group_cols = [c for c in group_cols if c in df_subset.columns]
    
    numeric_cols = df_subset.select_dtypes(include=[np.number]).columns.tolist()
    
    # Exclude grouping columns from numeric_cols to avoid errors during aggregation
    numeric_cols = [c for c in numeric_cols if c not in group_cols]
    
    if not numeric_cols:
        print("No numeric columns found for correlation.")
        return

    # Aggregate (mean) to handle duplicates
    if group_cols:
        df_agg = df_subset.groupby(group_cols)[numeric_cols].mean().reset_index()
    else:
        df_agg = df_subset

    # 2. Calculate Spearman Correlation
    # We only care about the correlation between the metrics (numeric_cols)
    analysis_df = df_agg[numeric_cols]
    
    # Drop columns with zero variance (constants) as they break correlation
    analysis_df = analysis_df.loc[:, analysis_df.std() > 0]
    
    if analysis_df.shape[1] < 2:
        print("Not enough valid numeric columns to calculate correlation.")
        return

    corr_matrix = analysis_df.corr(method='spearman')

    # 3. Generate Heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0, annot=False)
    plt.title(f"Spearman Correlation - {dataset_name}")
    plt.tight_layout()
    heatmap_path = os.path.join(output_dir, f"heatmap_{dataset_name}.png")
    plt.savefig(heatmap_path)
    plt.close()
    print(f"Saved Heatmap: {heatmap_path}")

    # 4. Identify Strong Pairs
    strong_pairs = get_strong_pairs(corr_matrix, threshold=0.7)
    
    if strong_pairs:
        print(f"Found {len(strong_pairs)} strong correlations (>= 0.7):")
        for col1, col2, val in strong_pairs[:10]: # Print top 10
            print(f"  {col1} vs {col2}: {val:.2f}")
            
        # 5. Generate Scatter Plot for the Strongest Pair
        top_col1, top_col2, top_val = strong_pairs[0]
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df_agg, x=top_col1, y=top_col2, alpha=0.6)
        plt.title(f"Strongest Pair: {top_col1} vs {top_col2} (r={top_val:.2f})")
        plt.xlabel(top_col1)
        plt.ylabel(top_col2)
        plt.tight_layout()
        scatter_path = os.path.join(output_dir, f"scatter_{dataset_name}_top_pair.png")
        plt.savefig(scatter_path)
        plt.close()
        print(f"Saved Scatter Plot: {scatter_path}")
    else:
        print("No strong pairs found above threshold 0.7.")

def main():
    parser = argparse.ArgumentParser(description="Analyze Feature Correlations")
    parser.add_argument("--file", type=str, required=True, help="Path to input CSV file.")
    parser.add_argument("--output", type=str, default=".", help="Directory to save plots.")
    
    args = parser.parse_args()
    
    df = load_data(args.file)
    
    # Determine the dataset grouping column
    # Checks for 'dataset' or 'domain' (common names in your data)
    if 'dataset' in df.columns:
        group_col = 'dataset'
    elif 'domain' in df.columns:
        group_col = 'domain'
    else:
        print("Error: Could not find 'dataset' or 'domain' column to group by.")
        print("Available columns:", list(df.columns))
        return

    # Process each dataset separately
    unique_datasets = df[group_col].unique()
    print(f"Found {len(unique_datasets)} unique datasets: {unique_datasets}")
    
    for dataset_name in unique_datasets:
        # Create a clean filename for the dataset
        clean_name = str(dataset_name).replace(" ", "_").replace("/", "-")
        
        subset = df[df[group_col] == dataset_name].copy()
        analyze_dataset(subset, clean_name, args.output)

if __name__ == "__main__":
    main()