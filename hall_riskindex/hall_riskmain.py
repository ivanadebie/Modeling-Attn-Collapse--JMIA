import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('metrics_per_head.csv')
results_nli = pd.read_csv('results_nli_labeled_with_negatives.csv')

# Ensure both keys are string type before merging
df['row_id'] = df['row_id'].astype(str)
#Extract numeric part from qa_id and remove leading zeros
results_nli['qa_id_num'] = results_nli['qa_id'].str.extract(r'(\d+)').astype(int).astype(str)

print("Sample row_id from metrics_per_head:", df['row_id'].unique()[:10])
print("Sample qa_id from results_nli:", results_nli['qa_id'].unique()[:10])

if 'row_id' in df.columns and 'qa_id_num' in results_nli.columns and 'label_not_hallu' in results_nli.columns:
    df = df.merge(results_nli[['qa_id_num', 'label_not_hallu']], left_on='row_id', right_on='qa_id_num', how='left')
    df = df.drop(columns=['qa_id_num'])  # Optional: remove qa_id after merge
else:
    print("Required columns not found for merging label_not_hallu.")

print("Unique label_not_hallu values:", df['label_not_hallu'].unique())
print("label_not_hallu value counts:\n", df['label_not_hallu'].value_counts(dropna=False))
print("Sample rows with label_not_hallu:\n", df[['row_id', 'label_not_hallu']].head(10))

# Add this after merging to check the result:
print("Merged df columns:", df.columns.tolist())
print(df.head())

# Step 1: Compute per-layer averages for each question (row_id)
layer_metrics = df.groupby(['row_id', 'layer']).agg({
    'entropy': 'mean',
    'eff_rank': 'mean',
    'head_sim': 'mean',
    'self_attn_ratio': 'mean',
}).reset_index()

# Add per-layer averages as new columns to the main DataFrame
df = df.merge(layer_metrics, on=['row_id', 'layer'], suffixes=('', '_per_layer'))

# Rename new columns for clarity
df = df.rename(columns={
    'entropy_per_layer': 'avg_entropy_per_layer',
    'eff_rank_per_layer': 'avg_effective_rank_per_layer',
    'head_sim_per_layer': 'avg_head_sim_per_layer',
    'self_attn_ratio_per_layer': 'avg_self_attn_ratio_per_layer',
})

# Pivot to wide format for easy plotting
pivot_entropy = layer_metrics.pivot(index='layer', columns='row_id', values='entropy')
pivot_rank = layer_metrics.pivot(index='layer', columns='row_id', values='eff_rank')
pivot_sim = layer_metrics.pivot(index='layer', columns='row_id', values='head_sim')
pivot_sar = layer_metrics.pivot(index='layer', columns='row_id', values='self_attn_ratio')


# Step 2: Plot layer-wise trends for hallucinated vs. non-hallucinated responses (using per-layer averages)
if 'label_not_hallu' in df.columns:
    for metric, ylabel in [
        ('avg_entropy_per_layer', 'Avg Entropy per Layer'),
        ('avg_effective_rank_per_layer', 'Avg Effective Rank per Layer'),
        ('avg_head_sim_per_layer', 'Avg Head Similarity per Layer'),
        ('avg_self_attn_ratio_per_layer', 'Avg Self-Attn Ratio per Layer'),
    ]:
        plt.figure(figsize=(8,5))
        for label, group in df.groupby('label_not_hallu'):
            avg = group.groupby('layer')[metric].mean()
            std = group.groupby('layer')[metric].std()
            plt.plot(avg.index, avg.values, label=f'Hallu={label}')
            plt.fill_between(avg.index, avg - std, avg + std, alpha=0.2)
        plt.xlabel('Layer')
        plt.ylabel(ylabel)
        plt.title(f'Layer-wise {ylabel} (Hallucinated vs. Non-Hallucinated)')
        plt.legend()
        plt.show()
else:
    print("Column 'label_not_hallu' not found in merged DataFrame. Skipping hallucination plots.")


# Step 3: Head-wise heatmaps for hallucinated vs. correct cases
if 'label_not_hallu' in df.columns:
    for metric in ['entropy', 'eff_rank', 'self_attn_ratio', 'head_sim']:
        for label in df['label_not_hallu'].dropna().unique():
            subset = df[df['label_not_hallu'] == label]
            heatmap_data = subset.groupby(['layer', 'head'])[metric].mean().unstack()
            plt.figure(figsize=(10,6))
            sns.heatmap(heatmap_data, cmap='viridis')
            plt.title(f'{metric} Heatmap (Hallu={label})')
            plt.xlabel('Head')
            plt.ylabel('Layer')
            plt.show()
else:
    print("Column 'label_not_hallu' not found in merged DataFrame. Skipping head-wise heatmaps.")

# Step 4: Scatter plot of Effective Rank vs. Entropy, colored by hallucination label
if 'label_not_hallu' in df.columns:
    plt.figure(figsize=(7,5))
    sns.scatterplot(data=df, x='eff_rank', y='entropy', hue='label_not_hallu', alpha=0.6)
    plt.title('Effective Rank vs. Entropy (Colored by Hallucination)')
    plt.xlabel('Effective Rank')
    plt.ylabel('Entropy')
    plt.show()
else:
    print("Column 'label_not_hallu' not found in merged DataFrame. Skipping scatter plot.")

# Load metrics_per_head if available
metrics_per_head = pd.read_csv('metrics_per_head.csv')  # Adjust path if needed

# Load results_nli_labeled_with_negatives if available
results_nli = pd.read_csv('results_nli_labeled_with_negatives.csv')  # Adjust path if needed

# Example: Merge metrics_per_head with results_nli on row_id (if both have this column)
if 'row_id' in metrics_per_head.columns and 'row_id' in results_nli.columns:
    merged_df = pd.merge(metrics_per_head, results_nli, on='row_id', how='left')
    # Now you can use merged_df for further analysis or plotting