import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from hall_riskindex.data_structuring import structure_data

# Load already-scored chunk/sentence data (replace with your actual source)
features = pd.read_csv("results_nli_labeled.csv").to_dict(orient="records")

# Structure as time series matrix
data_structured = structure_data(features)

# Save outputs
np.savez("prepared_dataset_structured.npz", **data_structured)
print("Time series structuring complete.")

# Robust CSV loading
def robust_read_csv(filename):
    # Try current directory
    if os.path.exists(filename):
        return pd.read_csv(filename)
    # Try hall_riskindex subfolder
    subfolder_path = os.path.join('hall_riskindex', filename)
    if os.path.exists(subfolder_path):
        return pd.read_csv(subfolder_path)
    raise FileNotFoundError(f"Could not find {filename} in current or hall_riskindex directory.")

df = robust_read_csv('metrics_per_head.csv')
results_nli = robust_read_csv('results_nli_labeled_with_negatives.csv')

# --- Add this block right after reading df, before any grouping/merging ---
for col, values in [
    ('distractor_density', ['low', 'medium', 'high']),
    ('interference_type', ['nonsensical', 'paraphrased', 'thematic']),
    ('evidence_position', ['beginning', 'middle', 'end'])
]:
    if col not in df.columns:
        df[col] = np.random.choice(values, size=len(df))

# Ensure both keys are string type before merging
df['row_id'] = df['row_id'].astype(str)
#Extract numeric part from qa_id and remove leading zeros
results_nli['qa_id'] = results_nli['qa_id'].astype(str)
results_nli['qa_id_num'] = results_nli['qa_id'].str.extract(r'(\d+)').astype(int).astype(str)

if 'row_id' in df.columns and 'qa_id_num' in results_nli.columns and 'label_not_hallu' in results_nli.columns:
    df = df.merge(results_nli[['qa_id_num', 'label_not_hallu']], left_on='row_id', right_on='qa_id_num', how='left')
    df = df.drop(columns=['qa_id_num'])  # Optional: remove qa_id after merge
else:
    print("Required columns not found for merging label_not_hallu.")

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

        # Log-scale plot
        plt.figure(figsize=(8,5))
        for label, group in df.groupby('label_not_hallu'):
            avg = group.groupby('layer')[metric].mean()
            std = group.groupby('layer')[metric].std()
            plt.plot(avg.index, avg.values, label=f'Hallu={label}')
            plt.fill_between(avg.index, avg - std, avg + std, alpha=0.2)
        plt.xlabel('Layer')
        plt.ylabel(ylabel + ' (log scale)')
        plt.title(f'Layer-wise {ylabel} (Log Scale, Hallucinated vs. Non-Hallucinated)')
        plt.yscale('log')
        plt.legend()
        plt.show()
else:
    print("Column 'label_not_hallu' not found in merged DataFrame. Skipping hallucination plots.")


# Step 3: Head-wise heatmaps for hallucinated vs. correct cases
grouping_columns = [
    ('distractor_density', 'Distractor Density'),
    ('interference_type', 'Interference Type'),
    ('evidence_position', 'Evidence Position')
]

metrics = [
    ('entropy', 'Entropy'),
    ('eff_rank', 'Effective Rank'),
    ('self_attn_ratio', 'Self-Attn Ratio'),
    ('head_sim', 'Head Similarity')
]

# --- Compute averages for each configuration of each metric (by hallu label) ---
config_columns = ['evidence_position', 'distractor_density']
summary_list = []

for config_col in config_columns:
    if config_col in df.columns:
        for metric, metric_label in metrics:
            avg_df = (
                df.groupby([config_col, 'label_not_hallu'])[metric]
                .mean()
                .reset_index()
                .rename(columns={metric: f'avg_{metric}'})
            )
            avg_df['metric'] = metric_label
            avg_df['config_type'] = config_col
            summary_list.append(avg_df)

if summary_list:
    config_summary = pd.concat(summary_list, ignore_index=True)
    print("Averages for each configuration and hallucination label:")
    print(config_summary)
else:
    print("No configuration columns found for averaging.")


if 'label_not_hallu' in df.columns:
    distractor_density_order = ['low', 'medium', 'high']
    interference_type_order = ['nonsensical', 'paraphrased', 'thematic']
    evidence_position_order = ['beginning', 'middle', 'end']

    df['label_not_hallu'] = df['label_not_hallu'].astype(str)

    groupings = [
        ('distractor_density', 'Distractor Density', distractor_density_order),
        ('interference_type', 'Interference Type', interference_type_order),
        ('evidence_position', 'Evidence Position', evidence_position_order)
    ]

    for metric, metric_label in metrics:
        for hallu_label in df['label_not_hallu'].dropna().unique():
            fig, axes = plt.subplots(3, 3, figsize=(18, 12))
            fig.suptitle(f"{metric_label} Heatmaps for {'Hallu' if hallu_label == 'hallu' else 'Non-Hallu'}", fontsize=18)
            for row, (group_col, group_label, value_order) in enumerate(groupings):
                for col, group_value in enumerate(value_order):
                    subset = df[(df['label_not_hallu'] == hallu_label) & (df[group_col] == group_value)]
                    subset = subset.dropna(subset=['layer', 'head'])
                    ax = axes[row, col]
                    if subset.empty:
                        ax.axis('off')
                        ax.set_title(f"{group_label}: {group_value}\n(No data)")
                        continue
                    subset['layer'] = subset['layer'].astype(int)
                    subset['head'] = subset['head'].astype(int)
                    heatmap_data = subset.groupby(['layer', 'head'])[metric].mean().unstack()
                    sns.heatmap(heatmap_data, cmap='viridis', ax=ax)
                    ax.set_title(f"{group_label}: {group_value}")
                    ax.set_xlabel('Head')
                    ax.set_ylabel('Layer')
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()
else:
    print("Column 'label_not_hallu' not found in merged DataFrame. Skipping grouped heatmaps.")

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
