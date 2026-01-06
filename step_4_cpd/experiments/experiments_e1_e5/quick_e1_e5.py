#!/usr/bin/env python3
"""
Quick E1-E5 Experiments Runner
==============================
Streamlined CPD experiments using sampling.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import ruptures as rpt
import json
import warnings
warnings.filterwarnings('ignore')

os.chdir(str(Path(__file__).parent.parent))

# Settings
DATA_FILE = "dataset_prep_code/prepared_dataset_cpd_with_attn_metrics_FINAL.csv"
OUTPUT_DIR = "output/experiments_e1_e5"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("E1-E5 EXPERIMENTS RUNNER")
print("=" * 80)

# Load data with sample
print("\nLoading data (sampling 10% for speed)...")
df = pd.read_csv(DATA_FILE, low_memory=False)
print(f"Full dataset: {len(df)} rows")

# Sample for faster processing
np.random.seed(42)
sample_indices = np.random.choice(len(df), size=max(len(df)//10, 5000), replace=False)
df = df.iloc[sample_indices].reset_index(drop=True)
print(f"Sampled dataset: {len(df)} rows")

# Add features
print("\nEngineering features...")
sort_cols = ['domain', 'question_id']
if 'config_idx' in df.columns:
    sort_cols.append('config_idx')
if 'chunk_index' in df.columns:
    sort_cols.append('chunk_index')

df = df.sort_values(sort_cols, na_position='last').reset_index(drop=True)

# E2 features
if 'att_avg_entropy' in df.columns:
    df['delta_avg_attn_entropy'] = df.groupby(['domain', 'question_id'])['att_avg_entropy'].diff().fillna(0)

if 'distractor_attention_max' in df.columns:
    df['delta_distractor_max'] = df.groupby(['domain', 'question_id'])['distractor_attention_max'].diff().fillna(0)

if 'cross_attention_mass_to_gold' in df.columns:
    df['gold_attn_drop'] = -df.groupby(['domain', 'question_id'])['cross_attention_mass_to_gold'].diff().fillna(0)

# E2.5 features
if 'attention_entropy' in df.columns:
    df['entropy_variance'] = df.groupby(['domain', 'question_id'])['attention_entropy'].transform(
        lambda x: pd.Series(x.values).rolling(window=3, center=True).var().fillna(0).values
    )
    df['attn_reallocation_rate'] = df.groupby(['domain', 'question_id'])['attention_entropy'].diff().fillna(0).abs()

print("Features engineered")

def run_cpd_on_feature_set(feature_cols, penalty=1.0, cost_model='l2', max_seqs=50):
    """Run CPD on a feature set and return metrics."""
    valid_cols = [c for c in feature_cols if c in df.columns and df[c].dtype in [np.float64, np.float32, int, np.int64]]
    if not valid_cols:
        return None
    
    # Get valid rows
    df_valid = df[df[valid_cols].notna().all(axis=1)].copy()
    
    if 'hallu_label' not in df_valid.columns:
        return None
    
    # Group by sequence
    groups = list(df_valid.groupby(['domain', 'question_id']))[:max_seqs]
    
    tp, fp, fn = 0, 0, 0
    seqs_processed = 0
    
    for seq_id, seq_df in groups:
        if 'chunk_index' in seq_df.columns:
            seq_df = seq_df.sort_values('chunk_index').reset_index(drop=True)
        
        if len(seq_df) < 5:
            continue
        
        seqs_processed += 1
        
        # Build signal from features
        signal_parts = []
        for col in valid_cols:
            s = seq_df[col].astype(float).values
            s = np.nan_to_num(s, nan=0)
            # Normalize
            s = (s - np.median(s)) / (np.std(s) + 1e-10)
            signal_parts.append(s)
        
        if not signal_parts:
            continue
        
        signal = np.column_stack(signal_parts) if len(signal_parts) > 1 else signal_parts[0].reshape(-1, 1)
        
        # Find ground truth CP
        labels = seq_df['hallu_label'].values
        if len(labels) > 0 and np.any(labels):
            true_cp = np.argmax(labels)
            
            # Run PELT
            try:
                algo = rpt.Pelt(model=cost_model, min_size=3, jump=1)
                algo.fit(signal)
                bkps = algo.predict(pen=penalty)
                detected_cps = [cp for cp in bkps[:-1] if cp < len(signal)]
                
                # Compute metrics
                tolerance = 2
                detected_near = any(abs(cp - true_cp) <= tolerance for cp in detected_cps)
                
                if detected_near:
                    tp += 1
                else:
                    fn += 1
                
                fp += max(0, len(detected_cps) - 1)
            except:
                pass
    
    if seqs_processed == 0:
        return None
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + 1) if tp > 0 else 0  # Simplified - 1 change point per sequence
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'sequences': seqs_processed
    }

# Run experiments
results = []

print("\n" + "=" * 80)
print("E1: PENALTY SWEEP")
print("=" * 80)

for penalty in [0.1, 0.25, 0.5]:
    print(f"\nPenalty = {penalty}...")
    metrics = run_cpd_on_feature_set(
        ['evidence_pos_category', 'interference_score_lexical_wrt_distractors'],
        penalty=penalty
    )
    if metrics:
        print(f"  P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1']:.3f}")
        results.append({
            'exp_id': f'E1_pen{penalty}',
            'feature_set': 'evidence_pos + interference_score',
            'cpd_method': f'PELT(pen={penalty})',
            **metrics
        })

print("\n" + "=" * 80)
print("E2: FEATURE DIFFERENCING")
print("=" * 80)

print("\nRunning CPD...")
metrics = run_cpd_on_feature_set(
    ['delta_avg_attn_entropy', 'gold_attn_drop', 'delta_distractor_max'],
    penalty=1.0
)
if metrics:
    print(f"  P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1']:.3f}")
    results.append({
        'exp_id': 'E2',
        'feature_set': 'Δ avg_attn_entropy + gold_attn_drop + Δ distractor_max',
        'cpd_method': 'PELT(pen=1.0)',
        **metrics
    })

print("\n" + "=" * 80)
print("E2.5: ATTENTION LEVEL CHANGE")
print("=" * 80)

print("\nRunning CPD...")
metrics = run_cpd_on_feature_set(
    ['entropy_variance', 'attn_reallocation_rate'],
    penalty=1.0
)
if metrics:
    print(f"  P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1']:.3f}")
    results.append({
        'exp_id': 'E2.5',
        'feature_set': 'entropy_variance + attn_reallocation_rate',
        'cpd_method': 'PELT(pen=1.0)',
        **metrics
    })

print("\n" + "=" * 80)
print("E3: ROBUST COST (L1/HUBER)")
print("=" * 80)

for model in ['l1', 'huber']:
    print(f"\nModel = {model}...")
    metrics = run_cpd_on_feature_set(
        ['evidence_pos_category', 'interference_score_lexical_wrt_distractors'],
        penalty=1.0,
        cost_model=model
    )
    if metrics:
        print(f"  P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1']:.3f}")
        results.append({
            'exp_id': f'E3_{model}',
            'feature_set': 'evidence_pos + interference_score',
            'cpd_method': f'PELT({model})',
            **metrics
        })

print("\n" + "=" * 80)
print("E3.5: ROBUST COST (GAUSSIAN)")
print("=" * 80)

print("\nRunning CPD...")
metrics = run_cpd_on_feature_set(
    ['evidence_pos_category', 'interference_score_lexical_wrt_distractors'],
    penalty=1.0,
    cost_model='l2'
)
if metrics:
    print(f"  P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1']:.3f}")
    results.append({
        'exp_id': 'E3.5_gaussian',
        'feature_set': 'evidence_pos + interference_score',
        'cpd_method': 'PELT(l2/Gaussian)',
        **metrics
    })

# Save results
print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

csv_path = Path(OUTPUT_DIR) / "experiment_results_e1_e5.csv"
json_path = Path(OUTPUT_DIR) / "experiment_results_e1_e5.json"

df_results = pd.DataFrame(results)
df_results.to_csv(csv_path, index=False)
print(f"\nSaved: {csv_path}")

with open(json_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"Saved: {json_path}")

print("\n" + "=" * 100)
print("RESULTS TABLE")
print("=" * 100)
print(df_results[['exp_id', 'cpd_method', 'precision', 'recall', 'f1', 'sequences']].to_string(index=False))

print("\nDone!")
