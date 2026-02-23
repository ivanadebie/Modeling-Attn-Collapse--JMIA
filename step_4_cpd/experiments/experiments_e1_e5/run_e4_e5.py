#!/usr/bin/env python3
"""
E4 & E5 Experiments: Onset Features and CUSUM-based Features
=============================================================
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
print("E4-E5 EXPERIMENTS: ONSET FEATURES & CUSUM VARIANTS")
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

# ============================================================================
# E4: Onset Features (Novelty/Sudden Changes)
# ============================================================================

print("\nEngineering E4 onset features...")

# E4 features: sudden changes/novelty indicators
if 'evid_overlap_ngram' in df.columns:
    df['delta_evidence_overlap_ngram'] = df.groupby(['domain', 'question_id'])['evid_overlap_ngram'].diff().fillna(0)

# Inference score lexical (if exists, otherwise approximate with interference score)
if 'interference_score_lexical_wrt_distractors' in df.columns:
    df['delta_interference_score'] = df.groupby(['domain', 'question_id'])['interference_score_lexical_wrt_distractors'].diff().fillna(0)

# Novelty spike detection (large changes)
if 'delta_evidence_overlap_ngram' in df.columns:
    df['is_novelty_spike'] = (np.abs(df['delta_evidence_overlap_ngram']) > df['delta_evidence_overlap_ngram'].std()).astype(float)

# ============================================================================
# E5: CUSUM-based Features (Cumulative Sum Control Charts)
# ============================================================================

print("Engineering E5 CUSUM features...")

def compute_cusum(signal, threshold=1.0):
    """Compute CUSUM (Cumulative Sum Control Chart)."""
    signal = np.asarray(signal, dtype=float)
    signal = np.nan_to_num(signal, nan=0)
    
    # Normalize
    mean = np.mean(signal)
    std = np.std(signal)
    if std < 1e-10:
        std = 1.0
    
    normalized = (signal - mean) / std
    
    # CUSUM computation
    cusum_pos = np.zeros_like(normalized)
    cusum_neg = np.zeros_like(normalized)
    
    for i in range(1, len(normalized)):
        cusum_pos[i] = max(0, cusum_pos[i-1] + normalized[i] - threshold)
        cusum_neg[i] = max(0, cusum_neg[i-1] - normalized[i] - threshold)
    
    return cusum_pos, cusum_neg

# CUSUM on gold attention
if 'cross_attention_mass_to_gold' in df.columns:
    cusum_results = df.groupby(['domain', 'question_id'])['cross_attention_mass_to_gold'].apply(
        lambda x: pd.Series({
            'cusum_gold_pos': compute_cusum(x.values)[0],
            'cusum_gold_neg': compute_cusum(x.values)[1]
        })
    )
    # Flatten and merge
    for idx, row in cusum_results.iterrows():
        if isinstance(row['cusum_gold_pos'], np.ndarray):
            df.loc[df[['domain', 'question_id']].apply(tuple, axis=1) == idx, 'cusum_gold_max'] = np.max(row['cusum_gold_pos'])
        else:
            # Handle case where apply returns Series
            pass

# Simplified CUSUM: just track cumulative change direction
if 'attention_entropy' in df.columns:
    df['entropy_cusum'] = df.groupby(['domain', 'question_id'])['attention_entropy'].diff().fillna(0)
    df['entropy_cusum_cumsum'] = df.groupby(['domain', 'question_id'])['entropy_cusum'].cumsum()

# CUSUM on evidence overlap
if 'evid_overlap_ngram' in df.columns:
    df['evidence_cusum'] = df.groupby(['domain', 'question_id'])['evid_overlap_ngram'].diff().fillna(0)
    df['evidence_cusum_cumsum'] = df.groupby(['domain', 'question_id'])['evidence_cusum'].cumsum()

print("Features engineered")

def run_cpd_on_feature_set(feature_cols, penalty=1.0, cost_model='l2', max_seqs=50):
    """Run CPD on a feature set and return metrics."""
    valid_cols = [c for c in feature_cols if c in df.columns and df[c].dtype in [np.float64, np.float32, int, np.int64]]
    if not valid_cols:
        print(f"  Warning: No valid columns from {feature_cols}")
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
    recall = tp / (tp + 1) if tp > 0 else 0
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
print("E4: ONSET FEATURES (Novelty/Sudden Changes)")
print("=" * 80)

print("\nRunning CPD with onset features...")
metrics = run_cpd_on_feature_set(
    ['delta_evidence_overlap_ngram', 'delta_interference_score', 'is_novelty_spike'],
    penalty=1.0
)
if metrics:
    print(f"  P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1']:.3f}")
    results.append({
        'exp_id': 'E4',
        'feature_set': 'Δ evidence_overlap_ngram + Δ interference_score + novelty_spike',
        'cpd_method': 'PELT(pen=1.0)',
        **metrics
    })

print("\n" + "=" * 80)
print("E5: CUSUM-BASED FEATURES (Cumulative Sum Control Charts)")
print("=" * 80)

print("\nRunning CPD with CUSUM features...")
metrics = run_cpd_on_feature_set(
    ['entropy_cusum_cumsum', 'evidence_cusum_cumsum'],
    penalty=1.0
)
if metrics:
    print(f"  P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1']:.3f}")
    results.append({
        'exp_id': 'E5',
        'feature_set': 'entropy_cusum_cumsum + evidence_cusum_cumsum',
        'cpd_method': 'PELT(pen=1.0)',
        **metrics
    })

# Save results (append to existing)
print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

csv_path = Path(OUTPUT_DIR) / "experiment_results_e1_e5.csv"
json_path = Path(OUTPUT_DIR) / "experiment_results_e1_e5.json"

# Load existing results
try:
    existing_df = pd.read_csv(csv_path)
    existing_results = existing_df.to_dict('records')
except:
    existing_results = []

# Combine
all_results = existing_results + results

# Save
df_all = pd.DataFrame(all_results)
df_all.to_csv(csv_path, index=False)
print(f"\nUpdated: {csv_path}")

with open(json_path, 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"Updated: {json_path}")

print("\n" + "=" * 80)
print("UPDATED RESULTS TABLE")
print("=" * 80)
print(df_all[['exp_id', 'cpd_method', 'precision', 'recall', 'f1', 'sequences']].to_string(index=False))

print("\nDone!")
