#!/usr/bin/env python3
"""Quick E4-E5 with aggressive sampling"""
import sys, os, pandas as pd, numpy as np, ruptures as rpt, json, warnings
warnings.filterwarnings('ignore')
os.chdir("c:\\Users\\ivana\\OneDrive\\Documents\\Modeling-Attn-Collapse--JMIA\\step_4_cpd")

print("E4-E5 EXPERIMENTS - Quick Run")
print("="*60)

# Load with heavy sampling
print("Loading data...")
df = pd.read_csv("dataset_prep_code/prepared_dataset_cpd_with_attn_metrics_FINAL.csv", low_memory=False)
df = df.sample(frac=0.05, random_state=42).reset_index(drop=True)  # 5% sample
print(f"Working with {len(df)} rows")

# Sort
sort_cols = ['domain', 'question_id']
if 'config_idx' in df.columns:
    sort_cols.append('config_idx')
df = df.sort_values(sort_cols, na_position='last').reset_index(drop=True)

# E4 features
print("\nEngineering E4 onset features...")
if 'evid_overlap_ngram' in df.columns:
    df['delta_evidence_overlap_ngram'] = df.groupby(['domain', 'question_id'])['evid_overlap_ngram'].diff().fillna(0)
if 'interference_score_lexical_wrt_distractors' in df.columns:
    df['delta_interference_score'] = df.groupby(['domain', 'question_id'])['interference_score_lexical_wrt_distractors'].diff().fillna(0)

# E5 features
print("Engineering E5 CUSUM features...")
if 'attention_entropy' in df.columns:
    df['entropy_cusum'] = df.groupby(['domain', 'question_id'])['attention_entropy'].diff().fillna(0).cumsum()
if 'evid_overlap_ngram' in df.columns:
    df['evidence_cusum'] = df.groupby(['domain', 'question_id'])['evid_overlap_ngram'].diff().fillna(0).cumsum()

def run_cpd(feature_cols, label=''):
    """Quick CPD run"""
    valid_cols = [c for c in feature_cols if c in df.columns]
    if not valid_cols:
        return None
    
    df_valid = df[df[valid_cols].notna().all(axis=1) & df['hallu_label'].notna()].copy()
    groups = list(df_valid.groupby(['domain', 'question_id']))[:30]  # Only 30 sequences
    
    tp, fp, fn = 0, 0, 0
    seqs = 0
    
    for seq_id, seq_df in groups:
        if 'chunk_index' in seq_df.columns:
            seq_df = seq_df.sort_values('chunk_index')
        if len(seq_df) < 5:
            continue
        seqs += 1
        
        # Build signal
        signals = []
        for col in valid_cols:
            s = seq_df[col].astype(float).values
            s = np.nan_to_num(s, nan=0)
            s = (s - np.median(s)) / (np.std(s) + 1e-10)
            signals.append(s)
        
        signal = np.column_stack(signals) if len(signals) > 1 else signals[0].reshape(-1, 1)
        
        # Find true CP
        labels = seq_df['hallu_label'].values
        if not np.any(labels):
            continue
        true_cp = np.argmax(labels)
        
        # Run PELT
        try:
            algo = rpt.Pelt(model='l2', min_size=3, jump=1)
            algo.fit(signal)
            bkps = [b for b in algo.predict(pen=1.0)[:-1] if b < len(signal)]
            
            detected_near = any(abs(b - true_cp) <= 2 for b in bkps)
            if detected_near:
                tp += 1
            else:
                fn += 1
            fp += max(0, len(bkps) - 1)
        except:
            pass
    
    if seqs == 0:
        return None
    
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + 1) if tp > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    
    return {'precision': prec, 'recall': rec, 'f1': f1, 'tp': tp, 'fp': fp, 'sequences': seqs}

# Run E4
print("\nE4: Onset features...")
m4 = run_cpd(['delta_evidence_overlap_ngram', 'delta_interference_score'], 'E4')
if m4:
    print(f"  F1={m4['f1']:.4f}, P={m4['precision']:.4f}, R={m4['recall']:.4f}")

# Run E5
print("E5: CUSUM features...")
m5 = run_cpd(['entropy_cusum', 'evidence_cusum'], 'E5')
if m5:
    print(f"  F1={m5['f1']:.4f}, P={m5['precision']:.4f}, R={m5['recall']:.4f}")

# Append to existing results
print("\nUpdating results...")
df_existing = pd.read_csv("output/experiments_e1_e5/experiment_results_e1_e5.csv")

new_rows = []
if m4:
    new_rows.append({'exp_id': 'E4', 'feature_set': 'delta_evidence_overlap_ngram + delta_interference_score', 'cpd_method': 'PELT(pen=1.0)', **m4})
if m5:
    new_rows.append({'exp_id': 'E5', 'feature_set': 'entropy_cusum + evidence_cusum', 'cpd_method': 'PELT(pen=1.0)', **m5})

df_updated = pd.concat([df_existing, pd.DataFrame(new_rows)], ignore_index=True)
df_updated.to_csv("output/experiments_e1_e5/experiment_results_e1_e5.csv", index=False)

print("\nUPDATED RESULTS:")
print(df_updated[['exp_id', 'precision', 'recall', 'f1']].tail(5).to_string(index=False))
print("\nComplete!")
