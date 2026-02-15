#!/usr/bin/env python3
"""
Negative Control Experiments for CPD Hallucination Detection
============================================================

Experiments:
1. Model Architecture Controls:
   - Simple standard CPD (standard params + rbf kernel)
   - CPD + classifier setup

2. Data Controls:
   - Sequences without hallucinations check
   - Random CP baselines

3. Feature Ablation Controls:
   - Within-sequence feature shuffling (break temporal structure)
   - Across-sequence feature shuffling (swap time series between sequences)
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import ruptures as rpt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from scipy.ndimage import uniform_filter1d
import json
import warnings
warnings.filterwarnings('ignore')

# Setup paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
os.chdir(str(PROJECT_ROOT))

DATA_FILE = "dataset_prep_code/prepared_dataset_cpd_with_attn_metrics_FINAL.csv"
OUTPUT_DIR = SCRIPT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Best raw feature set from E5 experiments
BEST_FEATURES = [
    'interference_score_lexical_wrt_distractors',
    'evid_overlap_ngram',
    'evid_overlap_emb',
    'evidence_position',
    'normalized_chunk_index',
]

# Attention features (for shuffling experiments)
ATTENTION_FEATURES = [
    'att_avg_entropy',
    'attention_entropy',
    'cross_attention_mass_to_gold',
    'distractor_attention_max'
]


def load_data(sample_frac=0.1, seed=42):
    """Load and optionally sample data."""
    print("Loading data...")
    df = pd.read_csv(DATA_FILE, low_memory=False)
    print(f"Full dataset: {len(df)} rows")

    if sample_frac < 1.0:
        np.random.seed(seed)
        sample_size = max(int(len(df) * sample_frac), 5000)
        sample_indices = np.random.choice(len(df), size=sample_size, replace=False)
        df = df.iloc[sample_indices].reset_index(drop=True)
        print(f"Sampled dataset: {len(df)} rows")

    # Sort by sequence
    sort_cols = ['domain', 'question_id']
    if 'config_idx' in df.columns:
        sort_cols.append('config_idx')
    if 'chunk_index' in df.columns:
        sort_cols.append('chunk_index')
    df = df.sort_values(sort_cols, na_position='last').reset_index(drop=True)

    return df


def get_sequences(df, max_seqs=None):
    """Get list of (sequence_id, sequence_df) tuples."""
    groups = list(df.groupby(['domain', 'question_id']))
    if max_seqs:
        groups = groups[:max_seqs]
    return groups


def prepare_signal(seq_df, feature_cols):
    """Prepare signal array from sequence dataframe."""
    valid_cols = [c for c in feature_cols if c in seq_df.columns]
    if not valid_cols:
        return None, []

    # Build signal
    signal_parts = []
    for col in valid_cols:
        s = seq_df[col].astype(float).values
        s = np.nan_to_num(s, nan=0)
        # Normalize per feature
        std = np.std(s)
        if std > 1e-10:
            s = (s - np.median(s)) / std
        signal_parts.append(s)

    if not signal_parts:
        return None, []

    signal = np.column_stack(signal_parts) if len(signal_parts) > 1 else signal_parts[0].reshape(-1, 1)
    return signal, valid_cols


def get_true_cp(seq_df):
    """Get ground truth change point (first hallucination)."""
    if 'hallu_label' not in seq_df.columns:
        return None
    labels = seq_df['hallu_label'].values
    if len(labels) > 0 and np.any(labels > 0):
        return np.argmax(labels > 0)
    return None


def compute_metrics(detected_cps, true_cp, tolerance=2):
    """Compute precision, recall, F1 for detected vs true change points."""
    if true_cp is None:
        return None

    detected_near = any(abs(cp - true_cp) <= tolerance for cp in detected_cps)
    tp = 1 if detected_near else 0
    fp = max(0, len(detected_cps) - (1 if detected_near else 0))
    fn = 1 - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'num_detected': len(detected_cps)
    }


# =============================================================================
# EXPERIMENT 1: Standard CPD with RBF Kernel
# =============================================================================

def exp_standard_cpd_rbf(df, max_seqs=100):
    """
    Simple standard CPD modeling with standard parameters and RBF kernel.
    This tests whether a vanilla CPD approach works for hallucination detection.
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: Standard CPD with RBF Kernel")
    print("=" * 80)

    results = {'tp': 0, 'fp': 0, 'fn': 0, 'seqs': 0}
    seq_results = []

    groups = get_sequences(df, max_seqs)

    for seq_id, seq_df in groups:
        if 'chunk_index' in seq_df.columns:
            seq_df = seq_df.sort_values('chunk_index').reset_index(drop=True)

        if len(seq_df) < 5:
            continue

        signal, _ = prepare_signal(seq_df, BEST_FEATURES)
        if signal is None or signal.shape[0] < 5:
            continue

        true_cp = get_true_cp(seq_df)
        if true_cp is None:
            continue

        results['seqs'] += 1

        try:
            # Standard RBF kernel CPD
            algo = rpt.Pelt(model="rbf", min_size=2, jump=1)
            algo.fit(signal)
            bkps = algo.predict(pen=1.0)
            detected_cps = [cp for cp in bkps[:-1] if cp < len(signal)]

            metrics = compute_metrics(detected_cps, true_cp)
            if metrics:
                results['tp'] += metrics['tp']
                results['fp'] += metrics['fp']
                results['fn'] += metrics['fn']
                seq_results.append({
                    'seq_id': str(seq_id),
                    'true_cp': true_cp,
                    'detected_cps': detected_cps,
                    **metrics
                })
        except Exception as e:
            print(f"  Error on sequence {seq_id}: {e}")

    # Compute overall metrics
    total_tp = results['tp']
    total_fp = results['fp']
    total_fn = results['fn']

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\nResults (RBF Kernel):")
    print(f"  Sequences processed: {results['seqs']}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")

    return {
        'experiment': 'E1_RBF_Kernel',
        'description': 'Standard CPD with RBF kernel (standard params)',
        'sequences': results['seqs'],
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn,
        'seq_results': seq_results
    }


# =============================================================================
# EXPERIMENT 2: CPD + Classifier Setup
# =============================================================================

def exp_cpd_plus_classifier(df, max_seqs=100):
    """
    Hybrid approach: Use CPD to detect candidates, then classifier to filter.
    CPD detects potential change points, classifier validates if they're hallucinations.
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: CPD + Classifier Setup")
    print("=" * 80)

    results = {'tp': 0, 'fp': 0, 'fn': 0, 'seqs': 0}
    seq_results = []

    groups = get_sequences(df, max_seqs)

    # First pass: collect training data
    print("Phase 1: Collecting training data from sequences...")
    training_data = []
    training_labels = []

    for seq_id, seq_df in groups[:max_seqs//2]:  # Use half for training
        if 'chunk_index' in seq_df.columns:
            seq_df = seq_df.sort_values('chunk_index').reset_index(drop=True)

        if len(seq_df) < 5:
            continue

        valid_cols = [c for c in BEST_FEATURES if c in seq_df.columns]
        if not valid_cols:
            continue

        for idx, row in seq_df.iterrows():
            features = [row.get(c, 0) for c in valid_cols]
            features = [0 if pd.isna(f) else f for f in features]
            training_data.append(features)
            # Label: 1 if this is start of hallucination (hallu_label > 0 and prev was 0)
            is_cp = 0
            if 'hallu_label' in seq_df.columns:
                curr_label = row.get('hallu_label', 0)
                prev_label = seq_df.iloc[max(0, idx-1)].get('hallu_label', 0) if idx > 0 else 0
                if curr_label > 0 and prev_label == 0:
                    is_cp = 1
            training_labels.append(is_cp)

    if len(training_data) < 10:
        print("  Not enough training data")
        return None

    # Train classifier
    print(f"Phase 2: Training classifier on {len(training_data)} samples...")
    X_train = np.array(training_data)
    y_train = np.array(training_labels)

    # Handle class imbalance
    clf = RandomForestClassifier(n_estimators=50, max_depth=5, class_weight='balanced', random_state=42)
    clf.fit(X_train, y_train)

    # Second pass: CPD + classifier validation
    print("Phase 3: Running CPD + classifier on test sequences...")

    for seq_id, seq_df in groups[max_seqs//2:]:  # Use other half for testing
        if 'chunk_index' in seq_df.columns:
            seq_df = seq_df.sort_values('chunk_index').reset_index(drop=True)

        if len(seq_df) < 5:
            continue

        signal, valid_cols = prepare_signal(seq_df, BEST_FEATURES)
        if signal is None or signal.shape[0] < 5:
            continue

        true_cp = get_true_cp(seq_df)
        if true_cp is None:
            continue

        results['seqs'] += 1

        try:
            # Step 1: CPD to find candidates
            algo = rpt.Pelt(model="l1", min_size=2, jump=1)
            algo.fit(signal)
            bkps = algo.predict(pen=0.5)  # Lower penalty for more candidates
            cpd_candidates = [cp for cp in bkps[:-1] if cp < len(signal)]

            # Step 2: Classifier to filter candidates
            validated_cps = []
            for cp in cpd_candidates:
                if cp < len(seq_df):
                    features = [seq_df.iloc[cp].get(c, 0) for c in valid_cols]
                    features = [0 if pd.isna(f) else f for f in features]
                    pred = clf.predict([features])[0]
                    if pred == 1:
                        validated_cps.append(cp)

            # If no validated CPs, fall back to CPD candidates
            final_cps = validated_cps if validated_cps else cpd_candidates[:1]

            metrics = compute_metrics(final_cps, true_cp)
            if metrics:
                results['tp'] += metrics['tp']
                results['fp'] += metrics['fp']
                results['fn'] += metrics['fn']
                seq_results.append({
                    'seq_id': str(seq_id),
                    'true_cp': true_cp,
                    'cpd_candidates': cpd_candidates,
                    'validated_cps': validated_cps,
                    'final_cps': final_cps,
                    **metrics
                })
        except Exception as e:
            print(f"  Error on sequence {seq_id}: {e}")

    # Compute overall metrics
    total_tp = results['tp']
    total_fp = results['fp']
    total_fn = results['fn']

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\nResults (CPD + Classifier):")
    print(f"  Sequences processed: {results['seqs']}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")

    return {
        'experiment': 'E2_CPD_Classifier',
        'description': 'CPD candidates + Random Forest classifier validation',
        'sequences': results['seqs'],
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn,
        'seq_results': seq_results
    }


# =============================================================================
# EXPERIMENT 3: Check for Sequences Without Hallucinations
# =============================================================================

def exp_check_no_hallucination_sequences(df):
    """
    Check if there are sequences without any hallucinations in the dataset.
    These would be useful for negative control experiments.
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: Check for No-Hallucination Sequences")
    print("=" * 80)

    if 'hallu_label' not in df.columns:
        print("  ERROR: hallu_label column not found")
        return None

    # Group by sequence and get max hallucination label
    seq_max_hallu = df.groupby(['domain', 'question_id'])['hallu_label'].max()

    total_seqs = len(seq_max_hallu)
    no_hallu_seqs = (seq_max_hallu == 0).sum()
    hallu_seqs = (seq_max_hallu > 0).sum()

    print(f"\nDataset Analysis:")
    print(f"  Total unique sequences: {total_seqs}")
    print(f"  Sequences WITH hallucination: {hallu_seqs} ({100*hallu_seqs/total_seqs:.1f}%)")
    print(f"  Sequences WITHOUT hallucination: {no_hallu_seqs} ({100*no_hallu_seqs/total_seqs:.1f}%)")

    # Per-domain breakdown
    print(f"\nPer-domain breakdown:")
    for domain in df['domain'].unique():
        domain_df = df[df['domain'] == domain]
        domain_max = domain_df.groupby('question_id')['hallu_label'].max()
        domain_no_hallu = (domain_max == 0).sum()
        domain_total = len(domain_max)
        print(f"  {domain}: {domain_total} seqs, {domain_no_hallu} no-hallu ({100*domain_no_hallu/max(1,domain_total):.1f}%)")

    result = {
        'experiment': 'E3_NoHallu_Check',
        'description': 'Check for sequences without hallucinations',
        'total_sequences': total_seqs,
        'sequences_with_hallu': hallu_seqs,
        'sequences_without_hallu': no_hallu_seqs,
        'has_no_hallu_sequences': no_hallu_seqs > 0
    }

    if no_hallu_seqs == 0:
        print("\n*** FINDING: No sequences without hallucinations exist in current dataset ***")
        print("*** To run negative controls on no-hallucination sequences, need to prepare")
        print("    model responses for cases WITHOUT distractor content ***")
        result['recommendation'] = 'Need to prepare model responses without distractor content'

    return result


# =============================================================================
# EXPERIMENT 4: Random Change Point Baseline
# =============================================================================

def exp_random_cp_baseline(df, max_seqs=100, n_random_trials=10):
    """
    Select random change points as candidates and compare with ground truth.
    This establishes a random baseline for CPD performance.
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 4: Random Change Point Baseline")
    print("=" * 80)

    np.random.seed(42)

    all_trial_results = []
    groups = get_sequences(df, max_seqs)

    for trial in range(n_random_trials):
        results = {'tp': 0, 'fp': 0, 'fn': 0, 'seqs': 0}

        for seq_id, seq_df in groups:
            if 'chunk_index' in seq_df.columns:
                seq_df = seq_df.sort_values('chunk_index').reset_index(drop=True)

            if len(seq_df) < 5:
                continue

            true_cp = get_true_cp(seq_df)
            if true_cp is None:
                continue

            results['seqs'] += 1

            # Generate random change point(s)
            seq_len = len(seq_df)
            n_random_cps = np.random.randint(1, 4)  # 1-3 random CPs
            random_cps = np.random.choice(range(1, seq_len), size=min(n_random_cps, seq_len-1), replace=False).tolist()

            metrics = compute_metrics(random_cps, true_cp)
            if metrics:
                results['tp'] += metrics['tp']
                results['fp'] += metrics['fp']
                results['fn'] += metrics['fn']

        total_tp = results['tp']
        total_fp = results['fp']
        total_fn = results['fn']

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        all_trial_results.append({
            'trial': trial,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'seqs': results['seqs']
        })

    # Aggregate across trials
    avg_precision = np.mean([r['precision'] for r in all_trial_results])
    avg_recall = np.mean([r['recall'] for r in all_trial_results])
    avg_f1 = np.mean([r['f1'] for r in all_trial_results])
    std_f1 = np.std([r['f1'] for r in all_trial_results])

    print(f"\nRandom Baseline Results ({n_random_trials} trials):")
    print(f"  Sequences per trial: {all_trial_results[0]['seqs']}")
    print(f"  Avg Precision: {avg_precision:.4f}")
    print(f"  Avg Recall: {avg_recall:.4f}")
    print(f"  Avg F1 Score: {avg_f1:.4f} (+/- {std_f1:.4f})")

    return {
        'experiment': 'E4_Random_Baseline',
        'description': f'Random change point selection ({n_random_trials} trials)',
        'n_trials': n_random_trials,
        'sequences': all_trial_results[0]['seqs'],
        'avg_precision': avg_precision,
        'avg_recall': avg_recall,
        'avg_f1': avg_f1,
        'std_f1': std_f1,
        'trial_results': all_trial_results
    }


# =============================================================================
# EXPERIMENT 5: Within-Sequence Feature Shuffling
# =============================================================================

def exp_within_sequence_shuffle(df, max_seqs=100):
    """
    Shuffle feature values within each sequence to break temporal structure.
    If CPD still works, it suggests the method is not relying on temporal patterns.
    Expected: Performance should DROP significantly.
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 5: Within-Sequence Feature Shuffling")
    print("=" * 80)

    np.random.seed(42)

    results = {'tp': 0, 'fp': 0, 'fn': 0, 'seqs': 0}
    seq_results = []

    groups = get_sequences(df, max_seqs)

    for seq_id, seq_df in groups:
        if 'chunk_index' in seq_df.columns:
            seq_df = seq_df.sort_values('chunk_index').reset_index(drop=True)

        if len(seq_df) < 5:
            continue

        signal, valid_cols = prepare_signal(seq_df, BEST_FEATURES)
        if signal is None or signal.shape[0] < 5:
            continue

        true_cp = get_true_cp(seq_df)
        if true_cp is None:
            continue

        results['seqs'] += 1

        # SHUFFLE: Randomly permute each feature column independently
        shuffled_signal = signal.copy()
        for col_idx in range(shuffled_signal.shape[1]):
            np.random.shuffle(shuffled_signal[:, col_idx])

        try:
            algo = rpt.Pelt(model="l1", min_size=2, jump=1)
            algo.fit(shuffled_signal)
            bkps = algo.predict(pen=1.0)
            detected_cps = [cp for cp in bkps[:-1] if cp < len(shuffled_signal)]

            metrics = compute_metrics(detected_cps, true_cp)
            if metrics:
                results['tp'] += metrics['tp']
                results['fp'] += metrics['fp']
                results['fn'] += metrics['fn']
                seq_results.append({
                    'seq_id': str(seq_id),
                    'true_cp': true_cp,
                    'detected_cps': detected_cps,
                    **metrics
                })
        except Exception as e:
            print(f"  Error on sequence {seq_id}: {e}")

    total_tp = results['tp']
    total_fp = results['fp']
    total_fn = results['fn']

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\nWithin-Sequence Shuffling Results:")
    print(f"  Sequences processed: {results['seqs']}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"\n  (If F1 drops significantly vs baseline, temporal structure matters)")

    return {
        'experiment': 'E5_Within_Shuffle',
        'description': 'Feature values shuffled within each sequence (break temporal structure)',
        'sequences': results['seqs'],
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn,
        'seq_results': seq_results
    }


# =============================================================================
# EXPERIMENT 6: Across-Sequence Feature Shuffling
# =============================================================================

def exp_across_sequence_shuffle(df, max_seqs=100):
    """
    Swap feature time series between different sequences.
    Uses features from one sequence with labels from another.
    Expected: Performance should DROP significantly.
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 6: Across-Sequence Feature Shuffling")
    print("=" * 80)

    np.random.seed(42)

    results = {'tp': 0, 'fp': 0, 'fn': 0, 'seqs': 0}
    seq_results = []

    groups = get_sequences(df, max_seqs)

    # Prepare all sequences first
    all_signals = []
    all_true_cps = []
    all_seq_ids = []

    for seq_id, seq_df in groups:
        if 'chunk_index' in seq_df.columns:
            seq_df = seq_df.sort_values('chunk_index').reset_index(drop=True)

        if len(seq_df) < 5:
            continue

        signal, _ = prepare_signal(seq_df, BEST_FEATURES)
        if signal is None or signal.shape[0] < 5:
            continue

        true_cp = get_true_cp(seq_df)
        if true_cp is None:
            continue

        all_signals.append(signal)
        all_true_cps.append(true_cp)
        all_seq_ids.append(seq_id)

    if len(all_signals) < 2:
        print("  Not enough sequences for across-sequence shuffling")
        return None

    # Shuffle the signal-label pairings
    n_seqs = len(all_signals)
    shuffled_indices = np.random.permutation(n_seqs)

    # Run CPD with mismatched signals and labels
    for i in range(n_seqs):
        results['seqs'] += 1

        # Use signal from sequence i, but evaluate against true_cp from shuffled sequence
        signal = all_signals[i]
        # Mismatch: use true_cp from a DIFFERENT sequence
        mismatched_idx = shuffled_indices[i]
        # Ensure we're actually using a different sequence's label
        while mismatched_idx == i and n_seqs > 1:
            mismatched_idx = (mismatched_idx + 1) % n_seqs

        true_cp = all_true_cps[mismatched_idx]

        # Adjust true_cp if it exceeds signal length
        if true_cp >= len(signal):
            true_cp = len(signal) - 1

        try:
            algo = rpt.Pelt(model="l1", min_size=2, jump=1)
            algo.fit(signal)
            bkps = algo.predict(pen=1.0)
            detected_cps = [cp for cp in bkps[:-1] if cp < len(signal)]

            metrics = compute_metrics(detected_cps, true_cp)
            if metrics:
                results['tp'] += metrics['tp']
                results['fp'] += metrics['fp']
                results['fn'] += metrics['fn']
                seq_results.append({
                    'signal_seq_id': str(all_seq_ids[i]),
                    'label_seq_id': str(all_seq_ids[mismatched_idx]),
                    'true_cp': true_cp,
                    'detected_cps': detected_cps,
                    **metrics
                })
        except Exception as e:
            print(f"  Error: {e}")

    total_tp = results['tp']
    total_fp = results['fp']
    total_fn = results['fn']

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\nAcross-Sequence Shuffling Results:")
    print(f"  Sequences processed: {results['seqs']}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"\n  (If F1 drops vs baseline, features are sequence-specific)")

    return {
        'experiment': 'E6_Across_Shuffle',
        'description': 'Feature time series swapped between sequences (mismatched signal-label pairs)',
        'sequences': results['seqs'],
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn,
        'seq_results': seq_results
    }


# =============================================================================
# BASELINE: Standard L1 CPD (for comparison)
# =============================================================================

def exp_baseline_l1_cpd(df, max_seqs=100):
    """
    Baseline: Standard L1 cost CPD (same as best from previous experiments).
    This provides a reference point for negative control comparisons.
    """
    print("\n" + "=" * 80)
    print("BASELINE: Standard L1 CPD (Reference)")
    print("=" * 80)

    results = {'tp': 0, 'fp': 0, 'fn': 0, 'seqs': 0}
    seq_results = []

    groups = get_sequences(df, max_seqs)

    for seq_id, seq_df in groups:
        if 'chunk_index' in seq_df.columns:
            seq_df = seq_df.sort_values('chunk_index').reset_index(drop=True)

        if len(seq_df) < 5:
            continue

        signal, _ = prepare_signal(seq_df, BEST_FEATURES)
        if signal is None or signal.shape[0] < 5:
            continue

        true_cp = get_true_cp(seq_df)
        if true_cp is None:
            continue

        results['seqs'] += 1

        try:
            algo = rpt.Pelt(model="l1", min_size=2, jump=1)
            algo.fit(signal)
            bkps = algo.predict(pen=1.0)
            detected_cps = [cp for cp in bkps[:-1] if cp < len(signal)]

            metrics = compute_metrics(detected_cps, true_cp)
            if metrics:
                results['tp'] += metrics['tp']
                results['fp'] += metrics['fp']
                results['fn'] += metrics['fn']
                seq_results.append({
                    'seq_id': str(seq_id),
                    'true_cp': true_cp,
                    'detected_cps': detected_cps,
                    **metrics
                })
        except Exception as e:
            print(f"  Error on sequence {seq_id}: {e}")

    total_tp = results['tp']
    total_fp = results['fp']
    total_fn = results['fn']

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\nBaseline L1 CPD Results:")
    print(f"  Sequences processed: {results['seqs']}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")

    return {
        'experiment': 'Baseline_L1_CPD',
        'description': 'Standard L1 cost PELT CPD (reference baseline)',
        'sequences': results['seqs'],
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn,
        'seq_results': seq_results
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("NEGATIVE CONTROL EXPERIMENTS FOR CPD HALLUCINATION DETECTION")
    print("=" * 80)

    # Load data (10% sample for speed)
    df = load_data(sample_frac=0.1)

    all_results = []

    # Run baseline first for reference
    baseline = exp_baseline_l1_cpd(df, max_seqs=100)
    if baseline:
        all_results.append(baseline)

    # Experiment 1: Standard CPD with RBF kernel
    result1 = exp_standard_cpd_rbf(df, max_seqs=100)
    if result1:
        all_results.append(result1)

    # Experiment 2: CPD + Classifier
    result2 = exp_cpd_plus_classifier(df, max_seqs=100)
    if result2:
        all_results.append(result2)

    # Experiment 3: Check for no-hallucination sequences
    result3 = exp_check_no_hallucination_sequences(df)
    if result3:
        all_results.append(result3)

    # Experiment 4: Random CP baseline
    result4 = exp_random_cp_baseline(df, max_seqs=100, n_random_trials=10)
    if result4:
        all_results.append(result4)

    # Experiment 5: Within-sequence shuffling
    result5 = exp_within_sequence_shuffle(df, max_seqs=100)
    if result5:
        all_results.append(result5)

    # Experiment 6: Across-sequence shuffling
    result6 = exp_across_sequence_shuffle(df, max_seqs=100)
    if result6:
        all_results.append(result6)

    # Save results
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    # Summary table
    summary_rows = []
    for r in all_results:
        if 'precision' in r:
            summary_rows.append({
                'Experiment': r['experiment'],
                'Description': r['description'],
                'Sequences': r.get('sequences', 0),
                'Precision': r.get('precision', r.get('avg_precision', 0)),
                'Recall': r.get('recall', r.get('avg_recall', 0)),
                'F1': r.get('f1', r.get('avg_f1', 0))
            })
        elif 'has_no_hallu_sequences' in r:
            summary_rows.append({
                'Experiment': r['experiment'],
                'Description': r['description'],
                'Sequences': r.get('total_sequences', 0),
                'Precision': 'N/A',
                'Recall': 'N/A',
                'F1': f"No-hallu seqs: {r['sequences_without_hallu']}"
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_file = OUTPUT_DIR / "negative_control_results.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"\nSaved summary to: {summary_file}")

    # Full results JSON
    json_file = OUTPUT_DIR / "negative_control_results_full.json"
    # Remove seq_results for cleaner JSON (they're verbose)
    clean_results = []
    for r in all_results:
        clean_r = {k: v for k, v in r.items() if k not in ['seq_results', 'trial_results']}
        clean_results.append(clean_r)

    with open(json_file, 'w') as f:
        json.dump(clean_results, f, indent=2, default=str)
    print(f"Saved full results to: {json_file}")

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(summary_df.to_string(index=False))

    # Print interpretation
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    baseline_f1 = baseline['f1'] if baseline else 0

    for r in all_results:
        if r['experiment'] == 'Baseline_L1_CPD':
            continue

        exp_name = r['experiment']

        if 'f1' in r:
            f1 = r['f1']
            diff = f1 - baseline_f1
            pct = 100 * diff / baseline_f1 if baseline_f1 > 0 else 0
            print(f"\n{exp_name}:")
            print(f"  F1 = {f1:.4f} (baseline: {baseline_f1:.4f}, diff: {diff:+.4f}, {pct:+.1f}%)")

            if 'Shuffle' in exp_name:
                if f1 < baseline_f1 * 0.7:
                    print(f"  -> GOOD: F1 dropped significantly, temporal/sequence structure matters")
                else:
                    print(f"  -> WARNING: F1 didn't drop much, method may not rely on temporal structure")
            elif 'Random' in exp_name:
                if f1 < baseline_f1 * 0.5:
                    print(f"  -> GOOD: Random baseline much worse, CPD is meaningful")
                else:
                    print(f"  -> WARNING: Random baseline competitive, CPD may not add much value")

        elif 'has_no_hallu_sequences' in r:
            if r['sequences_without_hallu'] == 0:
                print(f"\n{exp_name}:")
                print(f"  -> No sequences without hallucinations exist in dataset")
                print(f"  -> Recommendation: {r.get('recommendation', 'Prepare new data')}")

    print("\nDone!")


if __name__ == "__main__":
    main()
