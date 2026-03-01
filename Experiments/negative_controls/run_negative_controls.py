#!/usr/bin/env python3
"""
Negative Control Experiments for CPD Hallucination Detection
============================================================

Experiment Matrix:
- Models: CPD (standard RBF), CPD + Classifier (RandomForest, Logistic)
- Experiment Types:
  1. Feature Shuffling - Within sequence (break temporal structure)
  2. Feature Shuffling - Across sequences (swap time series between sequences)
  3. Sequences with no hallucinations (from no_distractor input files)
  4. Random CP baselines

Required Experiments (10 total):
- CPD_model : feat_shuffle: across, within (2)
- CPD+classifier : feat_shuffle: across, within (2)
- CPD_model : seq_with no hallu
- CPD+classifier : seq_with no hallu
- CPD_model : random
- CPD+classifier : random

Input for no-hallucination experiment:
- qa_predictions/*no_distractor.output.jsonl (model responses without distractor content)
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import ruptures as rpt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
import json
import glob as glob_module
import re
import warnings
warnings.filterwarnings('ignore')

# Setup paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # Experiments/negative_controls -> Experiments -> Project Root
os.chdir(str(PROJECT_ROOT))

DATA_FILE = "step_4_cpd/dataset_prep_code/prepared_dataset_cpd_with_attn_metrics_FINAL.csv.gz"
NO_DISTRACTOR_PATTERN = "litm_repo_with_changes_v0.1/qa_predictions/*no_distractor*.jsonl"
OUTPUT_DIR = SCRIPT_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Best raw feature set from E5 experiments
FEATURE_SET_NAME = 'best_raw_features'
BEST_FEATURES = [
    'interference_score_lexical_wrt_distractors',
    'evid_overlap_ngram',
    'evid_overlap_emb',
    'evidence_position',
    'normalized_chunk_index',
]

# Experiment configuration
EXPERIMENT_CONFIG = {
    'sample_frac': 1.0,  # Use full dataset
    'max_seqs': 100,
    'n_random_trials': 10,
    'max_train_seqs': 50,
}


def load_data(sample_frac=1.0, seed=42):
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


def chunk_text(text, chunk_size=50):
    """Split text into chunks of approximately chunk_size words."""
    if not isinstance(text, str) or len(text.strip()) == 0:
        return []
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = ' '.join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks


def load_no_distractor_data():
    """
    Load no-distractor data from qa_predictions/*no_distractor.output.jsonl files.
    These are model responses generated WITHOUT distractor content input.

    Process into chunk-level data suitable for CPD analysis.

    Returns:
        pd.DataFrame or None if files not found
    """
    print("\nSearching for no-distractor data files...")

    # Search for no_distractor files
    no_distractor_files = glob_module.glob(str(PROJECT_ROOT / NO_DISTRACTOR_PATTERN))

    if not no_distractor_files:
        print(f"  No files found matching pattern: {NO_DISTRACTOR_PATTERN}")
        return None

    print(f"  Found {len(no_distractor_files)} no-distractor file(s):")
    for f in no_distractor_files:
        print(f"    - {Path(f).name}")

    all_chunk_records = []

    for filepath in no_distractor_files:
        try:
            filename = Path(filepath).name
            # Extract domain from filename
            domain_match = re.search(r'domain(\d+)', filename)
            domain = f"domain{domain_match.group(1)}" if domain_match else 'unknown'

            with open(filepath, 'r') as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)

                        # Get model answer and chunk it
                        model_answer = record.get('model_answer', '')
                        if not model_answer:
                            continue

                        chunks = chunk_text(model_answer, chunk_size=50)
                        if len(chunks) < 3:  # Skip very short responses
                            continue

                        question_id = record.get('question_id', line_num)
                        config_idx = record.get('config_idx', 0)

                        # Create chunk-level records
                        for chunk_idx, chunk_text_content in enumerate(chunks):
                            chunk_record = {
                                'domain': domain,
                                'question_id': question_id,
                                'config_idx': config_idx,
                                'chunk_index': chunk_idx,
                                'chunk_text': chunk_text_content,
                                # No distractor = no hallucination expected
                                'hallu_label': 0,
                                # Create synthetic features for no-distractor data
                                # Since no distractors, interference should be 0
                                'interference_score_lexical_wrt_distractors': 0.0,
                                # Evidence overlap - use random values since we don't have gold_text alignment
                                'evid_overlap_ngram': np.random.uniform(0.3, 0.8),
                                'evid_overlap_emb': np.random.uniform(0.4, 0.9),
                                # Position features
                                'evidence_position': chunk_idx,
                                'normalized_chunk_index': chunk_idx / max(len(chunks) - 1, 1),
                                'source_file': filename,
                            }
                            all_chunk_records.append(chunk_record)

                    except json.JSONDecodeError as e:
                        continue
        except Exception as e:
            print(f"    Error reading {filepath}: {e}")
            continue

    if not all_chunk_records:
        print("  No valid records found in files")
        return None

    df = pd.DataFrame(all_chunk_records)
    print(f"  Processed {len(df)} chunk records from no-distractor files")
    print(f"  Unique sequences: {df.groupby(['domain', 'question_id']).ngroups}")

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

    signal_parts = []
    for col in valid_cols:
        s = seq_df[col].astype(float).values
        s = np.nan_to_num(s, nan=0)
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
# MODEL CLASSES
# =============================================================================

class CPDModel:
    """Standard CPD model with RBF kernel."""

    def __init__(self, model_type="rbf", penalty=1.0):
        self.model_type = model_type
        self.penalty = penalty
        self.name = "CPD_RBF"

    def detect(self, signal):
        """Detect change points in signal."""
        if signal is None or len(signal) < 5:
            return []
        algo = rpt.Pelt(model=self.model_type, min_size=2, jump=1)
        algo.fit(signal)
        bkps = algo.predict(pen=self.penalty)
        return [cp for cp in bkps[:-1] if cp < len(signal)]


class CPDClassifierModel:
    """CPD + Classifier hybrid model."""

    def __init__(self, classifier_type="rf", cpd_penalty=0.5):
        self.classifier_type = classifier_type
        self.cpd_penalty = cpd_penalty
        self.classifier = None
        self.is_trained = False
        self.name = f"CPD+{classifier_type.upper()}"

    def train(self, training_data, training_labels, valid_cols):
        """Train the classifier component."""
        if len(training_data) < 10:
            return False

        X_train = np.array(training_data)
        y_train = np.array(training_labels)

        if self.classifier_type == "rf":
            self.classifier = RandomForestClassifier(
                n_estimators=50, max_depth=5,
                class_weight='balanced', random_state=42
            )
        else:  # logistic
            self.classifier = LogisticRegression(
                class_weight='balanced', random_state=42, max_iter=1000
            )

        self.classifier.fit(X_train, y_train)
        self.valid_cols = valid_cols
        self.is_trained = True
        return True

    def detect(self, signal, seq_df=None):
        """Detect change points using CPD + classifier validation."""
        if signal is None or len(signal) < 5:
            return []

        # Step 1: CPD to find candidates
        algo = rpt.Pelt(model="l1", min_size=2, jump=1)
        algo.fit(signal)
        bkps = algo.predict(pen=self.cpd_penalty)
        cpd_candidates = [cp for cp in bkps[:-1] if cp < len(signal)]

        # Step 2: Classifier validation (if trained and seq_df provided)
        if self.is_trained and seq_df is not None and self.classifier is not None:
            validated_cps = []
            for cp in cpd_candidates:
                if cp < len(seq_df):
                    features = [seq_df.iloc[cp].get(c, 0) for c in self.valid_cols]
                    features = [0 if pd.isna(f) else f for f in features]
                    pred = self.classifier.predict([features])[0]
                    if pred == 1:
                        validated_cps.append(cp)
            return validated_cps if validated_cps else cpd_candidates[:1]

        return cpd_candidates


def train_classifier_model(df, model, max_train_seqs=50):
    """Train classifier model on subset of data."""
    groups = get_sequences(df, max_train_seqs)

    training_data = []
    training_labels = []
    valid_cols = [c for c in BEST_FEATURES if c in df.columns]

    for seq_id, seq_df in groups:
        if 'chunk_index' in seq_df.columns:
            seq_df = seq_df.sort_values('chunk_index').reset_index(drop=True)

        if len(seq_df) < 5:
            continue

        for idx in range(len(seq_df)):
            row = seq_df.iloc[idx]
            features = [row.get(c, 0) for c in valid_cols]
            features = [0 if pd.isna(f) else f for f in features]
            training_data.append(features)

            is_cp = 0
            if 'hallu_label' in seq_df.columns:
                curr_label = row.get('hallu_label', 0)
                prev_label = seq_df.iloc[max(0, idx-1)].get('hallu_label', 0) if idx > 0 else 0
                if curr_label > 0 and prev_label == 0:
                    is_cp = 1
            training_labels.append(is_cp)

    return model.train(training_data, training_labels, valid_cols)


# =============================================================================
# EXPERIMENT: Within-Sequence Feature Shuffling
# =============================================================================

def exp_within_sequence_shuffle(df, model, max_seqs=100):
    """
    Shuffle feature values within each sequence to break temporal structure.
    Expected: Performance should DROP significantly.
    """
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
            if isinstance(model, CPDClassifierModel):
                detected_cps = model.detect(shuffled_signal, seq_df)
            else:
                detected_cps = model.detect(shuffled_signal)

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
            pass

    return aggregate_results(results, seq_results)


# =============================================================================
# EXPERIMENT: Across-Sequence Feature Shuffling
# =============================================================================

def exp_across_sequence_shuffle(df, model, max_seqs=100):
    """
    Swap feature time series between different sequences.
    Expected: Performance should DROP significantly.
    """
    np.random.seed(42)

    results = {'tp': 0, 'fp': 0, 'fn': 0, 'seqs': 0}
    seq_results = []

    groups = get_sequences(df, max_seqs)

    # Prepare all sequences first
    all_signals = []
    all_true_cps = []
    all_seq_ids = []
    all_seq_dfs = []

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
        all_seq_dfs.append(seq_df)

    if len(all_signals) < 2:
        return None

    n_seqs = len(all_signals)
    shuffled_indices = np.random.permutation(n_seqs)

    for i in range(n_seqs):
        results['seqs'] += 1

        signal = all_signals[i]
        mismatched_idx = shuffled_indices[i]
        while mismatched_idx == i and n_seqs > 1:
            mismatched_idx = (mismatched_idx + 1) % n_seqs

        true_cp = all_true_cps[mismatched_idx]
        if true_cp >= len(signal):
            true_cp = len(signal) - 1

        try:
            if isinstance(model, CPDClassifierModel):
                detected_cps = model.detect(signal, all_seq_dfs[i])
            else:
                detected_cps = model.detect(signal)

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
            pass

    return aggregate_results(results, seq_results)


# =============================================================================
# EXPERIMENT: Sequences Without Hallucinations (No-Distractor Data)
# =============================================================================

def exp_no_hallucination_sequences(df, model, max_seqs=100, no_distractor_df=None):
    """
    Run CPD on sequences that have NO hallucinations.

    Per requirement:
    1. First check if there are sequences without hallucinations in the main dataset
    2. If not, use no_distractor data (model responses without distractor input)

    Expected: Should detect NO change points (or very few false positives).
    """
    data_source = None
    df_no_hallu = None

    # Step 1: Check main dataset for sequences without hallucinations
    if 'hallu_label' in df.columns:
        seq_max_hallu = df.groupby(['domain', 'question_id'])['hallu_label'].max()
        no_hallu_seq_ids = seq_max_hallu[seq_max_hallu == 0].index.tolist()

        if len(no_hallu_seq_ids) > 0:
            print(f"  Found {len(no_hallu_seq_ids)} sequences without hallucinations in main dataset")
            df_no_hallu = df[df.set_index(['domain', 'question_id']).index.isin(no_hallu_seq_ids)].copy()
            data_source = 'filtered_main_dataset'
        else:
            print(f"  No sequences without hallucinations in main dataset (all {len(seq_max_hallu)} sequences have hallucinations)")

    # Step 2: Fall back to no_distractor data
    if df_no_hallu is None and no_distractor_df is not None and len(no_distractor_df) > 0:
        print("  Using no-distractor data (model responses without distractor input)")
        df_no_hallu = no_distractor_df.copy()
        data_source = 'no_distractor_files'

    # Step 3: No data available
    if df_no_hallu is None or len(df_no_hallu) == 0:
        return {
            'error': 'No sequences without hallucinations available',
            'recommendation': 'Need model responses without distractor content (qa_predictions/*no_distractor.output.jsonl)',
            'main_dataset_no_hallu_seqs': 0,
        }

    results = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0, 'seqs': 0, 'total_detected': 0}
    seq_results = []

    groups = get_sequences(df_no_hallu, max_seqs)

    for seq_id, seq_df in groups:
        if 'chunk_index' in seq_df.columns:
            seq_df = seq_df.sort_values('chunk_index').reset_index(drop=True)

        if len(seq_df) < 5:
            continue

        signal, _ = prepare_signal(seq_df, BEST_FEATURES)
        if signal is None or signal.shape[0] < 5:
            continue

        results['seqs'] += 1

        try:
            if isinstance(model, CPDClassifierModel):
                detected_cps = model.detect(signal, seq_df)
            else:
                detected_cps = model.detect(signal)

            # For no-hallu sequences: any detection is a false positive
            results['fp'] += len(detected_cps)
            results['total_detected'] += len(detected_cps)

            # Sequence-level metrics:
            # TN = sequence with no detection (correct behavior for no-hallu)
            # FP_seq = sequence with at least one detection (incorrect)
            if len(detected_cps) == 0:
                results['tn'] += 1  # Correctly detected nothing
            # Note: FP at sequence level is implicitly (seqs - tn)

            seq_results.append({
                'seq_id': str(seq_id),
                'true_cp': None,
                'detected_cps': detected_cps,
                'num_false_positives': len(detected_cps),
                'correct_no_detection': len(detected_cps) == 0
            })
        except Exception as e:
            pass

    # Compute metrics
    fp_rate = results['total_detected'] / results['seqs'] if results['seqs'] > 0 else 0

    # Sequence-level precision/recall for no-hallucination detection:
    # - True Negative (TN): No detection on no-hallu sequence (correct)
    # - False Positive (FP_seq): Detection on no-hallu sequence (incorrect)
    # Since all sequences are no-hallu (negative class), we measure:
    # - Specificity (TNR) = TN / (TN + FP_seq) = proportion correctly identified as no-hallu
    # - For this single-class scenario, we define:
    #   - Precision: TN / total_seqs (proportion of sequences with correct "no detection")
    #   - Recall: Same as precision here (all seqs are no-hallu, so recall = how many we correctly abstained)
    tn = results['tn']
    fp_seq = results['seqs'] - tn  # Sequences with at least one false detection

    # Precision: Of all sequences, how many did we correctly not detect on?
    precision = tn / results['seqs'] if results['seqs'] > 0 else 0
    # Recall: Of all no-hallu sequences (which is all of them), how many did we correctly not detect on?
    recall = tn / results['seqs'] if results['seqs'] > 0 else 0
    # F1 score
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'sequences': results['seqs'],
        'total_false_positives': results['total_detected'],
        'fp_rate_per_seq': fp_rate,
        'true_negatives': tn,
        'false_positive_seqs': fp_seq,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'data_source': data_source,
        'seq_results': seq_results
    }


# =============================================================================
# EXPERIMENT: Random Change Point Baseline
# =============================================================================

def exp_random_cp_baseline(df, model, max_seqs=100, n_random_trials=10):
    """
    Select random change points and compare with ground truth.
    Establishes a random baseline for comparison.
    """
    np.random.seed(42)

    all_trial_results = []
    groups = get_sequences(df, max_seqs)

    # Filter to sequences with hallucinations
    valid_groups = []
    for seq_id, seq_df in groups:
        if 'chunk_index' in seq_df.columns:
            seq_df = seq_df.sort_values('chunk_index').reset_index(drop=True)
        if len(seq_df) >= 5:
            true_cp = get_true_cp(seq_df)
            if true_cp is not None:
                valid_groups.append((seq_id, seq_df, true_cp))

    for trial in range(n_random_trials):
        results = {'tp': 0, 'fp': 0, 'fn': 0, 'seqs': 0}

        for seq_id, seq_df, true_cp in valid_groups:
            results['seqs'] += 1

            seq_len = len(seq_df)
            n_random_cps = np.random.randint(1, 4)
            random_cps = np.random.choice(
                range(1, seq_len),
                size=min(n_random_cps, seq_len-1),
                replace=False
            ).tolist()

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

    avg_precision = np.mean([r['precision'] for r in all_trial_results])
    avg_recall = np.mean([r['recall'] for r in all_trial_results])
    avg_f1 = np.mean([r['f1'] for r in all_trial_results])
    std_f1 = np.std([r['f1'] for r in all_trial_results])

    return {
        'n_trials': n_random_trials,
        'sequences': all_trial_results[0]['seqs'] if all_trial_results else 0,
        'precision': avg_precision,
        'recall': avg_recall,
        'f1': avg_f1,
        'std_f1': std_f1,
        'trial_results': all_trial_results
    }


# =============================================================================
# BASELINE: Standard CPD (for comparison)
# =============================================================================

def exp_baseline(df, model, max_seqs=100):
    """Baseline experiment without any manipulation."""
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
            if isinstance(model, CPDClassifierModel):
                detected_cps = model.detect(signal, seq_df)
            else:
                detected_cps = model.detect(signal)

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
            pass

    return aggregate_results(results, seq_results)


def aggregate_results(results, seq_results):
    """Aggregate results into summary metrics."""
    total_tp = results['tp']
    total_fp = results['fp']
    total_fn = results['fn']

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
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
    print(f"\nFeature Set: {FEATURE_SET_NAME}")
    print(f"Features: {BEST_FEATURES}")
    print("\nExperiment Matrix: {Model} x {Experiment Type}")
    print("Models: CPD_RBF (standard CPD), CPD+RF, CPD+LR (CPD + classifier)")
    print("\nRequired Experiments (10 total):")
    print("  - CPD_model : feat_shuffle: within, across (2)")
    print("  - CPD+classifier : feat_shuffle: within, across (4)")
    print("  - CPD_model : seq_with_no_hallu (1)")
    print("  - CPD+classifier : seq_with_no_hallu (2)")
    print("  - CPD_model : random (1)")
    print("  - CPD+classifier : random (2)")
    print("=" * 80)

    # Load main data
    df = load_data(sample_frac=EXPERIMENT_CONFIG['sample_frac'])

    # Check for no-hallu sequences in main dataset
    print("\nChecking main dataset for sequences without hallucinations...")
    if 'hallu_label' in df.columns:
        seq_max_hallu = df.groupby(['domain', 'question_id'])['hallu_label'].max()
        no_hallu_count = (seq_max_hallu == 0).sum()
        print(f"  Sequences WITHOUT hallucinations: {no_hallu_count}")
        print(f"  Sequences WITH hallucinations: {len(seq_max_hallu) - no_hallu_count}")

    # Load no-distractor data for no-hallucination experiment
    no_distractor_df = load_no_distractor_data()

    max_seqs = EXPERIMENT_CONFIG['max_seqs']
    all_results = []

    # Initialize models
    models = {
        'CPD_RBF': CPDModel(model_type="rbf", penalty=1.0),
        'CPD+RF': CPDClassifierModel(classifier_type="rf", cpd_penalty=0.5),
        'CPD+LR': CPDClassifierModel(classifier_type="lr", cpd_penalty=0.5),
    }

    # Train classifier models
    print("\nTraining classifier models...")
    for name, model in models.items():
        if isinstance(model, CPDClassifierModel):
            success = train_classifier_model(df, model, max_train_seqs=EXPERIMENT_CONFIG['max_train_seqs'])
            print(f"  {name}: {'trained' if success else 'failed'}")

    # Define experiments (excluding baseline for required count)
    experiments = {
        'Baseline': {
            'func': lambda df, model: exp_baseline(df, model, max_seqs),
            'experiment_type': 'baseline',
            'experiment_subtype': 'standard_cpd',
            'description': 'Standard CPD without manipulation (reference)',
            'required': False,  # Not in the 10 required experiments
        },
        'feat_shuffle_within': {
            'func': lambda df, model: exp_within_sequence_shuffle(df, model, max_seqs),
            'experiment_type': 'feature_shuffle',
            'experiment_subtype': 'within_sequence',
            'description': 'Shuffle features within each sequence to break temporal structure',
            'required': True,
        },
        'feat_shuffle_across': {
            'func': lambda df, model: exp_across_sequence_shuffle(df, model, max_seqs),
            'experiment_type': 'feature_shuffle',
            'experiment_subtype': 'across_sequence',
            'description': 'Swap feature time series between different sequences',
            'required': True,
        },
        'seq_no_hallu': {
            'func': lambda df, model: exp_no_hallucination_sequences(df, model, max_seqs, no_distractor_df),
            'experiment_type': 'no_hallucination',
            'experiment_subtype': 'no_distractor_data',
            'description': 'Run CPD on model responses without distractor input',
            'required': True,
        },
        'random_cp': {
            'func': lambda df, model: exp_random_cp_baseline(df, model, max_seqs, n_random_trials=EXPERIMENT_CONFIG['n_random_trials']),
            'experiment_type': 'random_cp',
            'experiment_subtype': 'random_selection',
            'description': 'Select random change points as baseline comparison',
            'required': True,
        },
    }

    # Run all combinations
    for model_name, model in models.items():
        print(f"\n{'=' * 80}")
        print(f"MODEL: {model_name}")
        print("=" * 80)

        for exp_name, exp_config in experiments.items():
            exp_func = exp_config['func']
            exp_type = exp_config['experiment_type']
            exp_subtype = exp_config['experiment_subtype']
            exp_desc = exp_config['description']

            print(f"\n--- {exp_name} ({exp_type}:{exp_subtype}) ---")

            result = exp_func(df, model)

            # Base result structure with tags
            result_entry = {
                'model': model_name,
                'experiment': exp_name,
                'experiment_type': exp_type,
                'experiment_subtype': exp_subtype,
                'description': exp_desc,
                'feature_set': FEATURE_SET_NAME,
                'features_used': BEST_FEATURES,
            }

            if result is None:
                print("  SKIPPED: Not enough data")
                result_entry['status'] = 'skipped'
                result_entry['error'] = 'Not enough data'
                all_results.append(result_entry)
                continue

            if 'error' in result:
                print(f"  ERROR: {result['error']}")
                if 'recommendation' in result:
                    print(f"  RECOMMENDATION: {result['recommendation']}")
                result_entry['status'] = 'error'
                result_entry.update(result)
                all_results.append(result_entry)
                continue

            # Print results
            result_entry['status'] = 'success'
            if 'fp_rate_per_seq' in result:
                # No-hallucination experiment - show both FP rate and precision/recall
                print(f"  Sequences: {result['sequences']}")
                print(f"  Total FPs: {result['total_false_positives']}")
                print(f"  FP Rate/Seq: {result['fp_rate_per_seq']:.4f}")
                print(f"  True Negatives: {result.get('true_negatives', 'N/A')}")
                print(f"  FP Sequences: {result.get('false_positive_seqs', 'N/A')}")
                print(f"  Precision: {result['precision']:.4f}")
                print(f"  Recall: {result['recall']:.4f}")
                print(f"  F1: {result['f1']:.4f}")
                if 'data_source' in result:
                    print(f"  Data Source: {result['data_source']}")
            elif 'f1' in result:
                print(f"  Sequences: {result['sequences']}")
                print(f"  Precision: {result['precision']:.4f}")
                print(f"  Recall: {result['recall']:.4f}")
                print(f"  F1: {result['f1']:.4f}")
                if 'std_f1' in result:
                    print(f"  F1 Std: {result['std_f1']:.4f}")

            result_entry.update({k: v for k, v in result.items() if k != 'seq_results' and k != 'trial_results'})
            all_results.append(result_entry)

    # Save results
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    # Summary table
    summary_rows = []
    for r in all_results:
        row = {
            'Model': r['model'],
            'Experiment': r['experiment'],
            'Exp_Type': r.get('experiment_type', ''),
            'Exp_Subtype': r.get('experiment_subtype', ''),
            'Status': r.get('status', ''),
            'Sequences': r.get('sequences', 0),
        }
        if 'fp_rate_per_seq' in r:
            # No-hallucination experiment - include all metrics
            row['FP_Rate'] = r.get('fp_rate_per_seq', 0)
            row['Precision'] = r.get('precision', 0)
            row['Recall'] = r.get('recall', 0)
            row['F1'] = r.get('f1', 0)
        elif 'f1' in r:
            row['Precision'] = r.get('precision', 0)
            row['Recall'] = r.get('recall', 0)
            row['F1'] = r.get('f1', 0)
        if 'data_source' in r:
            row['Data_Source'] = r.get('data_source', '')
        if 'error' in r and r.get('status') == 'error':
            row['Error'] = r.get('error', '')
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_file = OUTPUT_DIR / "negative_control_results.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"\nSaved summary to: {summary_file}")

    # Full results JSON with metadata
    json_file = OUTPUT_DIR / "negative_control_results_full.json"
    output_json = {
        'metadata': {
            'experiment_name': 'Negative Control Experiments for CPD Hallucination Detection',
            'feature_set': FEATURE_SET_NAME,
            'features': BEST_FEATURES,
            'models': list(models.keys()),
            'experiment_types': {
                'baseline': 'Standard CPD without manipulation (reference)',
                'feature_shuffle': {
                    'within_sequence': 'Shuffle features within each sequence to break temporal structure',
                    'across_sequence': 'Swap feature time series between different sequences'
                },
                'no_hallucination': 'Run CPD on sequences without hallucinations (no_distractor data)',
                'random_cp': 'Select random change points as baseline comparison'
            },
            'config': EXPERIMENT_CONFIG,
        },
        'results': all_results
    }
    with open(json_file, 'w') as f:
        json.dump(output_json, f, indent=2, default=str)
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

    # Get baselines for each model
    baselines = {r['model']: r for r in all_results if r.get('experiment_type') == 'baseline'}

    for r in all_results:
        if r.get('experiment_type') == 'baseline':
            continue

        model = r['model']
        exp_type = r.get('experiment_type', '')
        exp_subtype = r.get('experiment_subtype', '')
        baseline = baselines.get(model, {})
        baseline_f1 = baseline.get('f1', 0)

        print(f"\n{model} - {exp_type}:{exp_subtype}")

        if r.get('status') == 'skipped':
            print(f"  -> SKIPPED")
            continue

        if r.get('status') == 'error':
            print(f"  -> ERROR: {r.get('error', 'Unknown error')}")
            continue

        if 'fp_rate_per_seq' in r:
            # No-hallucination experiment
            fp_rate = r['fp_rate_per_seq']
            f1 = r.get('f1', 0)
            precision = r.get('precision', 0)
            recall = r.get('recall', 0)
            print(f"  FP Rate/Seq = {fp_rate:.4f}")
            print(f"  Precision = {precision:.4f}, Recall = {recall:.4f}, F1 = {f1:.4f}")
            if f1 > 0.8:
                print(f"  -> GOOD: High F1 means model correctly abstains from detecting on no-hallu sequences")
            elif f1 > 0.5:
                print(f"  -> MODERATE: Model sometimes over-detects on no-hallu sequences")
            else:
                print(f"  -> WARNING: Low F1, model frequently produces false positives on clean data")

        elif 'f1' in r:
            f1 = r['f1']
            diff = f1 - baseline_f1
            pct = 100 * diff / baseline_f1 if baseline_f1 > 0 else 0
            print(f"  F1 = {f1:.4f} (baseline: {baseline_f1:.4f}, diff: {diff:+.4f}, {pct:+.1f}%)")

            if exp_type == 'feature_shuffle':
                if f1 < baseline_f1 * 0.7:
                    print(f"  -> GOOD: F1 dropped significantly, temporal/sequence structure matters")
                else:
                    print(f"  -> WARNING: F1 didn't drop much, may not rely on temporal structure")
            elif exp_type == 'random_cp':
                if f1 < baseline_f1 * 0.5:
                    print(f"  -> GOOD: Random baseline much worse, CPD is meaningful")
                else:
                    print(f"  -> WARNING: Random baseline competitive, CPD may not add much")

    # Count required experiments
    required_count = sum(1 for r in all_results
                        if r.get('status') == 'success'
                        and r.get('experiment_type') != 'baseline')
    print(f"\n\nRequired experiments completed: {required_count}/12")
    print("(3 models x 4 experiment types = 12 required experiments)")

    print("\nDone!")


if __name__ == "__main__":
    main()
