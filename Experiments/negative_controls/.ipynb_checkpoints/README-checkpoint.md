# Negative Control Experiments for CPD Hallucination Detection

This directory contains negative control experiments to validate the Change Point Detection (CPD) approach for detecting hallucinations in LLM outputs.

## Overview

These experiments test whether the CPD method is genuinely detecting hallucination boundaries or if the results could be explained by simpler baselines or artifacts.

### Experiment Matrix

The experiments run a full matrix of **3 models × 5 experiment types = 15 combinations**.

| Model | Description |
|-------|-------------|
| `CPD_RBF` | Standard CPD with RBF kernel |
| `CPD+RF` | CPD + Random Forest classifier |
| `CPD+LR` | CPD + Logistic Regression classifier |

| Experiment Type | Subtype | Description |
|-----------------|---------|-------------|
| `baseline` | `standard_cpd` | Standard CPD without manipulation (reference) |
| `feature_shuffle` | `within_sequence` | Shuffle features within each sequence to break temporal structure |
| `feature_shuffle` | `across_sequence` | Swap feature time series between different sequences |
| `no_hallucination` | `gold_only` | Run CPD on sequences without hallucinations |
| `random_cp` | `random_selection` | Select random change points as baseline comparison |

---

## Prerequisites

### 1. Install Dependencies

```bash
pip3 install ruptures scikit-learn scipy pandas numpy
```

### 2. Pull Data Files (Git LFS)

The dataset files are stored in Git LFS. Pull them before running:

```bash
cd /path/to/Modeling-Attn-Collapse--JMIA
git lfs install
git lfs pull
```

### 3. Verify Data

```bash
# Check that data file is not just an LFS pointer
head -1 step_4_cpd/dataset_prep_code/prepared_dataset_cpd_with_attn_metrics_FINAL.csv
# Should show column headers, NOT "version https://git-lfs.github.com/spec/v1"
```

---

## Directory Structure

```
negative_controls/
├── README.md                      # This file
├── run_negative_controls.py       # Main experiment runner
├── prepare_no_hallu_data.py       # Prepare no-hallucination data
├── output/                        # Experiment results
│   ├── negative_control_results.csv
│   └── negative_control_results_full.json
└── no_hallu_data/                 # No-hallucination experiment data
    ├── domain1_no_hallu_prompts.jsonl
    ├── domain1_no_hallu_prompts.csv
    └── run_inference.py
```

---

## How to Run

### Quick Start

```bash
# From project root directory
cd /path/to/Modeling-Attn-Collapse--JMIA

# Run all negative control experiments
python step_4_cpd/experiments/negative_controls/run_negative_controls.py
```

### Full Process (Including No-Hallucination Data)

#### Step 1: Run Main Experiments

```bash
python step_4_cpd/experiments/negative_controls/run_negative_controls.py
```

This runs all experiment combinations except "no_hallucination" (which requires additional data).

#### Step 2: Prepare No-Hallucination Data

```bash
python step_4_cpd/experiments/negative_controls/prepare_no_hallu_data.py
```

This creates prompts using only `gold_text` (no distractors) for domain1.

**Output:**
- `no_hallu_data/domain1_no_hallu_prompts.jsonl` - Prompts for inference
- `no_hallu_data/domain1_no_hallu_prompts.csv` - Reference CSV
- `no_hallu_data/run_inference.py` - Inference script

#### Step 3: Run Model Inference (Requires GPU)

```bash
# Requires: transformers, torch, GPU
python step_4_cpd/experiments/negative_controls/no_hallu_data/run_inference.py
```

This generates LongChat-13B responses for the gold-only prompts.

#### Step 4: Score and Integrate

After inference:
1. Score responses using hallucination scoring pipeline
2. Process through CPD data preparation
3. Re-run experiments

---

## Configuration

Edit `EXPERIMENT_CONFIG` in `run_negative_controls.py`:

```python
EXPERIMENT_CONFIG = {
    'sample_frac': 0.1,      # Data sampling fraction (0.1 = 10%)
    'max_seqs': 100,         # Max sequences per experiment
    'n_random_trials': 10,   # Trials for random baseline
    'max_train_seqs': 50,    # Sequences for classifier training
}
```

### Full Dataset (Slower, More Accurate)

```python
EXPERIMENT_CONFIG = {
    'sample_frac': 1.0,      # Use full dataset
    'max_seqs': 500,         # More sequences
    'n_random_trials': 20,   # More random trials
    'max_train_seqs': 100,   # More training data
}
```

---

## Experiments Detail

### Baseline: Standard CPD
- **Type:Subtype**: `baseline:standard_cpd`
- **Description**: Reference baseline using PELT algorithm with RBF kernel
- **Purpose**: Establish reference performance for comparison

### Feature Shuffle - Within Sequence
- **Type:Subtype**: `feature_shuffle:within_sequence`
- **Description**: Shuffle feature values within each sequence
- **Method**: For each feature column, randomly permute values within the sequence
- **Purpose**: Break temporal structure while preserving feature distributions
- **Expected**: If CPD relies on temporal patterns, F1 should DROP significantly

### Feature Shuffle - Across Sequence
- **Type:Subtype**: `feature_shuffle:across_sequence`
- **Description**: Swap feature time series between different sequences
- **Method**: Use features from sequence A with ground truth labels from sequence B
- **Purpose**: Test if features are sequence-specific
- **Expected**: If features correlate with specific sequences, F1 should DROP

### No Hallucination Sequences
- **Type:Subtype**: `no_hallucination:gold_only`
- **Description**: Run CPD on sequences without any hallucinations
- **Method**: Use model responses generated with only gold_text (no distractors)
- **Purpose**: Test false positive rate on known-clean sequences
- **Expected**: Should detect NO change points (low FP rate)
- **Status**: Requires data preparation (see Step 2-4 above)

### Random CP Baseline
- **Type:Subtype**: `random_cp:random_selection`
- **Description**: Randomly select change points and compare with ground truth
- **Method**: Generate 1-3 random CPs per sequence, run 10 trials
- **Purpose**: Establish random chance baseline
- **Expected**: CPD should significantly OUTPERFORM random

---

## Output Files

### Location
```
step_4_cpd/experiments/negative_controls/output/
```

### CSV Output (negative_control_results.csv)

| Column | Description |
|--------|-------------|
| `Model` | Model name (CPD_RBF, CPD+RF, CPD+LR) |
| `Experiment` | Experiment name |
| `Exp_Type` | Experiment type tag |
| `Exp_Subtype` | Experiment subtype tag |
| `Status` | success/error/skipped |
| `Sequences` | Number of sequences processed |
| `Precision` | Detection precision |
| `Recall` | Detection recall |
| `F1` | F1 score |
| `FP_Rate` | False positive rate (for no-hallu only) |
| `Error` | Error message if failed |

### JSON Output (negative_control_results_full.json)

```json
{
  "metadata": {
    "experiment_name": "Negative Control Experiments...",
    "feature_set": "best_raw_features",
    "features": ["..."],
    "models": ["CPD_RBF", "CPD+RF", "CPD+LR"],
    "experiment_types": {
      "baseline": {"standard_cpd": "..."},
      "feature_shuffle": {"within_sequence": "...", "across_sequence": "..."},
      "no_hallucination": {"gold_only": "..."},
      "random_cp": {"random_selection": "..."}
    },
    "config": {...}
  },
  "results": [
    {
      "model": "CPD_RBF",
      "experiment": "Baseline",
      "experiment_type": "baseline",
      "experiment_subtype": "standard_cpd",
      "feature_set": "best_raw_features",
      "features_used": ["..."],
      "status": "success",
      "sequences": 100,
      "precision": 0.01,
      "recall": 0.11,
      "f1": 0.018
    }
  ]
}
```

---

## Features Used

The experiments use the best raw feature set from E5 experiments:

```python
BEST_FEATURES = [
    'interference_score_lexical_wrt_distractors',  # Lexical overlap with distractors
    'evid_overlap_ngram',                          # N-gram overlap with gold text
    'evid_overlap_emb',                            # Embedding similarity with gold
    'evidence_position',                           # Position relative to evidence
    'normalized_chunk_index',                      # Normalized position in sequence
]
```

---

## Metrics Computation

### Ground Truth
- **True Change Point**: First chunk where `hallu_label > 0`

### Detection Criteria
- **Tolerance**: ±2 chunks
- **True Positive**: Detected CP within tolerance of ground truth
- **False Positive**: Detected CPs not within tolerance
- **False Negative**: Ground truth CP not detected

### Formulas
```
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 = 2 * Precision * Recall / (Precision + Recall)
```

---

## Interpreting Results

### Expected Outcomes

| Experiment | Expected Result | Good Sign |
|------------|-----------------|-----------|
| Feature Shuffle (within) | F1 drops significantly | Temporal structure matters |
| Feature Shuffle (across) | F1 drops significantly | Sequence-specific patterns exist |
| No Hallucination | Low FP rate (<0.5) | Model doesn't over-detect |
| Random CP | F1 much lower than baseline | CPD adds value over random |

### Warning Signs

- **Shuffle F1 doesn't drop**: Method may not rely on temporal structure
- **Random baseline competitive**: CPD may not add significant value
- **High FP rate on no-hallu**: Model over-detects change points

---

## Code Details

### run_negative_controls.py

**Main Components:**

1. **Model Classes**
   - `CPDModel`: Standard CPD using ruptures library with RBF kernel
   - `CPDClassifierModel`: Hybrid CPD + classifier (RF or LR)

2. **Experiment Functions**
   - `exp_baseline()`: Standard CPD detection
   - `exp_within_sequence_shuffle()`: Shuffle features within sequences
   - `exp_across_sequence_shuffle()`: Swap features between sequences
   - `exp_no_hallucination_sequences()`: Test on no-hallu sequences
   - `exp_random_cp_baseline()`: Random change point selection

3. **Result Tags**
   - `model`: CPD_RBF, CPD+RF, CPD+LR
   - `experiment_type`: baseline, feature_shuffle, no_hallucination, random_cp
   - `experiment_subtype`: standard_cpd, within_sequence, across_sequence, gold_only, random_selection

### prepare_no_hallu_data.py

**Purpose:** Create prompts with only gold_text (no distractors) to generate model responses that should NOT contain hallucinations.

**Prompt Format:**
```
A chat between a curious user and an artificial intelligence assistant...
USER: Question: {question}
Answer the question using the document below:

Document [1]
{gold_text}

A:
```

---

## Related Files

- **Data preparation**: `step_4_cpd/dataset_prep_code/prepare_dataset.py`
- **Main CPD experiments**: `step_4_cpd/experiments/experiments_e1_e5/`
- **Hallucination scoring**: `step_2_hallucination_scoring/`
