## How to Run

### Quick Start (10% Sample)

```bash
cd Modeling-Attn-Collapse--JMIA/Experiments/negative_controls
python3 run_negative_controls.py
```

### Full Dataset (Slower, More Accurate)

Edit `run_negative_controls.py` line 475:

```python
# Change from:
df = load_data(sample_frac=0.1)

# To:
df = load_data(sample_frac=1.0)
```

Then run:

```bash
python3 run_negative_controls.py
```

---

## Experiments

### Baseline: Standard L1 CPD
- **Description**: Reference baseline using PELT algorithm with L1 cost
- **Parameters**: `model="l1"`, `penalty=1.0`, `min_size=2`
- **Purpose**: Establish reference performance for comparison

### Experiment 1: RBF Kernel CPD
- **Description**: Standard CPD with Radial Basis Function kernel
- **Parameters**: `model="rbf"`, `penalty=1.0`
- **Purpose**: Test alternative kernel performance
- **Expected**: Compare RBF vs L1 cost models

### Experiment 2: CPD + Classifier
- **Description**: Hybrid approach combining CPD with Random Forest classifier
- **Method**:
  1. Split data: 50% training, 50% testing
  2. Train classifier on chunk-level features to predict change points
  3. CPD generates candidate CPs (low penalty=0.5)
  4. Classifier validates/filters candidates
- **Purpose**: Test if post-hoc filtering improves precision

### Experiment 3: No-Hallucination Sequences Check
- **Description**: Check if sequences without any hallucinations exist
- **Method**: Filter sequences where `max(hallu_label) == 0`
- **Purpose**: Identify data for true negative control
- **Finding**: Current dataset has 0 no-hallucination sequences

### Experiment 4: Random CP Baseline
- **Description**: Randomly select change points and compare with ground truth
- **Method**:
  - Generate 1-3 random CPs per sequence
  - Run 10 trials and average results
- **Purpose**: Establish random chance baseline
- **Expected**: CPD should significantly outperform random

### Experiment 5: Within-Sequence Feature Shuffling
- **Description**: Shuffle feature values within each sequence
- **Method**: For each feature column, randomly permute values within the sequence
- **Purpose**: Break temporal structure while preserving feature distributions
- **Expected**: If CPD relies on temporal patterns, F1 should drop significantly

### Experiment 6: Across-Sequence Feature Shuffling
- **Description**: Swap feature time series between different sequences
- **Method**: Use features from sequence A with ground truth labels from sequence B
- **Purpose**: Test if features are sequence-specific
- **Expected**: If features correlate with specific sequences, F1 should drop

---

## Output Files

### Location
```
Experiments/negative_controls/output/
```

### Files Generated

| File | Description |
|------|-------------|
| `negative_control_results.csv` | Summary table with metrics per experiment |
| `negative_control_results_full.json` | Full results with all parameters and details |

### CSV Columns

| Column | Description |
|--------|-------------|
| `Experiment` | Experiment identifier |
| `Description` | Brief description of the experiment |
| `Sequences` | Number of sequences processed |
| `Precision` | True positives / (True positives + False positives) |
| `Recall` | True positives / (True positives + False negatives) |
| `F1` | Harmonic mean of precision and recall |

---

## Output Summary

### Expected Results Table

| Experiment | F1 Score | vs Baseline | Interpretation |
|------------|----------|-------------|----------------|
| Baseline_L1_CPD | ~0.024 | - | Reference |
| E1_RBF_Kernel | ~0.018 | -22% | L1 better than RBF |
| E2_CPD_Classifier | ~0.038 | +63% | Hybrid approach helps |
| E4_Random_Baseline | ~0.056 | +137% | Random is competitive (warning) |
| E5_Within_Shuffle | ~0.029 | +21% | Temporal structure not critical |
| E6_Across_Shuffle | ~0.024 | +2% | Features not sequence-specific |

### Interpretation Guide

**Good Signs:**
- RBF kernel performs worse than L1 (confirms L1 choice)
- CPD + Classifier improves over baseline (hybrid approach viable)

**Warning Signs:**
- Random baseline competitive with CPD (method may not extract much signal)
- Shuffling experiments don't show significant F1 drop (temporal structure may not matter)

**Action Required:**
- No sequences without hallucinations exist
- Need to prepare model responses WITHOUT distractor content for true negative control

---

## Features Used

The experiments use the best raw feature set:

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


## File Structure

```
negative_controls/
├── README.md                      # This file
├── run_negative_controls.py       # Main experiment script
└── output/
    ├── negative_control_results.csv        # Summary results
    └── negative_control_results_full.json  # Full results
```

