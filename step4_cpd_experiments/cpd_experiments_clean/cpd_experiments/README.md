# CPD Hallucination Detection Experiments

Sentence-level attention extraction and Change Point Detection (CPD) experiments for detecting hallucination onset in LLM responses.

## Best Result

**E7.2 Two-Stage Detection: F1 = 0.957** (Precision = 0.989, Recall = 0.926)

## Package Structure

```
cpd_experiments/
├── __init__.py
├── run_experiments.py          # Main entry point
├── configs/
│   ├── __init__.py
│   └── config.py               # Configuration dataclasses
├── core/
│   ├── __init__.py
│   ├── attention_extraction.py # Sentence-level attention extraction
│   ├── feature_engineering.py  # EWMA, delta, CUSUM features
│   └── cpd_algorithms.py       # PELT, KernelCPD wrappers
├── experiments/
│   ├── __init__.py
│   ├── e1_e5_single_stage.py   # E1-E5: Single-stage CPD
│   ├── e7_two_stage.py         # E7: Two-stage detection
│   └── e8_ablation.py          # E8: Feature ablation
└── utils/
    ├── __init__.py
    ├── helpers.py              # Data parsing, metrics, etc.
    └── evaluation.py           # PR curves, evaluation functions
```

## Experiments

### E1-E5: Single-Stage CPD
Direct change point detection on feature signals:
- **E1**: PELT L2 with penalty sweep
- **E2**: Delta (difference) features  
- **E2.5**: Entropy variance + attention reallocation
- **E3**: KernelCPD with RBF kernel (best single-stage: F1=0.618)
- **E4**: Onset-specific features (overlap, interference)
- **E5**: CUSUM features

### E7: Two-Stage Detection
1. **Stage 1**: KernelCPD generates candidate change points (high recall)
2. **Stage 2**: Gradient Boosting classifier filters candidates (high precision)

Best result: **F1 = 0.957** with threshold = 0.315

### E8: Feature Ablation
Drop-one-group analysis to identify important features:
- `raw_ewma_base` features are most critical
- `attn_level` features may introduce noise

## Usage

### Command Line

```bash
# Run all experiments
python -m cpd_experiments.run_experiments \
    --data path/to/data.csv \
    --output ./output

# Run specific experiments
python -m cpd_experiments.run_experiments \
    --data path/to/data.csv \
    --experiments E7 \
    --domain 01_nq_closed_book

# Run E1-E5 only
python -m cpd_experiments.experiments.e1_e5_single_stage \
    --data path/to/data.csv \
    --output ./output/E1_E5
```

### Python API

```python
from cpd_experiments.core import engineer_all_features
from cpd_experiments.experiments import run_e7_experiment

# Load and prepare data
df = pd.read_csv("data.csv")
df = engineer_all_features(df)

# Run E7 two-stage experiment
results = run_e7_experiment(
    df,
    output_dir="./output/E7"
)

print(f"Best F1: {results['best_threshold_result']['F1']:.4f}")
print(f"AP: {results['ap']:.4f}")
```

## Key Features

### Sentence-Level Attention
Unlike chunk-level extraction (which gives same attention for 8-9 consecutive sentences), sentence-level extraction provides unique attention per sentence:
- `sent_avg_entropy`: Attention entropy
- `sent_cross_attention_to_gold`: Gold attention mass
- `sent_distractor_attention_max`: Max distractor attention

### Engineered Features
- **EWMA smoothing** (α=0.3): Temporal smoothing
- **Delta features**: First differences to capture changes
- **CUSUM**: Cumulative sum relative to median baseline
- **Entropy variance**: Rolling variance (window=5)
- **Reallocation rate**: |Δgold_attn| / (|gold_attn| + ε)

### Two-Stage Architecture
1. **Stage 1 (High Recall)**:
   - KernelCPD with RBF kernel
   - Penalty scaled by variance
   - Keep top-5 candidates per sequence
   
2. **Stage 2 (High Precision)**:
   - Window features around each candidate
   - Gradient Boosting classifier
   - Out-of-fold prediction to avoid bias

## Requirements

```
numpy
pandas
scikit-learn
ruptures
torch
transformers
tqdm
matplotlib
```

## Input Data Format

Required columns:
- `sequence_id`: Unique sequence identifier
- `sentence_index` or `chunk_index`: Time ordering
- `hallu_label`: Hallucination labels (0=none, 1+=hallucination)
- `model_answer_chunk`: Individual sentence text (for extraction)
- `gold_text_chunk`: Gold/evidence text
- `distractor_text`: Distractor text

Pre-extracted attention columns (if skipping extraction):
- `sent_avg_entropy`
- `sent_cross_attention_to_gold`
- `sent_distractor_attention_max`

## Outputs

Each experiment saves:
- Sweep results CSV with metrics per configuration
- Best model/threshold selection
- PR curves (PNG)
- Candidate features (for E7/E8)

## Citation

If you use this code, please cite the original work on hallucination detection in LLMs.

## License

MIT License
