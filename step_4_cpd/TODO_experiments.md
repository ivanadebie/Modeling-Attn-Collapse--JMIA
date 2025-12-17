# CPD Experiments - Tasks TODO

## Status: CPD Pipeline Complete ✅
All core functionality is implemented:
- ✅ Feature engineering (fixed_cpd.py)
- ✅ PELT change point detection (change_pt_detection.py)
- ✅ Performance metrics (precision, recall, F1, delay)
- ✅ Smoothing & normalization
- ✅ Result aggregation

---

## Tasks Remaining

### 1. Sanity Checks (Individual Features)
**Goal**: Validate CPD on single features before multivariate analysis

**Fixed Configuration**:
- Dataset: `domain1` (closed_book_qa)
- Distractor density: `Medium`
- Interference type: `nonsensical`

**Features to Test** (one at a time):
1. `sentence_level_hallu_score` (or `hallu_score` / `chunk_hallu_score`)
2. `distractor_density_chunk`
3. `evidence_overlap` (TBD: which column? `evidence_overlap_ratio`, `evid_overlap_emb`, or `evid_overlap_ngram`)
4. `interference_score_lexical_wrt_distractors`
5. `evidence_position` (or `evidence_position_encoded`)
6. `attn_avg_entropy` (or `att_avg_entropy`)
7. `cross_attn_gold` (or `cross_attention_mass_to_gold`)
8. `distractor_Attn_max` (or `distractor_attention_max`)

**TODO**:
- [ ] Create function `run_sanity_checks()` in new file `run_sanity_experiments.py`
- [ ] Filter dataset for fixed config: `(domain1, Medium, nonsensical)`
- [ ] For each feature, run PELT and evaluate
- [ ] Save results to `output/sanity_checks_results.csv`
- [ ] Generate plots showing which features detect change points

**Expected Outcome**:
- Identify which individual features are discriminative
- Establish baseline performance
- Verify ground truth labels are reasonable

---

### 2. Grid Search Experiments
**Goal**: Systematic evaluation across all configurations

**Dimensions**:
- **Datasets**: 7 domains (domain1-domain7)
- **Evidence Position**: Beg, Mid, End (3 levels)
- **Distractor Density**: Low, Medium, High (3 levels)
- **Interference Type**: nonsensical, related, etc. (check actual values in data)

**Total Combinations**: 7 × 3 × 3 × N = ~189+ unique configs

**TODO**:
- [ ] Create function `run_grid_search_experiments()` in `run_grid_experiments.py`
- [ ] Extract unique levels from dataset:
  ```python
  datasets = df['domain'].unique()
  evidence_pos_levels = df['evidence_pos_category'].unique()  # or extract from config
  density_levels = df['density_level'].unique()
  interference_levels = df['interference_type'].unique()
  ```
- [ ] Implement nested loop structure:
  ```python
  for dataset in datasets:
    for evidence_pos in evidence_pos_levels:
      for density in density_levels:
        for interference in interference_levels:
          key = (dataset, evidence_pos, density, interference)
          # Filter data for this config
          # Build signal
          # Run PELT
          # Store metrics
  ```
- [ ] Use existing `build_signal_for_key()` logic (or adapt from `run_cpd_on_features()`)
- [ ] Save per-config results to `output/grid_search_results.csv`
- [ ] Aggregate across configs to answer research questions

**Expected Outcome**:
- Understand how context manipulation affects hallucination onset
- Identify which configurations make change points harder to detect
- Provide evidence for attention collapse hypothesis

---

### 3. Column Name Verification
**TODO**:
- [ ] Check actual column names in `prepared_dataset_cpd_fixed.csv`
- [ ] Map requested features to actual column names:
  - `sentence_level_hallu_score` → `hallu_score`? `chunk_hallu_score`?
  - `evidence_overlap` → `evidence_overlap_ratio`? `evid_overlap_emb`?
  - `attn_avg_entropy` → `att_avg_entropy`?
  - `cross_attn_gold` → `cross_attention_mass_to_gold`?
  - `distractor_Attn_max` → `distractor_attention_max`?
- [ ] Update feature lists accordingly

---

### 4. Parameter Configuration
**Current Settings** (in change_pt_detection.py):
- Model: `l2` (L2 cost) ✓
- Penalty: `1.0`
- Min size: Not explicitly set (default=2)
- Smoothing window: `3`
- Normalization: `RobustScaler`

**TODO**:
- [ ] Set `min_size=5` as specified in requirements
- [ ] Consider making penalty tunable via argparse (for future experiments)
- [ ] Document why `l2` was chosen over other models

---

### 5. Integration & Execution
**TODO**:
- [ ] Create master script `run_all_experiments.py`:
  ```python
  # Step 1: Sanity checks
  run_sanity_checks(data_file, output_dir)
  
  # Step 2: Grid search (only if sanity checks pass)
  run_grid_search_experiments(data_file, output_dir)
  
  # Step 3: Aggregate and visualize
  aggregate_all_results(output_dir)
  generate_summary_plots(output_dir)
  ```
- [ ] Add command-line args for flexibility
- [ ] Add logging/progress tracking
- [ ] Estimate runtime (could be long for 189+ configs)

---

## File Structure (Proposed)

```
step_4_cpd/
├── cpd_code/
│   ├── change_pt_detection.py        [DONE - core CPD functions]
│   ├── analysis_utils.py              [DONE - utilities]
│   └── hall_riskmain.py               [DONE - main runner]
├── dataset_prep_code/
│   └── prepared_dataset_cpd_fixed.csv [DONE - feature-engineered data]
├── experiments/                       [NEW]
│   ├── run_sanity_experiments.py      [TODO]
│   ├── run_grid_experiments.py        [TODO]
│   └── run_all_experiments.py         [TODO - orchestrator]
├── output/
│   ├── sanity_checks_results.csv      [OUTPUT]
│   ├── grid_search_results.csv        [OUTPUT]
│   └── final_summary.csv              [OUTPUT]
└── TODO_experiments.md                [THIS FILE]
```

---

## Quick Start

### Step 1: Verify Data
```bash
cd step_4_cpd
python -c "import pandas as pd; df=pd.read_csv('dataset_prep_code/prepared_dataset_cpd_fixed.csv'); print(df.columns.tolist())"
```

### Step 2: Sanity Checks
```bash
python experiments/run_sanity_experiments.py --data_file dataset_prep_code/prepared_dataset_cpd_fixed.csv --output_dir output/sanity
```

### Step 3: Grid Search
```bash
python experiments/run_grid_experiments.py --data_file dataset_prep_code/prepared_dataset_cpd_fixed.csv --output_dir output/grid
```

### Step 4: Review Results
```bash
# Check sanity results
cat output/sanity/sanity_checks_results.csv

# Check grid results
cat output/grid/grid_search_results.csv
```

---

## Notes

- **Sanity checks MUST pass** before running grid search (validates pipeline)
- Grid search may take significant time (consider parallelization)
- Results will help determine if attention collapse is detectable via CPD
- May need to tune penalty parameter after initial results
