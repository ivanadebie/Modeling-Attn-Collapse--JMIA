from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path to allow for absolute imports
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import argparse
import ast
import json
import pandas as pd

from step_4_cpd.metrics_lib import compute_metrics_for_dataset

def load_and_prepare_data(file_paths: list[str]) -> pd.DataFrame:
    """
    Loads data from one or more CSV files, concatenates them,
    safely evaluates list-like columns, and prepares it for metric calculation.
    """
    df_list = []
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        df_list.append(df)
    
    combined_df = pd.concat(df_list, ignore_index=True)
    
    # Safely evaluate string representations of lists
    for col in ['Distractor_Sentences', 'Gold_Sentences', 'Combined_Sentences']:
        if col in combined_df.columns and not combined_df[combined_df[col].notna()].empty and isinstance(combined_df[combined_df[col].notna()].iloc[0], str):
            combined_df[col] = combined_df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else x)

    # Rename columns to match the expected format for the metrics library
    # This creates a "bridge" between the CSV format and the expected dict format
    combined_df = combined_df.rename(columns={
        'Prompt': 'prompt',
        'Question': 'question',
        'Answer': 'model_answer',
        'Gold_Sentences': 'gold_text', # Assuming gold_text is a list of sentences
        'Distractor_Sentences': 'distractors'
    })

    return combined_df

def main():
    # --- Hardcoded file paths ---
    input_files = [
        "step_4_cpd/scored_answers/answer_hallu_scored_01_nq_closed_book.csv",
        "step_4_cpd/scored_answers/answer_hallu_scored_02_hotpot_citation.csv",
        "step_4_cpd/scored_answers/answer_hallu_scored_03_novelhop_multihop.csv",
        "step_4_cpd/scored_answers/answer_hallu_scored_04_temporal_recency.csv",
        "step_4_cpd/scored_answers/answer_hallu_scored_05_stackmath_numerical.csv",
        "step_4_cpd/scored_answers/answer_hallu_scored_06_policy_compliance.csv",
        "step_4_cpd/scored_answers/answer_hallu_scored_07_expert_legal_med.csv",
    ]
    output_file = "step_4_cpd/prepared_dataset_cpd.csv"
    use_embeddings = True # Set to False to disable embedding calculations
    # --- End of hardcoded paths ---

    # 1. Load and prepare data
    print(f"Loading and preparing data from {len(input_files)} file(s)...")
    base_df = load_and_prepare_data(input_files)

    # The metrics library expects a list of dictionaries
    data_rows = base_df.to_dict('records')

    # 2. Compute metrics
    print("Calculating run-level and chunk-level metrics...")
    run_metrics, chunk_metrics = compute_metrics_for_dataset(
        data_rows, 
        use_embeddings=use_embeddings
    )

    # 3. Merge metrics back into the main DataFrame
    print("Merging computed metrics...")
    run_metrics_df = pd.DataFrame(run_metrics)
    
    # The 'run_id' from metrics corresponds to the original DataFrame index
    final_df = base_df.merge(run_metrics_df, left_index=True, right_on='run_id', how='left')

    # Add placeholder columns for attention-based features
    # In a real scenario, these would be calculated and merged from another source
    attention_features = [
        "attention_entropy",
        "effective_rank",
        "self_attention_ratio",
        "head_similarity_ratio",
        "cross_attention_mass_to_gold",
        "distractor_attention_max"
    ]
    for feat in attention_features:
        if feat not in final_df.columns:
            final_df[feat] = None # Or np.nan

    # For now, we are not using chunk-level metrics in the final flat file,
    # but you could save them to a separate file if needed:
    # chunk_metrics_df = pd.DataFrame(chunk_metrics)
    # chunk_metrics_df.to_csv('chunk_level_data.csv', index=False)

    # 4. Save the final dataset
    print(f"Saving prepared dataset to {output_file}...")
    final_df.to_csv(output_file, index=False)
    
    print("Processing complete.")

if __name__ == "__main__":
    main()
