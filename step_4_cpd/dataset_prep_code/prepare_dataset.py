from __future__ import annotations
import sys
from pathlib import Path
import argparse
import ast
import pandas as pd
import numpy as np
import re
from tqdm.auto import tqdm

# Add project root to path to allow for absolute imports
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Assuming metrics_lib is in the same directory or accessible via python path
from dataset_prep_code.metrics_lib import (
    jaccard, 
    ngram_set, 
    preprocess_tokens, 
    tokenize, 
    try_embedding_similarity,
    find_span,
    gold_position_from_span
)

# ==============================================================================
# 1. CHUNKING LOGIC
# ==============================================================================

def _split_into_chunks(text: str, max_tokens: int = 64) -> list[str]:
    """
    Splits text into sentences, with a fallback to split by token count for
    long sentences.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    sanitized_text = text.replace('\\n', ' ').replace('\\r', ' ').replace('\n', ' ').replace('\r', ' ').strip()
    
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', sanitized_text)
    
    chunks = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        tokens = tokenize(sentence)
        if len(tokens) > max_tokens:
            # If sentence is too long, split it into chunks by tokens
            for i in range(0, len(tokens), max_tokens):
                chunks.append(" ".join(tokens[i:i+max_tokens]))
        else:
            chunks.append(sentence)
            
    return chunks

# ==============================================================================
# 2. DATA LOADING AND PREPARATION
# ==============================================================================

def load_and_prepare_data(file_paths: list[str]) -> pd.DataFrame:
    """
    Loads data from CSV files, concatenates them, and performs initial cleaning.
    """
    print("Step 1: Loading and concatenating data...")
    df_list = [pd.read_csv(file_path) for file_path in file_paths]
    combined_df = pd.concat(df_list, ignore_index=True)
    
    # Drop rows where question_id is missing, as they are unusable for grouping
    combined_df.dropna(subset=['question_id'], inplace=True)

    # Safely evaluate string representations of lists
    for col in ['distractors', 'gold_text']:
        if col in combined_df.columns:
            combined_df[col] = combined_df[col].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) and x.strip().startswith('[') and x.strip().endswith(']') else ([] if pd.isna(x) else [str(x)])
            )

        combined_df = combined_df.rename(columns={'Answer': 'model_answer', 'Question': 'question', 'Context': 'prompt'})
    
    # Pre-compute full text for gold and distractors
    combined_df['full_gold_text'] = combined_df['gold_text'].apply(lambda x: " ".join(x) if isinstance(x, list) else "")
    combined_df['full_distractor_text'] = combined_df['distractors'].apply(lambda x: " ".join(x) if isinstance(x, list) else "")

    print(f"Loaded {len(combined_df)} total rows from {len(file_paths)} files.")
    return combined_df

# ==============================================================================
# 3. ROBUST CHUNKING AND EXPLODING
# ==============================================================================

def create_chunk_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs a robust 'explode' operation to transform the dataframe from
    one-row-per-answer to one-row-per-chunk.
    """
    print("Step 2: Splitting answers into chunks and exploding DataFrame...")
    
    # Apply the chunking function to the model_answer column
    df['model_answer_chunk'] = df['model_answer'].apply(_split_into_chunks)
    
    # Explode the dataframe based on the new chunks column
    chunk_df = df.explode('model_answer_chunk').reset_index(drop=True)

    print(f"Successfully exploded into {len(chunk_df)} chunks.")
    return chunk_df

# ==============================================================================
# 4. FEATURE CALCULATION
# ==============================================================================

def calculate_chunk_level_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates features for each individual chunk."""
    print("Step 3: Calculating chunk-level features...")
    
    # Create a temporary copy with unique question_ids for run-level calculations
    # that are then merged back. This is for efficiency.
    run_level_df = df.drop_duplicates(subset=['question_id']).copy()

    # --- Calculate Run-Level Features first ---
    def calculate_distractor_metrics(row):
        distr_tokens = set(tokenize(row['full_distractor_text']))
        gold_tokens = set(tokenize(row['full_gold_text']))
        
        denom = (len(gold_tokens) + len(distr_tokens)) or 1
        distractor_ratio = len(distr_tokens) / denom
        
        if distractor_ratio < 0.33:
            distractor_density = "low"
        elif distractor_ratio < 0.66:
            distractor_density = "medium"
        else:
            distractor_density = "high"
            
        return pd.Series([distractor_density])

    run_level_df[['distractor_density']] = run_level_df.apply(calculate_distractor_metrics, axis=1)
    dist_density_mapping = {"low": 0, "medium": 1, "high": 2}
    run_level_df['distractor_density'] = run_level_df['distractor_density'].map(dist_density_mapping)

    def calculate_other_run_level_metrics(row):
        search_context = str(row.get('prompt', '')) + " " + str(row.get('full_gold_text', ''))
        search_tokens = tokenize(search_context)
        gold_tokens = tokenize(row.get('full_gold_text', ''))
        
        gold_span_start, gold_span_end = find_span(search_tokens, gold_tokens)
        gold_position = gold_position_from_span(gold_span_start, gold_span_end)

        return pd.Series([gold_span_start, gold_span_end, gold_position])

    tqdm.pandas(desc="Calculating run-level metrics (spans, positions)")
    run_level_df[['gold_span_start', 'gold_span_end', 'gold_position']] = run_level_df.progress_apply(calculate_other_run_level_metrics, axis=1)

    cols_to_merge = [
        'question_id', 'distractor_density', 'gold_span_start', 'gold_span_end', 'gold_position'
    ]
    df = df.merge(run_level_df[cols_to_merge], on='question_id', how='left')

    # --- Calculate Chunk-Specific Features ---
    def calculate_features_for_row(row):
        chunk = row['model_answer_chunk']
        chunk_tokens = tokenize(chunk)
        chunk_token_set = set(chunk_tokens)
        
        # is_gold_binary (more robust check)
        gold_text_full = row['full_gold_text']
        is_gold_binary = 0
        if gold_text_full and chunk:
            chunk_ngrams = ngram_set(preprocess_tokens(chunk, drop_stop=True), 2)
            gold_ngrams = ngram_set(preprocess_tokens(gold_text_full, drop_stop=True), 2)
            if jaccard(chunk_ngrams, gold_ngrams) > 0.2:
                is_gold_binary = 1
        
        # interference_score_lexical
        distractor_tokens = set(tokenize(row['full_distractor_text']))
        interference_hits = len(chunk_token_set.intersection(distractor_tokens))
        interference_score = interference_hits / len(chunk_token_set) if chunk_token_set else 0

        # evidence_overlap_ratio (chunk vs full gold text)
        evid_overlap_ngram = jaccard(
            ngram_set(preprocess_tokens(chunk), 3),
            ngram_set(preprocess_tokens(row['full_gold_text']), 3)
        )
        
        # Embedding overlap (chunk vs full gold text)
        evid_overlap_emb = try_embedding_similarity(chunk, row['full_gold_text'])
                        
    tqdm.pandas(desc="Calculating chunk features")
    df[[
        'is_gold_binary', 'interference_score_lexical_wrt_distractors', 
        'evid_overlap_ngram', 'evid_overlap_emb', 'hallu_score',
        'total_response_tokens', 'interference_token_hits'
    ]] = df.progress_apply(calculate_features_for_row, axis=1)
    
    # This is a chunk-level metric representing lexical overlap with distractors.
    df['distractor_density_chunk'] = df['interference_score_lexical_wrt_distractors']

    # interference_type (categorical from score)
    conditions = [
        df['interference_score_lexical_wrt_distractors'].lt(0.15),
        df['interference_score_lexical_wrt_distractors'].lt(0.50)
    ]
    choices = [0, 1] # 0: Nonsensical, 1: Thematic
    df['interference_type_lexical_wrt_distractors'] = np.select(conditions, choices, default=2) # 2: Paraphrased

    # hallu_label (categorical from score)
    conditions = [
        df['hallu_score'] < 0.25,
        df['hallu_score'] < 0.50
    ]
    choices = [0, 1] # 0: Low, 1: Medium
    df['hallu_label'] = np.select(conditions, choices, default=2) # 2: High

    return df

def calculate_positional_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates features that depend on the position within a group of chunks."""
    print("Step 4: Calculating positional features...")

    def apply_positional(group):
        group = group.copy()
        group['chunk_index'] = range(len(group))
        total_chunks = len(group)
        
        if total_chunks <= 1:
            group['normalized_chunk_index'] = 0
            group['evidence_position'] = 1.0
            return group

        gold_chunk_indices = group.index[group['is_gold_binary'] == 1]
        
        if not gold_chunk_indices.empty:
            # Calculate distance to the nearest gold chunk for each chunk
            # Using index directly since we are in a group context
            distances = group.index.map(lambda i: min(abs(i - gc) for gc in gold_chunk_indices))
            group['evidence_position'] = distances / (total_chunks - 1)
        else:
            group['evidence_position'] = 1.0 # Max distance if no gold chunks
            
        group['normalized_chunk_index'] = group['chunk_index'] / (total_chunks - 1)
        return group

    tqdm.pandas(desc="Calculating positional features")
    # Group by the original question ID to process chunks from the same answer together
    df = df.groupby("question_id", group_keys=False).progress_apply(apply_positional)
    return df

# ==============================================================================
# 5. NORMALIZATION AND FINALIZATION
# ==============================================================================

def normalize_features(df: pd.DataFrame) -> pd.DataFrame:
    """Applies Z-score normalization to specified numeric columns."""
    print("Step 5: Applying Z-score normalization...")
    
    # Identify all numeric columns
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    # Define columns to exclude from normalization (identifiers, flags, pre-scaled)
    cols_to_exclude = {
        'question_id', 'chunk_index', 'is_gold_binary', 
        'interference_type_lexical_wrt_distractors', 'evidence_position', 
        'normalized_chunk_index', 'interference_score_lexical_wrt_distractors',
        'evidence_overlap_ratio', 'hallu_score',
        'distractor_density', 'evid_overlap_ngram', 'evid_overlap_emb',
        'gold_span_start', 'gold_span_end', 'gold_position', 'total_response_tokens', 'interference_token_hits'
    }
    
    cols_to_normalize = [col for col in numeric_cols if col not in cols_to_exclude and df[col].nunique() > 1]

    if not cols_to_normalize:
        print("No columns to normalize.")
        return df

    print(f"Normalizing columns: {cols_to_normalize}")
    for col in cols_to_normalize:
        mean = df[col].mean()
        std = df[col].std()
        if std > 0:
            df[f'{col}_z_score'] = (df[col] - mean) / std
        else:
            df[f'{col}_z_score'] = 0
            
    return df

def finalize_and_save(df: pd.DataFrame, output_file: str):
    """Selects final columns, reorders them, and saves the dataset."""
    print("Step 6: Finalizing and saving the dataset...")
    
    # Define the desired final column order
    final_columns = [
        'question_id', 'question', 'model_answer', 'distractors', 'model_answer_chunk', 'chunk_index',
        'is_gold_binary', 'evidence_position', 'normalized_chunk_index',
        'interference_score_lexical_wrt_distractors', 
        'interference_type_lexical_wrt_distractors',
        'evidence_overlap_ratio', 'hallu_score', 'hallu_label',
        'distractor_density', 'evid_overlap_ngram', 'evid_overlap_emb',
        'gold_span_start', 'gold_span_end', 'gold_position', 'total_response_tokens', 'interference_token_hits'
    ]
    
    # Add placeholder columns and set their values to empty
    placeholder_cols = [
        'att_avg_entropy', 'attention_entropy', 
        'cross_attention_mass_to_gold', 'distractor_attention_max'
    ]
    for col in placeholder_cols:
        df[col] = np.nan

    final_columns.extend(placeholder_cols)

    # Add any other original columns that might be useful, excluding temporary ones
    original_cols_to_keep = [c for c in df.columns if c not in final_columns and '_z_score' not in c and c not in ['full_gold_text', 'full_distractor_text', 'gold_text', 'prompt', 'context_for_search']]
    
    # Ensure all columns in final_columns exist in the dataframe, add if missing
    for col in final_columns:
        if col not in df.columns:
            df[col] = np.nan

    # Reorder and select
    final_df = df[final_columns + original_cols_to_keep]

    print(f"Dataset shape: {final_df.shape}")
    print(f"Saving prepared dataset to {output_file}...")
    final_df.to_csv(output_file, index=False)
    print("Done.")

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """Main pipeline for preparing the change-point detection dataset."""
    # --- Configuration ---
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
    
    # --- Pipeline ---
    base_df = load_and_prepare_data(input_files)
    chunked_df = create_chunk_df(base_df)
    # All feature calculations are now performed on the chunked dataframe
    features_df = calculate_chunk_level_features(chunked_df)
    positional_df = calculate_positional_features(features_df)
    normalized_df = normalize_features(positional_df)
    finalize_and_save(normalized_df, output_file)

if __name__ == "__main__":
    main()
