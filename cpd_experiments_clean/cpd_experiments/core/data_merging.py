"""
Data merging for CPD experiments.

This module handles merging lexical features (from prepare_dataset.py) with
sentence-level attention metrics (from attention_extraction.py).

Merge Key: original_index (row index from main dataset)

Usage:
    from cpd_experiments.core.data_merging import load_and_merge_all
    
    df = load_and_merge_all(
        main_path="prepared_dataset_cpd_with_attn_metrics_FINAL_v2.csv",
        sent_path="sentence_level_attention_01_nq_closed_book.csv",
        domain="01_nq_closed_book"
    )
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Tuple, Dict

from ..utils import parse_list_column


# =============================================================================
# CONFIGURATION
# =============================================================================

# Default file paths
DEFAULT_MAIN_PATH = "prepared_dataset_cpd_with_attn_metrics_FINAL_v2.csv"
DEFAULT_SENT_PATH = "sentence_level_attention_01_nq_closed_book.csv"
DEFAULT_DOMAIN = "01_nq_closed_book"

# Columns to bring from sentence-level attention CSV
SENTENCE_ATTENTION_COLS = [
    "original_index",
    "sent_avg_entropy",
    "sent_distractor_attention_max",
    "sent_cross_attention_to_gold",
    "sent_seq_len",
    "sent_resp_len",
    "sent_gold_len",
    "sent_dist_len",
    "sent_gold_found",
    "sent_dist_found",
]

# Alternative column names for gold attention (varies across datasets)
GOLD_ATTN_ALIASES = [
    "sent_cross_attention_to_gold",
    "sent_cross_attention_mass_to_gold_raw",
    "sent_gold_attn",
]


# =============================================================================
# LOADING FUNCTIONS
# =============================================================================

def load_main_dataset(path: str, domain: Optional[str] = None) -> pd.DataFrame:
    """
    Load the main lexical features dataset.
    
    Args:
        path: Path to prepared_dataset CSV
        domain: Optional domain filter (e.g., "01_nq_closed_book")
        
    Returns:
        DataFrame with original_index column added
    """
    print(f"Loading main dataset from: {path}")
    df = pd.read_csv(path, low_memory=False)
    
    # Normalize column names (strip whitespace)
    df.columns = df.columns.astype(str).str.strip()
    
    print(f"  Loaded: {len(df)} rows, {len(df.columns)} columns")
    
    # Filter by domain if specified
    if domain and "domain" in df.columns:
        df = df[df["domain"] == domain].copy()
        print(f"  Filtered to domain '{domain}': {len(df)} rows")
    
    # Create original_index from row index (this is the merge key)
    df = df.reset_index().rename(columns={"index": "original_index"})
    
    return df


def load_sentence_attention(path: str) -> pd.DataFrame:
    """
    Load the sentence-level attention metrics dataset.
    
    Args:
        path: Path to sentence_level_attention CSV
        
    Returns:
        DataFrame with attention metrics
    """
    print(f"Loading sentence attention from: {path}")
    df = pd.read_csv(path, low_memory=False)
    
    # Normalize column names
    df.columns = df.columns.astype(str).str.strip()
    
    print(f"  Loaded: {len(df)} rows, {len(df.columns)} columns")
    
    # Verify original_index exists (required for merge)
    if "original_index" not in df.columns:
        raise KeyError(
            f"Sentence attention CSV missing 'original_index' column. "
            f"Available columns: {df.columns.tolist()[:20]}"
        )
    
    return df


# =============================================================================
# MERGING FUNCTIONS
# =============================================================================

def resolve_gold_attention_column(df: pd.DataFrame) -> str:
    """
    Find the gold attention column name (varies across datasets).
    
    Args:
        df: Sentence attention DataFrame
        
    Returns:
        Column name for gold attention
    """
    for alias in GOLD_ATTN_ALIASES:
        if alias in df.columns:
            return alias
    
    # Try to find any column with 'gold' and 'attn' in name
    gold_cols = [c for c in df.columns if 'gold' in c.lower() and 'attn' in c.lower()]
    if gold_cols:
        return gold_cols[0]
    
    raise KeyError(
        f"Could not find gold attention column. "
        f"Tried: {GOLD_ATTN_ALIASES}. "
        f"Available: {[c for c in df.columns if 'gold' in c.lower() or 'attn' in c.lower()]}"
    )


def merge_datasets(
    df_main: pd.DataFrame,
    df_sent: pd.DataFrame,
    validate: bool = True
) -> pd.DataFrame:
    """
    Merge main dataset with sentence-level attention metrics.
    
    Merge key: original_index
    
    Args:
        df_main: Main dataset with lexical features (must have original_index)
        df_sent: Sentence attention dataset (must have original_index)
        validate: Whether to validate one-to-one merge
        
    Returns:
        Merged DataFrame
    """
    print("\nMerging datasets...")
    
    # Resolve gold attention column name
    gold_col = resolve_gold_attention_column(df_sent)
    print(f"  Using gold attention column: {gold_col}")
    
    # Columns to keep from sentence attention
    sent_cols_needed = [
        "original_index", 
        "sent_avg_entropy", 
        "sent_distractor_attention_max"
    ]
    if gold_col not in sent_cols_needed:
        sent_cols_needed.append(gold_col)
    
    # Add optional columns if they exist
    optional_cols = ["sent_seq_len", "sent_resp_len", "sent_gold_len", 
                     "sent_dist_len", "sent_gold_found", "sent_dist_found"]
    for col in optional_cols:
        if col in df_sent.columns and col not in sent_cols_needed:
            sent_cols_needed.append(col)
    
    # Filter to existing columns
    sent_cols_available = [c for c in sent_cols_needed if c in df_sent.columns]
    print(f"  Columns from sentence attention: {sent_cols_available}")
    
    # Create subset to avoid column collisions
    df_sent_small = df_sent[sent_cols_available].copy()
    
    # Check for overlapping columns (excluding merge key)
    main_cols = set(df_main.columns) - {"original_index"}
    sent_cols = set(df_sent_small.columns) - {"original_index"}
    overlap = main_cols & sent_cols
    
    if overlap:
        print(f"  Warning: Overlapping columns will use sentence version: {overlap}")
        df_main = df_main.drop(columns=list(overlap))
    
    # Perform merge
    df_merged = df_main.merge(
        df_sent_small,
        on="original_index",
        how="left",
        validate="one_to_one" if validate else None
    )
    
    print(f"  Merged: {len(df_merged)} rows, {len(df_merged.columns)} columns")
    
    # Normalize gold attention column name to canonical form
    canonical_gold_col = "sent_cross_attention_to_gold"
    if gold_col != canonical_gold_col and gold_col in df_merged.columns:
        df_merged[canonical_gold_col] = df_merged[gold_col]
        print(f"  Aliased {gold_col} -> {canonical_gold_col}")
    
    # Report missing rates
    attention_cols = ["sent_avg_entropy", "sent_distractor_attention_max", canonical_gold_col]
    attention_cols = [c for c in attention_cols if c in df_merged.columns]
    
    print("\n  Missing rates:")
    for col in attention_cols:
        missing_pct = 100 * df_merged[col].isna().mean()
        print(f"    {col}: {missing_pct:.2f}%")
    
    return df_merged


def add_sentence_index(
    df: pd.DataFrame,
    seq_col: str = "sequence_id",
    chunk_col: str = "chunk_index"
) -> pd.DataFrame:
    """
    Add sentence_index column (cumulative index within each sequence).
    
    Args:
        df: Merged DataFrame
        seq_col: Sequence ID column
        chunk_col: Chunk index column (for sorting)
        
    Returns:
        DataFrame with sentence_index added
    """
    print("\nAdding sentence_index...")
    
    # Determine sort columns
    sort_cols = [seq_col]
    if chunk_col in df.columns:
        sort_cols.append(chunk_col)
    if "sentence_in_chunk" in df.columns:
        sort_cols.append("sentence_in_chunk")
    elif "original_index" in df.columns:
        sort_cols.append("original_index")
    
    print(f"  Sorting by: {sort_cols}")
    
    # Sort and create cumulative index
    df = df.sort_values(sort_cols).reset_index(drop=True)
    df["sentence_index"] = df.groupby(seq_col).cumcount()
    
    print(f"  Created sentence_index: 0 to {df['sentence_index'].max()}")
    
    return df


def validate_merge(df: pd.DataFrame) -> bool:
    """
    Validate that merged DataFrame has required columns.
    
    Args:
        df: Merged DataFrame
        
    Returns:
        True if valid, raises error otherwise
    """
    required_cols = [
        "sequence_id",
        "sentence_index", 
        "hallu_label",
        "sent_avg_entropy",
        "sent_distractor_attention_max",
    ]
    
    missing = [c for c in required_cols if c not in df.columns]
    
    if missing:
        print(f"\n⚠️  WARNING: Missing required columns: {missing}")
        print(f"   Available columns: {df.columns.tolist()}")
        return False
    
    print("\n✅ Validation passed: All required columns present")
    print(f"   Sequences: {df['sequence_id'].nunique()}")
    print(f"   Rows: {len(df)}")
    
    # Label distribution
    if "hallu_label" in df.columns:
        label_dist = df["hallu_label"].value_counts().to_dict()
        print(f"   Label distribution: {label_dist}")
    
    return True


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def load_and_merge_all(
    main_path: str = DEFAULT_MAIN_PATH,
    sent_path: str = DEFAULT_SENT_PATH,
    domain: Optional[str] = DEFAULT_DOMAIN,
    output_path: Optional[str] = None,
    validate: bool = True
) -> pd.DataFrame:
    """
    Complete pipeline: load both CSVs, merge, add sentence_index.
    
    This is the main entry point for the merge logic.
    
    Args:
        main_path: Path to main dataset (lexical features)
        sent_path: Path to sentence-level attention CSV
        domain: Domain to filter (e.g., "01_nq_closed_book")
        output_path: Optional path to save merged CSV
        validate: Whether to validate the merge
        
    Returns:
        Merged DataFrame ready for feature engineering
        
    Example:
        >>> from cpd_experiments.core.data_merging import load_and_merge_all
        >>> df = load_and_merge_all(
        ...     main_path="prepared_dataset_cpd_with_attn_metrics_FINAL_v2.csv",
        ...     sent_path="sentence_level_attention_01_nq_closed_book.csv",
        ...     domain="01_nq_closed_book",
        ...     output_path="merged_dataset.csv"
        ... )
    """
    print("=" * 70)
    print("DATA MERGING PIPELINE")
    print("=" * 70)
    
    # Step 1: Load main dataset
    df_main = load_main_dataset(main_path, domain)
    
    # Step 2: Load sentence attention
    df_sent = load_sentence_attention(sent_path)
    
    # Step 3: Merge
    df_merged = merge_datasets(df_main, df_sent, validate=False)
    
    # Step 4: Add sentence index
    df_merged = add_sentence_index(df_merged)
    
    # Step 5: Validate
    if validate:
        validate_merge(df_merged)
    
    # Step 6: Save if output path provided
    if output_path:
        df_merged.to_csv(output_path, index=False)
        print(f"\n💾 Saved merged dataset to: {output_path}")
    
    print("=" * 70)
    print("MERGE COMPLETE")
    print("=" * 70)
    
    return df_merged


def get_merge_info() -> Dict:
    """
    Return information about the merge process.
    
    Returns:
        Dict with merge configuration details
    """
    return {
        "merge_key": "original_index",
        "main_dataset": DEFAULT_MAIN_PATH,
        "sentence_attention": DEFAULT_SENT_PATH,
        "default_domain": DEFAULT_DOMAIN,
        "sentence_attention_cols": SENTENCE_ATTENTION_COLS,
        "gold_attn_aliases": GOLD_ATTN_ALIASES,
    }


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Merge lexical features with sentence-level attention metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python -m cpd_experiments.core.data_merging --main prepared_dataset.csv --sent sentence_attention.csv
    
    # With domain filter and output path
    python -m cpd_experiments.core.data_merging \\
        --main prepared_dataset_cpd_with_attn_metrics_FINAL_v2.csv \\
        --sent sentence_level_attention_01_nq_closed_book.csv \\
        --domain 01_nq_closed_book \\
        --output merged_01_nq_closed_book.csv
        """
    )
    
    parser.add_argument(
        "--main", 
        default=DEFAULT_MAIN_PATH,
        help=f"Path to main dataset CSV (default: {DEFAULT_MAIN_PATH})"
    )
    parser.add_argument(
        "--sent", 
        default=DEFAULT_SENT_PATH,
        help=f"Path to sentence attention CSV (default: {DEFAULT_SENT_PATH})"
    )
    parser.add_argument(
        "--domain", 
        default=None,
        help="Domain to filter (e.g., 01_nq_closed_book). If not specified, uses all domains."
    )
    parser.add_argument(
        "--output", 
        default="merged_features.csv",
        help="Output path for merged CSV (default: merged_features.csv)"
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip validation checks"
    )
    
    args = parser.parse_args()
    
    # Run merge
    df = load_and_merge_all(
        main_path=args.main,
        sent_path=args.sent,
        domain=args.domain,
        output_path=args.output,
        validate=not args.no_validate
    )
    
    # Print summary
    print(f"\nFinal dataset shape: {df.shape}")
