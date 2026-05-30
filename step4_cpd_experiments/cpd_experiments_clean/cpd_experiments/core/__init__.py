"""
Core modules for CPD experiments.

This package contains the core functionality for:
- Attention extraction (attention_extraction.py)
- CPD algorithms (cpd_algorithms.py)
- Feature engineering (feature_engineering.py)
- Data merging (data_merging.py)
"""

from .attention_extraction import AttentionExtractor, concat_gold_chunks
from .cpd_algorithms import safe_pelt, safe_kernel_cpd
from .feature_engineering import (
    add_sentence_index,
    add_ewma_features,
    add_delta_features,
    add_cusum_features,
    add_variance_features,
    add_reallocation_rate,
    add_gold_attention_drop,
    engineer_all_features,
    extract_window_features,
    extract_jump_features
)
from .data_merging import (
    load_and_merge_all,
    load_main_dataset,
    load_sentence_attention,
    merge_datasets,
    add_sentence_index,
    validate_merge,
    get_merge_info
)

__all__ = [
    # Attention extraction
    "AttentionExtractor",
    "concat_gold_chunks",
    
    # CPD algorithms
    "safe_pelt",
    "safe_kernel_cpd",
    
    # Feature engineering
    "add_sentence_index",
    "add_ewma_features",
    "add_delta_features",
    "add_cusum_features",
    "add_variance_features",
    "add_reallocation_rate",
    "add_gold_attention_drop",
    "engineer_all_features",
    "extract_window_features",
    "extract_jump_features",
    
    # Data merging
    "load_and_merge_all",
    "load_main_dataset",
    "load_sentence_attention",
    "merge_datasets",
    "validate_merge",
    "get_merge_info",
]
