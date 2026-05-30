"""
Configuration settings for CPD experiments.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path


@dataclass
class ModelConfig:
    """Model configuration."""
    model_name: str = "lmsys/longchat-13b-16k"
    max_tokens: int = 4096
    device_map: str = "auto"
    torch_dtype: str = "float16"
    attn_implementation: str = "eager"


@dataclass
class DataConfig:
    """Data configuration."""
    dataset_path: str = "prepared_dataset_cpd_with_attn_metrics_FINAL_v2.csv"
    domain: str = "01_nq_closed_book"
    response_col: str = "model_answer_chunk"  # Sentence-level (not chunk)
    sequence_col: str = "sequence_id"
    label_col: str = "hallu_label"
    
    # Column mappings
    gold_col: str = "gold_text_chunk"
    distractor_col: str = "distractor_text"
    chunk_index_col: str = "chunk_index"


@dataclass
class FeatureConfig:
    """Feature engineering configuration."""
    ewma_alpha: float = 0.3
    rolling_window: int = 5
    
    # Base sentence-level features
    sent_features: List[str] = field(default_factory=lambda: [
        "sent_avg_entropy",
        "sent_distractor_attention_max",
        "sent_cross_attention_mass_to_gold_raw"
    ])
    
    # Derived features
    delta_features: List[str] = field(default_factory=lambda: [
        "delta_entropy",
        "gold_attn_drop",
        "delta_distractor_max"
    ])
    
    # CPD 1.5 features
    cpd_features: List[str] = field(default_factory=lambda: [
        "entropy_variance",
        "attn_reallocation_rate",
        "delta_evid_overlap_emb",
        "delta_interference_lex"
    ])
    
    # CUSUM features
    cusum_features: List[str] = field(default_factory=lambda: [
        "gold_cusum",
        "entropy_cusum",
        "overlap_cusum"
    ])


@dataclass
class CPDConfig:
    """CPD algorithm configuration."""
    # Ground truth
    gt_mode: str = "onset"  # First 0 → label>0 transition
    tolerance: int = 2  # Sentences
    min_size: int = 2
    
    # PELT parameters
    pelt_penalties: List[float] = field(default_factory=lambda: [0.05, 0.1, 0.25, 0.5, 1.0])
    pelt_cost: str = "l2"
    
    # KernelCPD parameters
    kernel_type: str = "rbf"
    kernel_penalties: List[float] = field(default_factory=lambda: [0.05, 0.1, 0.25, 0.5, 1.0])


@dataclass
class E7Config:
    """E7 Two-Stage configuration."""
    # Stage 1: Candidate generation
    stage1_pen_scales: List[float] = field(default_factory=lambda: [0.05, 0.1, 0.25, 0.5, 1.0])
    max_candidates_per_seq: int = 5
    
    # Stage 2: Classifier
    classifier_type: str = "GradientBoosting"
    window_size: int = 5
    test_size: float = 0.3
    random_state: int = 42
    
    # Threshold optimization
    threshold_range: List[float] = field(default_factory=lambda: [
        0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
        0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9
    ])
    
    # Feature groups for Stage 2
    clf_base_features: List[str] = field(default_factory=lambda: [
        "entropy_variance", "delta_entropy", "gold_attn_drop", "delta_distractor_max",
        "attn_reallocation_rate", "delta_evid_overlap_emb", "delta_interference_lex",
        "sent_avg_entropy_ewma", "sent_distractor_attention_max_ewma",
        "sent_cross_attention_mass_to_gold_raw_ewma",
        "evidence_position_num_ewma", "interference_score_lexical_wrt_distractors_ewma",
        "evid_overlap_emb_ewma",
        "gold_cusum", "entropy_cusum", "overlap_cusum"
    ])


@dataclass
class OutputConfig:
    """Output configuration."""
    base_dir: Path = Path("./output")
    
    # Subdirectories
    extraction_dir: str = "extraction"
    cpd_single_stage_dir: str = "cpd_single_stage"
    cpd_two_stage_dir: str = "cpd_two_stage"
    ablation_dir: str = "ablation"
    visualization_dir: str = "visualization"
    
    def get_path(self, subdir: str) -> Path:
        """Get full path for a subdirectory."""
        path = self.base_dir / subdir
        path.mkdir(parents=True, exist_ok=True)
        return path


@dataclass
class ExperimentConfig:
    """Master experiment configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    cpd: CPDConfig = field(default_factory=CPDConfig)
    e7: E7Config = field(default_factory=E7Config)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    # Experiment selection
    run_extraction: bool = True
    run_e1_e5: bool = True
    run_e7: bool = True
    run_e8_ablation: bool = True
    run_visualization: bool = True


# Default configuration instance
DEFAULT_CONFIG = ExperimentConfig()
