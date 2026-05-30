"""
Experiment modules for CPD experiments.

This package contains experiment implementations:
- E1-E5: Single-stage CPD experiments (e1_e5_single_stage.py)
- E7: Two-stage CPD experiment (e7_two_stage.py, run_e7_experiment.py)
- E8: Feature ablation experiments (e8_ablation.py)
"""

from .e1_e5_single_stage import (
    EXPERIMENTS as E1_E5_EXPERIMENTS,
    run_single_experiment,
    run_all_experiments as run_e1_e5,
    create_summary
)

from .e7_two_stage import (
    generate_candidates,
    evaluate_stage1,
    sweep_stage1_penalties,
    build_candidate_features,
    predict_candidates_oof,
    evaluate_two_stage,
    optimize_threshold,
    evaluate_sequence_level,
    run_e7_experiment
)

from .e8_ablation import (
    define_feature_groups,
    cv_predict_proba,
    run_ablation,
    analyze_ablation_results,
    run_e8_ablation
)

# NEW: Import from run_e7_experiment.py (standalone E7 pipeline)
from .run_e7_experiment import (
    run_e7_pipeline,
    Config as E7Config,
    generate_candidates as generate_candidates_v2,
    build_candidate_features as build_candidate_features_v2,
    train_classifier_oof,
    threshold_sweep,
    evaluate_threshold,
    engineer_features,
    build_onset_map
)

__all__ = [
    # E1-E5
    "E1_E5_EXPERIMENTS",
    "run_single_experiment",
    "run_e1_e5",
    "create_summary",
    
    # E7 (from e7_two_stage.py)
    "generate_candidates",
    "evaluate_stage1",
    "sweep_stage1_penalties",
    "build_candidate_features",
    "predict_candidates_oof",
    "evaluate_two_stage",
    "optimize_threshold",
    "evaluate_sequence_level",
    "run_e7_experiment",
    
    # E7 (from run_e7_experiment.py - NEW)
    "run_e7_pipeline",
    "E7Config",
    "generate_candidates_v2",
    "build_candidate_features_v2",
    "train_classifier_oof",
    "threshold_sweep",
    "evaluate_threshold",
    "engineer_features",
    "build_onset_map",
    
    # E8
    "define_feature_groups",
    "cv_predict_proba",
    "run_ablation",
    "analyze_ablation_results",
    "run_e8_ablation"
]
