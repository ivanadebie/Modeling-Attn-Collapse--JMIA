#!/usr/bin/env python3
"""
Main entry point for CPD Hallucination Detection Experiments.

Usage:
    # Run all experiments
    python run_experiments.py --data path/to/data.csv --output ./output

    # Run specific experiments
    python run_experiments.py --data path/to/data.csv --experiments E1-E5 E7

    # Run with specific domain
    python run_experiments.py --data path/to/data.csv --domain 01_nq_closed_book
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

from cpd_experiments.configs import ExperimentConfig, DEFAULT_CONFIG
from cpd_experiments.core import engineer_all_features, add_sentence_index
from cpd_experiments.experiments import run_e1_e5, run_e7_experiment, run_e8_ablation


def load_and_prepare_data(data_path: str, domain: str = None) -> pd.DataFrame:
    """Load and prepare data for experiments."""
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path, low_memory=False)
    
    print(f"Loaded {len(df)} rows, {df['sequence_id'].nunique() if 'sequence_id' in df.columns else 'N/A'} sequences")
    
    # Filter domain if specified
    if domain and "domain" in df.columns:
        df = df[df["domain"] == domain]
        print(f"Filtered to domain '{domain}': {len(df)} rows")
    
    # Ensure sentence_index exists
    if "sentence_index" not in df.columns:
        print("Adding sentence_index...")
        df = add_sentence_index(df)
    
    return df


def run_extraction(df: pd.DataFrame, output_dir: Path, config: ExperimentConfig) -> pd.DataFrame:
    """Run attention extraction (if needed)."""
    # Check if extraction already done
    extraction_file = output_dir / "sentence_level_attention.csv"
    
    if extraction_file.exists():
        print(f"Loading existing extraction from {extraction_file}")
        return pd.read_csv(extraction_file)
    
    print("\n" + "="*60)
    print("ATTENTION EXTRACTION")
    print("Note: This requires GPU and the LongChat model.")
    print("If extraction already exists, place it at:")
    print(f"  {extraction_file}")
    print("="*60)
    
    # For now, assume extraction is pre-done
    # Full extraction code would go here
    print("Skipping extraction - using pre-extracted features")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Run CPD Hallucination Detection Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_experiments.py --data data.csv --output ./output
  python run_experiments.py --data data.csv --experiments E7 --domain 01_nq_closed_book
        """
    )
    
    parser.add_argument("--data", required=True, help="Path to input data CSV")
    parser.add_argument("--output", default="./output", help="Output directory")
    parser.add_argument("--domain", default=None, help="Filter to specific domain")
    parser.add_argument(
        "--experiments", 
        nargs="+", 
        default=["E1-E5", "E7", "E8"],
        choices=["E1-E5", "E7", "E8", "all"],
        help="Experiments to run"
    )
    parser.add_argument("--skip-extraction", action="store_true",
                        help="Skip attention extraction (use pre-extracted features)")
    
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Expand "all" to all experiments
    experiments = args.experiments
    if "all" in experiments:
        experiments = ["E1-E5", "E7", "E8"]
    
    print("="*60)
    print("CPD HALLUCINATION DETECTION EXPERIMENTS")
    print("="*60)
    print(f"Data: {args.data}")
    print(f"Output: {args.output}")
    print(f"Domain: {args.domain or 'all'}")
    print(f"Experiments: {experiments}")
    print("="*60)
    
    # Load data
    df = load_and_prepare_data(args.data, args.domain)
    
    # Feature engineering (if not already done)
    if "sent_avg_entropy_ewma" not in df.columns:
        print("\nEngineering features...")
        df = engineer_all_features(df)
    
    # Run experiments
    results = {}
    
    if "E1-E5" in experiments:
        print("\n" + "="*60)
        print("RUNNING E1-E5: Single-Stage CPD")
        print("="*60)
        e1_e5_dir = output_dir / "E1_E5"
        results["E1-E5"] = run_e1_e5(df, output_dir=e1_e5_dir)
    
    if "E7" in experiments:
        print("\n" + "="*60)
        print("RUNNING E7: Two-Stage Detection")
        print("="*60)
        e7_dir = output_dir / "E7"
        results["E7"] = run_e7_experiment(df, output_dir=e7_dir)
    
    if "E8" in experiments:
        print("\n" + "="*60)
        print("RUNNING E8: Feature Ablation")
        print("="*60)
        
        # E8 requires E7 candidate features
        if "E7" in results and "candidate_features" in results["E7"]:
            cand_df = results["E7"]["candidate_features"]
        else:
            # Try to load from file
            cand_file = output_dir / "E7" / "candidate_features.csv"
            if cand_file.exists():
                cand_df = pd.read_csv(cand_file)
            else:
                print("Warning: E8 requires E7 candidate features. Run E7 first.")
                cand_df = None
        
        if cand_df is not None:
            e8_dir = output_dir / "E8"
            results["E8"] = run_e8_ablation(cand_df, df, output_dir=e8_dir)
    
    # Final summary
    print("\n" + "="*60)
    print("EXPERIMENTS COMPLETE")
    print("="*60)
    print(f"Results saved to: {output_dir}")
    
    if "E7" in results:
        best = results["E7"].get("best_threshold_result", {})
        print(f"\nBest E7 Two-Stage result:")
        print(f"  F1: {best.get('F1', 'N/A'):.4f}")
        print(f"  Precision: {best.get('Precision', 'N/A'):.4f}")
        print(f"  Recall: {best.get('Recall', 'N/A'):.4f}")
        print(f"  AP: {results['E7'].get('ap', 'N/A'):.4f}")


if __name__ == "__main__":
    main()
