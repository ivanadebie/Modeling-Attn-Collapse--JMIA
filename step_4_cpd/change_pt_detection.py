import os
from pathlib import Path
from typing import List, Optional
import pandas as pd


def get_all_sequence_files(data_dir: str, config: str = "") -> List[str]:
    """Get all sequence files from the data directory."""
    data_path = Path(data_dir)
    if not data_path.exists():
        return []
    
    # Find all CSV or relevant data files
    sequence_files = []
    for file_path in data_path.rglob("*.csv"):
        sequence_files.append(str(file_path))
    
    return sequence_files


def run_cpd_on_all(data_dir: str, output_dir: str, config: str = "", overwrite: bool = False):
    """Run change point detection on all sequences."""
    print(f"Running CPD on data from: {data_dir}")
    print(f"Output directory: {output_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all sequence files
    sequence_files = get_all_sequence_files(data_dir, config)
    
    print(f"Found {len(sequence_files)} sequence files")
    
    # TODO: Implement actual change point detection logic
    # For now, create a placeholder results file
    results_file = Path(output_dir) / "cpd_results.csv"
    if not results_file.exists() or overwrite:
        placeholder_data = {
            "sequence_id": ["seq_1", "seq_2", "seq_3"],
            "change_point": [100, 150, 200],
            "confidence": [0.85, 0.92, 0.78]
        }
        df = pd.DataFrame(placeholder_data)
        df.to_csv(results_file, index=False)
        print(f"Saved CPD results to: {results_file}")


def aggregate_results(output_dir: str, config: str = ""):
    """Aggregate results from all sequences."""
    print(f"Aggregating results from: {output_dir}")
    
    results_file = Path(output_dir) / "cpd_results.csv"
    if results_file.exists():
        df = pd.read_csv(results_file)
        print(f"Loaded {len(df)} results")
        
        # Create aggregated summary
        summary = {
            "total_sequences": len(df),
            "avg_change_point": df["change_point"].mean(),
            "avg_confidence": df["confidence"].mean()
        }
        
        summary_file = Path(output_dir) / "aggregated_summary.csv"
        pd.DataFrame([summary]).to_csv(summary_file, index=False)
        print(f"Saved aggregated summary to: {summary_file}")
    else:
        print("No results file found to aggregate")