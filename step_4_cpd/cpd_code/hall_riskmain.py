import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from cpd_code.change_pt_detection import (
    get_all_sequence_files,
    load_features_from_csv,
    plot_advanced_visualizations,
    run_experiments,
    aggregate_results,
    plot_results,
)
from cpd_code.analysis_utils import add_config_columns, UNKNOWN_VALUE


def parse_args() -> argparse.Namespace:
# ... (argument parsing remains the same) ...
    parser = argparse.ArgumentParser(
        description="Run change point detection on all sequences and aggregate results."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../dataset_prep_code/prepared_dataset_cpd.csv",
        help="Directory or file containing the input data (default: ../dataset_prep_code/prepared_dataset_cpd.csv).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../output",
        help="Directory where the output results will be saved (default: ../output).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Configuration string for selecting specific sequences.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # --- CPD Analysis ---
    print("--- Running Change Point Detection Analysis ---")
    # Step 1: Run change point detection on all sequences
    run_experiments(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        config=args.config,
        overwrite=True,  # Always overwrite
    )

    # Step 2: Plot the CPD results
    aggregated_summary_path = Path(args.output_dir) / "aggregated_summary.csv"
    if aggregated_summary_path.exists():
        aggregated_df = pd.read_csv(aggregated_summary_path)
        plot_results(aggregated_df, args.output_dir)
        print("Finished plotting basic CPD results.")
    else:
        print("Skipping basic CPD plotting as aggregated summary file was not found.")

    # --- Advanced Visualizations ---
    print("\n--- Generating Advanced Visualizations ---")
    # Step 4: Find all feature files (.csv)
    sequence_files = get_all_sequence_files(args.data_dir, args.config)
    if not sequence_files:
        print(f"No .csv files found in {args.data_dir}. Cannot generate advanced visualizations.")
        return

    # Step 5: Load features from all files into a single DataFrame
    features_df = load_features_from_csv(sequence_files)
    if features_df.empty:
        print("No features were loaded for advanced visualizations. Exiting.")
        return

    # Step 6: Add configuration columns to the DataFrame for filtering/grouping
    features_df = add_config_columns(features_df)

    # Step 7: Generate the advanced visualizations
    plot_advanced_visualizations(features_df, args.output_dir)

    print("\nScript finished.")


if __name__ == "__main__":
    main()
