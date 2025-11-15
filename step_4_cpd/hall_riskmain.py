import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from step_4_cpd.change_pt_detection import (
    get_all_sequence_files,
    load_features_from_npz,
    plot_advanced_visualizations,
    run_cpd_on_all,
    aggregate_results,
    plot_results,
)
from step_4_cpd.config_utils import add_config_columns, UNKNOWN_VALUE


def parse_args() -> argparse.Namespace:
# ... (argument parsing remains the same) ...
    parser = argparse.ArgumentParser(
        description="Run change point detection on all sequences and aggregate results."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory where the input data is stored.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where the output results will be saved.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Configuration string for selecting specific sequences.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing results if any.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # --- CPD Analysis ---
    print("--- Running Change Point Detection Analysis ---")
    # Step 1: Run change point detection on all sequences
    run_cpd_on_all(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        config=args.config,
        overwrite=args.overwrite,
    )

    # Step 2: Aggregate CPD results
    aggregate_results(
        output_dir=args.output_dir,
        config=args.config,
    )

    # Step 3: Plot the CPD results
    aggregated_summary_path = Path(args.output_dir) / "aggregated_summary.csv"
    if aggregated_summary_path.exists():
        aggregated_df = pd.read_csv(aggregated_summary_path)
        plot_results(aggregated_df, args.output_dir)
        print("Finished plotting basic CPD results.")
    else:
        print("Skipping basic CPD plotting as aggregated summary file was not found.")

    # --- Advanced Visualizations ---
    print("\n--- Generating Advanced Visualizations ---")
    # Step 4: Find all feature files (.npz)
    sequence_files = get_all_sequence_files(args.data_dir, args.config)
    if not sequence_files:
        print(f"No .npz files found in {args.data_dir}. Cannot generate advanced visualizations.")
        return

    # Step 5: Load features from all files into a single DataFrame
    features_df = load_features_from_npz(sequence_files)
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
