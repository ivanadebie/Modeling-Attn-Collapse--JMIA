from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

# --- Configuration ---
DEFAULT_THRESHOLD = 0.4
DEFAULT_INPUT_PATTERN = "answer_hallu_scored_*.csv"
DEFAULT_OUTPUT_SUFFIX = "_labeled.csv"

def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for the script."""
    parser = argparse.ArgumentParser(
        description="Adds a hallucination label to scored answer files."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Scores > threshold are 'not hallu'. Default: {DEFAULT_THRESHOLD}",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("."),
        help="Directory containing the input CSV files. Default: current directory.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=DEFAULT_INPUT_PATTERN,
        help=f"Pattern to find input files. Default: '{DEFAULT_INPUT_PATTERN}'",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default=DEFAULT_OUTPUT_SUFFIX,
        help=f"Suffix for the output files. Default: '{DEFAULT_OUTPUT_SUFFIX}'",
    )
    return parser.parse_args()

def label_dataframe(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Adds a 'label' column to the DataFrame based on the 'hallucination_score'.
    
    Logic: A higher score indicates higher semantic similarity, meaning it is
    LESS likely to be a hallucination.
    """
    if "hallucination_score" not in df.columns:
        raise KeyError("Input file must have a 'hallucination_score' column.")

    # Ensure score is numeric, treating errors as 0
    scores = pd.to_numeric(df["hallucination_score"], errors="coerce").fillna(0)

    # If score > threshold, it's "not hallu". Otherwise, it's "hallu".
    df["label"] = "hallu"
    df.loc[scores > threshold, "label"] = "not hallu"
    
    return df

def main() -> None:
    """Finds, processes, and saves labeled data files."""
    args = parse_args()
    
    # The script is in step_4_cpd, so we look for files in its directory
    script_dir = Path(__file__).resolve().parent
    input_directory = script_dir / args.input_dir

    # Find all source files that do NOT have the output suffix
    source_files = [
        p for p in input_directory.glob(args.pattern) 
        if not p.name.endswith(args.suffix)
    ]

    if not source_files:
        print(f"Error: No source files found matching '{args.pattern}' in '{input_directory}'.")
        return

    print(f"Found {len(source_files)} files to process with threshold {args.threshold}.")
    print("-" * 40)

    for file_path in source_files:
        try:
            # Read the source file
            df = pd.read_csv(file_path)

            # Add the label column
            labeled_df = label_dataframe(df, args.threshold)

            # Report statistics
            hallu_count = (labeled_df["label"] == "hallu").sum()
            total_count = len(labeled_df)
            hallu_percent = (hallu_count / total_count * 100) if total_count > 0 else 0
            
            print(f"Processing '{file_path.name}':")
            print(f"  - Hallucinations: {hallu_count}/{total_count} ({hallu_percent:.2f}%)")

            # Save the newly labeled file
            output_filename = file_path.stem + args.suffix
            output_path = file_path.with_name(output_filename)
            labeled_df.to_csv(output_path, index=False)
            print(f"  - Saved labeled output to '{output_path.name}'")

        except Exception as e:
            print(f"Failed to process {file_path.name}: {e}")
    
    print("-" * 40)
    print("Processing complete.")

if __name__ == "__main__":
    main()
