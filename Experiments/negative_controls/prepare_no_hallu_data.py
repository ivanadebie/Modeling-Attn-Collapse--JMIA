#!/usr/bin/env python3
"""
Prepare No-Hallucination Data for Negative Control Experiments
==============================================================

Creates model response data using ONLY gold_text (no distractors).
Target: domain1 (nq_closed_book) with longchat-13b-16k model.

Without distractor content, the model should NOT hallucinate,
providing ground truth for negative control experiments.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import json
from ast import literal_eval

# Setup paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
os.chdir(str(PROJECT_ROOT))

# Input: Original domain1 data with distractors
INPUT_JSONL = "step_2_hallucination_scoring/model answers_longchat/domain1_data_nq_closed_book_8000_converted.output.jsonl"

# Output directory
OUTPUT_DIR = SCRIPT_DIR / "no_hallu_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_original_data(filepath, max_samples=None):
    """Load original JSONL data."""
    print(f"Loading data from: {filepath}")

    data = []
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            try:
                record = json.loads(line.strip())
                data.append(record)
            except json.JSONDecodeError:
                continue

    print(f"Loaded {len(data)} records")
    return data


def create_gold_only_prompt(question, gold_text, model_format="longchat"):
    """
    Create a prompt using ONLY gold_text (no distractors).
    This should result in no hallucination since model only sees correct info.
    """
    # Prompt structure matching original but without distractor documents
    if model_format == "longchat":
        system_msg = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."

        user_content = f"""Question: {question}
Answer the question using the document below:

Document [1]
{gold_text}

ASSISTANT:"""

        full_prompt = f"{system_msg} USER: {user_content}"

    else:  # Plain format
        full_prompt = f"""Question: {question}
Answer the question using the document below:

Document [1]
{gold_text}

Answer:"""

    return full_prompt


def prepare_no_hallu_dataset(data, output_path):
    """
    Prepare dataset with gold_text only prompts.
    Returns data ready for model inference.
    """
    print("\nPreparing gold-only (no distractor) prompts...")

    no_hallu_records = []

    # Get unique questions (each question has multiple configs with distractors)
    questions_seen = set()

    for record in data:
        question = record.get('question', '')
        gold_text = record.get('gold_text', '')

        # Skip if we've already processed this question
        question_key = question[:100]  # Use first 100 chars as key
        if question_key in questions_seen:
            continue
        questions_seen.add(question_key)

        if not question or not gold_text:
            continue

        # Create gold-only prompt
        gold_only_prompt = create_gold_only_prompt(question, gold_text)

        no_hallu_record = {
            'question': question,
            'gold_text': gold_text,
            'distractors': [],  # Empty - no distractors
            'config': 'NoDistractor',  # Special config for no-distractor case
            'model': 'lmsys/longchat-13b-16k',
            'prompt': gold_only_prompt,
            'model_prompt': gold_only_prompt,
            'model_answer': None,  # To be filled by model inference
            'expected_hallu_label': 0,  # Expected: no hallucination
            'experiment_type': 'negative_control',
            'experiment_subtype': 'no_hallucination_sequence'
        }

        no_hallu_records.append(no_hallu_record)

    print(f"Created {len(no_hallu_records)} unique question prompts (gold-only)")

    # Save as JSONL for model inference
    jsonl_path = output_path / "domain1_no_hallu_prompts.jsonl"
    with open(jsonl_path, 'w') as f:
        for record in no_hallu_records:
            f.write(json.dumps(record) + '\n')
    print(f"Saved prompts to: {jsonl_path}")

    # Also save as CSV for reference
    csv_path = output_path / "domain1_no_hallu_prompts.csv"
    df = pd.DataFrame(no_hallu_records)
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV reference to: {csv_path}")

    return no_hallu_records



def main():
    print("=" * 80)
    print("PREPARE NO-HALLUCINATION DATA FOR NEGATIVE CONTROLS")
    print("=" * 80)
    print("\nDomain: domain1 (nq_closed_book)")
    print("Model: longchat-13b-16k")
    print("Setup: Gold text ONLY (no distractors)")
    print("=" * 80)

    # Load original data
    data = load_original_data(INPUT_JSONL)

    if not data:
        print("ERROR: Could not load data")
        return

    # Show sample of original data structure
    print("\nSample original record keys:", list(data[0].keys()))

    # Prepare no-hallucination dataset
    no_hallu_records = prepare_no_hallu_dataset(data, OUTPUT_DIR)

    # Create inference script
    create_inference_script(OUTPUT_DIR)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total unique questions: {len(no_hallu_records)}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("\nNext steps:")
    print("1. Run inference script to generate model responses")
    print("2. Score responses for hallucinations (should be ~0)")
    print("3. Use in negative control experiments")
    print("=" * 80)


if __name__ == "__main__":
    main()
