#!/usr/bin/env python3
"""
Run LongChat-13B inference on gold-only (no distractor) prompts.
Requires: transformers, torch
"""

import json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Paths
SCRIPT_DIR = Path(__file__).parent
INPUT_FILE = SCRIPT_DIR / "domain1_no_hallu_prompts.jsonl"
OUTPUT_FILE = SCRIPT_DIR / "domain1_no_hallu_responses.jsonl"

# Model config
MODEL_NAME = "lmsys/longchat-13b-16k"
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.0
TOP_P = 1.0


def main():
    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    print(f"Loading prompts from: {INPUT_FILE}")
    with open(INPUT_FILE, 'r') as f:
        records = [json.loads(line) for line in f]

    print(f"Running inference on {len(records)} prompts...")

    results = []
    for i, record in enumerate(records):
        if (i + 1) % 10 == 0:
            print(f"  Processing {i+1}/{len(records)}")

        prompt = record['model_prompt']

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                do_sample=False
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the generated part (after prompt)
        model_answer = response[len(prompt):].strip()

        record['model_answer'] = model_answer
        results.append(record)

    print(f"Saving results to: {OUTPUT_FILE}")
    with open(OUTPUT_FILE, 'w') as f:
        for record in results:
            f.write(json.dumps(record) + '\n')

    print("Done!")


if __name__ == "__main__":
    main()
