import json
import argparse
import numpy as np

def tokenize(text):
    return text.lower().split()

def interference_ratio_from_response(model_answer, distractors):
    if not model_answer or not distractors:
        return 0.0
    answer_tokens = set(tokenize(model_answer))
    if not answer_tokens:
        return 0.0
    ratios = []
    for d in distractors:
        d_tokens = set(tokenize(d))
        if not d_tokens:
            continue
        overlap = answer_tokens & d_tokens
        ratios.append(len(overlap) / len(answer_tokens))
    return max(ratios) if ratios else 0.0

def compute_distractor_density(record):
    docs = record.get("documents", [])
    gold = record.get("gold_answer", "")
    distractors = [d for d in docs if d.strip() != gold.strip()]
    total_tokens = sum(len(tokenize(d)) for d in docs) if docs else 1
    distractor_tokens = sum(len(tokenize(d)) for d in distractors)
    ratio = distractor_tokens / total_tokens
    if ratio < 0.33:
        density = "low"
    elif ratio < 0.66:
        density = "medium"
    else:
        density = "high"
    return density, ratio

def compute_gold_position(record):
    docs = record.get("documents", [])
    gold = record.get("gold_answer", "")
    if not docs or gold not in docs:
        return "unknown"
    idx = docs.index(gold)
    if idx == 0:
        return "beginning"
    elif idx == len(docs) - 1:
        return "end"
    else:
        return "middle"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.input, "r") as f:
        records = [json.loads(line) for line in f]

    for r in records:
        density, ratio = compute_distractor_density(r)
        r["distractor_density"] = density
        r["distractor_ratio"] = ratio
        r["gold_position"] = compute_gold_position(r)
        r["interference_ratio"] = interference_ratio_from_response(
            r.get("model_answer", ""), 
            [d for d in r.get("documents", []) if d.strip() != r.get("gold_answer", "").strip()]
        )

    with open(args.output, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

if __name__ == "__main__":
    main()
