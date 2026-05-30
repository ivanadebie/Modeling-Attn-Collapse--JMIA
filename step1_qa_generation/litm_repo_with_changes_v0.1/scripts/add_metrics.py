import csv
import argparse

def tokenize(text):
    return text.lower().split()

def compute_num_distractors(record):
    distractors = []
    for i in range(1, 11):
        d = record.get(f"Distractor {i} Text", "").strip()
        if d:
            distractors.append(d)
    return len(distractors)

def compute_distractor_density(record):
    # All documents including distractors
    docs = []
    for i in range(1, 11):
        d = record.get(f"Distractor {i} Text", "").strip()
        if d:
            docs.append(d)

    gold = record.get("Gold Text", "").strip()
    all_tokens = tokenize(gold)
    distractor_tokens = []
    for d in docs:
        distractor_tokens.extend(tokenize(d))

    total_tokens = len(all_tokens) + len(distractor_tokens)
    if total_tokens == 0:
        ratio = 0.0
    else:
        ratio = len(distractor_tokens) / total_tokens

    # Density level
    if ratio < 0.33:
        density = "low"
    elif ratio < 0.66:
        density = "medium"
    else:
        density = "high"

    return density, round(ratio, 3)

def find_gold_token_span(full_prompt, gold_text):
    full_tokens = tokenize(full_prompt)
    gold_tokens = tokenize(gold_text)

    total_tokens = len(full_tokens)
    if total_tokens == 0 or len(gold_tokens) == 0:
        return {"span_start": -1.0, "span_end": -1.0}

    # Search for gold tokens in full prompt
    for i in range(total_tokens - len(gold_tokens) + 1):
        if full_tokens[i:i+len(gold_tokens)] == gold_tokens:
            span_start_norm = i / total_tokens
            span_end_norm = (i + len(gold_tokens)) / total_tokens
            return {"span_start": round(span_start_norm, 3),
                    "span_end": round(span_end_norm, 3)}

    return {"span_start": -1.0, "span_end": -1.0}

def get_gold_position(span_start_norm, span_end_norm):
    if span_start_norm < 0:
        return "unknown"
    if span_end_norm <= 0.33:
        return "beginning"
    elif span_start_norm >= 0.66:
        return "end"
    else:
        return "middle"

def compute_gold_metrics(full_prompt, gold_text):
    spans = find_gold_token_span(full_prompt, gold_text)
    position = get_gold_position(spans["span_start"], spans["span_end"])
    return {**spans, "gold_position": position}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.input, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        records = [row for row in reader]

    for r in records:
        r["num_distractors"] = compute_num_distractors(r)
        density, ratio = compute_distractor_density(r)
        r["distractor_density"] = density
        r["distractor_ratio"] = ratio

        full_prompt = r.get("Prompt", "")  # use full prompt for token positions
        gold_text = r.get("Gold Text", "")
        gold_metrics = compute_gold_metrics(full_prompt, gold_text)
        r["gold_position"] = gold_metrics["gold_position"]
        r["gold_span_start"] = gold_metrics["span_start"]
        r["gold_span_end"] = gold_metrics["span_end"]

    fieldnames = list(records[0].keys())
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

if __name__ == "__main__":
    main()
