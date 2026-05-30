import gzip
import json
import csv

input_file = "qa_predictions/output.jsonl.gz"
output_file = "qa_predictions/output.csv"

rows = []

# Read compressed JSONL
with gzip.open(input_file, "rt", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            obj = json.loads(line)
            rows.append(obj)

# Collect all keys across objects (so CSV has all columns)
all_keys = set()
for row in rows:
    all_keys.update(row.keys())
all_keys = list(all_keys)

# Write CSV
with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=all_keys)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

print(f"✅ Converted {input_file} → {output_file} with {len(rows)} rows and {len(all_keys)} columns")
