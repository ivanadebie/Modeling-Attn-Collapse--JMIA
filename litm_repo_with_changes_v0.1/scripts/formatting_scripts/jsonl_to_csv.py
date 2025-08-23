import pandas as pd
import json

jsonl_path = "qa_predictions/synthetic_qa_with_density.jsonl"
csv_path = "qa_predictions/synthetic_qa_with_density.csv"

records = []
with open(jsonl_path, "r", encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

df = pd.DataFrame(records)
df.to_csv(csv_path, index=False)
