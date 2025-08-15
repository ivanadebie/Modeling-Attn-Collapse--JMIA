import gzip
import json

path = "qa_data/synthetic_qa.jsonl.gz"

count = 0
missing_question = 0

with gzip.open(path, "rt", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        count += 1
        if not data.get("question"):
            missing_question += 1

print(f"Total entries: {count}")
print(f"Entries missing 'question': {missing_question}")
