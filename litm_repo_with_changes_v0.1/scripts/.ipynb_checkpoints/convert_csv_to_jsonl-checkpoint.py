import csv
import json
import gzip
import sys

def empty_to_none(value):
    return value.strip() if value and value.strip() != "" else None

def csv_to_jsonl_gz(csv_path, jsonl_gz_path):
    with open(csv_path, newline='', encoding='utf-8') as csvfile, \
         gzip.open(jsonl_gz_path, 'wt', encoding='utf-8') as jsonlfile:

        reader = csv.DictReader(csvfile)  # removed delimiter='\t'
        for row in reader:
            question = empty_to_none(row.get("Questions"))
            if not question:
                continue  # skip rows without a valid question

            json_obj = {
                "qa_id": empty_to_none(row.get("QA ID")),
                "question": question,
                "answer": empty_to_none(row.get("Answer (Gold Doc) 200-250T")),
                "density": empty_to_none(row.get("Density")),
                "interference": empty_to_none(row.get("Interference")),
                "gold_position": empty_to_none(row.get("Gold Position")),
                "status": empty_to_none(row.get("Status")),
            }
            jsonlfile.write(json.dumps(json_obj) + "\n")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_csv_to_jsonl.py input.csv output.jsonl.gz")
        sys.exit(1)
    csv_path = sys.argv[1]
    jsonl_gz_path = sys.argv[2]

    csv_to_jsonl_gz(csv_path, jsonl_gz_path)
    print(f"Converted {csv_path} to {jsonl_gz_path}")
