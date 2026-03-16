import json
import sys

def fix_question_ids(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    for task in tasks:
        data = task.get("data", {})

        question_id = data.get("question_id")

        config_id = data.get("config_idx")

        annotations = task.get("annotations", [])

        for ann in annotations:
            ann["question_id"] = question_id
            ann["config_id"] = config_id

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2)

    print(f"Fixed JSON written to: {output_file}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fix_question_ids.py input.json output.json")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    fix_question_ids(input_file, output_file)
