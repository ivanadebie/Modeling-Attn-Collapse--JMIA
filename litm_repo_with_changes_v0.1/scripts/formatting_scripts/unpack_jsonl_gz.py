import gzip
import json
import sys

def reformat_item(item):
    # Adjust this mapping depending on your raw data structure
    return {
        "instruction": "Answer the question using the documents below:",
        "question": item.get("question", ""),
        "answer": item.get("answer", ""),
        "model_documents": item.get("model_documents", []),
        "model_prompt": item.get("model_prompt", ""),
        "model_answer": item.get("model_answer", ""),
        "model": item.get("model", ""),
        "model_temperature": item.get("model_temperature", 0),
        "model_top_p": item.get("model_top_p", 1),
        "model_prompt_mention_random_ordering": item.get("model_prompt_mention_random_ordering", False),
        "model_use_random_ordering": item.get("model_use_random_ordering", False),
    }

def unpack_jsonl_gz(input_path, output_path=None):
    with gzip.open(input_path, 'rt', encoding='utf-8') as f:
        lines = [json.loads(line) for line in f]

    reformatted = [reformat_item(item) for item in lines]

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as out_file:
            for item in reformatted:
                json.dump(item, out_file)
                out_file.write('\n')
        print(f"Reformatted JSONL written to '{output_path}'")
    else:
        for item in reformatted:
            print(json.dumps(item, indent=2))

if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python unpack_jsonl_gz.py input.jsonl.gz [output.jsonl]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) == 3 else None

    unpack_jsonl_gz(input_path, output_path)
