import gzip
import json
import sys

def unpack_jsonl_gz(input_path, output_path=None):
    with gzip.open(input_path, 'rt', encoding='utf-8') as f:
        lines = [json.loads(line) for line in f]

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as out_file:
            for item in lines:
                json.dump(item, out_file)
                out_file.write('\n')
        print(f"Unpacked JSONL written to '{output_path}'")
    else:
        for item in lines:
            print(json.dumps(item, indent=2))

if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python unpack_jsonl_gz.py input.jsonl.gz [output.jsonl]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) == 3 else None

    unpack_jsonl_gz(input_path, output_path)
