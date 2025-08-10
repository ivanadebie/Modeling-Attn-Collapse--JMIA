import gzip
import json

def read_and_print_file(file_path, num_lines=5):
    """
    Opens a gzipped JSONL file, reads the first few lines,
    and prints them to the console.
    """
    print(f"Opening and reading the first {num_lines} lines of: {file_path}")
    
    try:
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= num_lines:
                    break
                data = json.loads(line)
                print(f"Line {i+1}: {json.dumps(data, indent=2)}")

    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
    except UnicodeDecodeError:
        print("Error: The file is not a valid UTF-8 text file. Please check the file's encoding.")
    except json.JSONDecodeError:
        print("Error: Failed to parse JSON. The file might be corrupted or malformed.")

if __name__ == "__main__":
    file_path_to_open = "qa_data/my_data/preprocessed_prompts.jsonl.gz"
    read_and_print_file(file_path_to_open)