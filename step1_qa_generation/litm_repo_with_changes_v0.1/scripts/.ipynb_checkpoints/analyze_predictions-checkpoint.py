import json
import gzip
from data_processing import (
    count_tokens,
    distractor_ratio,
    bucket_density,
    classify_evidence_position,
    find_distractor_tokens_in_response
)

def preprocess_and_save_data(input_path, output_path):
    """
    Reads an input .jsonl.gz file, classifies evidence position, calculates
    distractor density, and saves the enhanced data to a new .jsonl.gz file.
    """
    print(f"Starting data pre-processing from {input_path}...")
    
    try:
        with gzip.open(input_path, 'rt', encoding='utf-8') as f_in, \
             gzip.open(output_path, 'wt', encoding='utf-8') as f_out:
            
            for line in f_in:
                data = json.loads(line)
                
                ctxs = data.get("ctxs", [])
                gold_idx = data.get("gold_doc_index")
                
                if gold_idx is not None and ctxs:
                    pos = classify_evidence_position(gold_idx, len(ctxs))
                    ratio = distractor_ratio(ctxs, gold_idx)
                    density = bucket_density(ratio)
                    
                    data["gold_index"] = gold_idx
                    data["evidence_position"] = pos
                    data["distractor_ratio"] = ratio
                    data["distractor_density"] = density
                
                f_out.write(json.dumps(data) + '\n')
    
    except FileNotFoundError:
        print(f"Error: The file {input_path} was not found.")
        return
    except (KeyError, json.JSONDecodeError) as e:
        print(f"Error processing line. A key might be missing or JSON is invalid: {e}")
        return
    
    print(f"Pre-processing complete. Output saved to {output_path}")

if __name__ == "__main__":
    # Set the paths to your input and output files
    input_file_path = "qa_data/fixed_synthetic_prompts.jsonl.gz"
    output_file_path = "qa_data/preprocessed_prompts.jsonl.gz"
    
    preprocess_and_save_data(input_file_path, output_file_path)