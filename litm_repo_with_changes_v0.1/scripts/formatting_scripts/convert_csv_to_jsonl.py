import pandas as pd
import json
import gzip
import os
import re

def normalize_headers(df):
    """Lowercase and strip column headers for consistent access."""
    df.columns = [col.strip().lower() for col in df.columns]
    return df

def convert_csv_to_jsonl_gz(csv_file_path, jsonl_gz_file_path):
    """
    Converts a CSV file to a compressed JSONL.GZ file in proper LongChat (Lost in the Middle) format.
    """
    try:
        if not os.path.exists(csv_file_path):
            print(f"Error: The file '{csv_file_path}' was not found.")
            return

        df = pd.read_csv(csv_file_path)
        df = normalize_headers(df)

        # Expected columns
        question_col = 'question'
        answer_col = 'gold text'
        prompt_col = 'prompt'

        # Validate
        for col in [question_col, answer_col, prompt_col]:
            if col not in df.columns:
                print(f"Error: Missing required column '{col}'. Found: {df.columns.tolist()}")
                return

        with gzip.open(jsonl_gz_file_path, 'wt', encoding='utf-8') as gz_file:
            for _, row in df.iterrows():
                question = str(row[question_col]).strip()
                answer = str(row[answer_col]).strip()
                prompt_text = str(row[prompt_col]).strip().strip('"')

                # Split documents more robustly
                documents = re.split(r'Document\s*\[\d+\]\s*', prompt_text, flags=re.IGNORECASE)
                documents = [doc.strip() for doc in documents if doc.strip()]

                model_documents = []
                for idx, doc_text in enumerate(documents):
                    is_gold = answer.lower() in doc_text.lower()
                    model_documents.append({
                        "id": f"doc_{idx}",
                        "text": doc_text,
                        "is_gold": is_gold
                    })

                json_data = {
                    "question": question,
                    "answer": answer,
                    "model_documents": model_documents,
                    "model_prompt": (
                        "A chat between a curious user and an AI assistant. "
                        "The assistant gives helpful, detailed, and polite answers.\n\n"
                        f"USER: Question: {question}\n\nASSISTANT:"
                    ),
                    "model_answer": "",
                    "model": "lmsys/longchat-13b-16k",
                    "model_temperature": 0,
                    "model_top_p": 1,
                    "model_prompt_mention_random_ordering": False,
                    "model_use_random_ordering": False
                }

                gz_file.write(json.dumps(json_data) + '\n')

        print(f"Successfully converted '{csv_file_path}' → '{jsonl_gz_file_path}'")

    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    input_csv_file = 'qa_data/true_synthetic_qa.csv'
    output_jsonl_gz_file = 'qa_data/true_synthetic_qa.jsonl.gz'
    convert_csv_to_jsonl_gz(input_csv_file, output_jsonl_gz_file)
