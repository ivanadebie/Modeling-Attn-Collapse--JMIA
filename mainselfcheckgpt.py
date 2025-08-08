import pandas as pd
import time
from tqdm import tqdm 
from dataclasses import asdict

import llm_apihandler
from metrics_selfcheck import HallucinationScorer, QARecord

INPUT_DATA_PATH = 'Synthetic QA.xlsx'   
OUTPUT_DATA_PATH = 'results_nli_labeled.csv' 
MODEL_TO_USE = "mistralai/mistral-7b-instruct" 

def main():

    # --- Part 1: Load data from JSONL and create initial records ---
    print("--- Starting Part 1: Loading Data ---")
    import json
    INPUT_DATA_PATH = 'nq-open-oracle-mpt-30b-clean_synthetic-predictions.jsonl'
    records = []
    with open(INPUT_DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            # Use 'id', 'prompt', and optionally 'context' if present
            record = QARecord(
                qa_id=item.get('id', ''),
                context=item.get('prompt', ''),  # If you want to use a separate context field, adjust here
                prompt=item.get('prompt', '')
            )
            records.append(record)
    print(f"{len(records)} records created.")

    # --- Part 2: Get Initial Answers from the LLM ---
    print("--- Starting Part 2: Generating Initial Answers ---")
    for record in tqdm(records, desc="Generating Answers"):
        record.answer = llm_apihandler.get_llm_response(record.prompt, MODEL_TO_USE)
        time.sleep(1)
   
    # --- Part 3: Score Answers for Hallucination ---
    print("\n--- Starting Part 3: Scoring Answers with NLI ---")
    scorer = HallucinationScorer()
    
    for record in tqdm(records, desc="Scoring Answers"):
        record.confidence = scorer.score_answer(record.answer, record.prompt, MODEL_TO_USE)

    # --- Part 4: Save Final Results ---
    print("\n--- Starting Part 4: Saving Results ---")  
 
    # Convert the list of QARecord objects to a list of dictionaries for saving
    results_data = [asdict(record) for record in records]
    
    # Convert the confidence dictionary to a string for CSV compatibility
    for item in results_data:
        item['confidence'] = str(item['confidence'])

    df_output = pd.DataFrame(results_data)
    
    df_output.to_csv(OUTPUT_DATA_PATH, index=False)
    print(f"\nPipeline finished! Results saved to {OUTPUT_DATA_PATH}")

if __name__ == "__main__":
    main()
