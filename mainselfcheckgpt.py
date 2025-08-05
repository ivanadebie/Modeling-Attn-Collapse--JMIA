import pandas as pd
import time
from tqdm import tqdm 
from dataclasses import asdict

import llm_apihandler
from metrics import HallucinationScorer

INPUT_DATA_PATH = 'Synthetic QA.xlsx'   
OUTPUT_DATA_PATH = 'results_nli_labeled.csv' 
MODEL_TO_USE = "meta-llama/llama-3-8b-instruct" 

def main():

    # --- Part 1: Load data from Excel and create initial records ---
    print("--- Starting Part 1: Loading Data ---")
    df_input = pd.read_excel(INPUT_DATA_PATH)
    
    records = []
    for _, row in df_input.iterrows():
        # Excel columns are named 'full_context' and 'question'
        full_prompt = row['full_context'] + "\n\nQuestion: " + row['question']
        record = Synthetic QA(
            qa_id=row['qa_id'],
            context=row['full_context'],
            prompt=full_prompt
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
