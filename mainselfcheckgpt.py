import pandas as pd
import time
from tqdm import tqdm 

# Import your custom modules
import llm_apihandler
from metrics import HallucinationScorer

# --- Configuration ---
INPUT_DATA_PATH = 'Synthetic QA.xlsx'   
OUTPUT_DATA_PATH = 'results_nli_labeled.csv' 
MODEL_TO_USE = "modelchoice" 

def main():
    # --- Part 1: Get Initial Answers from the LLM ---
    print("--- Starting Part 1: Generating Initial Answers ---")
    df = pd.read_csv(INPUT_DATA_PATH)
    
    generated_answers = []
    
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Generating Answers"):
        prompt = row['full_context'] + "\n\nQuestion: " + row['question']
        answer = llm_apihandler.get_llm_response(prompt, MODEL_TO_USE)
        generated_answers.append(answer)
        time.sleep(1) 

    df['generated_answer'] = generated_answers
    print("\nPart 1 Complete. All initial answers have been generated.")
    
    # --- Part 2: Score Answers for Hallucination ---
    print("\n--- Starting Part 2: Scoring Answers with NLI ---")
    
    scorer = HallucinationScorer()
    
    # This list will now store sentence-wise score dicts as strings
    # Each dict will map sentences to their NLI contradiction scores
    # e.g., {'sentence1': score1, 'sentence2': score2, ...}
    # This allows us to keep track of individual sentence scores
    nli_sentence_scores = []

    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Scoring Answers"):
        original_answer = row['generated_answer']
        prompt = row['full_context'] + "\n\nQuestion: " + row['question']
        
        # Get sentence-wise scores
        sentence_scores = scorer.score_answer(original_answer, prompt, MODEL_TO_USE)
        #Store as string for CSV
        nli_sentence_scores.append(str(sentence_scores))

    # Add the list of sentence-wise scores as a new column to the DataFrame
    df['selfcheck_nli_sentence_scores'] = nli_sentence_scores
    print("\nPart 2 Complete. All answers have been scored.")

    # --- Part 3: Save Final Results ---
    df.to_csv(OUTPUT_DATA_PATH, index=False)
    print(f"\nPipeline finished! Results saved to {OUTPUT_DATA_PATH}")


if __name__ == "__main__":
    main()
