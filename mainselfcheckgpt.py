import pandas as pd
from tqdm import tqdm 
from dataclasses import asdict
import numpy as np

import llm_apihandler
from metrics_selfcheck import HallucinationScorer, QARecord, confidence_to_binary, aggregate_confidence_scores, aggregate_binary_scores

INPUT_DATA_PATH = '81QA-longchat-13b-16k-predictions.jsonl'   
OUTPUT_DATA_PATH = 'results_nli_labeled.csv' 
DIAGNOSTICS_OUTPUT_PATH = 'results_diagnostics.csv'
MODEL_TO_USE = "openai/gpt-3.5-turbo" 

def save_results_to_csv(records, output_path):
    """Helper function to convert records and save to CSV"""
    results_data = [asdict(record) for record in records]
    
    # Add aggregated numeric columns and convert dictionaries to strings
    for item in results_data:
        # Remove fields that are not populated by this script
        item.pop('per_sample_scores', None)
        item.pop('risk_score_mean', None)
        item.pop('risk_score_p95', None)
        item.pop('self_consistency_vote', None)

        # Convert dictionaries to strings for CSV compatibility
        item['confidence'] = str(item['confidence'])
        item['binary_confidence'] = str(item['binary_confidence'])
        item['nli_probabilities'] = str(item['nli_probabilities']) # Also convert this new field

    df_output = pd.DataFrame(results_data)
    df_output.to_csv(output_path, index=False)

def save_diagnostics_to_csv(records, output_path):
    """Saves per-item diagnostics to a separate CSV file."""
    diagnostics_data = []
    for record in records:
        conf_dict = record.confidence if record.confidence else {}
        bin_dict = record.binary_confidence if record.binary_confidence else {}
        
        diag_item = {'qa_id': record.qa_id}
        if conf_dict:
            scores = list(conf_dict.values())
            diag_item['n_sentences'] = len(scores)
            diag_item['std_conf'] = np.std(scores) if len(scores) > 1 else 0.0
            diag_item['pos_rate'] = np.mean(list(bin_dict.values())) if bin_dict else 0.0
        else:
            diag_item['n_sentences'] = 0
            diag_item['std_conf'] = 0.0
            diag_item['pos_rate'] = 0.0
        diagnostics_data.append(diag_item)
        
    df_diagnostics = pd.DataFrame(diagnostics_data)
    df_diagnostics.to_csv(output_path, index=False)

def main():
    # Load data from JSONL and create initial records
    import json
    records = []
    full_model_prompts = []  # Store prompts separately to avoid re-reading file
    
    with open(INPUT_DATA_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            item = json.loads(line)
            
            # Get the question and model prompt
            question = item.get('question', '')
            full_model_prompts.append(item.get('model_prompt', ''))  # Store for later use
            
            # Extract context from the ctxs field (list of documents)
            context_docs = item.get('ctxs', [])
            context_parts = []
            for doc in context_docs:
                if doc.get('title') and doc.get('text'):
                    context_parts.append(f"Document: {doc['title']}\n{doc['text']}")
            context_part = "\n\n".join(context_parts)
            
            record = QARecord(
                qa_id=f"QA_{i+1:03d}",  # Generate ID like QA_001, QA_002, etc.
                context=context_part,  # Combined documents as context
                prompt=question,       # Just the question text
                answer=item.get('model_answer', ''),  # Use existing model answer
                nli_probabilities={} # Initialize new field
            )
            records.append(record)

    # Score Answers for Hallucination (using existing model answers)
    scorer = HallucinationScorer()
    
    for i, record in enumerate(tqdm(records, desc="Scoring Answers")):
        print(f"Processing question {i+1}/81: {record.qa_id}")
        
        # Use the stored full model prompt for generating consistency samples
        full_model_prompt = full_model_prompts[i]
        
        # Pass a seed for reproducible sample generation (i is a simple choice)
        score_results = scorer.score_answer(record.answer, full_model_prompt or record.prompt, MODEL_TO_USE, seed=i)
        
        if score_results:
            record.confidence = score_results.get('hallucination_scores', {})
            # NLI probabilities are no longer returned, so we leave the field empty
            record.nli_probabilities = {}
        else:
            record.confidence = {}
            record.nli_probabilities = {}
            
        record.binary_confidence = confidence_to_binary(record.confidence, threshold=0.85)

        # Aggregate sentence-level scores and store them in the record
        conf_agg = aggregate_confidence_scores(record.confidence)
        record.conf_agg_max = conf_agg['conf_agg_max']
        record.conf_agg_mean = conf_agg['conf_agg_mean']
        
        bin_agg = aggregate_binary_scores(record.binary_confidence)
        record.bin_majority = bin_agg['bin_majority']
        
        # Save intermediate results every 10 questions
        if (i + 1) % 10 == 0:
            print(f"Completed {i+1} questions, saving checkpoint...")
            save_results_to_csv(records, OUTPUT_DATA_PATH)
            save_diagnostics_to_csv(records, DIAGNOSTICS_OUTPUT_PATH)
            print(f"Checkpoint saved to {OUTPUT_DATA_PATH} and {DIAGNOSTICS_OUTPUT_PATH}")

    # Final save after completing all records
    save_results_to_csv(records, OUTPUT_DATA_PATH)
    save_diagnostics_to_csv(records, DIAGNOSTICS_OUTPUT_PATH)
    print(f"Results saved to {OUTPUT_DATA_PATH}")
    print(f"Diagnostics saved to {DIAGNOSTICS_OUTPUT_PATH}")

if __name__ == "__main__":
    main()
