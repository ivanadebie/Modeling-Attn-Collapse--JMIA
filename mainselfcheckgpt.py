import pandas as pd
from tqdm import tqdm 
from dataclasses import asdict
import numpy as np

import llm_apihandler
from data_requirements import QARecord, deduplicate_samples 
from metrics_selfcheck import HallucinationScorer
from data_utils import save_results_to_csv, save_diagnostics_to_csv

INPUT_DATA_PATH = '81QA-longchat-13b-16k-predictions.jsonl'   
OUTPUT_DATA_PATH = 'results_nli_labeled.csv' 
DIAGNOSTICS_OUTPUT_PATH = 'results_diagnostics.csv'
MODEL_TO_USE = "openai/gpt-3.5-turbo" 


def main():
    # Load data from JSONL and create initial records


    # loading_data()
    # ...
    ...
    # ...

    # # evaluating scores()
    # ....

    # # saving results()
    # ...




    # 1. Load data from JSONL and create initial records
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
        sentence_level_hallu_scores = scorer.get_sentence_level_hallucination_scores(record.answer, full_model_prompt or record.prompt, MODEL_TO_USE, seed=i)
        sample_level_hallu_scores = scorer.get_sample_level_hallucination_scores(full_model_prompt or record.prompt, MODEL_TO_USE, seed=i)
        
        whole_answer_hallu_score = scorer.aggregate_confidence_scores(sentence_level_hallu_scores)
        whole_answer_hallu_label = scorer.confidence_to_binary(whole_answer_hallu_score['conf_agg_mean'], threshold=0.85) #1-hallu / 0-not hallu

        model_response_uncertainty = scorer.aggregate_confidence_scores(sample_level_hallu_scores)['conf_agg_mean']
        
        sample_level_95th_percentile = scorer.calculate_95th_percentile(
            sample_level_hallu_scores.get('sample_level_hallucination_scores', [])
        )

        record.hallucination_label = whole_answer_hallu_label
        record.hallucination_score = whole_answer_hallu_score['conf_agg_mean']
        record.model_response_uncertainty = model_response_uncertainty
        record.confidence = 1 - model_response_uncertainty
        record.sentence_level_hallu_scores = sentence_level_hallu_scores['sentence_level_hallucination_scores']
        record.sample_level_hallu_scores = sample_level_hallu_scores['sample_level_hallucination_scores']
        record.sample_level_95th_percentile = sample_level_95th_percentile

        # if sentence_level_hallu_scores:
        #     record.confidence = sentence_level_hallu_scores.get('sentence_level_hallucination_scores', {})
        #     # NLI probabilities are no longer returned, so we leave the field empty
        #     record.nli_probabilities = {}
        # else:
        #     record.confidence = {}
        #     record.nli_probabilities = {}
            
        # record.binary_confidence = confidence_to_binary(record.confidence, threshold=0.85)

        # Aggregate sentence-level scores and store them in the record
        # conf_agg = aggregate_confidence_scores(record.confidence)
        # record.conf_agg_max = conf_agg['conf_agg_max']
        # record.conf_agg_mean = conf_agg['conf_agg_mean']
        
        # bin_agg = aggregate_binary_scores(record.binary_confidence)
        # record.bin_majority = bin_agg['bin_majority']
        
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
