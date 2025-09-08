import pandas as pd
from pyparsing import col
from tqdm import tqdm 
from dataclasses import asdict
import numpy as np

import llm_apihandler
from data_requirements import QARecord, deduplicate_samples 
from metrics_selfcheck import HallucinationScorer
from data_utils import save_results_to_csv

INPUT_DATA_PATH = 'true_synthetic_qa_with_metadata_final.csv'   
OUTPUT_DATA_PATH = 'results_nli_labeled.csv' 
MODEL_TO_USE = "openai/gpt-3.5-turbo" 


def main():
    # Load data from CSV and create initial records
    df = pd.read_csv(INPUT_DATA_PATH)

    records = []
    full_model_prompts = []  # Store prompts separately to avoid re-reading file
    
    for i, row in df.iterrows():
        qa_id = str(row.get('Column 1', ''))
        prompt = row.get('Prompt', '')
        full_model_prompts.append(prompt)

        # Combine only Gold Text, Distractor 1 Text and Distractor 2 Text
        context_parts = []
        for col in ['Gold Text', 'Distractor 1 Text', 'Distractor 2 Text']:
                val = row.get(col, '')
                if isinstance(val, str) and val.strip():
                    context_parts.append(val)
        context_part = "\n\n".join(context_parts)
            
        record = QARecord(
            qa_id=qa_id, 
            context=context_part,  # Combined documents as context
            prompt=prompt,       # Just the question text
            answer=row.get('model_answer', ''),  # Use existing model answer
            distractor_density_class=row.get('Distractor Density Class', ''),
            interference_type=row.get('Interference Type', ''),
            evidence_position=row.get('Evidence Position', ''),
        )
        records.append(record)

    # Score Answers for Hallucination (using existing model answers)
    scorer = HallucinationScorer()
    for i, record in enumerate(tqdm(records, desc="Scoring Answers")):
        print(f"Processing question {i+1}/{len(records)}: {record.qa_id}")
        
        # Use the stored full model prompt for generating consistency samples
        full_model_prompt = full_model_prompts[i]
        
        # Pass a seed for reproducible sample generation (i is a simple choice)
        sentence_level_hallu_scores = scorer.get_sentence_level_hallucination_scores(record.answer, full_model_prompt or record.prompt, MODEL_TO_USE, seed=i)
        # sample_level_hallu_scores = scorer.get_sample_level_hallucination_scores(full_model_prompt or record.prompt, MODEL_TO_USE, seed=i)
      
        whole_answer_hallu_score = scorer.aggregate_confidence_scores(sentence_level_hallu_scores)
        whole_answer_hallu_label = 1 if whole_answer_hallu_score['conf_agg_mean'] >= 0.85 else 0  # 1-hallu / 0-not hallu

        # model_response_uncertainty = scorer.aggregate_confidence_scores(sample_level_hallu_scores)['conf_agg_mean']
        
        # sample_level_95th_percentile = scorer.calculate_95th_percentile(
        #     sample_level_hallu_scores.get('sample_level_hallucination_scores', [])
        # )

        record.hallucination_label = whole_answer_hallu_label
        record.hallucination_score = whole_answer_hallu_score['conf_agg_mean']
        # record.model_response_uncertainty = model_response_uncertainty
        # record.confidence = 1 - model_response_uncertainty
        if 'sentence_level_hallucination_scores' in sentence_level_hallu_scores:
            record.sentence_level_hallu_scores = sentence_level_hallu_scores['sentence_level_hallucination_scores']
        else:
            record.sentence_level_hallu_scores = {}
        # record.sample_level_hallu_scores = sample_level_hallu_scores['sample_level_hallucination_scores']
        # record.sample_level_95th_percentile = sample_level_95th_percentile

        # Save intermediate results every 10 questions
        if (i + 1) % 10 == 0:
            print(f"Completed {i+1} questions, saving checkpoint...")
            save_results_to_csv(records, OUTPUT_DATA_PATH)
            print(f"Checkpoint saved to {OUTPUT_DATA_PATH}")

    # Final save after completing all records
    save_results_to_csv(records, OUTPUT_DATA_PATH)
    print(f"Results saved to {OUTPUT_DATA_PATH}")

if __name__ == "__main__":
    main()
