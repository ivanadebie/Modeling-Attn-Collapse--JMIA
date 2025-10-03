import pandas as pd
from pyparsing import col
from sympy import re
import re
from tqdm import tqdm 
from dataclasses import asdict
import numpy as np
import os
import llm_apihandler
from data_requirements import QARecord, deduplicate_samples 
from metrics_selfcheck import HallucinationScorer
from data_utils import save_results_to_csv

INPUT_DATA_PATH = os.path.join(os.path.dirname(__file__), 'true_synthetic_qa_with_metadata_final.csv')
#filepath: c:\Users\anand\Documents\Modeling-Attn-Collapse--JMIA\selfcheckgpt\mainselfcheckgpt.py
OUTPUT_DATA_PATH = os.path.join(os.path.dirname(__file__), 'results_nli_labeled.csv')
MODEL_TO_USE = "openai/gpt-3.5-turbo"


def main():
    # Load data from CSV and create initial records
    df = pd.read_csv(INPUT_DATA_PATH)

    records = []
    full_model_prompts = []  # Store prompts separately to avoid re-reading file
    
    for i, row in df.iterrows():
        qa_id = str(row.get('Column 1', ''))
        prompt = row.get('Prompt', '')
        question = row.get('Question', '')
        gold_text = row.get('Gold Text', '')
        distractor_1_text = row.get('Distractor 1 Text', '')
        distractor_2_text = row.get('Distractor 2 Text', '')
        full_model_prompts.append(prompt)


        record = QARecord(
            qa_id=qa_id,
            prompt=prompt,
            question=question,
            gold_text=gold_text,
            distractor_1_text=distractor_1_text,
            distractor_2_text=distractor_2_text,
            answer=row.get('model_answer', ''),
            distractor_density_class=row.get('Distractor Density Class', ''),
            interference_type=row.get('Interference Type', ''),
            evidence_position=row.get('Evidence Position', ''),
        )
        records.append(record)


    # Score Answers for Hallucination (using existing model answers)
    scorer = HallucinationScorer()
    results = []

    for i, record in enumerate(tqdm(records, desc="Scoring Answers")):
        print(f"Processing question {i+1}/{len(records)}: {record.qa_id}")
        # Use the stored full model prompt for generating consistency samples
        full_model_prompt = full_model_prompts[i]

        print(f"Answer: {record.answer!r}")
        print(f"Prompt/context for scoring: {full_model_prompt or record.prompt!r}")
        sentence_level_hallu_scores = scorer.get_sentence_level_hallucination_scores(
            record.answer, full_model_prompt or record.prompt, MODEL_TO_USE, seed=i
        )
        print(f"Sentence-level scores: {sentence_level_hallu_scores}")

        # Pass a seed for reproducible sample generation (i is a simple choice)
        sentence_level_hallu_scores = scorer.get_sentence_level_hallucination_scores(record.answer, record.context, MODEL_TO_USE, seed=i)
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

        # New: For each sentence/chunk, add is_gold_binary and gold_text_chunk
        gold_text_norm = record.gold_text.lower().strip()
        for chunk, score in sentence_level_hallu_scores.get('sentence_level_hallucination_scores', {}).items():
            chunk_norm = chunk.lower().strip()
            is_gold_binary = int(chunk_norm == gold_text_norm)  # Simple string match; replace with semantic if needed
            result = {
                "qa_id": record.qa_id,
                "chunk": chunk,
                "score": score,
                "is_gold_binary": is_gold_binary,
                "gold_text_chunk": record.gold_text,
                "hallucination_label": 1 if score >= 0.85 else 0,
                "distractor_density_class": record.distractor_density_class,
                "interference_type": record.interference_type,
                "evidence_position": record.evidence_position
            }
            results.append(result)

        # No intermediate chunk file output; only main results file will be written

    # Final save after completing all records
    save_results_to_csv(records, OUTPUT_DATA_PATH)
    print(f"Results saved to {OUTPUT_DATA_PATH}")

if __name__ == "__main__":
    main()
