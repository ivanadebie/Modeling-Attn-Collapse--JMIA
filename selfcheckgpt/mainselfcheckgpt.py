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

INPUT_JSONL_FILES = [
    os.path.join(os.path.dirname(__file__), 'domain1_data_nq_closed_book_8000_converted.output.jsonl'),
    os.path.join(os.path.dirname(__file__), 'domain2_data_hotpot_citation_8000_converted.output.jsonl'),
    os.path.join(os.path.dirname(__file__), 'domain3_data_novelhop_multihop_8000_converted.output.jsonl'),
    os.path.join(os.path.dirname(__file__), 'domain4_data_temporal_recency_8000_converted.output.jsonl'),
]
OUTPUT_DATA_PATH = os.path.join(os.path.dirname(__file__), 'answer_hallu_scored.csv')
MODEL_TO_USE = "all-MiniLM-L6-v2"


def main():
    # Load data from JSONL files and create initial records
    import json
    records = []
    for jsonl_path in INPUT_JSONL_FILES:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                row = json.loads(line)
                # Map JSONL fields to QARecord fields, handling missing and type mismatches
                record = QARecord(
                    question_id=str(row.get('question_id', '')),
                    config_idx=int(row.get('config_idx', 0)),
                    config=row.get('config', ''),
                    model_prompt=row.get('model_prompt', ''),
                    question=row.get('question', ''),
                    gold_text=row.get('gold_text', ''),
                    distractors=', '.join(row.get('distractors', [])) if isinstance(row.get('distractors', []), list) else str(row.get('distractors', '')),
                    row_index=int(row.get('row_index', 0)),
                    model_answer=row.get('model_answer', ''),
                    hallucination_score=0.0,
                    sentence_level_hallu_scores={},
                    is_gold_binary=0,
                    gold_text_chunk='',
                    domain=row.get('domain', ''),
                )
                records.append(record)


    # Score Answers for Hallucination (using existing model answers)
    scorer = HallucinationScorer()
    results = []

    for i, record in enumerate(tqdm(records, desc="Scoring Answers")):
        print(f"Processing question {i+1}/{len(records)}: {record.question_id}")
        # Use the stored full model prompt for generating consistency samples
        # full_model_prompt = full_model_prompts[i]

        print(f"Answer: {record.model_answer!r}")
        print(f"Gold/reference text: {record.gold_text!r}")

        # Check for missing answer or gold text
        if not record.model_answer or not record.gold_text:
            print(f"[SKIP] Missing answer or gold_text for question_id {record.question_id}")
            continue

        # Use gold_text as the reference for cosine similarity scoring
        sentence_level_hallu_scores = scorer.get_sentence_level_hallucination_scores(
            record.model_answer, record.gold_text, MODEL_TO_USE, seed=i
        )
        print(f"Sentence-level scores: {sentence_level_hallu_scores}")

        # Pass a seed for reproducible sample generation (i is a simple choice)
        sentence_level_hallu_scores = scorer.get_sentence_level_hallucination_scores(record.model_answer, getattr(record, 'context', ''), MODEL_TO_USE, seed=i)
        # sample_level_hallu_scores = scorer.get_sample_level_hallucination_scores(full_model_prompt or record.prompt, MODEL_TO_USE, seed=i)
        if not sentence_level_hallu_scores or not sentence_level_hallu_scores.get('sentence_level_hallucination_scores'):
            print(f"[SKIP] No sentence/chunk scores produced for question_id {record.question_id}")
            continue

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
        record.sentence_level_hallu_scores = sentence_level_hallu_scores['sentence_level_hallucination_scores']

        # For each sentence/chunk, add is_gold_binary and gold_text_chunk
        gold_text_norm = record.gold_text.lower().strip()
        model_answer_norm = record.model_answer.lower()
        for chunk, score in sentence_level_hallu_scores.get('sentence_level_hallucination_scores', {}).items():
            chunk_norm = chunk.lower().strip()
            # is_gold_binary: 1 if gold_text is present anywhere in the model answer
            is_gold_binary = int(gold_text_norm in model_answer_norm)
            result = {
                "question_id": record.question_id,
                "model_prompt": record.model_prompt,
                "config_idx": record.config_idx,
                "domain": getattr(record, "domain", "unknown"),
                "config": record.config,
                "question": record.question,
                "gold_text": record.gold_text,
                "distractors": record.distractors,
                "row_index": record.row_index,
                "model_answer": record.model_answer,
                "hallucination_score": record.hallucination_score,
                "sentence_level_hallu_scores": record.sentence_level_hallu_scores,
                "is_gold_binary": is_gold_binary,
                "gold_text_chunk": record.gold_text,
            }
            results.append(result)
        # No intermediate chunk file output; only main results file will be written

    # Separate results by domain and save to different files
    import collections
    domain_results = collections.defaultdict(list)
    for result in results:
        domain = result.get('domain', 'unknown')
        domain_results[domain].append(result)

    for domain, domain_group in domain_results.items():
        domain_str = str(domain).replace(' ', '_')
        out_path = os.path.join(os.path.dirname(__file__), f'answer_hallu_scored_{domain_str}.csv')
        pd.DataFrame(domain_group).to_csv(out_path, index=False)
        print(f"Results saved to {out_path}")

if __name__ == "__main__":
    main()
