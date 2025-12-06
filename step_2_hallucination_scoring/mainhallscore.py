import pandas as pd
from pyparsing import col
from sympy import re
import re
from tqdm import tqdm 
from dataclasses import asdict
import numpy as np
import os
from data_requirements import QARecord 
from metrics_hallscore import HallucinationScorer
from data_utils import save_results_to_csv
import json
import csv
import argparse

MODEL_TO_USE = "all-MiniLM-L6-v2"

def get_input_files(model_type):
    """Get input JSONL files based on model type (longchat or mistral)."""
    base_dir = os.path.dirname(__file__)
    
    if model_type == 'longchat':
        folder = os.path.join(base_dir, 'model answers_longchat')
        file_patterns = [
            'domain1_data_nq_closed_book_8000_converted.output.jsonl',
            'domain2_data_hotpot_citation_8000_converted.output.jsonl',
            'domain3_data_novelhop_multihop_8000_converted.output.jsonl',
            'domain4_data_temporal_recency_8000_converted.output.jsonl',
            'domain5_data_05_stackmath_numerical_8000_converted.output.jsonl',
            'domain6_data_06_policy_compliance_8000_converted.output.jsonl',
            'domain7_data_07_expert_legal_med_8000_converted.output.jsonl',
        ]
    elif model_type == 'mistral':
        folder = os.path.join(base_dir, 'model answers_mistral')
        file_patterns = [
            'domain1_data_nq_closed_book_8000_converted.mistral.jsonl',
            'domain2_data_hotpot_citation_8000_converted.mistral.jsonl',
            'domain3_data_novelhop_multihop_8000_converted.mistral.jsonl',
            'domain4_data_temporal_recency_8000_converted.mistral.jsonl',
            'domain5_data_05_stackmath_numerical_8000_converted.mistral.jsonl',
            'domain6_data_06_policy_compliance_8000_converted.mistral.jsonl',
            'domain7_data_07_expert_legal_med_8000_converted.mistral.jsonl',
        ]
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'longchat' or 'mistral'.")
    
    return [os.path.join(folder, pattern) for pattern in file_patterns]

def get_output_folder(model_type):
    """Get output folder based on model type."""
    base_dir = os.path.dirname(__file__)
    if model_type == 'longchat':
        return os.path.join(base_dir, 'scored answers_longchat')
    elif model_type == 'mistral':
        return os.path.join(base_dir, 'scored answers_mistral')
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def main(model_type):
    # Get input files and output folder based on model type
    INPUT_JSONL_FILES = get_input_files(model_type)
    output_folder = get_output_folder(model_type)
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"Processing {model_type} model answers...")
    print(f"Input files: {len(INPUT_JSONL_FILES)} domains")
    print(f"Output folder: {output_folder}")
    
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
                    evidence_position=0.0, #must fix for evidence posiiton; the gold text position is found then the difference between the 
                    domain=row.get('domain', ''),
                )
                # Keep the entire original input row so we can preserve all input headings/fields in outputs
                setattr(record, 'original_row', row)
                records.append(record)


    # Score Answers for Hallucination (using existing model answers)
    scorer = HallucinationScorer()
    results = []

    for i, record in enumerate(tqdm(records, desc="Scoring Answers")):
        print(f"Processing question {i+1}/{len(records)}: {record.question_id}")
        # full_model_prompt = full_model_prompts[i]

        print(f"Answer: {record.model_answer!r}")
        print(f"Gold/reference text: {record.gold_text!r}")

        # Check for missing answer or gold text
        if not record.model_answer or not record.gold_text:
            print(f"[SKIP] Missing answer or gold_text for question_id {record.question_id}")
            continue

        # Use gold_text as the reference for semantic similarity scoring
        sentence_level_hallu_scores = scorer.get_sentence_level_sem_similarity_scores(
            record.model_answer, record.gold_text, MODEL_TO_USE, seed=i
        )
        print(f"Sentence-level scores: {sentence_level_hallu_scores}")
        # sample_level_hallu_scores = scorer.get_sample_level_sem_similarity_scores(full_model_prompt or record.prompt, MODEL_TO_USE, seed=i)
        if not sentence_level_hallu_scores or not sentence_level_hallu_scores.get('sentence_level_sem_similarity_scores'):
            print(f"[SKIP] No sentence/chunk scores produced for question_id {record.question_id}")
            continue

        # aggregate_confidence_scores returns aggregated similarity metrics (higher = more similar)
        whole_answer_hallu_score = scorer.aggregate_confidence_scores(sentence_level_hallu_scores)
        # convert similarity -> hallucination likelihood: hallucination = 1 - similarity
        similarity_mean = whole_answer_hallu_score.get('conf_agg_mean', 0.0)
        hallucination_mean = 1.0 - similarity_mean
        # whole_answer_hallu_label = 1 if hallucination_mean >= 0.85 else 0  # 1-hallu / 0-not hallu

        record.hallucination_score = hallucination_mean
        record.sem_similarity_score = similarity_mean
        record.hallucination_label = 1 if hallucination_mean >= 0.5 else 0

        if 'sentence_level_sem_similarity_scores' in sentence_level_hallu_scores:
            record.sentence_level_sem_similarity_scores = sentence_level_hallu_scores['sentence_level_sem_similarity_scores']
        else:
            record.sentence_level_sem_similarity_scores = {}

        # For each sentence/chunk, add is_gold_binary and evidence_position
        gold_text_norm = record.gold_text.lower().strip()
        model_answer_norm = record.model_answer.lower()
        chunk_scores = sentence_level_hallu_scores.get('sentence_level_sem_similarity_scores', {})
        chunk_gold_matches = {}
        matched_chunks = []
        for chunk, score in chunk_scores.items():
            chunk_norm = chunk.lower().strip()
            is_match = int(chunk_norm in gold_text_norm)
            chunk_gold_matches[chunk] = is_match
            if is_match:
                matched_chunks.append(chunk)
        
        if record.gold_text and record.model_prompt:
            gold_position = record.model_prompt.lower().find(record.gold_text.lower())
            if gold_position != -1:
                record.evidence_position = gold_position
        

        if hasattr(record, 'original_row') and isinstance(record.original_row, dict):
            result = dict(record.original_row)
        else:
            result = asdict(record)

        def _sanitize(v):
            if v is None:
                return v
            if isinstance(v, str):
                return " ".join(v.split())
            if isinstance(v, (list, dict)):
                return json.dumps(v, ensure_ascii=False)
            return v

        for k in list(result.keys()):
            try:
                result[k] = _sanitize(result[k])
            except Exception:
                result[k] = str(result[k])

        canonical_fields = {
            "question_id": record.question_id,
            "model_prompt": record.model_prompt,
            "config_idx": record.config_idx,
            "domain": record.domain,
            "config": record.config,
            "question": record.question,
            "gold_text": record.gold_text,
            "distractors": record.distractors,
            "row_index": record.row_index,
            "model_answer": record.model_answer,
        }
        canonical_fields = {k: _sanitize(v) for k, v in canonical_fields.items()}
        result.update(canonical_fields)

        result["sem_similarity_score"] = record.sem_similarity_score
        result["hallucination_label"] = record.hallucination_label
        result["sentence_level_sem_similarity_scores"] = _sanitize(record.sentence_level_sem_similarity_scores)
        result["is_gold_binary"] = int(bool(matched_chunks))
        result["evidence_position"] = record.evidence_position
        result["chunk_level_gold_matches"] = _sanitize(chunk_gold_matches)
        results.append(result)

    # Separate results by domain and save to different files
    import collections
    domain_results = collections.defaultdict(list)
    for result in results:
        domain = result.get('domain', 'unknown')
        domain_results[domain].append(result)

    for domain, domain_group in domain_results.items():
        domain_str = str(domain).replace(' ', '_')
        out_path = os.path.join(output_folder, f'answer_hallu_scored_{domain_str}.csv')
        if not domain_group:
            print(f"No results for domain {domain}")
            continue
        # Each domain_group item is a dict whose insertion order should reflect original input fields
        import pandas as pd
        df = pd.DataFrame(domain_group)
        # Determine preferred ordering: start with keys from the first row
        first_keys = list(domain_group[0].keys())
        # Common scoring columns we append at the end if present
        scoring_cols = ["sem_similarity_score", "sentence_level_sem_similarity_scores", "is_gold_binary", "evidence_position"]
        final_cols = [k for k in first_keys if k not in scoring_cols]
        final_cols += [k for k in scoring_cols if k in df.columns and k not in final_cols]
        # Reindex dataframe to the final column order (will add missing cols as NaN)
        df = df.reindex(columns=final_cols)
        # Truncate extremely long textual columns for CSV readability
        def _truncate_series(s, limit=1000):
            if s.dtype == object:
                return s.apply(lambda v: (v[:limit] + '...') if isinstance(v, str) and len(v) > limit else v)
            return s

        for col in ['prompt', 'model_prompt', 'model_documents', 'model_answer', 'gold_text']:
            if col in df.columns:
                df[col] = _truncate_series(df[col], limit=1000)

        # Write CSV with quoting to avoid column-jumping from embedded separators/newlines
        df.to_csv(out_path, index=False, quoting=csv.QUOTE_ALL)
        print(f"Saved results for domain {domain} to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Score hallucinations in model answers')
    parser.add_argument('--model', type=str, required=True, choices=['longchat', 'mistral'],
                        help='Model type to process: longchat or mistral')
    
    args = parser.parse_args()
    main(args.model)
