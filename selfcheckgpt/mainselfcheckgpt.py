import pandas as pd
from pyparsing import col
from sympy import re
import re
from tqdm import tqdm 
from dataclasses import asdict
import numpy as np
import os
# import llm_apihandler
from data_requirements import QARecord # deduplicate_samples 
from metrics_selfcheck import HallucinationScorer
from data_utils import save_results_to_csv
import json
import csv

INPUT_JSONL_FILES = [
    os.path.join(os.path.dirname(__file__), 'domain1_data_nq_closed_book_8000_converted.output.jsonl'),
    os.path.join(os.path.dirname(__file__), 'domain2_data_hotpot_citation_8000_converted.output.jsonl'),
    os.path.join(os.path.dirname(__file__), 'domain3_data_novelhop_multihop_8000_converted.output.jsonl'),
    os.path.join(os.path.dirname(__file__), 'domain4_data_temporal_recency_8000_converted.output.jsonl'),
    os.path.join(os.path.dirname(__file__), 'domain5_data_05_stackmath_numerical_8000_converted.output.jsonl'),
    os.path.join(os.path.dirname(__file__), 'domain6_data_06_policy_compliance_8000_converted.output.jsonl'),
    os.path.join(os.path.dirname(__file__), 'domain7_data_07_expert_legal_med_8000_converted.output.jsonl'),
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
                # Keep the entire original input row so we can preserve all input headings/fields in outputs
                setattr(record, 'original_row', row)
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

        # Use gold_text as the reference for semantic similarity scoring
        sentence_level_hallu_scores = scorer.get_sentence_level_hallucination_scores(
            record.model_answer, record.gold_text, MODEL_TO_USE, seed=i
        )
        print(f"Sentence-level scores: {sentence_level_hallu_scores}")
        # sample_level_hallu_scores = scorer.get_sample_level_hallucination_scores(full_model_prompt or record.prompt, MODEL_TO_USE, seed=i)
        if not sentence_level_hallu_scores or not sentence_level_hallu_scores.get('sentence_level_hallucination_scores'):
            print(f"[SKIP] No sentence/chunk scores produced for question_id {record.question_id}")
            continue

        # aggregate_confidence_scores returns aggregated similarity metrics (higher = more similar)
        whole_answer_hallu_score = scorer.aggregate_confidence_scores(sentence_level_hallu_scores)
        # convert similarity -> hallucination likelihood: hallucination = 1 - similarity
        similarity_mean = whole_answer_hallu_score.get('conf_agg_mean', 0.0)
        hallucination_mean = 1.0 - similarity_mean
        whole_answer_hallu_label = 1 if hallucination_mean >= 0.85 else 0  # 1-hallu / 0-not hallu

        # model_response_uncertainty = scorer.aggregate_confidence_scores(sample_level_hallu_scores)['conf_agg_mean']

        # sample_level_95th_percentile = scorer.calculate_95th_percentile(
        #     sample_level_hallu_scores.get('sample_level_hallucination_scores', [])
        # )

        record.hallucination_label = whole_answer_hallu_label
        # store hallucination_score as a probability-like value where higher means more likely hallucination
        record.hallucination_score = hallucination_mean
        # record.model_response_uncertainty = model_response_uncertainty
        # record.confidence = 1 - model_response_uncertainty
        if 'sentence_level_hallucination_scores' in sentence_level_hallu_scores:
            record.sentence_level_hallu_scores = sentence_level_hallu_scores['sentence_level_hallucination_scores']
        else:
            record.sentence_level_hallu_scores = {}
        # record.sample_level_hallu_scores = sample_level_hallu_scores['sample_level_hallucination_scores']
        # record.sample_level_95th_percentile = sample_level_95th_percentile
        # Attach sentence-level scores to the record (if present)
        if 'sentence_level_hallucination_scores' in sentence_level_hallu_scores:
            record.sentence_level_hallu_scores = sentence_level_hallu_scores['sentence_level_hallucination_scores']
        else:
            record.sentence_level_hallu_scores = {}

        # For each sentence/chunk, add is_gold_binary and gold_text_chunk
        gold_text_norm = record.gold_text.lower().strip()
        model_answer_norm = record.model_answer.lower()
        chunk_scores = sentence_level_hallu_scores.get('sentence_level_hallucination_scores', {})
        chunk_gold_matches = {}
        matched_chunks = []
        for chunk, score in chunk_scores.items():
            chunk_norm = chunk.lower().strip()
            is_match = int(chunk_norm in gold_text_norm)
            chunk_gold_matches[chunk] = is_match
            if is_match:
                matched_chunks.append(chunk)

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

        result["hallucination_score"] = record.hallucination_score
        result["hallucination_label"] = record.hallucination_label
        result["sentence_level_hallu_scores"] = _sanitize(record.sentence_level_hallu_scores)
        result["is_gold_binary"] = int(bool(matched_chunks))
        result["gold_text_chunk"] = _sanitize(matched_chunks)
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
        out_path = os.path.join(os.path.dirname(__file__), f'answer_hallu_scored_{domain_str}.csv')
        if not domain_group:
            print(f"No results for domain {domain}")
            continue
        # Preserve column order from the original input row where possible.
        # Each domain_group item is a dict whose insertion order should reflect original input fields
        import pandas as pd
        df = pd.DataFrame(domain_group)
        # Determine preferred ordering: start with keys from the first row
        first_keys = list(domain_group[0].keys())
        # Common scoring columns we append at the end if present
        scoring_cols = ["hallucination_score", "sentence_level_hallu_scores", "is_gold_binary", "gold_text_chunk"]
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
        # Also write a JSONL with full sanitized content (one JSON per line) for exact downstream use
        jsonl_out = os.path.splitext(out_path)[0] + '.jsonl'
        with open(jsonl_out, 'w', encoding='utf-8') as jf:
            for row in domain_group:
                jf.write(json.dumps(row, ensure_ascii=False) + '\n')
        print(f"Results saved to {out_path}")

if __name__ == "__main__":
    main()
