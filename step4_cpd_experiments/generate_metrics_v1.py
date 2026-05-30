#!/usr/bin/env python3
"""
Inputs
- JSON file with a list of rows. Expected (but flexible) fields:
  - 'prompt' (str): raw prompt that may include "Document [k]" sections.
  - 'question' (str): the question text (optional).
  - 'gold_text' (str): gold/evidence text. (preferred; else see extract_gold_text heuristic)
  - 'distractors' (list[str]): distractor passages if available. (preferred; else heuristic)
  - 'model_answer' | 'response' | 'answer' (str): model/system answer (optional but needed for some metrics)
  - 'row_index' (int): stable identifier for the item; if missing, index is used.

Outputs
- run_level_metrics.csv
- chunk_level_metrics.csv
- metrics_output.xlsx (two sheets) if --excel is passed (default: true)

Metrics (Run Level)
- run_id                               : stable row id (row['row_index'] if present else enumerate index)
- num_distractors                      
- distractor_ratio                      : distractor_tokens / (gold_tokens + distractor_tokens)
- distractor_density                    : bucketed from ratio (<0.33=low, <0.66=medium, else high)
- gold_span_start, gold_span_end        : normalized indices of gold_text inside prompt tokens; -1.0 if not found
- gold_position                         : beginning/middle/end/unknown from span
- total_response_tokens                
- interference_token_hits               : # unique response tokens also in distractor vocab (after canonical processing)
- interference_score_lexical            : interference_token_hits / total_response_tokens
- interference_type_lexical             : <0.15=Nonsensical, <0.5=Thematic, else Paraphrased
- evid_overlap_ngram                    : Jaccard 3-gram overlap resp vs gold_text (optional; falls back to 0 if no response)
- response_alignment                    : alignment score = Jaccard(resp,gold) - Jaccard(resp,distractors_merged)
- hallu_score                           : fraction of response tokens not present in (prompt ∪ gold_text); 0..1
- hallu_label                           : Low/Medium/High thresholded from hallu_score
- evid_overlap_emb                      : cosine similarity between embeddings of response and gold (if --embeddings and package is available)
- interference_score_gpt, interference_type_gpt : placeholders; can be merged later from external evals

Metrics (Chunk Level)
- run_id, chunk_idx, chunk_text
- is_gold                               : 1 if chunk shares any token with gold_text
- evidence_position                     : min distance to any gold chunk / (T-1)
- distractor_density_chunk              : |chunk_tokens ∩ distractor_vocab| / |chunk_tokens|
- normalized_idx                        : chunk_idx / (T-1)

Usage
python generate_metrics.py /path/to/data.json --outdir ./out
python generate_metrics.py /path/to/data.json --no-excel
python generate_metrics.py /path/to/data.json --embeddings  # optional; requires 'sentence_transformers'
"""

import argparse, json, math, os, re, statistics, sys
from collections import Counter
from typing import List, Dict, Any, Tuple, Set

import pandas as pd

# Text preprocessing utils
_STOPWORDS = {
    "a","an","the","and","or","but","if","then","else","when","while","of","in","on","at","to","for","from","by","with",
    "without","between","among","over","under","into","through","as","is","are","was","were","be","been","being","have",
    "has","had","having","do","does","did","doing","can","could","may","might","must","will","would","should","shall",
    "not","no","nor","it","its","this","that","these","those","i","you","he","she","we","they","them","me","him","her",
    "us","our","your","their","my","mine","yours","theirs","ours","who","whom","whose","which","what","where","when",
    "why","how","about","above","below","before","after","again","further","here","there","all","any","both","each",
    "few","more","most","other","some","such","only","own","same","so","than","too","very"
}

def canon(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(text: str) -> List[str]:
    return canon(text).split()

def simple_lemmatize(tokens: List[str]) -> List[str]:
    out = []
    for t in tokens:
        for suf in ("ing","ed","es","s"):
            if t.endswith(suf) and len(t) > len(suf) + 2:
                t = t[:-len(suf)]
                break
        out.append(t)
    return out

def preprocess_tokens(text: str, drop_stop=True) -> List[str]:
    toks = simple_lemmatize(tokenize(text))
    if drop_stop:
        toks = [t for t in toks if t not in _STOPWORDS]
    return toks

def ngram_set(tokens: List[str], n: int) -> Set[Tuple[str, ...]]:
    return set(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)) if n > 0 else set()

def jaccard(a: Set, b: Set) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return float(inter) / union if union else 0.0

# Field extraction
def extract_gold_text(row: Dict[str, Any]) -> str:
    for key in ("gold_text","gold","gold_document","gold_doc","gold_passages","gold_source"):
        if key in row and row[key]:
            val = row[key]
            if isinstance(val, str):
                return val
            if isinstance(val, list) and val:
                return " ".join(map(str, val))
    # fallback: try to pick a "Document [k]" block that looks most contentful
    prompt = str(row.get("prompt", "") or row.get("Prompt", ""))
    docs = re.split(r"(?:^|\n)Document\s*\[\d+\]\s*", prompt, flags=re.IGNORECASE)
    candidates = [d.strip() for d in docs[1:]] if len(docs) > 1 else ([prompt] if prompt else [])
    def score(doc: str) -> int:
        t = canon(doc)
        s = 0
        if re.search(r"\b(19|20)\d{2}\b", doc): s += 2
        if re.search(r"[A-Z][a-z]+ [A-Z][a-z]+", doc): s += 1
        if re.search(r"\b(composer|lyrics|history|film|reception|chart|population|chemistry)\b", t): s += 2
        return s
    return max(candidates, key=score) if candidates else ""

def extract_distractors(row: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    if isinstance(row.get("distractors"), list):
        out.extend([d for d in row["distractors"] if isinstance(d, str) and d.strip()])
    if out:
        # dedupe (preserve order)
        seen = set()
        dedup = []
        for d in out:
            k = d[:200]
            if k not in seen:
                seen.add(k); dedup.append(d)
        return dedup
    # fallback: parse whimsical docs from prompt
    prompt = str(row.get("prompt", "") or row.get("Prompt", ""))
    docs = re.split(r"(?:^|\n)Document\s*\[\d+\]\s*", prompt, flags=re.IGNORECASE)
    cands = [d.strip() for d in docs[1:]] if len(docs) > 1 else []
    bad_terms = r"\b(pancake|unicorn|licorice|jellybean|giraffe|tapir|trampoline|marshmallow|waffle|spaghetti|corgi|zebra|pogo)\b"
    return [d for d in cands if re.search(bad_terms, canon(d))]

# Span & position utilities
def find_span(tokens: List[str], subseq: List[str]) -> Tuple[float, float]:
    if not tokens or not subseq:
        return (-1.0, -1.0)
    n, m = len(tokens), len(subseq)
    for i in range(0, n - m + 1):
        if tokens[i:i+m] == subseq:
            denom = max(1, n - 1)
            return (round(i / denom, 6), round((i + m - 1) / denom, 6))
    return (-1.0, -1.0)

def gold_position_from_span(start: float, end: float) -> str:
    if start < 0 or end < 0:
        return "unknown"
    if end <= 0.33:
        return "beginning"
    if start >= 0.66:
        return "end"
    return "middle"

# Optional embedding similarity
def try_embedding_similarity(a: str, b: str, model_name: str = "all-MiniLM-L6-v2") -> float:
    try:
        from sentence_transformers import SentenceTransformer, util
    except Exception:
        return float("nan")
    if not a or not b:
        return float("nan")
    try:
        model = SentenceTransformer(model_name)
        ea = model.encode(a, normalize_embeddings=True)
        eb = model.encode(b, normalize_embeddings=True)
        # cosine similarity of normalized vectors = dot product
        sim = float((ea * eb).sum())
        return sim
    except Exception:
        return float("nan")

# Chunking
def split_into_chunks(prompt: str, max_subchunk_tokens: int = 64) -> List[str]:
    text = (prompt or "").strip()
    if not text:
        return []
    parts = re.split(r'(?<=[\.!?])\s+', text)
    parts = [p.strip() for p in parts if p.strip()]
    if not parts:
        return [text]

    # average sentence length
    lens = [len(tokenize(p)) for p in parts] or [0]
    avg_len = statistics.mean(lens) if lens else 0
    subchunks: List[str] = []
    for p in parts:
        toks = tokenize(p)
        # split if much longer than average
        if avg_len and len(toks) > int(0.8 * avg_len):
            for i in range(0, len(toks), max_subchunk_tokens):
                subchunks.append(" ".join(toks[i:i+max_subchunk_tokens]))
        else:
            subchunks.append(p)
    return subchunks

# Core computations
def compute_run_and_chunk_rows(rows: List[Dict[str, Any]], use_embeddings: bool=False, max_subchunk_tokens: int=64):
    run_rows = []
    chunk_rows = []

    for idx, row in enumerate(rows):
        run_id = row.get("row_index", idx)
        prompt = str(row.get("prompt") or row.get("Prompt") or "")
        question = str(row.get("question") or "")
        gold_text = str(extract_gold_text(row) or "")
        distractors = extract_distractors(row)
        model_answer = str(row.get("model_answer") or row.get("response") or row.get("answer") or "")

        # Token bags
        gold_tokens = tokenize(gold_text) if gold_text else []
        distr_tokens = tokenize(" ".join(distractors)) if distractors else []
        prompt_tokens = tokenize(prompt)
        resp_tokens = preprocess_tokens(model_answer) if model_answer else []

        # Distractor stats
        num_distractors = sum(1 for d in distractors if isinstance(d, str) and d.strip())
        denom = (len(gold_tokens) + len(distr_tokens)) or 1
        distractor_ratio = round(len(distr_tokens) / denom, 6)
        distractor_density = "low" if distractor_ratio < 0.33 else ("medium" if distractor_ratio < 0.66 else "high")

        # Gold span + position
        gold_span_start, gold_span_end = find_span(prompt_tokens, gold_tokens)
        gold_position = gold_position_from_span(gold_span_start, gold_span_end)

        # Lexical interference vs distractor vocab
        distractor_vocab = set(preprocess_tokens(" ".join(distractors))) if distractors else set()
        unique_resp = set(resp_tokens)
        interference_token_hits = sum(1 for t in unique_resp if t in distractor_vocab)
        total_response_tokens = len(resp_tokens)
        interference_score_lexical = (interference_token_hits / total_response_tokens) if total_response_tokens > 0 else 0.0
        if interference_score_lexical < 0.15:
            interference_type_lexical = "Nonsensical"
        elif interference_score_lexical < 0.5:
            interference_type_lexical = "Thematic"
        else:
            interference_type_lexical = "Paraphrased"

        # N-gram overlap based signals (response vs gold_text / distractors)
        resp_toks_ng = preprocess_tokens(model_answer, drop_stop=True)
        gold_toks_ng = preprocess_tokens(gold_text, drop_stop=True)
        distr_toks_ng = preprocess_tokens(" ".join(distractors), drop_stop=True) if distractors else []

        resp_tri = ngram_set(resp_toks_ng, 3)
        gold_tri = ngram_set(gold_toks_ng, 3)
        distr_tri = ngram_set(distr_toks_ng, 3) if distr_toks_ng else set()

        evid_overlap_ngram = jaccard(resp_tri, gold_tri) if resp_tri or gold_tri else 0.0
        j_gold = jaccard(resp_tri, gold_tri)
        j_distr = jaccard(resp_tri, distr_tri) if distr_tri else 0.0
        response_alignment = j_gold - j_distr  # positive => more aligned with gold than distractors

        # Simple hallucination proxy: fraction of response tokens not in (prompt ∪ gold)
        grounding_vocab = set(preprocess_tokens(prompt)) | set(preprocess_tokens(gold_text))
        hallu_oov = sum(1 for t in resp_tokens if t not in grounding_vocab)
        hallu_score = (hallu_oov / total_response_tokens) if total_response_tokens > 0 else 0.0
        if hallu_score < 0.25:
            hallu_label = "Low"
        elif hallu_score < 0.5:
            hallu_label = "Medium"
        else:
            hallu_label = "High"

        # Optional embedding similarity
        evid_overlap_emb = try_embedding_similarity(model_answer, gold_text) if use_embeddings else float("nan")

        run_rows.append({
            "run_id": run_id,
            "num_distractors": num_distractors,
            "distractor_ratio": round(distractor_ratio, 6),
            "distractor_density": distractor_density,
            "gold_span_start": gold_span_start,
            "gold_span_end": gold_span_end,
            "gold_position": gold_position,
            "total_response_tokens": total_response_tokens,
            "interference_token_hits": interference_token_hits,
            "interference_score_lexical": round(interference_score_lexical, 6),
            "interference_type_lexical": interference_type_lexical,
            "evid_overlap_ngram": round(evid_overlap_ngram, 6),
            "response_alignment": round(response_alignment, 6),
            "hallu_score": round(hallu_score, 6),
            "hallu_label": hallu_label,
            "evid_overlap_emb": round(evid_overlap_emb, 6) if isinstance(evid_overlap_emb, float) and not math.isnan(evid_overlap_emb) else None,
            # Placeholders for external/advanced metrics
            "interference_score_gpt": None,
            "interference_type_gpt": None,
            "att_avg_entropy": None,
            "att_std_entropy": None,
            "att_avg_effective_rank": None,
        })

        # Chunk-level
        chunks = split_into_chunks(prompt, max_subchunk_tokens=max_subchunk_tokens)
        T = len(chunks)
        gold_can = set(tokenize(gold_text)) if gold_text else set()
        distr_vocab_basic = set(tokenize(" ".join(distractors))) if distractors else set()

        gold_chunks = set()
        for i, ch in enumerate(chunks):
            ch_set = set(tokenize(ch))
            if gold_can and len(gold_can & ch_set) > 0:
                gold_chunks.add(i)

        for i, ch in enumerate(chunks):
            ch_tokens = tokenize(ch)
            is_gold = 1 if i in gold_chunks else 0
            evid_pos = (min(abs(i-g) for g in gold_chunks) / (T-1)) if (gold_chunks and T > 1) else (1.0 if T > 1 else 0.0)
            dd_chunk = (len(distr_vocab_basic & set(ch_tokens)) / len(ch_tokens)) if ch_tokens else 0.0
            norm_idx = (i / (T-1)) if T > 1 else 0.0

            chunk_rows.append({
                "run_id": run_id,
                "chunk_idx": i,
                "chunk_text": ch,
                "is_gold": is_gold,
                "evidence_position": round(evid_pos, 6),
                "distractor_density_chunk": round(dd_chunk, 6),
                "normalized_idx": round(norm_idx, 6),
            })

    return run_rows, chunk_rows

# Main
def main():
    p = argparse.ArgumentParser()
    #p.add_argument("json_path", default=".", help="Path to input JSON (list of dicts).")
    p.add_argument("--outdir", default=".", help="Output directory (default: current dir).")
    p.add_argument("--no-excel", dest="excel", action="store_false", help="Do not write Excel workbook.")
    p.add_argument("--excel", dest="excel", action="store_true", help="Write Excel workbook (default).")
    p.add_argument("--embeddings", action="store_true", help="Attempt embedding similarity if sentence_transformers is available.")
    p.add_argument("--max-subchunk-tokens", type=int, default=64, help="Chunking granularity for long sentences.")
    p.set_defaults(excel=True)
    args = p.parse_args()

    args.outdir = "/Users/skuo/skuo/Js/output_v2/"
    args.json_path = "/Users/skuo/skuo/Js/0928-input/domain1_data_1questions_8000_final_copy.json"
    os.makedirs(args.outdir, exist_ok=True)
    with open(args.json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = data if isinstance(data, list) else [data]

    run_rows, chunk_rows = compute_run_and_chunk_rows(rows, use_embeddings=args.embeddings, max_subchunk_tokens=args.max_subchunk_tokens)
    run_df = pd.DataFrame(run_rows)
    chunk_df = pd.DataFrame(chunk_rows)

    run_csv = os.path.join(args.outdir, "run_level_metrics.csv")
    chunk_csv = os.path.join(args.outdir, "chunk_level_metrics.csv")
    run_df.to_csv(run_csv, index=False)
    chunk_df.to_csv(chunk_csv, index=False)

    if args.excel:
        xlsx_path = os.path.join(args.outdir, "metrics_output.xlsx")
        with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
            run_df.to_excel(writer, index=False, sheet_name="run_level")
            chunk_df.to_excel(writer, index=False, sheet_name="chunk_level")

    print(f"Wrote:\n- {run_csv}\n- {chunk_csv}")
    if args.excel:
        print(f"- {os.path.join(args.outdir, 'metrics_output.xlsx')}")

if __name__ == "__main__":
    main()
