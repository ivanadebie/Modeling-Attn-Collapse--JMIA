#!/usr/bin/env python3
"""
Library of functions for calculating various metrics related to text quality,
hallucination, and interference. Originally refactored from generate_metrics_v1.py.
"""

import re, statistics, math, json
import os
from collections import Counter
from typing import List, Dict, Any, Tuple, Set
from tqdm import tqdm
import pandas as pd

try:
    import openai
except ImportError:
    openai = None

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
_EMBEDDING_MODEL = None
def try_embedding_similarity(a: str, b: str, model_name: str = "all-MiniLM-L6-v2") -> float:
    global _EMBEDDING_MODEL
    try:
        from sentence_transformers import SentenceTransformer, util
    except Exception:
        return float("nan")
    
    if not a or not b:
        return float("nan")
    
    try:
        if _EMBEDDING_MODEL is None:
            print("Loading embedding model...")
            _EMBEDDING_MODEL = SentenceTransformer(model_name)
            print("Embedding model loaded.")

        model = _EMBEDDING_MODEL
        ea = model.encode(a, normalize_embeddings=True)
        eb = model.encode(b, normalize_embeddings=True)
        # cosine similarity of normalized vectors = dot product
        sim = float((ea * eb).sum())
        return sim
    except Exception as e:
        print(f"Error during embedding: {e}")
        return float("nan")

# GPT-based interference scoring
_OPENAI_CLIENT = None
def get_gpt_interference_score(question: str, answer: str, distractors: List[str], model_name: str = "gpt-3.5-turbo") -> Tuple[float | None, str | None]:
    """
    Uses a GPT model to score the level of interference from distractors in an answer.
    """
    global _OPENAI_CLIENT
    if openai is None:
        # This check is mostly for type-hinting, the real check is in the calling function.
        return None, None

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        # This check is also a safeguard.
        return None, None

    if not answer or not distractors:
        return 0.0, "Nonsensical"

    if _OPENAI_CLIENT is None:
        _OPENAI_CLIENT = openai.OpenAI(api_key=api_key)

    client = _OPENAI_CLIENT
    distractor_text = "\n".join(f"- {d}" for d in distractors)

    system_prompt = """
You are an expert evaluator. Your task is to determine if a given 'Answer' to a 'Question' has been improperly influenced by 'Distractor Passages'.
The 'Answer' should be based on general knowledge or the evidence provided in a prompt (which you cannot see), not the 'Distractor Passages'.
Analyze the 'Answer' and determine the level of interference from the distractors.

Respond with a JSON object containing two keys:
1. "interference_score": A float between 0.0 (no interference) and 1.0 (heavy interference/plagiarism from distractors).
2. "interference_type": A string, one of "Nonsensical", "Thematic", or "Paraphrased".
   - "Nonsensical": The answer is irrelevant to the question, or the interference score is very low (< 0.15).
   - "Thematic": The answer correctly addresses the question's topic but incorporates concepts or specific terms from the distractors.
   - "Paraphrased": The answer directly copies or closely paraphrases sentences or phrases from the distractors.
"""
    user_prompt = f"""
**Question:**
{question}

**Distractor Passages:**
{distractor_text}

**Answer:**
{answer}

**Evaluation (JSON response):**
"""

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)
        score = float(result.get("interference_score", 0.0))
        itype = str(result.get("interference_type", "Nonsensical"))
        return score, itype
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return None, None

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
def compute_run_level_metrics(row: Dict[str, Any], use_embeddings: bool=True, use_gpt_scoring: bool=False) -> Dict[str, Any]:
    """Computes run-level metrics for a single data row."""
    prompt = str(row.get("prompt") or row.get("Prompt") or "")
    question = str(row.get("question") or "")
    gold_text = str(extract_gold_text(row) or "")
    distractors = extract_distractors(row)
    model_answer = str(row.get("model_answer") or row.get("response") or row.get("answer") or "")

    # Return early if key fields are missing to avoid errors
    if not model_answer or not gold_text:
        # Return a dictionary with default/None values for all keys
        return {
            "distractor_density": "low",
            "gold_span_start": -1.0, "gold_span_end": -1.0, "gold_position": "unknown",
            "total_response_tokens": 0, "interference_token_hits": 0,
            "interference_score_lexical": 0.0, "interference_type_lexical": "Nonsensical",
            "evid_overlap_ngram": 0.0, "response_alignment": 0.0, "hallu_score": 0.0,
            "hallu_label": "Low", "evid_overlap_emb": None, "interference_score_gpt": None,
            "interference_type_gpt": None, "att_avg_entropy": None
        }

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

    # Optional GPT-based interference scoring
    interference_score_gpt, interference_type_gpt = None, None
    if use_gpt_scoring:
        interference_score_gpt, interference_type_gpt = get_gpt_interference_score(question, model_answer, distractors)

    return {
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
        "interference_score_gpt": interference_score_gpt,
        "interference_type_gpt": interference_type_gpt,
        "att_avg_entropy": None,
    }

def compute_chunk_level_metrics(row: Dict[str, Any], max_subchunk_tokens: int=64) -> List[Dict[str, Any]]:
    """Computes chunk-level metrics for a single data row."""
    run_id = row.get("row_index", row.get("id", -1))
    prompt = str(row.get("prompt") or row.get("Prompt") or "")
    gold_text = str(extract_gold_text(row) or "")
    distractors = extract_distractors(row)
    
    chunk_rows = []
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
    return chunk_rows

def compute_metrics_for_dataset(rows: List[Dict[str, Any]], use_embeddings: bool=True, use_gpt_scoring: bool=False, max_subchunk_tokens: int=64):
    """
    Takes a list of data rows (dicts) and computes run-level and chunk-level metrics for the whole dataset.
    """
    all_run_metrics = []
    all_chunk_metrics = []
    for idx, row in enumerate(tqdm(rows, desc="Calculating metrics")):
        # Ensure a stable ID exists for linking run and chunk data
        if "row_index" not in row:
            row["row_index"] = idx
        
        run_metrics = compute_run_level_metrics(row, use_embeddings=use_embeddings, use_gpt_scoring=use_gpt_scoring)
        run_metrics["run_id"] = row["row_index"]
        all_run_metrics.append(run_metrics)

        chunk_metrics = compute_chunk_level_metrics(row, max_subchunk_tokens=max_subchunk_tokens)
        all_chunk_metrics.extend(chunk_metrics)
        
    return all_run_metrics, all_chunk_metrics

def calculate_chunk_level_features(row: pd.Series) -> pd.Series:
    """
    Calculates all required features for a single chunk (sentence).
    This is designed to be used with df.apply.
    """
    # Ensure gold_text and distractors are lists of strings
    gold_text_list = row.get('gold_text', [])
    if isinstance(gold_text_list, str):
        gold_text_list = [gold_text_list]
    
    distractors_list = row.get('distractors', [])
    if isinstance(distractors_list, str):
        distractors_list = [distractors_list]

    gold_text_full = " ".join(gold_text_list)
    distractors_full = " ".join(distractors_list)
    
    sentence = row['sentence_text']

    # is_gold_binary: Check for semantic overlap with any gold sentence
    is_gold_binary = 0
    if gold_text_full and sentence:
        # A simple string-based check
        if sentence.lower() in gold_text_full.lower():
            is_gold_binary = 1
        else:
            # A more robust n-gram check
            sent_ngrams = ngram_set(preprocess_tokens(sentence, drop_stop=True), 2)
            gold_ngrams = ngram_set(preprocess_tokens(gold_text_full, drop_stop=True), 2)
            if jaccard(sent_ngrams, gold_ngrams) > 0.2: # Threshold for semantic match
                is_gold_binary = 1
    
    # evidence_position: Calculated later, requires view of all chunks
    # This will be calculated in the main script after grouping.
    
    # distractor_density_chunk
    sent_tokens = set(tokenize(sentence))
    distractor_tokens = set(tokenize(distractors_full))
    distractor_density_chunk = len(sent_tokens.intersection(distractor_tokens)) / len(sent_tokens) if sent_tokens else 0

    # evidence_overlap_ratio (ngram-based)
    evidence_overlap_ratio = jaccard(
        ngram_set(preprocess_tokens(sentence), 3),
        ngram_set(preprocess_tokens(gold_text_full), 3)
    )

    # interference_score_lexical for the chunk
    interference_token_hits = len(sent_tokens.intersection(distractor_tokens))
    total_chunk_tokens = len(sent_tokens)
    interference_score_lexical_wrt_distractors = interference_token_hits / total_chunk_tokens if total_chunk_tokens > 0 else 0

    # interference_type_lexical
    if interference_score_lexical_wrt_distractors < 0.15:
        interference_type_lexical_wrt_distractors = "Nonsensical"
    elif interference_score_lexical_wrt_distractors < 0.5:
        interference_type_lexical_wrt_distractors = "Thematic"
    else:
        interference_type_lexical = "Paraphrased"
        
    # model_response_alignment_score (semantic similarity between chunk and gold)
    # Using embedding similarity as a proxy for this
    model_response_alignment_score = try_embedding_similarity(sentence, gold_text_full)

    # normalized_chunk_index: Calculated later, requires view of all chunks

    return pd.Series({
        'is_gold_binary': is_gold_binary,
        'distractor_density_chunk': distractor_density_chunk,
        'evidence_overlap_ratio': evidence_overlap_ratio,
        'interference_score_lexical': interference_score_lexical_wrt_distractors,
        'interference_type_lexical': interference_type_lexical_wrt_distractors,
        'model_response_alignment_score': model_response_alignment_score,
    })
