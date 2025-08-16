import json
import gzip
import sys
import re
from collections import Counter
from statistics import mean
from typing import Iterable, Optional, Tuple

# Tokenization & Metrics

def tokenize_words(text: Optional[str]) -> list:
    return text.split() if text else []

def count_tokens(text: Optional[str]) -> int:
    return len(tokenize_words(text))

def classify_evidence_position(gold_idx: int, n_docs: int) -> str:
    if n_docs <= 0 or gold_idx is None or gold_idx < 0:
        return None
    pct = (gold_idx + 1) / n_docs
    if pct <= 0.33:
        return "beginning"
    elif pct <= 0.66:
        return "middle"
    else:
        return "end"

def distractor_ratio(ctxs: Iterable[dict], gold_idx: int) -> float:
    total = 0
    distractor = 0
    for i, d in enumerate(ctxs):
        title = d.get("title") or ""
        text = d.get("text") or ""
        tok = count_tokens(title + "\n" + text)
        total += tok
        if i != gold_idx:
            distractor += tok
    return (distractor / total) if total > 0 else 0.0

def bucket_density(ratio: float) -> str:
    if ratio < 0.25:
        return "low"
    elif ratio < 0.60:
        return "medium"
    else:
        return "high"

def interference_ratio_from_response(response: str, ctxs: Iterable[dict], gold_idx: int) -> Tuple[int, int, float]:
    if not response:
        return 0, 0, 0.0
    distractor_text = " ".join(
        (d.get("title") or "") + " " + (d.get("text") or "")
        for i, d in enumerate(ctxs) if i != gold_idx
    )
    resp_tokens = tokenize_words(response)
    dist_tokens = set(tokenize_words(distractor_text))
    hits = sum(1 for t in resp_tokens if t in dist_tokens)
    n = len(resp_tokens)
    return hits, n, (hits / n if n > 0 else 0.0)

def _open_in(path: str):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, "r", encoding="utf-8")

def _open_out(path: str):
    return gzip.open(path, "wt", encoding="utf-8") if path.endswith(".gz") else open(path, "w", encoding="utf-8")

def enrich(input_path: str, output_path: str):
    total, updated = 0, 0
    with _open_in(input_path) as fin, _open_out(output_path) as fout:
        for line in fin:
            if not line.strip():
                continue
            total += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            ctxs = rec.get("ctxs") or rec.get("documents") or rec.get("model_documents")
            gold_idx = rec.get("gold_doc_index")

            density_val = rec.get("density")
            interference_val = rec.get("interference")
            gold_pos_val = rec.get("gold_position")

            if isinstance(ctxs, list) and isinstance(gold_idx, int) and 0 <= gold_idx < len(ctxs):
                # metrics
                pos = classify_evidence_position(gold_idx, len(ctxs))
                ratio = distractor_ratio(ctxs, gold_idx)
                dens_bucket = bucket_density(ratio)

                response = rec.get("model_answer") or rec.get("answer_pred") or rec.get("prediction")
                hits, resp_len, irr_ratio = interference_ratio_from_response(response or "", ctxs, gold_idx)

                if not density_val:
                    rec["density"] = dens_bucket
                if not gold_pos_val:
                    rec["gold_position"] = pos
                if not interference_val:
                    rec["interference"] = irr_ratio

                # Always store these computed fields
                rec["distractor_ratio"] = ratio
                rec["distractor_density"] = dens_bucket
                rec["evidence_position"] = pos
                rec["interference_token_hits"] = hits
                rec["interference_token_ratio"] = irr_ratio
                rec["response_token_count"] = resp_len

                updated += 1
            else:
                rec.setdefault("distractor_ratio", None)
                rec.setdefault("distractor_density", None)
                rec.setdefault("evidence_position", None)
                rec.setdefault("density", None)
                rec.setdefault("interference", None)
                rec.setdefault("gold_position", None)
                if not rec.get("status"):
                    rec["status"] = "no_ctxs_or_gold_index"

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Processed: {total} lines")
    print(f"Enriched (had ctxs+gold_idx): {updated} lines")
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python add_distractor_density.py INPUT.jsonl[.gz] OUTPUT.jsonl[.gz]")
        sys.exit(1)
    enrich(sys.argv[1], sys.argv[2])
