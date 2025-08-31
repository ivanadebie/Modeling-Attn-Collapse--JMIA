#!/usr/bin/env python3
"""
Fill Interference for Synthetic_QA.csv by comparing distractors from
Nonsensical_Distractor.csv against each QA's question and gold answer.

Interference classes:
- paraphrased: high similarity to gold/question
- thematic:    moderate similarity
- nonsensical: low similarity

Outputs:
- Synthetic_QA_with_Interference.csv  (adds Interference and similarity columns)
- Interference_Details.csv            (per-qa, per-distractor similarity & type)
"""

import argparse
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in out.columns]
    return out

def find_col(df: pd.DataFrame, patterns) -> str | None:
    for p in patterns:
        for c in df.columns:
            if re.search(p, c, flags=re.I):
                return c
    return None

def get_distractor_text_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if re.search(r"(?i)\bDistractor\b.*\bText\b", c)]

def collect_distractors(copy_df: pd.DataFrame, qa_col: str, text_cols: list[str]) -> dict[str, list[str]]:
    dmap: dict[str, list[str]] = {}
    for _, row in copy_df.iterrows():
        qid = row.get(qa_col)
        if pd.isna(qid):
            continue
        texts = []
        for c in text_cols:
            val = row.get(c, None)
            if isinstance(val, str) and val.strip():
                texts.append(val.strip())
        if texts:
            dmap.setdefault(qid, []).extend(texts)
    return dmap

def tfidf_max_sim(question: str, gold: str, distractors: list[str]) -> tuple[float, float, float]:
    """Return (max_sim_to_gold, max_sim_to_question, overall_max)."""
    if not distractors:
        return 0.0, 0.0, 0.0
    base_texts, labels = [], []
    if isinstance(gold, str) and gold.strip():
        base_texts.append(gold)
        labels.append("gold")
    if isinstance(question, str) and question.strip():
        base_texts.append(question)
        labels.append("question")
    if not base_texts:
        return 0.0, 0.0, 0.0

    docs = base_texts + distractors
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    X = vec.fit_transform(docs)
    base = X[:len(base_texts)]
    dmat = X[len(base_texts):]
    sims = cosine_similarity(dmat, base)  # n_distractors x n_base

    sim_to_gold = 0.0
    sim_to_q = 0.0
    if "gold" in labels:
        idx_gold = labels.index("gold")
        sim_to_gold = float(sims[:, idx_gold].max())
    if "question" in labels:
        idx_q = labels.index("question")
        sim_to_q = float(sims[:, idx_q].max())
    return sim_to_gold, sim_to_q, max(sim_to_gold, sim_to_q)

def classify_adaptive(overall_sims: pd.Series) -> tuple[float, float]:
    """Return (thematic_threshold, paraphrased_threshold) from non-zero sims."""
    nz = overall_sims[overall_sims > 0]
    if len(nz) == 0:
        return 0.0, 0.0
    th_thematic = float(nz.quantile(0.50))   # median of non-zero sims
    th_para     = float(nz.quantile(0.90))   # top 10% of non-zero sims
    return th_thematic, th_para

def classify_fixed(s: float, th_thematic: float, th_para: float) -> str:
    if s >= th_para:
        return "paraphrased"
    if s >= th_thematic:
        return "thematic"
    return "nonsensical"


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", default="Synthetic_QA.csv", help="Path to Synthetic_QA.csv")
    ap.add_argument("--copy", default="Copy_of_Nonsensical_Distractor.csv",
                    help="Path to Copy_of_Nonsensical_Distractor.csv")
    ap.add_argument("--encoding", default="ISO-8859-1")
    # Threshold options
    ap.add_argument("--mode", choices=["adaptive", "fixed"], default="adaptive",
                    help="Thresholding mode for interference classification")
    ap.add_argument("--thematic", type=float, default=0.20, help="Fixed threshold for thematic (used if --mode fixed)")
    ap.add_argument("--paraphrased", type=float, default=0.35, help="Fixed threshold for paraphrased (used if --mode fixed)")
    # Outputs
    ap.add_argument("--out", default="Synthetic_QA_with_Interference.csv")
    ap.add_argument("--details_out", default="Interference_Details.csv")
    args = ap.parse_args()

    # Load inputs
    syn = norm_cols(pd.read_csv(args.synthetic, encoding=args.encoding))
    cdf = norm_cols(pd.read_csv(args.copy, encoding=args.encoding))

    # Identify columns
    qa_col_syn = find_col(syn, [r"^QA\s*ID$"])
    question_col = find_col(syn, [r"^Questions?$"])
    gold_col_syn = find_col(syn, [r"Answer\s*\(Gold Doc\)"])
    qa_col_copy = find_col(cdf, [r"^QA\s*ID$"])
    if not (qa_col_syn and question_col and gold_col_syn and qa_col_copy):
        raise ValueError("Required columns not found. Check 'QA ID', 'Questions', and 'Answer (Gold Doc)...' in Synthetic_QA and 'QA ID' in the copy file.")

    distractor_cols = get_distractor_text_cols(cdf)
    if not distractor_cols:
        raise ValueError("No distractor text columns found in Copy_of_Nonsensical_Distractor.csv")

    # Build d_map[QA ID] -> list of distractor texts
    d_map = collect_distractors(cdf, qa_col_copy, distractor_cols)

    # Compute similarities & collect per-distractor details
    per_qa = []
    details = []
    for _, row in syn.iterrows():
        qid = row[qa_col_syn]
        question = str(row.get(question_col, "") or "")
        gold = str(row.get(gold_col_syn, "") or "")
        d_texts = d_map.get(qid, [])
        if not d_texts:
            per_qa.append((qid, 0.0, 0.0, 0.0, 0, "nonsensical"))
            continue

        # Compute per-qa max sims
        sim_g, sim_q, sim_overall = tfidf_max_sim(question, gold, d_texts)
        per_qa.append((qid, sim_g, sim_q, sim_overall, len(d_texts), ""))

        # Also log each distractor’s own similarity/type later with shared thresholds
        # (We’ll compute thresholds first if using adaptive mode)
        details.append((qid, d_texts))  # delayed to second pass

    sim_df = pd.DataFrame(per_qa, columns=["QA ID", "sim_to_gold", "sim_to_question", "sim_overall", "d_count", "Interference"])

    # Choose thresholds
    if args.mode == "adaptive":
        th_thematic, th_para = classify_adaptive(sim_df["sim_overall"])
    else:
        th_thematic, th_para = args.thematic, args.paraphrased

    # Classify each QA based on overall similarity
    sim_df["Interference"] = sim_df["sim_overall"].apply(lambda s: classify_fixed(s, th_thematic, th_para))

    # Merge back and write output
    out = syn.merge(sim_df[["QA ID", "Interference", "sim_to_gold", "sim_to_question", "sim_overall", "d_count"]],
                    on="QA ID", how="left")
    out.to_csv(args.out, index=False, encoding=args.encoding)

    # Per-distractor details (optional, same thresholds)
    rows = []
    for qid, d_texts in details:
        # Rebuild small corpus for this QA to score each distractor independently
        question = str(syn.loc[syn[qa_col_syn] == qid, question_col].iloc[0])
        gold = str(syn.loc[syn[qa_col_syn] == qid, gold_col_syn].iloc[0])
        # Vectorize once per QA
        base_texts, labels = [], []
        if gold.strip():
            base_texts.append(gold)
            labels.append("gold")
        if question.strip():
            base_texts.append(question)
            labels.append("question")
        docs = base_texts + d_texts
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
        X = vec.fit_transform(docs)
        base = X[:len(base_texts)]
        dmat = X[len(base_texts):]
        sims = cosine_similarity(dmat, base) if base.shape[0] and dmat.shape[0] else None

        for i, dtext in enumerate(d_texts):
            if sims is None:
                sim_overall = 0.0
            else:
                sg = float(sims[i, labels.index("gold")]) if "gold" in labels else 0.0
                sq = float(sims[i, labels.index("question")]) if "question" in labels else 0.0
                sim_overall = max(sg, sq)
            itype = classify_fixed(sim_overall, th_thematic, th_para)
            rows.append({
                "QA ID": qid,
                "Distractor #": i + 1,
                "Similarity": round(sim_overall, 6),
                "Interference Type": itype
            })

    pd.DataFrame(rows).to_csv(args.details_out, index=False, encoding=args.encoding)

    print("✅ Done.")
    print(f"- Updated Synthetic: {args.out}")
    print(f"- Details log:       {args.details_out}")
    print(f"- Thresholds used:   thematic={th_thematic:.6f}, paraphrased={th_para:.6f}  (mode={args.mode})")


if __name__ == "__main__":
    main()
