#!/usr/bin/env python3
"""
CPD feature aggregation + Spearman correlation + visualization pipeline

Inputs
------
- prepared_dataset_cpd_with_attn_metrics_FINAL.csv  (or a .zip containing a single CSV)

Outputs (written to --out_dir; default: ./cpd_corr_outputs)
----------------------------------------------------------
1) Aggregated tables:
   - config_level_means_per_domain_question_config.csv
   - config_averaged_means_per_domain_question.csv
   - config_averaged_means_per_domain_question__zscore_within_domain.csv

2) Correlation tables + "strong" pairs:
   - spearman_pairs__<domain>__question_<qid>__across_configs.csv
   - strong_pairs__<domain>__question_<qid>__across_configs.csv
   - spearman_pairs__<domain>__across_persons.csv
   - strong_pairs__<domain>__across_persons.csv
   - strong_pair_frequency__<domain>__across_people.csv

3) Visualizations:
   - corr_heatmap__<domain>__question_<qid>__across_configs.png
   - corr_heatmap__<domain>__across_persons.png
   - scatter_pairs__<domain>__question_<qid>__batch_XX.pdf  (batched pairwise scatter plots)

4) Metadata:
   - analysis_meta.json

5) Optional zip of out_dir:
   - <out_dir>.zip  (enabled by default; use --no_zip to disable)

Notes
-----
- Treats `domain` as dataset and `question_id` as person.
- Filters to successful rows: (oom == False) and error is NaN.
- Aggregation:
    chunk -> (domain, question_id, config) means
    then config-averaged means per (domain, question_id)
- Standardization: z-score per domain across persons.

Example
-------
python run_cpd_corr_pipeline.py \
  --input prepared_dataset_cpd_with_attn_metrics_FINAL.csv \
  --out_dir cpd_corr_outputs
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import zipfile
from itertools import combinations
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import spearmanr


# ----------- Feature list (edit as needed) -----------
FEATURE_COLS_RAW = [
    "interference_score_lexical_wrt_distractors",
    "evidence_overlap_ratio",
    "hallu_score",
    "distractor_density",
    "evid_overlap_ngram",
    "evid_overlap_emb",
    "total_response_tokens",
    "interference_token_hits",
    "att_avg_entropy",
    "attention_entropy",
    "cross_attention_mass_to_gold",
    "distractor_attention_max",
    "hallucination_score",
    "distractor_density_chunk",
    "chunk_hallu_score",
    "evidence_overlap_emb",
    "seq_len",
    "resp_len",
    "gold_len",
    "dist_len",
    "cross_attention_mass_to_gold_raw",
    "gold_found",
    "dist_found",
    "gold_available",
]


def dedupe_preserve_order(xs: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


FEATURE_COLS = dedupe_preserve_order(FEATURE_COLS_RAW)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate CPD metrics and produce correlation visualizations.")
    p.add_argument("--input", required=True, help="Path to input CSV (or .zip containing a CSV).")
    p.add_argument("--out_dir", default="cpd_corr_outputs", help="Output directory.")
    p.add_argument("--domain", default=None, help="Domain/dataset to analyze. If omitted, auto-picks the domain with most people.")
    p.add_argument("--question_id", type=int, default=None, help="question_id/person to analyze (within chosen domain). If omitted, auto-picks person with most configs.")
    p.add_argument("--batch_size", type=int, default=20, help="Number of pairwise scatter plots per PDF batch.")
    p.add_argument("--max_annot", type=int, default=22, help="Annotate heatmap values only if number of features <= max_annot.")
    p.add_argument("--no_zip", action="store_true", help="Disable zipping the output directory.")
    return p.parse_args()


def _read_header_columns_csv(path: str) -> List[str]:
    return pd.read_csv(path, nrows=0).columns.tolist()


def _read_header_columns_zip(zip_path: str, inner_csv: str) -> List[str]:
    with zipfile.ZipFile(zip_path, "r") as z:
        with z.open(inner_csv) as f:
            return pd.read_csv(f, nrows=0).columns.tolist()


def discover_inner_csv(zip_path: str) -> str:
    with zipfile.ZipFile(zip_path, "r") as z:
        csvs = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if not csvs:
            raise ValueError(f"No .csv found inside zip: {zip_path}")
        if len(csvs) == 1:
            return csvs[0]
        # Prefer a CSV with the same stem as the zip if available
        stem = os.path.splitext(os.path.basename(zip_path))[0]
        for c in csvs:
            if os.path.splitext(os.path.basename(c))[0] == stem:
                return c
        # Otherwise pick the first deterministically
        csvs.sort()
        return csvs[0]


def load_input(path: str) -> pd.DataFrame:
    required_meta = ["domain", "question_id", "config", "oom", "error"]

    if path.lower().endswith(".zip"):
        inner_csv = discover_inner_csv(path)
        cols = _read_header_columns_zip(path, inner_csv)
        usecols = [c for c in required_meta + FEATURE_COLS if c in cols]
        with zipfile.ZipFile(path, "r") as z:
            with z.open(inner_csv) as f:
                df = pd.read_csv(f, usecols=usecols, low_memory=False)
        return df

    # Normal CSV
    cols = _read_header_columns_csv(path)
    usecols = [c for c in required_meta + FEATURE_COLS if c in cols]
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    return df


def filter_success(df: pd.DataFrame) -> pd.DataFrame:
    # Robustness: error column may be empty string, "None", etc.
    if "error" in df.columns:
        err = df["error"]
        ok_err = err.isna() | (err.astype(str).str.strip() == "") | (err.astype(str).str.lower().str.strip().isin(["none", "nan"]))
    else:
        ok_err = True

    if "oom" in df.columns:
        # oom may be bool or string; coerce
        oom = df["oom"]
        if oom.dtype == bool:
            ok_oom = (oom == False)
        else:
            ok_oom = ~(oom.astype(str).str.lower().str.strip().isin(["true", "1", "yes"]))
    else:
        ok_oom = True

    out = df[ok_err & ok_oom].copy()

    # Convert common boolean fields to int so averaging makes sense
    for c in ["gold_found", "dist_found", "gold_available", "oom"]:
        if c in out.columns and out[c].dtype == bool:
            out[c] = out[c].astype(int)

    return out


def corr_strength(abs_rho: float) -> str:
    if pd.isna(abs_rho):
        return "NA"
    if abs_rho >= 0.7:
        return "strong"
    if abs_rho >= 0.4:
        return "moderate"
    return "weak"


def pairwise_spearman(df_features: pd.DataFrame) -> pd.DataFrame:
    cols = df_features.columns.tolist()
    rows = []
    for a, b in combinations(cols, 2):
        x = df_features[a]
        y = df_features[b]
        mask = x.notna() & y.notna()
        n = int(mask.sum())
        if n < 3:
            rho, p = np.nan, np.nan
        else:
            rho, p = spearmanr(x[mask], y[mask])
        rows.append(
            dict(
                feature_1=a,
                feature_2=b,
                n=n,
                spearman_rho=rho,
                abs_rho=(abs(rho) if pd.notna(rho) else np.nan),
                p_value=p,
            )
        )
    out = pd.DataFrame(rows).sort_values(["abs_rho", "n"], ascending=[False, False], na_position="last").reset_index(drop=True)
    out["strength"] = out["abs_rho"].apply(corr_strength)
    out["direction"] = np.where(out["spearman_rho"] >= 0, "+", "-")
    return out


def heatmap_quick(corr: pd.DataFrame, title: str, out_png: str, max_annot: int = 22) -> None:
    labels = corr.columns.tolist()
    mat = corr.values.astype(float)

    plt.figure(figsize=(max(8, 0.42 * len(labels)), max(6, 0.42 * len(labels))))
    plt.imshow(mat, vmin=-1, vmax=1, aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.yticks(range(len(labels)), labels)
    plt.title(title)

    if len(labels) <= max_annot:
        for i in range(len(labels)):
            for j in range(len(labels)):
                v = mat[i, j]
                txt = "NA" if np.isnan(v) else f"{v:+.2f}"
                plt.text(j, i, txt, ha="center", va="center", fontsize=7)

    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


def zscore_within_domain(df_avg: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    df_z = df_avg.copy()
    for dom, idxs in df_avg.groupby("domain").groups.items():
        sub = df_avg.loc[idxs, feature_cols]
        mu = sub.mean(numeric_only=True)
        sigma = sub.std(numeric_only=True, ddof=0).replace(0, np.nan)
        df_z.loc[idxs, feature_cols] = ((sub - mu) / sigma).fillna(0.0)
    return df_z


def pick_domain_person(df_cfg: pd.DataFrame, df_avg: pd.DataFrame, domain: Optional[str], question_id: Optional[int]) -> Tuple[str, int]:
    if domain is None:
        # domain with most people (question_id unique)
        domain = df_avg.groupby("domain")["question_id"].nunique().sort_values(ascending=False).index[0]

    if question_id is None:
        cfg_counts = (
            df_cfg[df_cfg["domain"] == domain]
            .groupby("question_id")["config"]
            .nunique()
            .sort_values(ascending=False)
        )
        if cfg_counts.empty:
            raise ValueError(f"No configs found for domain={domain}.")
        question_id = int(cfg_counts.index[0])

    return domain, int(question_id)


def make_scatter_batches(
    sub_cfg: pd.DataFrame,
    nonconst_feats: List[str],
    domain: str,
    question_id: int,
    out_dir: str,
    batch_size: int,
) -> List[str]:
    # Deterministic pair order
    pairs = [(nonconst_feats[i], nonconst_feats[j]) for i in range(len(nonconst_feats)) for j in range(i + 1, len(nonconst_feats))]
    n_pairs = len(pairs)
    if n_pairs == 0:
        return []

    plt.rcParams["figure.autolayout"] = True  # avoid calling tight_layout per plot

    pdf_paths: List[str] = []
    n_batches = int(np.ceil(n_pairs / batch_size))
    for b in range(n_batches):
        start = b * batch_size
        end = min((b + 1) * batch_size, n_pairs)
        pdf_path = os.path.join(out_dir, f"scatter_pairs__{domain}__question_{question_id}__batch_{b+1:02d}.pdf")
        with PdfPages(pdf_path) as pdf:
            for k in range(start, end):
                a, c = pairs[k]
                x = sub_cfg[a]
                y = sub_cfg[c]
                mask = x.notna() & y.notna()
                n = int(mask.sum())
                if n < 3:
                    continue
                rho, p = spearmanr(x[mask], y[mask])
                abs_rho = abs(rho)
                strength = "strong" if abs_rho >= 0.7 else ("moderate" if abs_rho >= 0.4 else "weak")
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.scatter(x[mask], y[mask])
                ax.set_xlabel(a)
                ax.set_ylabel(c)
                ax.set_title(f"{a} vs {c}\nSpearman rho={rho:+.3f} ({strength}), n={n}")
                pdf.savefig(fig)
                plt.close(fig)

        pdf_paths.append(pdf_path)

    return pdf_paths


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = load_input(args.input)
    df = filter_success(df)

    # Determine which of the desired features are present
    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    if not feature_cols:
        raise ValueError("No expected feature columns were found in the input. Check FEATURE_COLS in the script.")
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"[WARN] Missing {len(missing)} expected feature columns (will be ignored). Example missing: {missing[:5]}")

    # ---------------- Aggregations ----------------
    df_cfg = df.groupby(["domain", "question_id", "config"], as_index=False)[feature_cols].mean()
    df_avg = df_cfg.groupby(["domain", "question_id"], as_index=False)[feature_cols].mean()
    df_avg_z = zscore_within_domain(df_avg, feature_cols)

    # Save aggregated tables
    cfg_csv = os.path.join(args.out_dir, "config_level_means_per_domain_question_config.csv")
    avg_csv = os.path.join(args.out_dir, "config_averaged_means_per_domain_question.csv")
    avgz_csv = os.path.join(args.out_dir, "config_averaged_means_per_domain_question__zscore_within_domain.csv")
    df_cfg.to_csv(cfg_csv, index=False)
    df_avg.to_csv(avg_csv, index=False)
    df_avg_z.to_csv(avgz_csv, index=False)

    # ---------------- Choose dataset/person for detailed plots ----------------
    domain_choice, question_choice = pick_domain_person(df_cfg, df_avg, args.domain, args.question_id)

    # ---------------- Per-person correlations across configs ----------------
    sub_cfg = df_cfg[(df_cfg["domain"] == domain_choice) & (df_cfg["question_id"] == question_choice)].copy()
    nonconst_feats_person = [c for c in feature_cols if sub_cfg[c].nunique(dropna=True) > 1]

    sub_X = sub_cfg[nonconst_feats_person]
    corr_person = sub_X.corr(method="spearman")
    pairs_person = pairwise_spearman(sub_X)

    pairs_person_csv = os.path.join(args.out_dir, f"spearman_pairs__{domain_choice}__question_{question_choice}__across_configs.csv")
    pairs_person.to_csv(pairs_person_csv, index=False)

    strong_person = pairs_person[pairs_person["strength"] == "strong"].copy()
    strong_person_csv = os.path.join(args.out_dir, f"strong_pairs__{domain_choice}__question_{question_choice}__across_configs.csv")
    strong_person.to_csv(strong_person_csv, index=False)

    person_heat_png = os.path.join(args.out_dir, f"corr_heatmap__{domain_choice}__question_{question_choice}__across_configs.png")
    heatmap_quick(
        corr_person,
        title=f"Spearman corr across configs ({domain_choice}, question_id={question_choice}, n={len(sub_cfg)})",
        out_png=person_heat_png,
        max_annot=args.max_annot,
    )

    # Pairwise scatter plots (batched)
    scatter_pdfs = make_scatter_batches(
        sub_cfg=sub_cfg,
        nonconst_feats=nonconst_feats_person,
        domain=domain_choice,
        question_id=question_choice,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
    )

    # ---------------- Dataset-level correlations across persons ----------------
    sub_avg = df_avg[df_avg["domain"] == domain_choice].copy()
    nonconst_feats_dom = [c for c in feature_cols if sub_avg[c].nunique(dropna=True) > 1]
    X_dom = sub_avg[nonconst_feats_dom]

    corr_dom = X_dom.corr(method="spearman")
    pairs_dom = pairwise_spearman(X_dom)

    pairs_dom_csv = os.path.join(args.out_dir, f"spearman_pairs__{domain_choice}__across_persons.csv")
    pairs_dom.to_csv(pairs_dom_csv, index=False)

    strong_dom = pairs_dom[pairs_dom["strength"] == "strong"].copy()
    strong_dom_csv = os.path.join(args.out_dir, f"strong_pairs__{domain_choice}__across_persons.csv")
    strong_dom.to_csv(strong_dom_csv, index=False)

    dom_heat_png = os.path.join(args.out_dir, f"corr_heatmap__{domain_choice}__across_persons.png")
    heatmap_quick(
        corr_dom,
        title=f"Spearman corr across persons ({domain_choice}, n_persons={len(sub_avg)})",
        out_png=dom_heat_png,
        max_annot=args.max_annot,
    )

    # ---------------- Strong-pair frequency across people (within chosen domain) ----------------
    people = sorted(df_cfg[df_cfg["domain"] == domain_choice]["question_id"].unique().tolist())

    def pair_key(a: str, b: str) -> str:
        return "||".join(sorted([a, b]))

    all_strong = []
    for qid in people:
        tmp = df_cfg[(df_cfg["domain"] == domain_choice) & (df_cfg["question_id"] == qid)].copy()
        tmp_feats = [c for c in feature_cols if tmp[c].nunique(dropna=True) > 1]
        if len(tmp_feats) < 2:
            continue
        tmp_pairs = pairwise_spearman(tmp[tmp_feats])
        tmp_strong = tmp_pairs[tmp_pairs["strength"] == "strong"].copy()
        tmp_strong["question_id"] = int(qid)
        tmp_strong["pair_key"] = tmp_strong.apply(lambda r: pair_key(r["feature_1"], r["feature_2"]), axis=1)
        all_strong.append(tmp_strong)

    if all_strong:
        strong_all = pd.concat(all_strong, ignore_index=True)
        strong_summary = (
            strong_all.groupby("pair_key", as_index=False)
            .agg(
                feature_1=("feature_1", "first"),
                feature_2=("feature_2", "first"),
                n_people_strong=("question_id", "nunique"),
                avg_abs_rho=("abs_rho", "mean"),
                avg_rho=("spearman_rho", "mean"),
            )
            .sort_values(["n_people_strong", "avg_abs_rho"], ascending=[False, False])
            .reset_index(drop=True)
        )
    else:
        strong_summary = pd.DataFrame(columns=["feature_1", "feature_2", "n_people_strong", "avg_abs_rho", "avg_rho"])

    strong_summary_csv = os.path.join(args.out_dir, f"strong_pair_frequency__{domain_choice}__across_people.csv")
    strong_summary.to_csv(strong_summary_csv, index=False)

    # ---------------- Metadata ----------------
    meta = dict(
        input=os.path.abspath(args.input),
        out_dir=os.path.abspath(args.out_dir),
        domain_choice=domain_choice,
        question_choice=question_choice,
        n_rows_after_filter=int(len(df)),
        n_configs_for_chosen_person=int(len(sub_cfg)),
        used_feature_cols=feature_cols,
        nonconst_feats_person=nonconst_feats_person,
        nonconst_feats_domain=nonconst_feats_dom,
        scatter_pdfs=[os.path.basename(p) for p in scatter_pdfs],
        files_written=[
            os.path.basename(cfg_csv),
            os.path.basename(avg_csv),
            os.path.basename(avgz_csv),
            os.path.basename(pairs_person_csv),
            os.path.basename(strong_person_csv),
            os.path.basename(person_heat_png),
            os.path.basename(pairs_dom_csv),
            os.path.basename(strong_dom_csv),
            os.path.basename(dom_heat_png),
            os.path.basename(strong_summary_csv),
        ] + [os.path.basename(p) for p in scatter_pdfs],
    )

    meta_path = os.path.join(args.out_dir, "analysis_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # ---------------- Zip outputs ----------------
    if not args.no_zip:
        zip_path = os.path.abspath(args.out_dir.rstrip(os.sep) + ".zip")
        if os.path.exists(zip_path):
            os.remove(zip_path)
        shutil.make_archive(args.out_dir.rstrip(os.sep), "zip", args.out_dir)
        print(f"[OK] Zipped outputs -> {zip_path}")

    print("\n[OK] Done.")
    print(f"Chosen domain: {domain_choice}")
    print(f"Chosen question_id: {question_choice}")
    print(f"Outputs in: {os.path.abspath(args.out_dir)}")
    print(f"Wrote metadata: {os.path.abspath(meta_path)}")


if __name__ == "__main__":
    main()
