#!/usr/bin/env python
import argparse
import ast
import math

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn.utils.rnn import pad_sequence


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute attention-based metrics for CPD dataset."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input CPD CSV (e.g., prepared_dataset_cpd_with_attn_metrics_windows512.csv)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output CSV with attention metrics merged in.",
    )
    parser.add_argument(
        "--model_name",
        default="lmsys/longchat-13b-16k",
        help="HF model name or path (default: lmsys/longchat-13b-16k)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for forward passes (default: 8).",
    )
    parser.add_argument(
        "--max_prefix_tokens",
        type=int,
        default=128,
        help="Max tokens for prefix (gold/distractor) segment.",
    )
    parser.add_argument(
        "--max_chunk_tokens",
        type=int,
        default=64,
        help="Max tokens for chunk (model_answer_chunk) segment.",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=None,
        help="Optional limit on number of rows to process (for debugging).",
    )
    return parser.parse_args()


def get_distractor_text(raw):
    """
    Turn the 'distractors' field into a single text string.
    Handles cases like "['doc1', 'doc2', ...]" (list in string form).
    """
    if pd.isna(raw):
        return ""
    txt = str(raw)
    try:
        val = ast.literal_eval(txt)
        if isinstance(val, list):
            return " ".join(str(x) for x in val)
        else:
            return str(val)
    except Exception:
        # If it's not a valid Python literal, just return as string
        return txt


def make_example(
    row_idx,
    prefix_text,
    chunk_text,
    kind,
    tokenizer,
    max_prefix_tokens,
    max_chunk_tokens,
    bos_id,
    sep_id,
):
    """
    Build a single attention example of the form:

        [BOS] prefix_tokens [SEP] chunk_tokens

    where:
      - prefix_tokens = tokens of gold text OR distractor text
      - chunk_tokens  = tokens of model_answer_chunk

    Returns a dict containing:
      - row_idx: original dataframe row index
      - kind: "gold" or "distr"
      - input_ids: full token sequence (Python list of ints)
      - pref_s, pref_e: start/end token indices of prefix segment
      - ch_s, ch_e:     start/end token indices of chunk segment
    """
    prefix_text = prefix_text or ""
    chunk_text = chunk_text or ""

    prefix_ids = tokenizer.encode(
        prefix_text,
        add_special_tokens=False,
    )[:max_prefix_tokens]

    chunk_ids = tokenizer.encode(
        chunk_text,
        add_special_tokens=False,
    )[:max_chunk_tokens]

    if len(chunk_ids) == 0:
        # No chunk tokens → nothing meaningful to compute
        return None

    input_ids = []
    if bos_id is not None:
        input_ids.append(bos_id)

    pref_s = len(input_ids)
    input_ids.extend(prefix_ids)
    pref_e = len(input_ids)

    input_ids.append(sep_id)

    ch_s = len(input_ids)
    input_ids.extend(chunk_ids)
    ch_e = len(input_ids)

    return {
        "row_idx": row_idx,
        "kind": kind,  # "gold" or "distr"
        "input_ids": input_ids,
        "pref_s": pref_s,
        "pref_e": pref_e,
        "ch_s": ch_s,
        "ch_e": ch_e,
    }


def run_batch(ex_batch, model, device, pad_id):
    """
    Run a batch of examples through the model and compute attention metrics.

    IMPORTANT COMMENT (for your mentor):

        For each example `ex`:

           - pref_s, pref_e:
                token index range for the *prefix* segment
                (i.e., the gold text OR the distractor text).

           - ch_s, ch_e:
                token index range for the *chunk* segment
                (i.e., the model_answer_chunk).

        We then look at the attention FROM chunk tokens
        TO all tokens, and summarize:

           - att_entropy_avg:
                average attention entropy over all heads and
                all query tokens within the chunk segment.

           - mass_avg, mass_max:
                average and maximum attention mass from chunk
                tokens to the prefix segment.
                (If prefix is gold, this is "mass to gold";
                 if prefix is distractors, this is
                 "mass to distractors".)
    """
    from torch.nn.utils.rnn import pad_sequence

    # 1) Convert variable-length sequences to a padded batch
    tensors = [torch.tensor(e["input_ids"], dtype=torch.long) for e in ex_batch]
    padded = pad_sequence(tensors, batch_first=True, padding_value=pad_id)
    attention_mask = (padded != pad_id).long()

    padded = padded.to(device)
    attention_mask = attention_mask.to(device)

    with torch.no_grad():
        out = model(
            padded,
            attention_mask=attention_mask,
            output_attentions=True,
        )

    # out.attentions: tuple(num_layers) of [B, H, Q, K]
    last_attn = out.attentions[-1]  # [B, H, Q, K]

    batch_results = []
    for b, ex in enumerate(ex_batch):
        # Attention tensor for this example: [H, seq_len, seq_len]
        attn = last_attn[b]

        pref_s, pref_e = ex["pref_s"], ex["pref_e"]
        ch_s, ch_e = ex["ch_s"], ex["ch_e"]

        seq_len = attn.size(-1)
        # Restrict to queries = chunk tokens
        chunk_attn = attn[:, ch_s:ch_e, :]  # [H, chunk_len, seq_len]

        # --- Average attention entropy over heads + chunk tokens ---
        # chunk_attn is already the softmax over K dimension (attention probs).
        probs = chunk_attn
        entropy = -(probs * (probs + 1e-12).log()).sum(dim=-1)  # [H, chunk_len]
        att_entropy_avg = entropy.mean().item()

        # --- Mass / max to prefix region (gold or distractors) ---
        mask_pref = torch.zeros(seq_len, device=device)
        if pref_e > pref_s:
            mask_pref[pref_s:pref_e] = 1.0

        # Broadcast mask over heads and queries
        mass = (chunk_attn * mask_pref.view(1, 1, -1)).sum(dim=-1)  # [H, chunk_len]

        mass_avg = mass.mean().item()
        mass_max = mass.max().item() if pref_e > pref_s else 0.0

        batch_results.append(
            {
                "row_idx": ex["row_idx"],
                "kind": ex["kind"],  # gold / distr
                "att_entropy_avg": att_entropy_avg,
                "mass_avg": mass_avg,
                "mass_max": mass_max,
            }
        )

    return batch_results


def main():
    args = parse_args()

    print(f"Loading CPD dataset from: {args.input}")
    df = pd.read_csv(args.input)
    df = df.reset_index(drop=True)
    print(f"Rows: {len(df)}, Columns: {len(df.columns)}")

    if args.max_rows is not None:
        df = df.head(args.max_rows).copy()
        print(f"DEBUG: limiting to first {len(df)} rows via --max_rows.")

    # -----------------------------
    #  Model + tokenizer setup
    # -----------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading tokenizer & model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    model.eval()

    # Prepare special token ids
    bos_id = tokenizer.bos_token_id
    sep_id = tokenizer.eos_token_id or tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        # For LLaMA-style models, use eos as pad
        tokenizer.pad_token = tokenizer.eos_token
        pad_id = tokenizer.pad_token_id

    # -----------------------------
    # Build attention examples
    # -----------------------------
    examples = []

    df_iter = df.iterrows()
    print("Preparing examples...")
    for idx, row in tqdm(df_iter, total=len(df), desc="Preparing examples"):
        chunk_text = str(row.get("model_answer_chunk", "") or "").strip()
        if not chunk_text:
            continue

        # GOLD side: prefer gold_text_chunk; fallback to gold_text if present
        gold_text = str(row.get("gold_text_chunk", "") or "").strip()
        if (not gold_text) and ("gold_text" in row.index):
            gold_text = str(row.get("gold_text", "") or "").strip()

        if gold_text:
            ex_gold = make_example(
                row_idx=idx,
                prefix_text=gold_text,
                chunk_text=chunk_text,
                kind="gold",
                tokenizer=tokenizer,
                max_prefix_tokens=args.max_prefix_tokens,
                max_chunk_tokens=args.max_chunk_tokens,
                bos_id=bos_id,
                sep_id=sep_id,
            )
            if ex_gold is not None:
                examples.append(ex_gold)

        # DISTRACTOR side
        distr_text = get_distractor_text(row.get("distractors", ""))
        distr_text = distr_text.strip()
        if distr_text:
            ex_distr = make_example(
                row_idx=idx,
                prefix_text=distr_text,
                chunk_text=chunk_text,
                kind="distr",
                tokenizer=tokenizer,
                max_prefix_tokens=args.max_prefix_tokens,
                max_chunk_tokens=args.max_chunk_tokens,
                bos_id=bos_id,
                sep_id=sep_id,
            )
            if ex_distr is not None:
                examples.append(ex_distr)

    print("Total attention examples:", len(examples))
    if len(examples) == 0:
        print("No examples to process. Exiting.")
        return

    # -----------------------------
    # Run model in batches
    # -----------------------------
    results = []
    print("Running model in batches...")
    for i in tqdm(range(0, len(examples), args.batch_size), desc="Running model in batches"):
        batch = examples[i : i + args.batch_size]
        batch_res = run_batch(
            batch,
            model=model,
            device=device,
            pad_id=pad_id,
        )
        results.extend(batch_res)

    print("Total results:", len(results))

    # -----------------------------
    # Aggregate results per row_idx
    # -----------------------------
    results_df = pd.DataFrame(results)
    print("results_df shape:", results_df.shape)

    # Split gold and distractor metrics
    gold_df = results_df[results_df["kind"] == "gold"].copy()
    distr_df = results_df[results_df["kind"] == "distr"].copy()

    print("gold_df shape:", gold_df.shape)
    print("distr_df shape:", distr_df.shape)

    # Use row_idx as index for easier joining
    gold_df = gold_df.set_index("row_idx")
    distr_df = distr_df.set_index("row_idx")

    # Initialize new columns with NaN so we keep ALL rows
    df["att_avg_entropy"] = pd.NA
    df["cross_attention_mass_to_gold"] = pd.NA
    df["distractor_attention_max"] = pd.NA

    # Fill from gold side
    common_gold_idx = gold_df.index.intersection(df.index)
    df.loc[common_gold_idx, "att_avg_entropy"] = gold_df.loc[
        common_gold_idx, "att_entropy_avg"
    ]
    df.loc[common_gold_idx, "cross_attention_mass_to_gold"] = gold_df.loc[
        common_gold_idx, "mass_avg"
    ]

    # Fill from distractor side
    common_distr_idx = distr_df.index.intersection(df.index)
    df.loc[common_distr_idx, "distractor_attention_max"] = distr_df.loc[
        common_distr_idx, "mass_max"
    ]

    # Optional: cast numeric columns to float
    df["att_avg_entropy"] = df["att_avg_entropy"].astype("float64")
    df["cross_attention_mass_to_gold"] = df["cross_attention_mass_to_gold"].astype(
        "float64"
    )
    df["distractor_attention_max"] = df["distractor_attention_max"].astype("float64")

    print("\nNon-null counts after merge:")
    print(df[["att_avg_entropy", "cross_attention_mass_to_gold", "distractor_attention_max"]].notna().sum())

    # -----------------------------
    # Save to CSV
    # -----------------------------
    print(f"\nSaving output to: {args.output}")
    df.to_csv(args.output, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
