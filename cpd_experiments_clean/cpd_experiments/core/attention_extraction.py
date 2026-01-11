"""
Sentence-level attention extraction for hallucination detection.

This module extracts attention-based features at the sentence level,
which provides unique attention values per sentence (unlike chunk-level
extraction which gives the same values for all sentences in a chunk).
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
import gc

from ..utils import (
    parse_list_column,
    clear_gpu_memory,
    get_gpu_memory_info
)


class AttentionExtractor:
    """
    Extract sentence-level attention metrics from LLM.
    
    Key insight: Using `model_answer_chunk` (individual sentence) instead of
    `sentence_text` (full chunk) gives unique attention per sentence.
    """
    
    def __init__(self, model, tokenizer, max_tokens: int = 4096, device: str = "cuda"):
        """
        Initialize extractor.
        
        Args:
            model: HuggingFace model with attention outputs
            tokenizer: Corresponding tokenizer
            max_tokens: Maximum sequence length
            device: Device to use
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.device = device
        
    def _tokenize_with_offsets(self, text: str) -> Tuple[List[int], List[Tuple[int, int]]]:
        """
        Tokenize text and return token IDs with character offsets.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (token_ids, offset_mapping)
        """
        enc = self.tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=False
        )
        return enc["input_ids"], enc["offset_mapping"]
    
    def _char_span_to_token_span(self,
                                  offset_mapping: List[Tuple[int, int]],
                                  char_start: int,
                                  char_end: int) -> Tuple[int, int]:
        """
        Convert character span to token span.
        
        Args:
            offset_mapping: List of (start, end) character offsets per token
            char_start: Start character index
            char_end: End character index
            
        Returns:
            Tuple of (token_start, token_end) indices
        """
        token_start = None
        token_end = None
        
        for i, (s, e) in enumerate(offset_mapping):
            if s == e == 0:
                continue
            if token_start is None and e > char_start:
                token_start = i
            if s < char_end:
                token_end = i + 1
                
        if token_start is None:
            token_start = 0
        if token_end is None:
            token_end = len(offset_mapping)
            
        return token_start, token_end
    
    def _build_input_sequence(self,
                               gold_text: str,
                               distractor_text: str,
                               response_text: str) -> Dict[str, Any]:
        """
        Build input sequence with order: [GOLD] -> [DIST] -> [RESP]
        
        This order is critical because causal attention means token i
        can only attend to tokens j where j <= i. Response tokens must
        come last to attend to gold and distractor.
        
        Args:
            gold_text: Gold/evidence text
            distractor_text: Distractor text
            response_text: Model response (sentence)
            
        Returns:
            Dict with input_ids, attention_mask, and span indices
        """
        # Build combined text
        combined = f"{gold_text} {distractor_text} {response_text}"
        
        # Tokenize with offsets
        input_ids, offsets = self._tokenize_with_offsets(combined)
        
        # Find spans
        gold_end = len(gold_text)
        dist_start = gold_end + 1
        dist_end = dist_start + len(distractor_text)
        resp_start = dist_end + 1
        resp_end = len(combined)
        
        gold_span = self._char_span_to_token_span(offsets, 0, gold_end)
        dist_span = self._char_span_to_token_span(offsets, dist_start, dist_end)
        resp_span = self._char_span_to_token_span(offsets, resp_start, resp_end)
        
        # Truncate if needed
        if len(input_ids) > self.max_tokens:
            input_ids = input_ids[:self.max_tokens]
            # Adjust spans
            gold_span = (gold_span[0], min(gold_span[1], self.max_tokens))
            dist_span = (dist_span[0], min(dist_span[1], self.max_tokens))
            resp_span = (resp_span[0], min(resp_span[1], self.max_tokens))
        
        return {
            "input_ids": torch.tensor([input_ids], device=self.device),
            "attention_mask": torch.ones(1, len(input_ids), device=self.device),
            "gold_span": gold_span,
            "dist_span": dist_span,
            "resp_span": resp_span,
            "seq_len": len(input_ids)
        }
    
    def extract_metrics(self,
                        gold_text: str,
                        distractor_text: str,
                        response_text: str) -> Dict[str, float]:
        """
        Extract attention metrics for a single sentence.
        
        Metrics:
        - sent_avg_entropy: Average attention entropy over response tokens
        - sent_cross_attention_to_gold: Total attention mass from response to gold
        - sent_distractor_attention_max: Max attention from any response token to distractors
        
        Args:
            gold_text: Gold/evidence text
            distractor_text: Distractor text
            response_text: Model response sentence
            
        Returns:
            Dict with attention metrics
        """
        result = {
            "sent_avg_entropy": np.nan,
            "sent_cross_attention_to_gold": np.nan,
            "sent_distractor_attention_max": np.nan,
            "sent_seq_len": 0,
            "sent_resp_len": 0,
            "sent_gold_len": 0,
            "sent_dist_len": 0,
            "sent_gold_found": False,
            "sent_dist_found": False,
            "sent_oom": False,
            "sent_error": None
        }
        
        # Handle empty inputs
        if not gold_text or not distractor_text or not response_text:
            return result
        
        try:
            # Build input
            inputs = self._build_input_sequence(gold_text, distractor_text, response_text)
            
            result["sent_seq_len"] = inputs["seq_len"]
            result["sent_resp_len"] = inputs["resp_span"][1] - inputs["resp_span"][0]
            result["sent_gold_len"] = inputs["gold_span"][1] - inputs["gold_span"][0]
            result["sent_dist_len"] = inputs["dist_span"][1] - inputs["dist_span"][0]
            result["sent_gold_found"] = result["sent_gold_len"] > 0
            result["sent_dist_found"] = result["sent_dist_len"] > 0
            
            # Skip if response span is empty
            if result["sent_resp_len"] == 0:
                return result
            
            # Forward pass with attention
            with torch.no_grad():
                outputs = self.model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    output_attentions=True
                )
            
            # Get last layer attention: [batch, heads, seq, seq]
            attn = outputs.attentions[-1][0]  # [heads, seq, seq]
            
            # Average over heads
            attn_avg = attn.mean(dim=0)  # [seq, seq]
            
            # Extract response token attention
            resp_start, resp_end = inputs["resp_span"]
            gold_start, gold_end = inputs["gold_span"]
            dist_start, dist_end = inputs["dist_span"]
            
            resp_attn = attn_avg[resp_start:resp_end, :]  # [resp_len, seq]
            
            # 1. Average entropy over response tokens
            # Entropy = -sum(p * log(p))
            eps = 1e-10
            entropy = -(resp_attn * torch.log(resp_attn + eps)).sum(dim=-1)
            result["sent_avg_entropy"] = float(entropy.mean().cpu())
            
            # 2. Cross-attention to gold
            if gold_end > gold_start:
                gold_attn = resp_attn[:, gold_start:gold_end].sum().cpu().item()
                result["sent_cross_attention_to_gold"] = float(gold_attn)
            
            # 3. Max attention to distractors
            if dist_end > dist_start:
                dist_attn = resp_attn[:, dist_start:dist_end]
                result["sent_distractor_attention_max"] = float(dist_attn.max().cpu())
            
            # Clean up
            del outputs, attn, attn_avg, resp_attn
            
        except torch.cuda.OutOfMemoryError:
            result["sent_oom"] = True
            clear_gpu_memory()
            
        except Exception as e:
            result["sent_error"] = str(e)
        
        return result
    
    def extract_batch(self,
                      df: pd.DataFrame,
                      gold_col: str = "gold_text_chunk",
                      dist_col: str = "distractor_text",
                      resp_col: str = "model_answer_chunk",
                      checkpoint_every: int = 500,
                      checkpoint_path: Optional[str] = None) -> pd.DataFrame:
        """
        Extract attention metrics for a batch of rows.
        
        Args:
            df: Input DataFrame
            gold_col: Column with gold text
            dist_col: Column with distractor text
            resp_col: Column with response text (sentence-level)
            checkpoint_every: Save checkpoint every N rows
            checkpoint_path: Path for checkpoint file
            
        Returns:
            DataFrame with extracted metrics
        """
        results = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting attention"):
            # Aggregate gold text if needed
            gold_text = row.get(gold_col, "")
            if isinstance(gold_text, list):
                gold_text = " ".join(str(g) for g in gold_text if g)
            gold_text = str(gold_text) if gold_text else ""
            
            # Get distractor and response
            dist_text = str(row.get(dist_col, "")) if row.get(dist_col) else ""
            resp_text = str(row.get(resp_col, "")) if row.get(resp_col) else ""
            
            # Extract metrics
            metrics = self.extract_metrics(gold_text, dist_text, resp_text)
            metrics["original_index"] = idx
            
            # Copy relevant columns from original row
            for col in ["sequence_id", "chunk_index", "hallu_label", "hallu_score"]:
                if col in row:
                    metrics[col] = row[col]
            
            results.append(metrics)
            
            # Checkpoint
            if checkpoint_path and len(results) % checkpoint_every == 0:
                pd.DataFrame(results).to_csv(checkpoint_path, index=False)
                clear_gpu_memory()
        
        return pd.DataFrame(results)


def concat_gold_chunks(df: pd.DataFrame,
                       seq_col: str = "sequence_id",
                       chunk_col: str = "chunk_index",
                       gold_col: str = "gold_text_chunk") -> pd.DataFrame:
    """
    Concatenate gold text chunks within each sequence.
    
    Args:
        df: Input DataFrame
        seq_col: Sequence ID column
        chunk_col: Chunk index column
        gold_col: Gold text column
        
    Returns:
        DataFrame with aggregated gold text per (sequence, chunk)
    """
    def agg_gold(group):
        texts = []
        for val in group[gold_col]:
            parsed = parse_list_column(val)
            texts.extend(str(t) for t in parsed if t)
        return " ".join(texts)
    
    df_sorted = df.sort_values([seq_col, chunk_col])
    gold_agg = df_sorted.groupby(seq_col).apply(agg_gold).reset_index()
    gold_agg.columns = [seq_col, "gold_text_aggregated"]
    
    return df.merge(gold_agg, on=seq_col, how="left")
