"""Utility for preparing per-sentence feature datasets for change-point detection."""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from step_4_cpd.data_structuring import structure_data
from step_4_cpd.feature_calculation import calculate_features_from_records

SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?])\s+")

def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Prepare sentence-level datasets for hallucination change-point detection.",
	)
	parser.add_argument(
		"inputs",
		nargs="+",
		help="CSV files containing model outputs, gold references, and scenario metadata.",
	)
	parser.add_argument(
		"--output-dir",
		default="hall_riskindex/prepared",
		help="Directory that will store prepared datasets and structured sequences.",
	)
	parser.add_argument(
		"--similarity-threshold",
		type=float,
		default=0.7,
		help="Cosine similarity threshold below which a sentence is treated as hallucinated.",
	)
	return parser.parse_args()

def _parse_serialised_mapping(value: Any) -> OrderedDict:
	if value is None or (isinstance(value, float) and np.isnan(value)):
		return OrderedDict()
	if isinstance(value, dict):
		return OrderedDict(value)

	text = str(value).strip()
	if not text:
		return OrderedDict()

	try:
		parsed = json.loads(text)
	except json.JSONDecodeError:
		parsed = ast.literal_eval(text)
	if isinstance(parsed, dict):
		return OrderedDict(parsed)
	if isinstance(parsed, list):
		return OrderedDict(parsed)
	return OrderedDict()

def _split_sentences(text: Any) -> List[str]:
	if not isinstance(text, str):
		return []

	sentences = [
		segment.strip()
		for segment in SENTENCE_SPLIT_REGEX.split(text)
		if segment and segment.strip()
	]
	if not sentences and text.strip():
		return [text.strip()]
	return sentences

def _sanitize_token(token: Any) -> str:
	if token is None:
		token = "unknown"
	text = str(token).strip()
	if not text:
		text = "unknown"
	return re.sub(r"[^A-Za-z0-9_.-]", "_", text)


def _make_sequence_id(row: pd.Series, fallback_index: int) -> str:
	question_id = row.get("question_id") or row.get("qa_id") or f"row_{fallback_index}"
	domain = row.get("domain") or "unknown"
	config = row.get("config") or ""

	parts = [question_id, domain, config]
	safe_parts = [_sanitize_token(part) for part in parts if part]
	if not safe_parts:
		safe_parts = [f"row_{fallback_index}"]
	return "__".join(safe_parts)


def _prepare_similarity_index(
	candidate_sentences: List[str], reference_sentences: List[str]
) -> Tuple[Optional[TfidfVectorizer], Optional[np.ndarray]]:
	clean_refs = [ref.strip() for ref in reference_sentences if isinstance(ref, str) and ref.strip()]
	clean_candidates = [
		sentence.strip()
		for sentence in candidate_sentences
		if isinstance(sentence, str) and sentence.strip()
	]

	if not clean_refs or not clean_candidates:
		return None, None

	corpus = clean_candidates + clean_refs
	vectorizer = TfidfVectorizer().fit(corpus)
	reference_matrix = vectorizer.transform(clean_refs)
	return vectorizer, reference_matrix


def build_sentence_records(
	df: pd.DataFrame,
	similarity_threshold: float,
) -> List[Dict[str, Any]]:
	records: List[Dict[str, Any]] = []

	for row_index, row in df.iterrows():
		gold_matches = _parse_serialised_mapping(row.get("chunk_level_gold_matches"))
		if gold_matches:
			sentence_iterable: Iterable[Tuple[str, Any]] = list(gold_matches.items())
		else:
			generated_sentences = _split_sentences(row.get("model_answer", ""))
			sentence_iterable = [(sentence, 0) for sentence in generated_sentences]

		if not sentence_iterable:
			continue

		sequence_id = _make_sequence_id(row, row_index)
		config = row.get("config", "")
		domain = row.get("domain", "")
		question_id = row.get("question_id", row.get("qa_id", ""))

		reference_sentences = _split_sentences(row.get("gold_text", ""))
		if not reference_sentences and row.get("gold_text"):
			reference_sentences = [row.get("gold_text", "")]

		candidate_sentences = [sentence for sentence, _ in sentence_iterable]
		vectorizer, reference_matrix = _prepare_similarity_index(candidate_sentences, reference_sentences)

		for chunk_idx, (sentence, gold_flag) in enumerate(sentence_iterable):
			gold_flag_int = 0
			if isinstance(gold_flag, (int, float, str)):
				try:
					gold_flag_int = int(float(gold_flag))
				except ValueError:
					gold_flag_int = 0

			if vectorizer is None or reference_matrix is None or reference_matrix.shape[0] == 0:
				similarity = 0.0
			else:
				sentence_clean = sentence.strip() if isinstance(sentence, str) else ""
				if sentence_clean:
					sentence_vec = vectorizer.transform([sentence_clean])
					sims = cosine_similarity(sentence_vec, reference_matrix)[0]
					similarity = float(np.clip(sims.max(), 0.0, 1.0)) if sims.size else 0.0
				else:
					similarity = 0.0

			similarity_flag = int(similarity < similarity_threshold)
			lack_of_evidence = int(1 - gold_flag_int)
			hallu_signal = int(similarity_flag and lack_of_evidence)

			record = {
				"sequence_id": sequence_id,
				"question_id": question_id,
				"domain": domain,
				"config": config,
				"chunk_index": chunk_idx,
				"sentence_text": sentence,
				"semantic_similarity": similarity,
				"is_gold_binary": gold_flag_int,
				"similarity_flag": similarity_flag,
				"lack_of_evidence_flag": lack_of_evidence,
				"hallu_signal": hallu_signal,
				"row_index": row.get("row_index", row_index),
				"config_idx": row.get("config_idx", 0),
			}
			records.append(record)

	return records


def save_structured_sequences(feature_df: pd.DataFrame, output_dir: Path) -> None:
	sequence_dir = output_dir / "structured_sequences"
	sequence_dir.mkdir(parents=True, exist_ok=True)

	for sequence_id, group in feature_df.groupby("sequence_id"):
		sequence_records = group.sort_values("chunk_index").to_dict(orient="records")
		structured = structure_data(sequence_records)
		safe_name = _sanitize_token(sequence_id)
		np.savez(sequence_dir / f"{safe_name}.npz", **structured)


def main() -> None:
	args = parse_args()
	input_paths = [Path(path) for path in args.inputs]

	frames: List[pd.DataFrame] = []
	for path in input_paths:
		if not path.exists():
			raise FileNotFoundError(f"Input file not found: {path}")
		frames.append(pd.read_csv(path))

	if not frames:
		raise ValueError("No input files provided for dataset preparation.")

	raw_df = pd.concat(frames, ignore_index=True)
	raw_df["row_index"] = raw_df.index
	if "config" not in raw_df.columns:
		raw_df["config"] = ""
	raw_df["config_idx"] = raw_df["config"].fillna("").astype("category").cat.codes

	sentence_records = build_sentence_records(raw_df, similarity_threshold=args.similarity_threshold)
	if not sentence_records:
		raise ValueError("No sentence-level records could be extracted from the provided inputs.")

	output_dir = Path(args.output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	sentence_df = pd.DataFrame(sentence_records)
	sentence_output = output_dir / "prepared_sentence_records.csv"
	sentence_df.to_csv(sentence_output, index=False)

	feature_rows = calculate_features_from_records(
		sentence_records,
		similarity_threshold=args.similarity_threshold,
	)
	if not feature_rows:
		raise ValueError("Feature calculation returned no rows; check input data integrity.")

	feature_df = pd.DataFrame(feature_rows)
	feature_csv = output_dir / "feature_dataframe.csv"
	feature_parquet = output_dir / "feature_dataframe.parquet"
	feature_df.to_csv(feature_csv, index=False)
	feature_df.to_parquet(feature_parquet, index=False)

	save_structured_sequences(feature_df, output_dir)

	print(f"Prepared sentence records written to {sentence_output}")
	print(f"Feature dataframe written to {feature_csv}")
	print(f"Parquet features written to {feature_parquet}")
	print(f"Structured sequences saved under {output_dir / 'structured_sequences'}")


if __name__ == "__main__":
	main()
