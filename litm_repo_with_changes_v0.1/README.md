# Example
python -u ./scripts/get_qa_responses_from_longchat.py \
  --input-path qa_data/synthetic_qa.jsonl.gz \
  --model lmsys/longchat-13b-16k \
  --output-path qa_predictions/synthetic_qa_output.jsonl.gz \
  --num-gpus 4 \
  --max-new-tokens 200 \
  --batch-size 8 \
  --max-memory-per-gpu 63 \
  --closedbook
