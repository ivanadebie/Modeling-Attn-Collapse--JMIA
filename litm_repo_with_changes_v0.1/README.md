# Example
python -u ./scripts/get_qa_responses_from_longchat.py \
  --input-path qa_data/true_synthetic_qa.jsonl.gz \
  --model lmsys/longchat-13b-16k \
  --output-path qa_predictions/output.jsonl.gz \
  --num-gpus 4 \
  --max-new-tokens 200 \
  --batch-size 8 \
  --max-memory-per-gpu 26 \
  --closedbook


## Run this before pip install -e.

```bash
 apt-get update
 apt-get install cmake zlib1g-dev
```

## More dependencies
