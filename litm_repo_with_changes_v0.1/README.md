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

 pip install -e .
```

## More dependencies
## memory needs at least 200G for storage and container
#1001 Fixed nump > 2 issue

pip uninstall -y numpy
pip install "numpy<2"  # installs 1.26.x
python -c "import numpy, torch; print('NumPy:', numpy.__version__, 'Torch:', torch.__version__)"
## 0930
python -u ./scripts/get_qa_responses_from_longchat.py \
  --input-path qa_data/domain1_data_1questions_8000_final_copy.jsonl.gz \
  --model lmsys/longchat-13b-16k \
  --output-path qa_predictions/output_3questions_8000.jsonl.gz \
  --num-gpus 1 \
  --max-new-tokens 200 \
  --batch-size 8 \
  --max-memory-per-gpu 26 \
  --closedbook

python -u ./scripts/get_qa_responses_from_longchat.py \
  --input-path qa_data/domain1_data_3questions_8000_final.jsonl \
  --model lmsys/longchat-13b-16k \
  --output-path qa_predictions/output_3questions_8000.jsonl.gz \
  --num-gpus 1 \
  --max-new-tokens 200 \
  --batch-size 8 \
  --max-memory-per-gpu 26 \
  --closedbook

  python -u ./scripts/get_qa_responses_from_longchat.py \
  --input-path qa_data/true_synthetic_qa.jsonl.gz \
  --model lmsys/longchat-13b-16k \
  --output-path qa_predictions/temp_output.jsonl.gz \
  --num-gpus 4 \
  --max-new-tokens 200 \
  --batch-size 8 \
  --max-memory-per-gpu 26 \
  --closedbook


 ##1001
 python -u ./scripts/get_qa_responses_from_longchat.py \
  --input-path qa_data/true_synthetic_qa.jsonl.gz \
  --model lmsys/longchat-13b-16k \
  --output-path qa_predictions/output.jsonl.gz \
  --num-gpus 2 \
  --max-new-tokens 200 \
  --batch-size 8 \
  --max-memory-per-gpu 26 \
  --closedbook


python -u ./scripts/evaluate_qa_responses.py \
--input-path qa_predictions/output.jsonl.gz \
--output-path qa_predictions/output-nq-open-oracle-longchat-13b-16k-closedbook-predictions-scored.jsonl.gz


 python -u ./scripts/get_qa_responses_from_longchat.py \
  --input-path qa_data/nq-open-oracle.jsonl.gz \
  --model lmsys/longchat-13b-16k \
  --output-path qa_predictions/output.nq-open-oracle.jsonl.gz \
  --num-gpus 2 \
  --max-new-tokens 200 \
  --batch-size 8 \
  --max-memory-per-gpu 26 \
  --closedbook

nq-open-oracle.jsonl.gz

python -u ./scripts/evaluate_qa_responses.py \
    --input-path qa_predictions/output.nq-open-oracle.jsonl.gz \
    --output-path qa_predictions/nq-open-oracle-longchat-13b-16k-closedbook-predictions-scored.jsonl.gz


python -u ./scripts/get_qa_responses_from_longchat.py \
    --input-path qa_data/nq-open-oracle.jsonl.gz \
    --num-gpus 1 \
    --max-new-tokens 100 \
    --batch-size 8 \
    --max-memory-per-gpu 40 \
    --num-gpus 1 \
    --closedbook \
    --model lmsys/longchat-13b-16k \
    --output-path qa_predictions/nq-open-oracle-longchat-13b-16k-closedbook-predictions.jsonl.gz


#1004 

python -u ./scripts/get_qa_responses_from_longchat.py \
  --input-path qa_data/domain1_data_nq_closed_book_8000.jsonl \
  --model lmsys/longchat-13b-16k \
  --output-path qa_predictions/domain1_data_nq_closed_book_8000.jsonl.output.gz \
  --num-gpus 2 \
  --max-new-tokens 200 \
  --batch-size 8 \
  --max-memory-per-gpu 26 \
  --closedbook


#customized
python -u ./scripts/get_qa_responses_from_longchat.py \
  --input-path qa_data/domain1_data_nq_closed_book_8000_jl.jsonl \
  --model lmsys/longchat-13b-16k \
  --output-path qa_predictions/domain1_data_nq_closed_book_8000_jl.jsonl.output.gz \
  --num-gpus 1 \
  --max-new-tokens 20 \
  --batch-size 8 \
  --max-memory-per-gpu 10 \
  --closedbook
  
#1006


python -u ./scripts/get_qa_responses_from_longchat.py \
  --input-path qa_data/domain1_data_nq_closed_book_8000_converted.jsonl \
  --model lmsys/longchat-13b-16k \
  --output-path qa_predictions/domain1_data_nq_closed_book_8000_converted.output.gz \
  --num-gpus 1 \
  --max-new-tokens 20 \
  --batch-size 8 \
  --max-memory-per-gpu 10 \
  --closedbook

#1008

python -u ./scripts/get_qa_responses_from_longchat.py \
  --input-path qa_data/domain1_data_nq_closed_book_8000_converted.jsonl \
  --model lmsys/longchat-13b-16k \
  --output-path qa_predictions/domain1_data_nq_closed_book_8000_converted.output.jsonl.gz \
  --num-gpus 2 \
  --max-new-tokens 200 \
  --batch-size 8 \
  --max-memory-per-gpu 26 \
  --closedbook


python -u ./scripts/get_qa_responses_from_longchat.py \
  --input-path qa_data/domain2_data_hotpot_citation_8000_converted.jsonl \
  --model lmsys/longchat-13b-16k \
  --output-path qa_predictions/domain2_data_hotpot_citation_8000_converted.output.jsonl.gz \
  --num-gpus 2 \
  --max-new-tokens 200 \
  --batch-size 8 \
  --max-memory-per-gpu 26 \
  --closedbook  

python -u ./scripts/get_qa_responses_from_longchat.py \
  --input-path qa_data/domain3_data_novelhop_multihop_8000_converted.jsonl \
  --model lmsys/longchat-13b-16k \
  --output-path qa_predictions/domain3_data_novelhop_multihop_8000_converted.output.jsonl.gz \
  --num-gpus 2 \
  --max-new-tokens 200 \
  --batch-size 8 \
  --max-memory-per-gpu 26 \
  --closedbook  


python -u ./scripts/get_qa_responses_from_longchat.py \
  --input-path qa_data/domain4_data_temporal_recency_8000_converted.jsonl \
  --model lmsys/longchat-13b-16k \
  --output-path qa_predictions/domain4_data_temporal_recency_8000_converted.output.jsonl.gz \
  --num-gpus 2 \
  --max-new-tokens 200 \
  --batch-size 8 \
  --max-memory-per-gpu 26 \
  --closedbook  