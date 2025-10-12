 python -u ./scripts/get_qa_responses_from_longchat_ori.py \
  --input-path qa_data/domain1_data_nq_closed_book_8000_converted.jsonl \
  --model lmsys/longchat-13b-16k \
  --output-path qa_predictions/domain1_data_nq_closed_book_8000_converted.output.jsonl.gz \
  --num-gpus 2 \
  --max-new-tokens 1000 \
  --batch-size 4 \
  --max-memory-per-gpu 50 \
  --closedbook


python -u ./scripts/get_qa_responses_from_longchat_ori.py \
  --input-path qa_data/domain2_data_hotpot_citation_8000_converted.jsonl \
  --model lmsys/longchat-13b-16k \
  --output-path qa_predictions/domain2_data_hotpot_citation_8000_converted.output.jsonl.gz \
  --num-gpus 2 \
  --max-new-tokens 1000 \
  --batch-size 4 \
  --max-memory-per-gpu 50 \
  --closedbook  


python -u ./scripts/get_qa_responses_from_longchat_ori.py \
  --input-path qa_data/domain3_data_novelhop_multihop_8000_converted.jsonl \
  --model lmsys/longchat-13b-16k \
  --output-path qa_predictions/domain3_data_novelhop_multihop_8000_converted.output.jsonl.gz \
  --num-gpus 2 \
  --max-new-tokens 1000 \
  --batch-size 4 \
  --max-memory-per-gpu 50 \
  --closedbook  

python -u ./scripts/get_qa_responses_from_longchat_ori.py \
  --input-path qa_data/domain4_data_temporal_recency_8000_converted.jsonl \
  --model lmsys/longchat-13b-16k \
  --output-path qa_predictions/domain4_data_temporal_recency_8000_converted.output.jsonl.gz \
  --num-gpus 2 \
  --max-new-tokens 1000 \
  --batch-size 4 \
  --max-memory-per-gpu 50 \
  --closedbook  
