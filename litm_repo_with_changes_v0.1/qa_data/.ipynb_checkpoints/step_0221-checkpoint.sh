# python -u ./scripts/get_qa_responses_from_longchat_ori.py \
#   --input-path qa_data/domain1_data_nq_closed_book_8000_converted_no_diatractor.jsonl.gz \
#   --model lmsys/longchat-13b-16k \
#   --output-path qa_predictions/domain1_data_nq_closed_book_8000_converted_no_diatractor.output.jsonl.gz \
#   --num-gpus 1 \
#   --max-new-tokens 1000 \
#   --batch-size 1 \
#   --max-memory-per-gpu 80 \
#   --closedbook 


python -u ./scripts/get_qa_responses_from_longchat_ori.py \
  --input-path qa_data/domain2_data_hotpot_citation_8000_converted_no_diatractor.jsonl.gz \
  --model lmsys/longchat-13b-16k \
  --output-path qa_predictions/domain2_data_hotpot_citation_8000_converted_no_diatractor.output.jsonl.gz \
  --num-gpus 1 \
  --max-new-tokens 1000 \
  --batch-size 1 \
  --max-memory-per-gpu 80 \
  --closedbook 

python -u ./scripts/get_qa_responses_from_longchat_ori.py \
  --input-path qa_data/domain3_data_novelhop_multihop_8000_converted_no_diatractor.jsonl.gz \
  --model lmsys/longchat-13b-16k \
  --output-path qa_predictions/domain3_data_novelhop_multihop_8000_converted_no_diatractor.output.jsonl.gz \
  --num-gpus 1 \
  --max-new-tokens 1000 \
  --batch-size 1 \
  --max-memory-per-gpu 80 \
  --closedbook 

python -u ./scripts/get_qa_responses_from_longchat_ori.py \
  --input-path qa_data/domain4_data_temporal_recency_8000_converted_no_diatractor.jsonl.gz \
  --model lmsys/longchat-13b-16k \
  --output-path qa_predictions/domain4_data_temporal_recency_8000_converted_no_diatractor.output.jsonl.gz \
  --num-gpus 1 \
  --max-new-tokens 1000 \
  --batch-size 1 \
  --max-memory-per-gpu 80 \
  --closedbook 

python -u ./scripts/get_qa_responses_from_longchat_ori.py \
  --input-path qa_data/domain5_data_05_stackmath_numerical_8000_converted_no_diatractor.jsonl.gz \
  --model lmsys/longchat-13b-16k \
  --output-path qa_predictions/domain5_data_05_stackmath_numerical_8000_converted_no_diatractor.output.jsonl.gz \
  --num-gpus 1 \
  --max-new-tokens 1000 \
  --batch-size 1 \
  --max-memory-per-gpu 80 \
  --closedbook 

python -u ./scripts/get_qa_responses_from_longchat_ori.py \
  --input-path qa_data/domain6_data_06_policy_compliance_8000_converted_no_diatractor.jsonl.gz \
  --model lmsys/longchat-13b-16k \
  --output-path qa_predictions/domain6_data_06_policy_compliance_8000_converted_no_diatractor.output.jsonl.gz \
  --num-gpus 1 \
  --max-new-tokens 1000 \
  --batch-size 1 \
  --max-memory-per-gpu 80 \
  --closedbook 

python -u ./scripts/get_qa_responses_from_longchat_ori.py \
  --input-path qa_data/domain7_data_07_expert_legal_med_8000_converted_no_diatractor.jsonl.gz \
  --model lmsys/longchat-13b-16k \
  --output-path qa_predictions/domain7_data_07_expert_legal_med_8000_converted_no_diatractor.output.jsonl.gz \
  --num-gpus 1 \
  --max-new-tokens 1000 \
  --batch-size 1 \
  --max-memory-per-gpu 80 \
  --closedbook 
