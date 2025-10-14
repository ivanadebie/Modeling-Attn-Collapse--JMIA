 # python -u ./scripts/get_qa_responses_from_longchat_ori.py \
 #  --input-path qa_data/domain5_data_05_stackmath_numerical_8000_converted.jsonl \
 #  --model lmsys/longchat-13b-16k \
 #  --output-path qa_predictions/domain5_data_05_stackmath_numerical_8000_converted.output.jsonl.gz \
 #  --num-gpus 2 \
 #  --max-new-tokens 1000 \
 #  --batch-size 4 \
 #  --max-memory-per-gpu 50 \
 #  --closedbook


python -u ./scripts/get_qa_responses_from_longchat_ori.py \
  --input-path qa_data/domain6_data_06_policy_compliance_8000_converted.jsonl \
  --model lmsys/longchat-13b-16k \
  --output-path qa_predictions/domain6_data_06_policy_compliance_8000_converted.output.jsonl.gz \
  --num-gpus 2 \
  --max-new-tokens 1000 \
  --batch-size 4 \
  --max-memory-per-gpu 50 \
  --closedbook  


python -u ./scripts/get_qa_responses_from_longchat_ori.py \
  --input-path qa_data/domain7_data_07_expert_legal_med_8000_converted.jsonl \
  --model lmsys/longchat-13b-16k \
  --output-path qa_predictions/domain7_data_07_expert_legal_med_8000_converted.jsonl.gz \
  --num-gpus 2 \
  --max-new-tokens 1000 \
  --batch-size 4 \
  --max-memory-per-gpu 50 \
  --closedbook  
