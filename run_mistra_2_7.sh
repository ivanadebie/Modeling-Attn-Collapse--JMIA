
# ./litm_repo_with_changes_v0.1/qa_data/domain1_data_nq_closed_book_8000_converted.jsonl
# ./litm_repo_with_changes_v0.1/qa_data/domain2_data_hotpot_citation_8000_converted.jsonl
# ./litm_repo_with_changes_v0.1/qa_data/domain3_data_novelhop_multihop_8000_converted.jsonl
# ./litm_repo_with_changes_v0.1/qa_data/domain4_data_temporal_recency_8000_converted.jsonl
# ./litm_repo_with_changes_v0.1/qa_data/domain5_data_05_stackmath_numerical_8000_converted.jsonl
# ./litm_repo_with_changes_v0.1/qa_data/domain6_data_06_policy_compliance_8000_converted.jsonl
# ./litm_repo_with_changes_v0.1/qa_data/domain7_data_07_expert_legal_med_8000_converted.jsonl



#2
python ./mistral/mistral_process.py \
    --input-path ./litm_repo_with_changes_v0.1/qa_data/domain2_data_hotpot_citation_8000_converted.jsonl \
    --output-path ./mistral/output/domain2_data_hotpot_citation_8000_converted.mistral.jsonl \
    --max-new-tokens 1000


#3
python ./mistral/mistral_process.py \
    --input-path ./litm_repo_with_changes_v0.1/qa_data/domain3_data_novelhop_multihop_8000_converted.jsonl \
    --output-path ./mistral/output/domain3_data_novelhop_multihop_8000_converted.mistral.jsonl \
    --max-new-tokens 1000

#4
python ./mistral/mistral_process.py \
    --input-path ./litm_repo_with_changes_v0.1/qa_data/domain4_data_temporal_recency_8000_converted.jsonl \
    --output-path ./mistral/output/domain4_data_temporal_recency_8000_converted.mistral.jsonl \
    --max-new-tokens 1000

#5
python ./mistral/mistral_process.py \
    --input-path ./litm_repo_with_changes_v0.1/qa_data/domain5_data_05_stackmath_numerical_8000_converted.jsonl \
    --output-path ./mistral/output/domain5_data_05_stackmath_numerical_8000_converted.mistral.jsonl \
    --max-new-tokens 1000

#6
python ./mistral/mistral_process.py \
    --input-path ./litm_repo_with_changes_v0.1/qa_data/domain6_data_06_policy_compliance_8000_converted.jsonl \
    --output-path ./mistral/output/domain6_data_06_policy_compliance_8000_converted.mistral.jsonl \
    --max-new-tokens 1000

#7
python ./mistral/mistral_process.py \
    --input-path ./litm_repo_with_changes_v0.1/qa_data/domain7_data_07_expert_legal_med_8000_converted.jsonl \
    --output-path ./mistral/output/domain7_data_07_expert_legal_med_8000_converted.mistral.jsonl \
    --max-new-tokens 1000