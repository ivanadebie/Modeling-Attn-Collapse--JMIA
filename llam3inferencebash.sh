#!/bin/bash
echo " Starting code testing"
git clone -b Atharv https://github.com/ivanadebie/Modeling-Attn-Collapse--JMIA.git
cd Modeling-Attn-Collapse--JMIA
echo "uploading git repo"
python3 -m venv atharv_env
source atharv_env/bin/activate
pip3 install -r requirements_atharv.txt
echo "Installing libs"
echo " run llama3 code"
python3 llama3.py --input-path domain1_data_nq_closed_book_8000.jsonl --output_path domain1_llama3.1-8b_model_responses.jsonl --max-new-tokens 1000
