#!/bin/bash
PATH_DIR=$(realpath .)
PATH_DATA_DIR=$(realpath ../../data)
RESULT=$1

# Formatter (multi-modal)
# python -m gpt2_dst.scripts.format_retrieval_results \
#     --model_output_file="${PATH_DIR}"/gpt2_dst/results/"${RESULT}"/simmc2_dials_dstc10_devtest_retrieval_predicted.txt \
#     --dialog_json_file="${PATH_DATA_DIR}"/simmc2_dials_dstc10_devtest.json \
#     --formatted_output_file="${PATH_DIR}"/gpt2_dst/results/"${RESULT}"/simmc2_dials_dstc10_devtest_retrieval_formatted.json \

# [boychaboy] formatter
python -m gpt2_dst.scripts.generation_to_retrieval \
    --predicted_path="${PATH_DIR}"/gpt2_dst/results/"${RESULT}"/simmc2_dials_dstc10_devtest_predicted.txt \
    --dials_path="${PATH_DATA_DIR}"/simmc2_dials_dstc10_devtest.json \
    --retrieval_candidate_path="${PATH_DATA_DIR}"/simmc2_dials_dstc10_devtest_retrieval_candidates.json

# Evaluate
python ../utils/retrieval_evaluation.py \
    --retrieval_json_path="${PATH_DATA_DIR}"/simmc2_dials_dstc10_devtest_retrieval_candidates.json \
    --model_score_path="${PATH_DIR}"/gpt2_dst/results/"${RESULT}"/simmc2_dials_dstc10_devtest_retrieved.json \
    


