#!/bin/bash
PATH_DIR=$(realpath .)
PATH_DATA_DIR=$(realpath ../../data)
RESULT=devtest/gpt2-large2
DOMAIN=devtest

# generation formatter
python -m gpt2_dst.scripts.subtask4_formatter \
    --predicted_path="${PATH_DIR}"/gpt2_dst/results/"${RESULT}"/simmc2_dials_dstc10_"${DOMAIN}"_predicted.txt \
    --dials_path="${PATH_DATA_DIR}"/simmc2_dials_dstc10_"${DOMAIN}".json \
    --domain="${DOMAIN}" \
    --retrieval_candidate_path="${PATH_DATA_DIR}"/simmc2_dials_dstc10_"${DOMAIN}"_retrieval_candidates.json \
    --output_path="${PATH_DIR}"/gpt2_dst/results/"${RESULT}" \
