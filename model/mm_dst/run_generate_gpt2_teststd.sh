#!/bin/bash
PATH_DIR=$(realpath .)
MODEL=gpt2-large2
RESULT=gpt2-large2

# Generate sentences (Furniture, multi-modal)
CUDA_VISIBLE_DEVICES=$1 python -m gpt2_dst.scripts.run_generation \
    --model_type=gpt2 \
    --model_name_or_path="${PATH_DIR}"/gpt2_dst/save/"${MODEL}"/ \
    --num_return_sequences=1 \
    --length=100 \
    --stop_token='<EOS>' \
    --prompts_from_file="${PATH_DIR}"/gpt2_dst/data/simmc2_dials_dstc10_teststd_predict.txt \
    --path_output="${PATH_DIR}"/gpt2_dst/results/"${RESULT}"/simmc2_dials_dstc10_teststd_predicted.txt \
    --num_beams=2
