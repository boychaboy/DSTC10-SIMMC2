#!/bin/bash
PATH_DIR=$(realpath .)
MODEL_TYPE=gpt2-large
MODEL_PATH=large3
EPOCH=4
DATA_DIR=data

# Train (multi-modal)
CUDA_VISIBLE_DEVICES=$1 python3 -m gpt2_dst.scripts.run_language_modeling \
    --output_dir="${PATH_DIR}"/gpt2_dst/save/"${MODEL_PATH}" \
    --model_type="${MODEL_TYPE}" \
    --model_name_or_path="${MODEL_TYPE}" \
    --line_by_line \
    --add_special_tokens="${PATH_DIR}"/gpt2_dst/"${DATA_DIR}"/simmc2_special_tokens.json \
    --do_train \
    --train_data_file="${PATH_DIR}"/gpt2_dst/"${DATA_DIR}"/simmc2_dials_dstc10_train_target.txt \
    --do_eval --eval_all_checkpoints \
    --eval_data_file="${PATH_DIR}"/gpt2_dst/"${DATA_DIR}"/simmc2_dials_dstc10_dev_target.txt \
    --num_train_epochs="${EPOCH}" \
    --overwrite_output_dir \
    --logging_steps=1000 \
    --warmup_steps=8000 \
    --save_steps=1000 \
    --learning_rate=5e-05 \
    --gradient_accumulation_steps=8 \
    --per_gpu_train_batch_size=1 \
    --per_gpu_eval_batch_size=1 \
    --save_total_limit=4 \
    --fp16
