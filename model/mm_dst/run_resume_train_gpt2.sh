#!/bin/bash
PATH_DIR=$(realpath .)

# Train (multi-modal)
CUDA_VISIBLE_DEVICES=$1 python3 -m gpt2_dst.scripts.run_language_modeling \
    --output_dir="${PATH_DIR}"/gpt2_dst/save/large \
    --model_type=gpt2-large \
    --model_name_or_path="${PATH_DIR}"/gpt2_dst/save/large/ \
    --line_by_line \
    --do_train \
    --train_data_file="${PATH_DIR}"/gpt2_dst/data/simmc2_dials_dstc10_train_target.txt \
    --do_eval --eval_all_checkpoints \
    --eval_data_file="${PATH_DIR}"/gpt2_dst/data/simmc2_dials_dstc10_dev_target.txt \
    --num_train_epochs=2 \
    --overwrite_output_dir \
    --logging_steps=2000 \
    --warmup_steps=4000 \
    --save_steps=2000 \
    --per_gpu_train_batch_size=2 \
    --per_gpu_eval_batch_size=2 \
    --save_total_limit=4 \
    --fp16
