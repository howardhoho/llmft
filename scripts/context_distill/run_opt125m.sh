#!/usr/bin/env bash

export PROJECT_DIR=/llmft
source $PROJECT_DIR/scripts/misc/setup.sh

deepspeed \
    --include localhost:0,1,2,3 \
    --master_port 29500 \
    $PROJECT_DIR/context_distill.py \
    --teacher_model facebook/opt-1.3b \
    --student_model facebook/opt-125m \
    --dataset_name wikitext \
    --dataset_config wikitext-2-raw-v1 \
    --max_length 128 \
    --batch_size 32 \
    --learning_rate 5e-5 \
    --num_epochs 3 \
    --output_dir /path/to/output \
    --deepspeed_config $PROJECT_DIR/deepspeed_configs/ds_config_zero2_small.json 