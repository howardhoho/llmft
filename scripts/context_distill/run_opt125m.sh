#!/usr/bin/env bash

# Set environment variables
export PROJECT_DIR=/llmft
source $PROJECT_DIR/scripts/misc/setup.sh

# Create output directory in GCS bucket
export OUTPUT_DIR=gs://your-bucket-name/models/context_distill/opt125m-rte
export CACHE_DIR=gs://your-bucket-name/cache

# Run context distillation
deepspeed \
    --include localhost:0,1,2,3 \
    --master_port 29500 \
    $PROJECT_DIR/context_distill.py \
    --teacher_model facebook/opt-1.3b \
    --student_model facebook/opt-125m \
    --dataset_name glue \
    --dataset_config rte \
    --max_length 128 \
    --batch_size 32 \
    --learning_rate 5e-5 \
    --num_epochs 3 \
    --output_dir $OUTPUT_DIR \
    --cache_dir $CACHE_DIR \
    --deepspeed_config $PROJECT_DIR/deepspeed_configs/ds_config_zero2_small.json