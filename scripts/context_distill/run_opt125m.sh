#!/usr/bin/env bash

# Set environment variables
export PROJECT_DIR=/llmft
source $PROJECT_DIR/scripts/misc/setup.sh

# Create output directory in GCS bucket
export BASE_OUTPUT_DIR=gs://llm-ft-cd-storage/models/context_distill/opt125m-rte
export CACHE_DIR=gs://llm-ft-cd-storage/cache

# Hyperparameter configurations
declare -a learning_rates=("1e-5" "3e-5")
declare -a batch_sizes=(8 16)
declare -a num_epochs=(3)
declare -a data_seeds=(0 1 2)

# Run grid search over hyperparameters
for lr in "${learning_rates[@]}"; do
    for bs in "${batch_sizes[@]}"; do
        for epochs in "${num_epochs[@]}"; do
            for data_seed in "${data_seeds[@]}"; do
                # Create unique output directory for this configuration
                export OUTPUT_DIR="${BASE_OUTPUT_DIR}/lr${lr}_bs${bs}_ep${epochs}_seed${data_seed}"
                
                echo "Running experiment with:"
                echo "Learning rate: ${lr}"
                echo "Batch size: ${bs}"
                echo "Epochs: ${epochs}"
                echo "Data seed: ${data_seed}"
                
                # Increment port for each run to avoid conflicts
                PORT=$((29500 + RANDOM % 1000))

                deepspeed \
                    --include localhost:0 \
                    --master_port ${PORT} \
                    $PROJECT_DIR/context_distill.py \
                    --teacher_model facebook/opt-1.3b \
                    --student_model facebook/opt-125m \
                    --dataset_name glue \
                    --dataset_config rte \
                    --max_length 128 \
                    --batch_size ${bs} \
                    --learning_rate ${lr} \
                    --num_epochs ${epochs} \
                    --output_dir ${OUTPUT_DIR} \
                    --cache_dir ${CACHE_DIR} \
                    --deepspeed_config $PROJECT_DIR/deepspeed_configs/ds_config_zero2_small.json \
                    --fp16 \
                    --seed 0 \
                    --data_seed ${data_seed} \
                    --report_to "none"
                
                # Wait between runs to avoid potential GPU memory issues
                sleep 30
            done
        done
    done
done