#!/usr/bin/env bash

export PROJECT_DIR=/content/llmft

source $PROJECT_DIR/scripts/misc/setup.sh
# source $PROJECT_DIR/llmft_env/bin/activate
export WANDB_MODE=offline


# args: task_name, max_train_samples, epochs, warmup_ratio, bsz, num_gpus, learning_rate, model_name_or_path, adapter_dim, lora_alpha, port
bash $PROJECT_DIR/scripts/pattern_verbalizer_ft/mnli/lora.sh mnli 32 80 0.25 4 8 2e-4 facebook/opt-125m 8 -1 60000
