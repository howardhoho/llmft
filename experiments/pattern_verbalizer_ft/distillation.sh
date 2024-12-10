#!/usr/bin/env bash

export PROJECT_DIR=/content/llmft

# Load necessary environment variables and activate Python environment
source $PROJECT_DIR/scripts/misc/setup.sh
# source $PROJECT_DIR/llmft_env/bin/activate
export WANDB_MODE=offline

# Arguments for distillation.sh:
# args: task_name, max_train_samples, epochs, warmup_ratio, bsz, num_gpus, learning_rate, student_model_name_or_path, teacher_model_path, adapter_dim, lora_alpha, port

# Example invocation for MNLI distillation task
bash $PROJECT_DIR/scripts/pattern_verbalizer_ft/mnli/distillation.sh mnli 32 80 0.25 4 1 2e-4 facebook/opt-125m facebook/opt-350m 8 -1 60000
