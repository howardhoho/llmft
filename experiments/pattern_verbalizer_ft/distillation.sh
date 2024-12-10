#!/usr/bin/env bash

export PROJECT_DIR=/content/llmft

# Load necessary environment variables and activate Python environment
source $PROJECT_DIR/scripts/misc/setup.sh
# source $PROJECT_DIR/llmft_env/bin/activate
export WANDB_MODE=offline

# Arguments for distillation.sh:
# args: task_name, max_train_samples, epochs, warmup_ratio, bsz, num_gpus, learning_rate, student_model_name_or_path, teacher_model_path, adapter_dim, lora_alpha, port

# Example invocation for MNLI distillation task
bash $PROJECT_DIR/scripts/pattern_verbalizer_ft/mnli/distillation.sh \
    mnli \                             # Task name
    32 \                               # Max training samples
    80 \                               # Number of epochs
    0.25 \                             # Warmup ratio
    4 \                                # Batch size
    8 \                                # Number of GPUs
    2e-4 \                             # Learning rate
    facebook/opt-125m \                # Student model name or path
    facebook/opt-350m \                # Teacher model name or path
    8 \                                # Adapter dimension
    -1 \                               # Lora alpha
    60000                              # Port
