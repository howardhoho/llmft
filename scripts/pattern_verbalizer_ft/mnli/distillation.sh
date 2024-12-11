#!/usr/bin/env bash

# args: task_name, max_train_samples, epochs, warmup_ratio, bsz, num_gpus, learning_rate, student_model_name_or_path, teacher_model_path, adapter_dim, lora_alpha, port

task_name=$1
max_train_samples=$2
epochs=$3
warmup_ratio=$4
bsz=$5
num_gpus=$6
learning_rate=$7
student_model_name_or_path=$8
teacher_model_path=$9
adapter_type="lora"
adapter_dim=${10}
lora_alpha=${11}
port=${12}
distillation_temperature=2.0  # Default value, modify if needed
alpha_distillation=0.5       # Default value, modify if needed

# Calculate logging steps
logging_steps=$((max_train_samples / (bsz * num_gpus)))

# Specify target tokens for the task
target_tokens="ĠYes,ĠNo"  # Update based on the model/task

for seed in "0"
do
    for data_seed in "0"
    do
        deepspeed \
            --include localhost:0 \
            --master_port $port \
            $PROJECT_DIR/distillation_ft.py \
            --wandb_project_name llmft-experiments \
            --wandb_group_name pattern-verbalizer-ft-$adapter_type \
            --student_model_name_or_path $student_model_name_or_path \
            --model_name_or_path $student_model_name_or_path \
            --teacher_model_path $teacher_model_path \
            --cache_dir $HF_MODELS_CACHE \
            --task_name $task_name \
            --pattern "{text1} {text2} ?" \
            --target_tokens $target_tokens \
            --dataset_cache_dir $HF_DATASETS_CACHE \
            --max_seq_length 256 \
            --output_dir $OUTPUT_DIR \
            --overwrite_output_dir \
            --do_train \
            --do_eval \
            --use_adapter \
            --adapter_type $adapter_type \
            --adapter_dim $adapter_dim \
            --lora_alpha $lora_alpha \
            --distillation_temperature $distillation_temperature \
            --alpha_distillation $alpha_distillation \
            --max_train_samples $max_train_samples \
            --per_device_train_batch_size $bsz \
            --gradient_accumulation_steps 1 \
            --num_train_epochs $epochs \
            --warmup_ratio $warmup_ratio \
            --logging_first_step true \
            --logging_steps $logging_steps \
            --learning_rate $learning_rate \
            --weight_decay 0.0 \
            --evaluation_strategy epoch \
            --per_device_eval_batch_size 10 \
            --eval_on_hans \
            --save_strategy no \
            --fp16 \
            --seed $seed \
            --data_seed $data_seed \
            --deepspeed $PROJECT_DIR/deepspeed_configs/ds_config_zero3.json \
            --deepspeed_stage 3 \
            --report_to "none"
    done
done
