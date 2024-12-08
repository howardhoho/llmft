#!/usr/bin/env python
# coding=utf-8

import os
import logging
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import (
    OPTForCausalLM,
    AutoTokenizer,
    set_seed,
    AdamW,
    get_linear_schedule_with_warmup
)
import math
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class MNLIDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=128):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_map = {0: "yes", 1: "maybe", 2: "no"}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        label = example["label"]

        prompt = (
            f"Premise: {example['premise']}\n"
            f"Hypothesis: {example['hypothesis']}\n"
            "Is the relationship yes, maybe, or no?\n"
            "Answer:"
        )

        encoded = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",  # Ensure uniform length
            return_tensors="pt"
        )

        label_token = self.label_map[label]

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label_id": label,
            "label_token": label_token,
        }

def main():
    set_seed(42)

    # Model and training parameters
    teacher_model_name = "facebook/opt-1.3b"
    student_model_name = "facebook/opt-350m"
    output_dir = "./context_distilled_student"
    batch_size = 4
    num_epochs = 1
    learning_rate = 1e-5
    alpha = 0.5
    temperature = 3.0

    os.makedirs(output_dir, exist_ok=True)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(student_model_name, use_fast=False)

    # Target tokens
    target_tokens = ["yes", "maybe", "no"]
    target_token_ids = tokenizer.convert_tokens_to_ids(target_tokens)
    if any(tid == tokenizer.unk_token_id for tid in target_token_ids):
        raise ValueError("One or more target tokens are not single tokens. Choose simpler tokens.")

    # Load teacher and student models
    teacher_model = OPTForCausalLM.from_pretrained(teacher_model_name).cuda().eval()
    student_model = OPTForCausalLM.from_pretrained(student_model_name).cuda().train()

    # Load dataset
    dataset = load_dataset("glue", "mnli")
    train_dataset = dataset["train"]

    mnli_train = MNLIDataset(train_dataset, tokenizer)
    train_dataloader = DataLoader(mnli_train, batch_size=batch_size, shuffle=True)

    # Optimizer and scheduler
    optimizer = AdamW(student_model.parameters(), lr=learning_rate)
    num_training_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    global_step = 0
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            optimizer.zero_grad()

            # Move inputs to GPU
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            label_tokens = batch["label_token"]

            # Inference with teacher (no grad)
            with torch.no_grad():
                teacher_outputs = teacher_model(input_ids=input_ids, attention_mask=attention_mask)
                teacher_last_token_logits = teacher_outputs.logits[:, -1, :]
                teacher_probs = F.softmax(teacher_last_token_logits / temperature, dim=-1)[:, target_token_ids]

            # Inference with student (requires grad)
            student_outputs = student_model(input_ids=input_ids, attention_mask=attention_mask)
            student_last_token_logits = student_outputs.logits[:, -1, :]
            student_logits_over_targets = student_last_token_logits[:, target_token_ids]

            # log_softmax for the student
            student_log_probs = F.log_softmax(student_logits_over_targets / temperature, dim=-1)

            # Distillation loss (KL Divergence)
            distill_loss = F.kl_div(
                student_log_probs,
                teacher_probs,
                reduction='batchmean'
            ) * (temperature ** 2)

            # CE loss with proper integer labels
            label_class_indices = torch.tensor(
                [target_tokens.index(lt) for lt in label_tokens],
                dtype=torch.long
            ).cuda()
            ce_loss = F.cross_entropy(student_logits_over_targets, label_class_indices)

            # Combined loss
            loss = (1 - alpha) * ce_loss + alpha * distill_loss

            print("loss.requires_grad:", loss.requires_grad)
            print("ce_loss.requires_grad:", ce_loss.requires_grad)
            print("distill_loss.requires_grad:", distill_loss.requires_grad)
            print("student_log_probs.requires_grad:", student_log_probs.requires_grad)
            print("student_logits_over_targets.requires_grad:", student_logits_over_targets.requires_grad)


            loss.backward()
            optimizer.step()
            scheduler.step()

            global_step += 1
            if global_step % 100 == 0:
                logger.info(f"Step {global_step}: loss={loss.item():.4f}, ce_loss={ce_loss.item():.4f}, distill_loss={distill_loss.item():.4f}")

    # Save the student model
    student_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Training complete. Model saved to {output_dir}.")

if __name__ == "__main__":
    main()