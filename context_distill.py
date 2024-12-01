import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import deepspeed
import argparse
from datasets import load_dataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_model", type=str, default="facebook/opt-1.3b")
    parser.add_argument("--student_model", type=str, default="facebook/opt-125m")
    parser.add_argument("--dataset_name", type=str, default="wikitext")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default="./distilled_model")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--deepspeed_config", type=str, default="deepspeed_configs/ds_config_zero2_small.json")
    return parser.parse_args()

def prepare_data(tokenizer, dataset, max_length):
    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt"
        )
    return dataset.map(tokenize, batched=True)

def main():
    args = parse_args()
    
    # Initialize teacher and student models
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher_model)
    student = AutoModelForCausalLM.from_pretrained(args.student_model)
    tokenizer = AutoTokenizer.from_pretrained(args.student_model)
    
    # Freeze teacher model
    for param in teacher.parameters():
        param.requires_grad = False
    
    # Load dataset
    dataset = load_dataset(args.dataset_name, args.dataset_config)
    train_dataset = prepare_data(tokenizer, dataset["train"], args.max_length)
    
    # Initialize DeepSpeed
    ds_config = args.deepspeed_config
    student_engine = deepspeed.initialize(
        model=student,
        config=ds_config,
        model_parameters=student.parameters()
    )[0]
    
    # Training loop
    student_engine.train()
    teacher.eval()
    
    for epoch in range(args.num_epochs):
        for batch in train_dataset:
            # Get teacher logits
            with torch.no_grad():
                teacher_outputs = teacher(**batch)
                teacher_logits = teacher_outputs.logits
            
            # Train student
            student_outputs = student_engine(**batch)
            student_logits = student_outputs.logits
            
            # Calculate distillation loss
            loss = torch.nn.functional.mse_loss(
                student_logits, teacher_logits
            )
            
            student_engine.backward(loss)
            student_engine.step()
            
        logger.info(f"Epoch {epoch+1}/{args.num_epochs}, Loss: {loss.item():.4f}")
    
    # Save the distilled model
    student_engine.save_checkpoint(args.output_dir)

if __name__ == "__main__":
    main() 