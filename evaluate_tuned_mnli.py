#!/usr/bin/env python
# coding=utf-8

import logging
import os
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoTokenizer, set_seed
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from models.opt_wrapper import OPTWithLMClassifier

logger = logging.getLogger(__name__)

class MNLIDataset(Dataset):
    def __init__(self, dataset, preprocessor):
        self.dataset = dataset
        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        processed = self.preprocessor(example)
        return processed

def evaluate_opt_mnli():
    # Set paths and configurations
    model_dir = "/content/drive/MyDrive/Colab_Notebooks/evaluation/model_dir"
    output_dir = "opt_mnli_results"
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure model directory exists
    assert os.path.isdir(model_dir), f"Model directory '{model_dir}' not found!"

    # Set random seed
    set_seed(42)

    # Load model and tokenizer from files
    config = AutoConfig.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = OPTWithLMClassifier.from_pretrained(model_dir, config=config)
    print("Model loaded successfully.")
    model.to(device)

    # Define placeholder target tokens
    target_tokens = ["yes", "maybe", "no"]
    target_tokens_ids = tokenizer.convert_tokens_to_ids(target_tokens)

    if any(token_id == tokenizer.unk_token_id for token_id in target_tokens_ids):
        raise ValueError(f"One or more placeholder tokens not found in vocabulary: {target_tokens}")
    print(f"Target token IDs: {target_tokens_ids}")

    def preprocess_function(example):
        template = "Premise: {text1}\nHypothesis: {text2}\nIs the relationship yes, maybe, or no?"
        prompt = template.format(text1=example["premise"], text2=example["hypothesis"])

        result = tokenizer(
            prompt,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt"
        )
        result["labels"] = torch.tensor([example["label"]])
        return result

    # Load MNLI dataset
    eval_dataset = load_dataset("glue", "mnli")["validation_matched"]
    eval_dataset = MNLIDataset(eval_dataset, preprocess_function)
    eval_dataloader = DataLoader(eval_dataset, batch_size=8)

    # Evaluation loop
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in eval_dataloader:
            inputs = {
                "input_ids": batch["input_ids"].squeeze(1).to(device),
                "attention_mask": batch["attention_mask"].squeeze(1).to(device),
            }
            # Pass the required arguments to the model
            outputs = model(**inputs, target_tokens=target_tokens, tokenizer=tokenizer)
            logits = outputs["logits"].cpu().numpy()
            predicted_classes = np.argmax(logits, axis=1)

            all_predictions.extend(predicted_classes)
            all_labels.extend(batch["labels"].cpu().numpy())

    # Compute metrics
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    accuracy = np.mean(all_labels == all_predictions)
    print(f"Accuracy: {accuracy:.4f}")

    # Save results
    results_df = pd.DataFrame([{"accuracy": accuracy}])
    results_file = os.path.join(output_dir, "opt125m_mnli_results.csv")
    results_df.to_csv(results_file, index=False)

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    print("Confusion Matrix:\n", cm)

    # Save Confusion Matrix as a CSV
    cm_file = os.path.join(output_dir, "opt125m_mnli_confusion_matrix.csv")
    pd.DataFrame(cm, index=target_tokens, columns=target_tokens).to_csv(cm_file)

    # Visualize Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_tokens, yticklabels=target_tokens)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    cm_image = os.path.join(output_dir, "opt125m_mnli_confusion_matrix.png")
    plt.savefig(cm_image)
    plt.close()

    print(f"Results saved to {results_file}")
    print(f"Confusion Matrix saved to {cm_file} and {cm_image}")

if __name__ == "__main__":
    evaluate_opt_mnli()



