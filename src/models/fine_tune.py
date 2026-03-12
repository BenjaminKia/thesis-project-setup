"""
src/models/fine_tune.py
Fine-tune Transformer models on market-informed sentiment labels.

Supports: BERT, FinBERT, RoBERTa, OPT-350M, OPT-2.7B
Grid search over learning rate, batch size, and epochs.
Temporal train/test split (no random shuffling).

Usage:
    # Fine-tune BERT with grid search
    python -m src.models.fine_tune --model bert --grid-search

    # Fine-tune with specific hyperparameters
    python -m src.models.fine_tune --model bert --lr 2e-5 --batch-size 16 --epochs 4

    # Fine-tune all models
    python -m src.models.fine_tune --model all --grid-search
"""

import argparse
import os
import yaml
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from itertools import product

from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Optional: experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ============================================================
# Configuration
# ============================================================
MODEL_REGISTRY = {
    "bert": "bert-base-uncased",
    "finbert": "ProsusAI/finbert",
    "roberta": "roberta-base",
    "opt-350m": "facebook/opt-350m",
}

# Grid search ranges (Devlin et al., 2019 recommendations)
GRID = {
    "learning_rate": [1e-5, 2e-5, 5e-5],
    "batch_size": [16, 32],
    "epochs": [3, 4],
}


# ============================================================
# Dataset
# ============================================================
class SentimentDataset(Dataset):
    """Dataset for news articles with market-informed labels."""

    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ============================================================
# Data loading
# ============================================================
def load_data(config):
    """
    Load matched news-returns data with temporal split.

    Expected input: a parquet file with columns:
        - text: str (news article text)
        - label: int (0 or 1)
        - date: datetime (publication date)

    Returns train, val, test DataFrames.
    """
    data_path = Path(config["paths"]["processed"]) / "matched_news_returns.parquet"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Matched data not found at {data_path}. "
            "Run the news-returns matching pipeline first."
        )

    df = pd.read_parquet(data_path)
    df["date"] = pd.to_datetime(df["date"])

    # Temporal split (no random shuffling!)
    train_end = pd.Timestamp(config["data"]["train_end"])
    test_start = pd.Timestamp(config["data"]["test_start"])

    train_full = df[df["date"] <= train_end].copy()
    test = df[df["date"] >= test_start].copy()

    # Validation: last 20% of training period (temporal, not random)
    val_cutoff = train_full["date"].quantile(0.8)
    train = train_full[train_full["date"] <= val_cutoff].copy()
    val = train_full[train_full["date"] > val_cutoff].copy()

    print(f"Data loaded:")
    print(f"  Train: {len(train):,} articles ({train['date'].min().date()} to {train['date'].max().date()})")
    print(f"  Val:   {len(val):,} articles ({val['date'].min().date()} to {val['date'].max().date()})")
    print(f"  Test:  {len(test):,} articles ({test['date'].min().date()} to {test['date'].max().date()})")
    print(f"  Label balance (train): {train['label'].mean():.3f} positive")

    return train, val, test


# ============================================================
# Training loop
# ============================================================
def train_one_epoch(model, dataloader, optimizer, scheduler, device, fp16=True):
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0
    scaler = torch.amp.GradScaler("cuda") if fp16 else None

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        if fp16:
            with torch.amp.autocast("cuda"):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """Evaluate model. Returns loss, accuracy, F1, precision, recall."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            total_loss += outputs.loss.item()

            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    metrics = {
        "loss": avg_loss,
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds, average="macro"),
        "precision": precision_score(all_labels, all_preds, average="macro"),
        "recall": recall_score(all_labels, all_preds, average="macro"),
    }
    return metrics


# ============================================================
# Full training run
# ============================================================
def train_model(
    model_name,
    hf_model_id,
    train_df,
    val_df,
    lr,
    batch_size,
    epochs,
    device,
    config,
    save_dir,
):
    """
    Train a single model with given hyperparameters.
    Returns validation metrics and saves the best checkpoint.
    """
    print(f"\n{'='*60}")
    print(f"Training {model_name} | lr={lr}, bs={batch_size}, epochs={epochs}")
    print(f"{'='*60}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        hf_model_id, num_labels=2
    ).to(device)

    # Handle OPT padding (decoder models need this)
    if "opt" in model_name.lower():
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    # Create datasets
    train_dataset = SentimentDataset(
        train_df["text"].tolist(),
        train_df["label"].tolist(),
        tokenizer,
    )
    val_dataset = SentimentDataset(
        val_df["text"].tolist(),
        val_df["label"].tolist(),
        tokenizer,
    )

    # Gradient accumulation for effective batch size
    if batch_size > 16 and "opt" not in model_name.lower():
        actual_batch = 16
        grad_accum = batch_size // 16
    elif "opt-2.7b" in model_name.lower():
        actual_batch = 4
        grad_accum = batch_size // 4
    else:
        actual_batch = min(batch_size, 8 if "opt" in model_name.lower() else 16)
        grad_accum = max(1, batch_size // actual_batch)

    train_loader = DataLoader(train_dataset, batch_size=actual_batch, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=actual_batch, shuffle=False)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=0.01
    )
    total_steps = len(train_loader) * epochs // grad_accum
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # Training loop with early stopping
    best_val_loss = float("inf")
    patience_counter = 0
    patience = 3
    best_metrics = None

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device
        )
        val_metrics = evaluate(model, val_loader, device)

        print(
            f"  Epoch {epoch+1}/{epochs} | "
            f"Train loss: {train_loss:.4f} | "
            f"Val loss: {val_metrics['loss']:.4f} | "
            f"Val acc: {val_metrics['accuracy']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f}"
        )

        # Early stopping check
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_metrics = val_metrics.copy()
            patience_counter = 0

            # Save best checkpoint
            checkpoint_dir = save_dir / f"{model_name}_best"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # Log best hyperparameters
    best_metrics["lr"] = lr
    best_metrics["batch_size"] = batch_size
    best_metrics["epochs_trained"] = epoch + 1
    best_metrics["model"] = model_name

    print(f"\n  Best val metrics: acc={best_metrics['accuracy']:.4f}, F1={best_metrics['f1']:.4f}")

    # Clean up GPU memory
    del model, optimizer, scheduler
    torch.cuda.empty_cache()

    return best_metrics


# ============================================================
# Grid search
# ============================================================
def grid_search(model_name, hf_model_id, train_df, val_df, device, config, save_dir):
    """
    Run grid search over hyperparameters.
    Returns best configuration and all results.
    """
    combinations = list(product(
        GRID["learning_rate"],
        GRID["batch_size"],
        GRID["epochs"],
    ))

    print(f"\nGrid search for {model_name}: {len(combinations)} combinations")

    all_results = []
    best_val_loss = float("inf")
    best_config = None

    for lr, bs, ep in combinations:
        metrics = train_model(
            model_name, hf_model_id,
            train_df, val_df,
            lr=lr, batch_size=bs, epochs=ep,
            device=device, config=config, save_dir=save_dir,
        )
        all_results.append(metrics)

        if metrics["loss"] < best_val_loss:
            best_val_loss = metrics["loss"]
            best_config = metrics.copy()

    print(f"\n{'='*60}")
    print(f"Grid search complete for {model_name}")
    print(f"Best config: lr={best_config['lr']}, bs={best_config['batch_size']}")
    print(f"Best val: acc={best_config['accuracy']:.4f}, F1={best_config['f1']:.4f}")
    print(f"{'='*60}")

    return best_config, all_results


# ============================================================
# Test set evaluation
# ============================================================
def evaluate_on_test(model_name, test_df, device, save_dir):
    """Load best checkpoint and evaluate on test set."""
    checkpoint_dir = save_dir / f"{model_name}_best"

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir).to(device)

    if "opt" in model_name.lower():
        tokenizer.pad_token = tokenizer.eos_token

    test_dataset = SentimentDataset(
        test_df["text"].tolist(),
        test_df["label"].tolist(),
        tokenizer,
    )
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    metrics = evaluate(model, test_loader, device)

    print(f"\nTest results for {model_name}:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")

    # Also generate predictions for portfolio construction
    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1)[:, 1]  # P(outperform)
            all_probs.extend(probs.cpu().numpy())

    test_df = test_df.copy()
    test_df[f"{model_name}_score"] = all_probs

    del model
    torch.cuda.empty_cache()

    return metrics, test_df


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Fine-tune models on sentiment labels")
    parser.add_argument("--model", type=str, required=True,
                        choices=list(MODEL_REGISTRY.keys()) + ["all"],
                        help="Model to fine-tune")
    parser.add_argument("--grid-search", action="store_true",
                        help="Run grid search over hyperparameters")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate (if not grid search)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size (if not grid search)")
    parser.add_argument("--epochs", type=int, default=4,
                        help="Number of epochs (if not grid search)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load data
    train_df, val_df, test_df = load_data(config)

    # Save directory
    save_dir = Path(config["paths"]["checkpoints"])
    save_dir.mkdir(parents=True, exist_ok=True)

    # Results directory
    results_dir = Path(config["paths"]["results"]) / "replication"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Determine which models to train
    models_to_train = (
        list(MODEL_REGISTRY.items())
        if args.model == "all"
        else [(args.model, MODEL_REGISTRY[args.model])]
    )

    all_test_results = {}

    for model_name, hf_model_id in models_to_train:
        print(f"\n{'#'*60}")
        print(f"# Model: {model_name} ({hf_model_id})")
        print(f"{'#'*60}")

        if args.grid_search:
            best_config, grid_results = grid_search(
                model_name, hf_model_id,
                train_df, val_df,
                device, config, save_dir,
            )
            # Save grid search results
            grid_df = pd.DataFrame(grid_results)
            grid_df.to_csv(
                results_dir / f"{model_name}_grid_search.csv", index=False
            )
        else:
            best_config = train_model(
                model_name, hf_model_id,
                train_df, val_df,
                lr=args.lr, batch_size=args.batch_size, epochs=args.epochs,
                device=device, config=config, save_dir=save_dir,
            )

        # Evaluate on test set
        test_metrics, test_df_with_scores = evaluate_on_test(
            model_name, test_df, device, save_dir
        )
        all_test_results[model_name] = test_metrics

        # Save predictions for portfolio construction
        test_df_with_scores.to_parquet(
            results_dir / f"{model_name}_test_predictions.parquet", index=False
        )

    # Save summary
    summary = pd.DataFrame(all_test_results).T
    summary.to_csv(results_dir / "test_results_summary.csv")
    print(f"\n{'='*60}")
    print("ALL TEST RESULTS")
    print(f"{'='*60}")
    print(summary.to_string())
    print(f"\nResults saved to {results_dir}/")


if __name__ == "__main__":
    main()
