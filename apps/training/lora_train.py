"""
LoRA fine-tuning script using HuggingFace PEFT + TRL SFTTrainer.

Supports:
  - 4-bit QLoRA (low VRAM, great for consumer GPUs)
  - Gradient checkpointing
  - Weights & Biases logging (optional)

Usage:
  python -m apps.training.lora_train \
    --dataset ./datasets/processed \
    --output ./models/lora/adapter \
    --epochs 3
"""
from __future__ import annotations
import argparse
from pathlib import Path

import torch
from datasets import load_from_disk
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

from apps.api.core.config import get_settings

settings = get_settings()


def train(
    dataset_path: str,
    output_dir: str,
    num_epochs: int = 3,
    batch_size: int = 4,
    lr: float = 2e-4,
    max_seq_length: int = 1024,
):
    """
    Run LoRA fine-tuning.

    Args:
        dataset_path:   Path to a HuggingFace DatasetDict saved to disk.
        output_dir:     Where to save the LoRA adapter.
        num_epochs:     Training epochs.
        batch_size:     Per-device batch size.
        lr:             Learning rate.
        max_seq_length: Max token length per sample.
    """
    # ── 1. Load tokenizer ───────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        settings.hf_base_model,
        token=settings.hf_token,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── 2. Load base model with 4-bit quantisation ──────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print(f"Loading base model: {settings.hf_base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        settings.hf_base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        token=settings.hf_token,
    )
    model = prepare_model_for_kbit_training(model)

    # ── 3. Attach LoRA adapters ──────────────────────────────────────────
    lora_config = LoraConfig(
        r=16,                            # rank – higher = more capacity
        lora_alpha=32,                   # scaling factor
        target_modules=["q_proj", "v_proj"],  # modules to adapt
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── 4. Load dataset ──────────────────────────────────────────────────
    dataset = load_from_disk(dataset_path)
    train_data = dataset["train"] if "train" in dataset else dataset
    eval_data = dataset.get("test")

    # ── 5. Training arguments ────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=lr,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=25,
        evaluation_strategy="epoch" if eval_data else "no",
        save_strategy="epoch",
        load_best_model_at_end=eval_data is not None,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="none",  # swap to "wandb" if you set WANDB_API_KEY
        gradient_checkpointing=True,
    )

    # ── 6. SFT Trainer ───────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        packing=False,
    )

    print("Starting LoRA fine-tuning ...")
    trainer.train()

    # ── 7. Save adapter only (NOT the full model) ────────────────────────
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"LoRA adapter saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA fine-tuning")
    parser.add_argument("--dataset", required=True, help="Path to processed dataset")
    parser.add_argument("--output", default="./models/lora/adapter")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    args = parser.parse_args()

    train(args.dataset, args.output, args.epochs, args.batch_size, args.lr, args.max_seq_length)
