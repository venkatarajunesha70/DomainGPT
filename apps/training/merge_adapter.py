"""
Merge LoRA adapter into the base model for standalone deployment.
Useful when you want a single deployable model without PEFT at runtime.

Usage:
  python -m apps.training.merge_adapter \
    --adapter ./models/lora/adapter \
    --output ./models/base/merged
"""
from __future__ import annotations
import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from apps.api.core.config import get_settings

settings = get_settings()


def merge_and_save(adapter_path: str, output_path: str) -> None:
    """
    Merge LoRA weights into the base model and save as a regular HF model.

    Args:
        adapter_path: Path to the saved PEFT adapter.
        output_path:  Where to save the merged full-precision model.
    """
    print(f"Loading base model: {settings.hf_base_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        settings.hf_base_model, token=settings.hf_token
    )

    # Load in float16 for merging (no quantization)
    base_model = AutoModelForCausalLM.from_pretrained(
        settings.hf_base_model,
        torch_dtype=torch.float16,
        device_map="cpu",
        token=settings.hf_token,
    )

    print(f"Attaching LoRA adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    print("Merging adapter weights into base model ...")
    merged = model.merge_and_unload()

    print(f"Saving merged model to: {output_path}")
    merged.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--adapter", required=True, help="Path to LoRA adapter directory")
    parser.add_argument("--output", required=True, help="Path for merged output model")
    args = parser.parse_args()
    merge_and_save(args.adapter, args.output)
