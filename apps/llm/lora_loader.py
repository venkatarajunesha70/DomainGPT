"""
LoRA adapter loader for local inference.
Loads a base HuggingFace model and merges or dynamically applies a PEFT adapter.
Used when running DomainGPT with a locally fine-tuned model instead of Groq.
"""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from apps.api.core.config import get_settings
from apps.api.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


@lru_cache(maxsize=1)
def load_lora_model():
    """
    Load the base model with 4-bit quantization and attach LoRA adapters.

    Returns:
        Tuple of (model, tokenizer).
    """
    adapter_path = Path(settings.lora_adapter_path)
    if not adapter_path.exists():
        raise FileNotFoundError(
            f"LoRA adapter not found at {adapter_path}. "
            "Run training first: python -m apps.training.lora_train"
        )

    logger.info("loading_base_model", model=settings.hf_base_model)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        settings.hf_base_model,
        token=settings.hf_token,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        settings.hf_base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        token=settings.hf_token,
    )

    logger.info("attaching_lora_adapter", path=str(adapter_path))
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model.eval()

    logger.info("lora_model_ready")
    return model, tokenizer
