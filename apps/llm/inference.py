"""
LLM inference abstraction layer.
Routes to either:
  - Groq API (fast cloud inference with open-weight models like Llama 3)
  - Local LoRA-tuned model (requires GPU)

Set USE_LOCAL_LLM=true in .env to use the local model.
"""
from __future__ import annotations
import os
from typing import AsyncGenerator

from apps.api.core.config import get_settings
from apps.api.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

USE_LOCAL_LLM = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"


async def generate_response(
    prompt: str,
    system_prompt: str = "",
    max_new_tokens: int = 512,
    temperature: float = 0.1,
) -> str:
    """
    Generate a response from the configured LLM backend.

    Args:
        prompt:         User prompt / question.
        system_prompt:  Optional system-level instructions.
        max_new_tokens: Maximum tokens to generate.
        temperature:    Sampling temperature (lower = more deterministic).

    Returns:
        Generated text string.
    """
    if USE_LOCAL_LLM:
        return await _local_generate(prompt, system_prompt, max_new_tokens, temperature)
    return await _groq_generate(prompt, system_prompt, max_new_tokens, temperature)


async def _groq_generate(
    prompt: str,
    system_prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    from langchain_groq import ChatGroq
    from langchain_core.messages import SystemMessage, HumanMessage

    llm = ChatGroq(
        api_key=settings.groq_api_key,
        model_name=settings.groq_model,
        temperature=temperature,
        max_tokens=max_new_tokens,
    )
    messages = []
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    messages.append(HumanMessage(content=prompt))

    result = await llm.ainvoke(messages)
    return result.content


async def _local_generate(
    prompt: str,
    system_prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    """Run local inference in a thread pool to avoid blocking the event loop."""
    import asyncio
    from functools import partial
    from apps.llm.lora_loader import load_lora_model
    import torch

    def _sync_generate():
        model, tokenizer = load_lora_model()
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
            )
        # Decode only the newly generated tokens
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True)

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _sync_generate)
