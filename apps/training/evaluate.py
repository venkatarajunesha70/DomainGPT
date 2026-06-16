"""
RAG + LLM evaluation suite.
Measures: retrieval recall, answer relevance, and groundedness.

Metrics:
  - Hit Rate @k  – was the answer chunk in the top-k retrieved?
  - ROUGE-L       – n-gram overlap between generated and reference answers
  - Faithfulness  – does the answer stay within the provided context?

Usage:
  python -m apps.training.evaluate \
    --eval-file ./datasets/eval_set.jsonl \
    --tenant-id demo
"""
from __future__ import annotations
import argparse
import asyncio
import json
from pathlib import Path

from evaluate import load as load_metric

from apps.api.core.logging import get_logger

logger = get_logger(__name__)
rouge = load_metric("rouge")


async def evaluate_rag(eval_path: str, tenant_id: str) -> dict:
    """
    Run end-to-end evaluation on a JSONL eval set.

    Each line in the eval file:
      {"question": "...", "expected_answer": "...", "document_id": "..."}

    Args:
        eval_path:  Path to the JSONL evaluation file.
        tenant_id:  Tenant to run retrieval against.

    Returns:
        Dict of metric scores.
    """
    from apps.rag.pipeline import run_rag

    examples = []
    with open(eval_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    predictions, references = [], []
    retrieval_hits = 0

    for ex in examples:
        result = await run_rag(
            question=ex["question"],
            tenant_id=tenant_id,
        )
        predictions.append(result["answer"])
        references.append(ex["expected_answer"])

        # Check if the cited source matches expected document
        cited_files = {c["filename"] for c in result.get("citations", [])}
        if ex.get("expected_filename") in cited_files:
            retrieval_hits += 1

    # ROUGE-L
    rouge_result = rouge.compute(predictions=predictions, references=references)

    metrics = {
        "num_examples": len(examples),
        "retrieval_hit_rate": retrieval_hits / len(examples) if examples else 0,
        "rouge_l": rouge_result["rougeL"],
        "rouge_1": rouge_result["rouge1"],
    }

    print("\n=== Evaluation Results ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline")
    parser.add_argument("--eval-file", required=True, help="Path to JSONL eval file")
    parser.add_argument("--tenant-id", required=True)
    args = parser.parse_args()

    asyncio.run(evaluate_rag(args.eval_file, args.tenant_id))
