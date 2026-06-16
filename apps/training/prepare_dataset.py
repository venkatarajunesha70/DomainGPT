"""
Dataset preparation for LoRA fine-tuning.

Expected input: a JSONL file where each line is:
  {"instruction": "...", "input": "...", "output": "..."}

Output: HuggingFace Dataset formatted as instruction-tuning pairs.
"""
from __future__ import annotations
import json
from pathlib import Path

from datasets import Dataset


PROMPT_TEMPLATE = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""


def load_jsonl(path: str | Path) -> list[dict]:
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def format_example(example: dict) -> dict:
    """
    Format a single instruction-tuning example.

    Args:
        example: Dict with 'instruction', optional 'input', and 'output'.

    Returns:
        Dict with 'text' key containing the full formatted prompt.
    """
    return {
        "text": PROMPT_TEMPLATE.format(
            instruction=example.get("instruction", ""),
            input=example.get("input", ""),
            output=example.get("output", ""),
        )
    }


def prepare_dataset(
    input_path: str | Path,
    output_path: str | Path | None = None,
    test_size: float = 0.1,
) -> tuple[Dataset, Dataset]:
    """
    Load, format, and split the dataset.

    Args:
        input_path:  Path to the .jsonl training file.
        output_path: Optional path to save the processed dataset.
        test_size:   Fraction of data to use as validation set.

    Returns:
        Tuple of (train_dataset, eval_dataset).
    """
    raw = load_jsonl(input_path)
    formatted = [format_example(ex) for ex in raw]

    dataset = Dataset.from_list(formatted)
    split = dataset.train_test_split(test_size=test_size, seed=42)

    print(f"Train samples: {len(split['train'])}")
    print(f"Eval  samples: {len(split['test'])}")

    if output_path:
        split.save_to_disk(str(output_path))
        print(f"Saved processed dataset to {output_path}")

    return split["train"], split["test"]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prepare dataset for LoRA fine-tuning")
    parser.add_argument("--input", required=True, help="Path to input .jsonl file")
    parser.add_argument("--output", default="./datasets/processed", help="Output directory")
    parser.add_argument("--test-size", type=float, default=0.1)
    args = parser.parse_args()

    prepare_dataset(args.input, args.output, args.test_size)
