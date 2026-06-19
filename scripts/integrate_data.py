#!/usr/bin/env python3
"""
Integrate external high-quality kernel datasets with our training pipeline.

Currently supported external datasets:
1. gzb666/linux-kernel-training-data — 730K kernel bug-fix + feature examples
   Format: {instruction, input, output, file_paths, commit_hash, author, author_date}

Output: data/processed/train.jsonl + valid.jsonl in ShareGPT format
"""

import json
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
EXTERNAL_DIR = DATA_DIR / "external"
PROCESSED_DIR = DATA_DIR / "processed"


def load_gzb666_samples(max_samples: int = 5000):
    """Load gzb666 dataset samples and convert to ShareGPT format."""
    filepath = EXTERNAL_DIR / "gzb666_samples.jsonl"
    if not filepath.exists():
        print(f"  [SKIP] {filepath} not found. Run download_external.py first.")
        return []

    samples = []
    with open(filepath) as f:
        for line in f:
            samples.append(json.loads(line))

    print(f"  Loaded {len(samples)} raw samples from gzb666")

    converted = []
    for s in samples:
        # Format: instruction + input as user prompt, output as assistant response
        if s["input"]:
            prompt = f"{s['instruction']}\n\n```c\n{s['input'][:3000]}\n```"
        else:
            prompt = s["instruction"]

        response = f"```c\n{s['output'][:3000]}\n```"

        converted.append({
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ],
            "type": "kernel_bugfix",
            "source": "gzb666",
            "file_paths": s.get("file_paths", []),
            "commit_hash": s.get("commit_hash", ""),
        })

    print(f"  Converted {len(converted)} samples to ShareGPT format")
    return converted


def load_local_kernel_samples(max_samples: int = 1000):
    """Load our locally generated kernel samples."""
    filepath = PROCESSED_DIR / "train.jsonl"
    if not filepath.exists():
        print(f"  [SKIP] {filepath} not found.")
        return []

    samples = []
    with open(filepath) as f:
        for line in f:
            samples.append(json.loads(line))

    print(f"  Loaded {len(samples)} local kernel samples")
    return samples[:max_samples]


def merge_and_split(external_samples, local_samples, split_ratio=0.9):
    """Merge external and local samples, then train/valid split."""
    all_samples = external_samples + local_samples
    random.shuffle(all_samples)

    split_idx = int(len(all_samples) * split_ratio)
    train = all_samples[:split_idx]
    valid = all_samples[split_idx:]

    print(f"\n  Merged dataset: {len(all_samples)} total")
    print(f"    External: {len(external_samples)}")
    print(f"    Local: {len(local_samples)}")
    print(f"    Train: {len(train)}")
    print(f"    Valid: {len(valid)}")

    return train, valid


def save_dataset(train, valid):
    """Save train/valid splits."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    with open(PROCESSED_DIR / "train.jsonl", "w") as f:
        for s in train:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    with open(PROCESSED_DIR / "valid.jsonl", "w") as f:
        for s in valid:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\n  Saved to {PROCESSED_DIR}:")
    print(f"    train.jsonl: {len(train)} samples")
    print(f"    valid.jsonl: {len(valid)} samples")


def main():
    print("=== Kernel Training Data Integration ===\n")

    # Load external datasets
    print("[1/3] Loading external datasets...")
    external = load_gzb666_samples(max_samples=5000)

    # Load local samples
    print("\n[2/3] Loading local kernel samples...")
    local = load_local_kernel_samples(max_samples=1000)

    # Merge and split
    print("\n[3/3] Merging and splitting...")
    train, valid = merge_and_split(external, local)

    # Save
    save_dataset(train, valid)

    print("\nDone!")


if __name__ == "__main__":
    main()
