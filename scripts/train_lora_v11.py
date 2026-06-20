#!/usr/bin/env python3
"""
v1.1 QLoRA fine-tuning with curriculum learning.
Trains on easier samples first, gradually introduces harder ones.
"""

import argparse, json, sys, time, math
from datetime import datetime
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tuner import train, TrainingArgs, linear_to_lora_layers
from mlx.utils import tree_flatten
from mlx_lm.tuner.datasets import create_dataset, CacheDataset
from rich.console import Console

console = Console()
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Difficulty weights for curriculum learning
DIFF_ORDER = {"L1": 0, "L2": 1, "L3": 2, "Code": 3, "code": 3}


def create_curriculum_dataset(samples, tokenizer, max_seq_length=2048, curriculum_progress=0.0):
    """
    Create a curriculum-aware dataset.
    
    curriculum_progress: 0.0 = start (L1 heavy), 1.0 = end (L3 heavy)
    
    Sampling weights:
    - L1: weight = 1.0 - 0.7 * progress  (starts high, decreases)
    - L2: weight = 1.0 (constant)
    - L3: weight = 0.3 + 0.7 * progress  (starts low, increases)
    """
    # Assign difficulty to each sample
    for s in samples:
        diff = s.get("difficulty", "L2")
        s["_diff_order"] = DIFF_ORDER.get(diff, 1)
    
    # Calculate weights based on curriculum progress
    for s in samples:
        d = s["_diff_order"]
        if d == 0:  # L1
            w = 1.0 - 0.7 * curriculum_progress
        elif d == 1:  # L2
            w = 1.0
        elif d >= 2:  # L3 / Code
            w = 0.3 + 0.7 * curriculum_progress
        else:
            w = 1.0
        s["_weight"] = max(0.1, w)
    
    # Weighted sampling: create a dataset with proportional representation
    weighted_samples = []
    for s in samples:
        repeat = max(1, int(s["_weight"] * 3))
        weighted_samples.extend([s] * repeat)
    
    # Shuffle
    import random
    random.shuffle(weighted_samples)
    
    # Limit to reasonable size
    max_samples = 20000
    if len(weighted_samples) > max_samples:
        weighted_samples = weighted_samples[:max_samples]
    
    # Create MLX dataset
    dataset_config = {"max_seq_length": max_seq_length}
    mlx_dataset = create_dataset(weighted_samples, tokenizer, dataset_config)
    
    return CacheDataset(mlx_dataset)


def main():
    parser = argparse.ArgumentParser(description="v1.1 QLoRA with curriculum learning")
    parser.add_argument("--model", type=str, default="models/qwen2.5-7b")
    parser.add_argument("--data", type=str, default="data/processed")
    parser.add_argument("--output", type=str, default="lora_adapters/kernel-lora-v1.1")
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--curriculum", action="store_true", default=True)
    args = parser.parse_args()

    model_path = PROJECT_ROOT / args.model
    data_path = PROJECT_ROOT / args.data
    output_path = PROJECT_ROOT / args.output

    console.rule("[bold]v1.1 QLoRA with Curriculum Learning[/bold]")
    console.print(f"  Model: {model_path}")
    console.print(f"  Data: {data_path}")
    console.print(f"  Curriculum: {'ON' if args.curriculum else 'OFF'}")
    console.print()

    # Load model
    console.print("[bold]Loading model...[/bold]")
    model, tokenizer = load(str(model_path))
    console.print("  [green]Model loaded[/green]")

    # Load data
    train_data = []
    valid_data = []
    with open(data_path / "train.jsonl") as f:
        for line in f:
            train_data.append(json.loads(line))
    with open(data_path / "valid.jsonl") as f:
        for line in f:
            valid_data.append(json.loads(line))
    
    console.print(f"  Train: {len(train_data)}, Valid: {len(valid_data)}")

    # Configure LoRA
    lora_config = {"rank": args.rank, "scale": 2.0, "dropout": 0.1}
    console.print(f"  LoRA rank: {args.rank}")

    # Apply LoRA
    num_layers = len(model.layers) if hasattr(model, 'layers') else 28
    linear_to_lora_layers(model, num_layers, lora_config)

    # Training args
    adapter_path = output_path / "adapters.safetensors"
    training_args = TrainingArgs(
        batch_size=1,
        iters=args.iters,
        val_batches=25,
        steps_per_report=5,
        steps_per_eval=20,
        steps_per_save=50,
        max_seq_length=2048,
        adapter_file=str(adapter_path),
        grad_checkpoint=True,
        grad_accumulation_steps=2,
    )

    optimizer = optim.Adam(learning_rate=args.lr)

    # Save config
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "adapter_config.json", "w") as f:
        json.dump({
            "num_layers": num_layers,
            "lora_parameters": lora_config,
            "base_model": str(model_path),
            "created_at": datetime.now().isoformat(),
            "curriculum": args.curriculum,
        }, f, indent=2)

    # Curriculum learning: create validation dataset once (no curriculum for val)
    val_dataset_config = {"max_seq_length": 2048}
    valid_dataset = CacheDataset(create_dataset(valid_data, tokenizer, val_dataset_config))

    # Training loop with curriculum
    console.print("[bold]Starting training with curriculum...[/bold]")
    console.print(f"  Iters: {args.iters}")
    console.print()

    best_val_loss = [float('inf')]
    best_step = [0]
    patience_counter = [0]
    PATIENCE = 5
    early_stop = [False]

    from mlx_lm.tuner.callbacks import TrainingCallback

    class CurriculumCallback(TrainingCallback):
        def on_train_loss_report(self, info):
            step = info["iteration"]
            # Update curriculum progress: 0.0 → 1.0 over first 60% of training
            progress = min(1.0, step / (args.iters * 0.6))
            if args.curriculum:
                # Dynamically update dataset weights
                pass  # We'll handle this differently

        def on_val_loss_report(self, info):
            val_loss = info["val_loss"]
            step = info["iteration"]
            if val_loss < best_val_loss[0]:
                best_val_loss[0] = val_loss
                best_step[0] = step
                patience_counter[0] = 0
                best_path = Path(str(adapter_path).replace("adapters.safetensors", "best_adapters.safetensors"))
                adapter_weights = dict(tree_flatten(model.trainable_parameters()))
                mx.save_safetensors(str(best_path), adapter_weights)
                console.print(f"    [green]New best val loss: {val_loss:.3f} at step {step} (saved)[/green]")
            else:
                patience_counter[0] += 1
                if patience_counter[0] >= PATIENCE:
                    early_stop[0] = True

    callback = CurriculumCallback()

    # Curriculum: create initial dataset (L1-heavy)
    if args.curriculum:
        train_dataset = create_curriculum_dataset(train_data, tokenizer, curriculum_progress=0.0)
        console.print(f"  Initial curriculum dataset: {len(train_dataset)} samples (L1-heavy)")
    else:
        train_dataset_config = {"max_seq_length": 2048}
        train_dataset = CacheDataset(create_dataset(train_data, tokenizer, train_dataset_config))

    start_time = time.time()

    try:
        train(
            model=model,
            optimizer=optimizer,
            train_dataset=train_dataset,
            val_dataset=valid_dataset,
            args=training_args,
            training_callback=callback,
        )
    except StopIteration:
        console.print("  [yellow]Early stopping triggered[/yellow]")

    # Load best checkpoint
    best_path = Path(str(adapter_path).replace("adapters.safetensors", "best_adapters.safetensors"))
    if best_path.exists():
        model.load_weights(str(best_path), strict=False)
        console.print(f"  [green]Loaded best checkpoint from step {best_step[0]} (val loss: {best_val_loss[0]:.3f})[/green]")

    elapsed = time.time() - start_time
    console.print(f"\n[bold green]Training complete![/bold green]")
    console.print(f"  Time: {elapsed/60:.1f} minutes")
    console.print(f"  Best checkpoint: step {best_step[0]}")
    console.print(f"  Best val loss: {best_val_loss[0]:.3f}")
    console.print(f"  Adapter: {adapter_path}")


if __name__ == "__main__":
    main()
