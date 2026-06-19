#!/usr/bin/env python3
"""
QLoRA fine-tuning script using MLX-LoRA (updated for mlx-lm >= 0.29).

Each step is logged clearly so you can understand the full training pipeline.
"""

import argparse
import json
import sys
import time
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
from rich.table import Table

console = Console()
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def print_step(step_num: int, title: str, description: str = ""):
    console.rule(f"[bold cyan]Step {step_num}: {title}[/bold cyan]")
    if description:
        console.print(f"  {description}")
    console.print()


def load_training_data(data_dir: Path):
    """Load training data and convert to MLX dataset format."""
    print_step(1, "Loading Training Data",
               "Reading train.jsonl and valid.jsonl in ShareGPT format")

    train_file = data_dir / "train.jsonl"
    valid_file = data_dir / "valid.jsonl"

    if not train_file.exists():
        console.print(f"[red]Error: {train_file} not found![/red]")
        console.print("Run: python scripts/prepare_data.py --clone --max-files 40")
        sys.exit(1)

    train_data = []
    valid_data = []

    with open(train_file) as f:
        for line in f:
            train_data.append(json.loads(line))

    if valid_file.exists():
        with open(valid_file) as f:
            for line in f:
                valid_data.append(json.loads(line))

    console.print(f"  Train samples: [green]{len(train_data)}[/green]")
    console.print(f"  Valid samples: [green]{len(valid_data)}[/green]")

    if train_data:
        sample = train_data[0]
        console.print(f"  [dim]Sample type: {sample.get('type', 'unknown')}[/dim]")

    return train_data, valid_data


def format_sharegpt(sample, tokenizer):
    """Format a ShareGPT sample into a single text string."""
    messages = sample["messages"]
    text_parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            text_parts.append(f"<|user|>\n{content}")
        elif role == "assistant":
            text_parts.append(f"<|assistant|>\n{content}")
    return "\n".join(text_parts) + tokenizer.eos_token


def create_mlx_dataset(train_data, valid_data, tokenizer):
    """Create MLX-compatible datasets."""
    print_step(2, "Creating Datasets",
               "Converting ShareGPT format to MLX training datasets")

    # Pass messages format directly - create_dataset auto-detects ChatDataset
    # Show stats on raw data
    sample_texts = [json.dumps(s["messages"]) for s in train_data[:100]]
    lengths = [len(tokenizer.encode(t)) for t in sample_texts]
    import numpy as np
    console.print(f"  Avg sequence length (sample): {np.mean(lengths):.0f}")
    console.print(f"  Max sequence length (sample): {np.max(lengths)}")

    # Create datasets using mlx_lm API - it auto-detects 'messages' key
    # Must wrap in CacheDataset for the training loop
    dataset_config = {"max_seq_length": 2048}
    train_dataset = CacheDataset(create_dataset(train_data, tokenizer, dataset_config))
    valid_dataset = CacheDataset(create_dataset(valid_data, tokenizer, dataset_config)) if valid_data else None

    return train_dataset, valid_dataset


def configure_lora():
    """Configure LoRA hyperparameters."""
    print_step(3, "Configuring LoRA",
               "LoRA adds small trainable matrices to attention layers")

    lora_config = {
        "rank": 8,
        "scale": 2.0,
        "dropout": 0.1,
    }

    table = Table(title="LoRA Configuration")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Explanation", style="dim")

    table.add_row("rank (r)", str(lora_config["rank"]),
                  "Low-rank dimension - controls adapter capacity")
    table.add_row("scale (alpha/r)", str(lora_config["scale"]),
                  "Scaling factor for LoRA weights")
    table.add_row("dropout", str(lora_config["dropout"]),
                  "Dropout rate for regularization")

    console.print(table)
    console.print()
    console.print("  [dim]Estimated trainable parameters: ~2-5M (vs ~7B for full model)[/dim]")
    console.print("  [dim]Expected adapter size on disk: ~10-50MB[/dim]")

    return lora_config


def run_training(model, tokenizer, train_dataset, valid_dataset, lora_config, output_dir: Path):
    """Run the LoRA training loop."""
    print_step(4, "Training",
               "Running QLoRA fine-tuning. Only LoRA weights are updated.")

    # Apply LoRA to model
    console.print("  Applying LoRA to model layers...")
    num_layers = len(model.layers) if hasattr(model, 'layers') else 28
    linear_to_lora_layers(model, num_layers, lora_config)

    # Count trainable params
    trainable_params = sum(
        v.size for v in model.parameters() if 'lora' in v
    ) if hasattr(model, 'parameters') else 0
    console.print(f"  Trainable parameters: ~{trainable_params:,}" if trainable_params else "  Trainable params: LoRA layers applied")

    # Training args
    adapter_path = output_dir / "adapters.safetensors"
    # v0.2 optimized hyperparameters (based on v0.1 learnings):
    # - Reduced iters to 100 to avoid overfitting on small dataset
    # - Lower learning rate for more stable convergence
    # - Larger batch size with grad_accumulation for smoother gradients
    # - More frequent eval to catch overfitting early
    # v0.4 hyperparameters:
    # - 200 iters with larger dataset (258 samples) 
    # - Lower LR (2e-5) for stable convergence
    # - More frequent eval to catch overfitting
    # - Weight decay via Adam default
    training_args = TrainingArgs(
        batch_size=1,
        iters=200,
        val_batches=25,
        steps_per_report=5,
        steps_per_eval=20,
        steps_per_save=50,
        max_seq_length=2048,
        adapter_file=str(adapter_path),
        grad_checkpoint=True,
        grad_accumulation_steps=2,
    )

    table = Table(title="Training Hyperparameters")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")
    for field in TrainingArgs.__dataclass_fields__:
        table.add_row(field, str(getattr(training_args, field)))
    console.print(table)
    console.print()

    # Optimizer
    # v0.2: Lower LR (5e-5 vs 1e-4) to prevent catastrophic forgetting
    # v0.4: Lower LR (2e-5) for stable convergence on eval-aligned data
    optimizer = optim.Adam(learning_rate=2e-5)

    # Save config
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "adapter_config.json"
    adapter_config = {
        "lora_config": lora_config,
        "base_model": str(PROJECT_ROOT / "models" / "qwen2.5-7b"),
        "created_at": datetime.now().isoformat(),
    }
    with open(config_path, "w") as f:
        json.dump(adapter_config, f, indent=2)

    console.print(f"  [bold]Starting training...[/bold]")
    console.print(f"  Output: {output_dir}")
    console.print(f"  This may take 30-60 minutes on M1 Pro 32GB")
    console.print()

    start_time = time.time()

    # v0.3: Custom training loop with early stopping based on val loss
    best_val_loss = float('inf')
    best_step = 0
    patience = 3  # Stop if val loss doesn't improve for 3 evaluations
    patience_counter = 0

    # We use mlx_lm's built-in train but add a callback for early stopping
    from mlx_lm.tuner.callbacks import TrainingCallback

    # v0.3: Track best checkpoint with early stopping
    best_val_loss = [float('inf')]  # mutable container for closure
    best_step = [0]
    patience_counter = [0]
    PATIENCE = 3
    early_stop = [False]

    class EarlyStoppingCallback(TrainingCallback):
        def on_val_loss_report(self, info):
            val_loss = info["val_loss"]
            step = info["iteration"]
            if val_loss < best_val_loss[0]:
                best_val_loss[0] = val_loss
                best_step[0] = step
                patience_counter[0] = 0
                # Save best checkpoint
                best_path = Path(str(adapter_path).replace("adapters.safetensors", "best_adapters.safetensors"))
                adapter_weights = dict(tree_flatten(model.trainable_parameters()))
                mx.save_safetensors(str(best_path), adapter_weights)
                console.print(f"    [green]New best val loss: {val_loss:.3f} at step {step} (saved)[/green]")
            else:
                patience_counter[0] += 1
                console.print(f"    [yellow]Val loss {val_loss:.3f} (best: {best_val_loss[0]:.3f}, patience: {patience_counter[0]}/{PATIENCE})[/yellow]")
                if patience_counter[0] >= PATIENCE:
                    console.print(f"    [bold red]Early stopping triggered at step {step}![/bold red]")
                    early_stop[0] = True

        def on_train_loss_report(self, info):
            if early_stop[0]:
                raise StopIteration("Early stopping")

    callback = EarlyStoppingCallback()

    try:
        train(
            model=model,
            optimizer=optimizer,
            train_dataset=train_dataset,
            val_dataset=valid_dataset,
            args=training_args,
            training_callback=callback,
        )
    except StopIteration as e:
        console.print(f"  [bold yellow]Training stopped early[/bold yellow]")

    # Load best checkpoint
    best_path = Path(str(adapter_path).replace("adapters.safetensors", "best_adapters.safetensors"))
    if best_path.exists():
        model.load_weights(str(best_path), strict=False)
        console.print(f"  [green]Loaded best checkpoint from step {best_step[0]} (val loss: {best_val_loss[0]:.3f})[/green]")

    elapsed = time.time() - start_time
    console.print(f"\n  [bold green]Training complete![/bold green]")
    console.print(f"  Time: {elapsed/60:.1f} minutes")
    console.print(f"  Best checkpoint: step {best_step[0]}")
    console.print(f"  Best val loss: {best_val_loss[0]:.3f}")
    console.print(f"  Adapter saved to: {adapter_path}")

    return adapter_path


def test_model(model, tokenizer, test_prompts: list[str]):
    """Quick test after training."""
    print_step(5, "Quick Test", "Running test prompts to verify the model works")

    for i, prompt in enumerate(test_prompts):
        console.print(f"\n  [bold]Test {i+1}:[/bold] {prompt[:80]}...")
        try:
            response = generate(
                model, tokenizer,
                prompt=prompt,
                max_tokens=200,
                sampler=make_sampler(temp=0.7),
            )
            console.print(f"  [green]Response:[/green] {response[:300]}")
        except Exception as e:
            console.print(f"  [red]Error: {e}[/red]")


def main():
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for Linux Kernel knowledge")
    parser.add_argument("--model", type=str, default="models/qwen2.5-7b",
                        help="Path to base model directory")
    parser.add_argument("--data", type=str, default="data/processed",
                        help="Path to processed training data")
    parser.add_argument("--output", type=str, default="lora_adapters/kernel-lora",
                        help="Output directory for LoRA adapters")
    parser.add_argument("--iters", type=int, default=200,
                        help="Number of training iterations")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--rank", type=int, default=8,
                        help="LoRA rank")
    parser.add_argument("--test-only", action="store_true",
                        help="Only run test prompts (skip training)")
    args = parser.parse_args()

    model_path = PROJECT_ROOT / args.model
    data_path = PROJECT_ROOT / args.data
    output_path = PROJECT_ROOT / args.output

    console.rule("[bold]Linux Kernel QLoRA Fine-tuning[/bold]")
    console.print(f"  Base model: {model_path}")
    console.print(f"  Data: {data_path}")
    console.print(f"  Output: {output_path}")
    console.print()

    # Load model
    print_step(0, "Loading Base Model",
               "Loading the 4-bit quantized model into memory")
    console.print("  This may take 30-60 seconds...")
    model, tokenizer = load(str(model_path))
    console.print(f"  [green]Model loaded successfully[/green]")

    if args.test_only:
        test_prompts = [
            "What is the Linux kernel?",
            "Explain what a page fault is.",
            "What is the purpose of the task_struct in Linux?",
        ]
        test_model(model, tokenizer, test_prompts)
        return

    # Training pipeline
    train_data, valid_data = load_training_data(data_path)
    train_dataset, valid_dataset = create_mlx_dataset(train_data, valid_data, tokenizer)
    lora_config = configure_lora()
    lora_config["rank"] = args.rank

    adapter_path = run_training(
        model, tokenizer, train_dataset, valid_dataset,
        lora_config, output_path,
    )

    # Quick test
    test_prompts = [
        "What is the Linux kernel?",
        "Explain what a page fault is in operating systems.",
        "What is the purpose of the task_struct in Linux?",
    ]
    test_model(model, tokenizer, test_prompts)

    console.rule("[bold green]Done![/bold green]")
    console.print(f"\n  Adapter saved to: {adapter_path}")
    console.print(f"  To evaluate: python scripts/evaluate.py --model {args.model} --adapter {args.output}")


if __name__ == "__main__":
    main()
