#!/usr/bin/env python3
"""
Interactive chat with the fine-tuned kernel model.
Usage: python scripts/chat.py [--adapter lora_adapters/kernel-lora-v1.0]
"""

import argparse, sys, time
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="lora_adapters/kernel-lora-v1.0")
    parser.add_argument("--model", default="models/qwen2.5-7b")
    args = parser.parse_args()

    print(f"Loading model ({args.model}) with adapter ({args.adapter})...", flush=True)
    model, tokenizer = load(args.model, adapter_path=args.adapter)
    sampler = make_sampler(temp=0.7)
    print("Loaded! Type 'exit' to quit.\n")

    history = []

    while True:
        try:
            prompt = input("\n>>> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if prompt.lower() == "exit":
            break

        if not prompt.strip():
            continue

        messages = history + [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        print("  [generating...]", end=" ", flush=True)
        start = time.time()

        response = generate(
            model, tokenizer,
            prompt=formatted,
            max_tokens=512,
            sampler=sampler,
        )

        elapsed = time.time() - start
        print(f"({elapsed:.1f}s)\n")
        print(response)

        history.append({"role": "user", "content": prompt})
        history.append({"role": "assistant", "content": response})

        # Keep last 5 turns
        if len(history) > 10:
            history = history[-10:]

if __name__ == "__main__":
    main()
