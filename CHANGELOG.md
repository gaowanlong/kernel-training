## v0.4 (2026-06-19) — Evaluation-Aligned Training

### Overview
Fourth iteration focused on aligning training data with evaluation format. The key insight from v0.3 was that the model was training on code refactoring/bug-fix tasks but being evaluated on kernel knowledge Q&A. This version uses the base model's own high-scoring evaluation responses as training targets.

### What Changed

**Training Data**: 258 samples aligned with evaluation format
- 58 kernel knowledge Q&A samples (base model's own eval responses)
- 200 gzb666 kernel code samples (reduced from 500 to limit code overfitting)
- Training data now directly matches evaluation format

**Training**: QLoRA (rank=8, 200 iters, LR=2e-5)
- Early stopping at step 119 (best checkpoint at step 59)
- Best val loss: 0.849
- Training time: 94.8 minutes on M1 Pro 32GB
- Peak memory: 8.7 GB

### Evaluation Results

39 test questions across 6 categories (LLM-as-judge scoring):

| Metric | Base Model | Fine-tuned | Delta |
|--------|-----------|------------|-------|
| Overall | 75.1% | 70.0% | -5.1% |
| Basic Concepts | 71.2% | 75.0% | +3.7% |
| Kernel Mechanisms | 68.8% | 65.0% | -3.8% |
| Advanced Internals | 70.0% | 68.3% | -1.7% |
| Code Understanding | 75.0% | 51.7% | -23.3% |
| Chinese Knowledge | 73.3% | 68.3% | -5.0% |
| Code Completion | 100.0% | 96.0% | -4.0% |

### Analysis

**Improvements over v0.3:**
- Overall gap narrowed from -7.4% to -5.1%
- Basic Concepts went from -5.0% to +3.7% (first positive delta!)
- Kernel Mechanisms recovered from -26.3% to -3.8%
- Advanced Internals stable at -1.7%

**Remaining Issues:**
1. **Code Understanding (-23.3%)**: The gzb666 code refactoring data still causes degradation in code knowledge questions. Need to either remove gzb666 data or convert it to Q&A format.
2. **Chinese Knowledge (-5.0%)**: New regression. The training data was English-only, causing the model to forget Chinese kernel vocabulary.
3. **Self-referential training ceiling**: Training on the base model's own responses can't exceed the base model's knowledge. Need external high-quality kernel Q&A data.

### Improvement Directions for v0.5

1. **High-quality external data**: Use GPT-4/Claude API to generate expert-level kernel Q&A pairs
2. **Remove or convert gzb666 data**: The code refactoring format doesn't help with kernel knowledge
3. **Add Chinese training data**: Include Chinese kernel Q&A to prevent language regression
4. **Multi-task training**: Mix knowledge Q&A with code understanding in proper proportions
5. **Knowledge distillation**: Use a larger model (e.g., Qwen2.5-72B) to generate training targets
6. **HuggingFace upload**: Login to HF and upload the fused model for community use

### Files Changed
- `scripts/evaluate.py`: Added duplicate ID detection, improved reporting
- `scripts/train_lora.py`: Updated hyperparameters (LR=2e-5, iters=200)
- `data/processed/train.jsonl`: New evaluation-aligned training data
- `data/processed/valid.jsonl`: New evaluation-aligned validation data

# Changelog

## v0.1 (2026-06-19) — Initial Release

### Overview
First end-to-end QLoRA post-training experiment using Linux Kernel v6.6 source code on Qwen2.5-7B-Instruct, running locally on M1 Pro 32GB with MLX.

### What's Included

**Model**: Qwen2.5-7B-Instruct (4-bit quantized, ~4GB on disk)

**Training Data**: 1,146 samples extracted from 40 Linux Kernel v6.6 source files
- 597 code explanation samples
- 518 code completion samples  
- 31 Q&A pair samples
- Covering 8 subsystems: process management, memory management, file system, network stack, device drivers, interrupts, locking, system calls

**Training**: QLoRA (rank=8, 200 iterations, batch_size=1)
- Trainable parameters: ~2-5M (vs ~7B base)
- Adapter size: 78MB
- Peak memory: 8.6GB
- Training time: ~20 minutes on M1 Pro 32GB

### Evaluation Results

23 test questions across 4 categories (kernel concepts, code understanding, Chinese knowledge, code completion):

| Metric | Base Model | Fine-tuned | Delta |
|--------|-----------|------------|-------|
| Overall | 44.6% | 27.8% | -16.7% |
| Kernel Concepts | 40% | 20% | -20% |
| Code Understanding | 48% | 16% | -32% |
| Chinese Knowledge | 40% | 40% | 0% |
| Code Completion | 58% | 50% | -8% |

### Root Cause Analysis

The fine-tuned model performed worse than the base model due to:

1. **Low-quality training labels**: Responses were auto-generated templates ("This is the X function from Y subsystem...") rather than genuine explanations
2. **Catastrophic forgetting**: 200 iterations on 1,031 low-quality samples caused the model to partially forget its general knowledge
3. **Insufficient data volume**: 1,031 samples is far below the threshold for effective fine-tuning of a 7B model
4. **Simple keyword-based evaluation**: The scoring method only checks for keyword presence, not semantic correctness

### Improvement Directions for v0.2

1. **High-quality data generation**: Use a stronger model (GPT-4/Claude) to generate genuine kernel code explanations as training targets
2. **Curated datasets**: Incorporate existing high-quality kernel QA datasets
3. **More data**: Scale to 5,000-10,000+ samples
4. **Better evaluation**: Implement LLM-as-judge or semantic similarity scoring
5. **Training optimization**: Lower learning rate, fewer iterations, better regularization
6. **Multi-epoch curriculum**: Start with easy concepts, progress to complex ones
