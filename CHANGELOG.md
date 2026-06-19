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
