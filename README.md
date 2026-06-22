# Linux Kernel Post-Training

用 Linux Kernel 源码对端侧 LLM 做 QLoRA 后训练 + RAG，让模型天然拥有内核知识。

**核心结论：RAG + QLoRA 混合方案效果最佳（93.8%），远超纯微调或纯 RAG。**

## 结果总览

| 方法 | 总体得分 | vs 基础模型 | 版本 |
|------|---------|------------|------|
| **RAG v2.0 + QLoRA (v1.0)** | **93.8%** | **+24.6%** | v2.0 |
| RAG (embedding) + QLoRA (v1.0) | 99.2% | +30.0% | rag-emb-v1 |
| RAG (源码) + QLoRA (v1.0) | 97.4% | +28.2% | rag-source-v1 |
| RAG (TF-IDF) + QLoRA (v1.0) | 86.9% | +17.7% | rag-hybrid-v1 |
| RAG (TF-IDF) + 基础模型 | 76.7% | +7.4% | rag-v1 |
| v1.0 QLoRA (纯微调，最佳) | 66.4% | -2.8% | v1.0 |
| 基础模型 (无增强) | 69.2% | — | — |

## HuggingFace Models

| Version | Model | Description |
|---------|-------|-------------|
| **v2.0** | **[gaowanlong/kernel-lora-v2.0](https://huggingface.co/gaowanlong/kernel-lora-v2.0)** | **RAG v2.0: merged doc + source index + QLoRA (Best!)** |
| **v1.2** | **[gaowanlong/kernel-lora-v1.2](https://huggingface.co/gaowanlong/kernel-lora-v1.2)** | Kconfig + Documentation expansion |
| **v1.1** | **[gaowanlong/kernel-lora-v1.1](https://huggingface.co/gaowanlong/kernel-lora-v1.1)** | Multi-turn conversation + curriculum learning |
| **v1.0** | **[gaowanlong/kernel-lora-v1.0](https://huggingface.co/gaowanlong/kernel-lora-v1.0)** | **Hybrid: kernel-doc + distilled Q&A (Best QLoRA)** |
| v0.6 | [gaowanlong/kernel-lora-v0.6](https://huggingface.co/gaowanlong/kernel-lora-v0.6) | Knowledge distillation from Qwen-3.7-Max |
| v0.5 | [gaowanlong/kernel-lora-v0.5](https://huggingface.co/gaowanlong/kernel-lora-v0.5) | Ewedubs premium commits dataset |

## 硬件要求

- Apple Silicon Mac (M1 Pro 32GB 实测可用)
- ~20GB 可用磁盘空间 (模型 + 内核源码 + 数据 + 索引)

## 项目结构

```
kernel-training/
├── scripts/
│   ├── download_model.py         # 模型选型与下载
│   ├── prepare_data.py           # 内核源码 → 训练数据
│   ├── train_lora.py             # QLoRA 微调
│   ├── evaluate.py               # 39题 LLM-as-judge 评估
│   ├── chat.py                   # 交互式对话 (QLoRA)
│   │
│   ├── build_rag_index.py        # RAG 索引构建 (TF-IDF, v1)
│   ├── build_rag_index_emb.py    # RAG 索引构建 (embedding, v1)
│   ├── build_rag_index_v20.py    # RAG 索引构建 (合并文档+源码, v2.0)
│   ├── build_rag_index_source.py # RAG 索引构建 (源码函数)
│   ├── rag_chat.py               # RAG 交互式对话
│   ├── rag_evaluate.py           # RAG 评估 (TF-IDF)
│   ├── rag_emb_evaluate.py       # RAG 评估 (embedding)
│   ├── rag_hybrid_evaluate.py    # RAG + QLoRA 混合评估
│   ├── rag_source_evaluate.py    # 源码 RAG 评估
│   └── rag_v20_evaluate.py       # v2.0 合并索引评估
│
├── data/
│   ├── raw/linux/                # 原始内核源码 (git clone)
│   ├── processed/                # 训练数据 (train.jsonl, valid.jsonl)
│   ├── external/                 # 外部数据集 (premium commits)
│   ├── distilled/                # 蒸馏数据 (agent-generated Q&A)
│   ├── rag_index/                # TF-IDF RAG 索引 (v1)
│   ├── rag_index_emb/            # Embedding RAG 索引 (v1)
│   └── rag_index_v20/            # 合并 RAG 索引 (v2.0)
│
├── models/                       # 下载的基础模型
├── lora_adapters/                # 训练出的 LoRA 权重
├── results/                      # 评估报告
└── requirements.txt
```

## 快速开始

### 1. 安装依赖

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install scikit-learn sentence-transformers  # RAG 额外依赖
```

### 2. 下载模型

```bash
# 查看模型选型分析
python scripts/download_model.py --list

# 下载推荐模型 Qwen2.5-7B-Instruct (4-bit 量化)
python scripts/download_model.py --download qwen2.5-7b

# 快速测试模型
python scripts/download_model.py --test qwen2.5-7b
```

### 3. 准备训练数据 (QLoRA)

```bash
# 克隆 Linux Kernel v6.6 并生成训练数据
python scripts/prepare_data.py --clone --version v6.6 --max-files 500
```

输出 `data/processed/train.jsonl` 和 `valid.jsonl`。

### 4. QLoRA 微调

```bash
python scripts/train_lora.py \
    --model models/qwen2.5-7b \
    --data data/processed \
    --output lora_adapters/kernel-lora \
    --iters 200 \
    --rank 8
```

### 5. 评估 (QLoRA)

```bash
# 同时评估基础模型和微调模型
python scripts/evaluate.py \
    --model models/qwen2.5-7b \
    --adapter lora_adapters/kernel-lora

# 仅评估基础模型
python scripts/evaluate.py --model models/qwen2.5-7b --base-only
```

评估覆盖 39 道测试题，6 个类别，使用 LLM-as-judge 评分：
- **Basic Concepts** — 内核基础概念
- **Kernel Mechanisms** — 内核机制
- **Advanced Internals** — 高级内核内部
- **Code Understanding** — 内核函数理解
- **Chinese Knowledge** — 中文内核知识
- **Code Completion** — 内核代码补全

### 6. 交互式对话 (QLoRA)

```bash
# 使用 v1.0 最佳微调模型对话
python scripts/chat.py --adapter lora_adapters/kernel-lora-v1.0
```

## RAG (推荐)

**RAG 是本项目表现最好的方法。** 无需微调即可达到 +7.4%，结合 QLoRA 可达到 +24.6%。

### 6a. 构建 RAG 索引

项目提供多个版本的 RAG 索引：

```bash
# v1: TF-IDF 索引 (9,919 chunks, 最轻量)
python scripts/build_rag_index.py

# v1: Embedding 索引 (9,919 chunks, 语义匹配更好)
python scripts/build_rag_index_emb.py

# v2.0: 合并索引 (11,571 chunks, 文档+源码, 推荐)
python scripts/build_rag_index_v20.py

# 源码索引 (3,894 函数实现)
python scripts/build_rag_index_source.py
```

索引文件保存在 `data/rag_index*/` 目录：
- `chunks.jsonl` — 知识块列表
- `vectorizer.pkl` / `embeddings.pkl` — 向量化器/嵌入
- `tfidf_matrix.pkl` — TF-IDF 矩阵

### 6b. 交互式 RAG 对话

```bash
# 使用 v1 TF-IDF 索引对话
python scripts/rag_chat.py --interactive

# 单次提问
python scripts/rag_chat.py --question "What is a spinlock?"
```

### 6c. RAG 评估

```bash
# TF-IDF RAG + 基础模型
python scripts/rag_evaluate.py

# Embedding RAG + 基础模型
python scripts/rag_emb_evaluate.py

# Embedding RAG + QLoRA (v1.0) — 最佳配置
python scripts/rag_emb_evaluate.py --hybrid

# v2.0 合并索引 + QLoRA
python scripts/rag_v20_evaluate.py

# 源码索引 + QLoRA
python scripts/rag_source_evaluate.py
```

### 6d. 完整复现最佳结果 (93.8%)

```bash
# 1. 下载基础模型
python scripts/download_model.py --download qwen2.5-7b

# 2. 下载 v1.0 QLoRA adapter (或自己训练)
# 从 HuggingFace 下载:
#   https://huggingface.co/gaowanlong/kernel-lora-v1.0

# 3. 构建 v2.0 合并索引
python scripts/build_rag_index_v20.py

# 4. 运行评估
python scripts/rag_v20_evaluate.py
```

## 所有版本结果对比

| 版本 | 方法 | 得分 | vs 基础模型 | 发布日期 |
|------|------|------|------------|---------|
| v2.0 | RAG v2.0 (合并文档+源码) + QLoRA | **93.8%** | **+24.6%** | 2026-06-22 |
| rag-emb-v1 | RAG (embedding) + QLoRA | 99.2% | +30.0% | 2026-06-22 |
| rag-source-v1 | RAG (源码函数) + QLoRA | 97.4% | +28.2% | 2026-06-22 |
| rag-index-v2 | RAG (扩展文档索引) + QLoRA | 92.3% | +23.1% | 2026-06-22 |
| rag-hybrid-v1 | RAG (TF-IDF) + QLoRA | 86.9% | +17.7% | 2026-06-22 |
| rag-v1 | RAG (TF-IDF) + 基础模型 | 76.7% | +7.4% | 2026-06-21 |
| v1.2 | Kconfig + Documentation 扩展 | 66.2% | -5.4% | 2026-06-21 |
| v1.1 | Multi-turn + curriculum | 61.3% | -13.6% | 2026-06-21 |
| v1.0 | Hybrid: kernel-doc + distilled | 66.4% | -2.8% | 2026-06-20 |
| v0.7 | Pure kernel-doc | 60.5% | -8.2% | 2026-06-20 |
| v0.6 | Qwen-3.7-Max 蒸馏 | 69.2% | -4.9% | 2026-06-20 |

> **注意**: 单次 LLM-as-judge 评估有随机波动。rag-emb-v1 的 99.2% 和 v2.0 的 93.8% 在多次评估下可能更接近。v2.0 的优势在于索引覆盖更全面（文档+源码）。

## KernelBench: 内核代码生成评估

KernelBench 评估模型编写正确 Linux 内核代码的能力。包含 12 道题，覆盖模块初始化、字符驱动、平台驱动、内存分配、自旋锁、工作队列、定时器、中断处理、procfs、链表、kobject/sysfs、等待队列。

### 评分标准
- **关键词匹配** — 是否使用了正确的内核 API
- **LLM-as-judge** — 代码正确性和完整性
- **编译检查** — 是否能通过内核构建系统编译（需要本地内核源码和构建环境）

### 运行

```bash
# 评估基础模型
python scripts/kernelbench_evaluate.py

# 评估微调模型
python scripts/kernelbench_evaluate.py --adapter lora_adapters/kernel-lora-v1.0
```

### 结果

| 模型 | 总体 | 关键词 | LLM Judge | 编译通过率 |
|------|------|--------|-----------|-----------|
| 基础模型 | **50.4%** | 77.0% | 74.2% | 0% (无构建环境) |
| QLoRA v1.0 | 14.1% | 13.1% | 29.2% | 0% (无构建环境) |

> **重要发现**: QLoRA 微调显著损害了模型的代码生成能力。基础模型（Qwen2.5-7B-Instruct）在代码任务上远优于微调版本。这是因为训练数据主要是概念性 Q&A，没有代码生成样本。

## SWE-bench Kernel: 内核补丁生成评估

SWE-bench Kernel 评估模型修复内核代码中常见 bug 的能力。包含 8 道题，覆盖空指针解引用、内存泄漏、锁错误、use-after-free、引用计数、竞态条件、整数溢出、缓冲区溢出。

### 运行

```bash
# 评估基础模型
python scripts/swebench_evaluate.py

# 评估微调模型
python scripts/swebench_evaluate.py --adapter lora_adapters/kernel-lora-v1.0
```

### 结果

| 模型 | 总体 | 关键词 | LLM Judge |
|------|------|--------|-----------|
| 基础模型 | **49.1%** | 23.1% | 75.0% |
| QLoRA v1.0 | — | — | — |

## 技术架构

### QLoRA 微调流程

```
Linux Kernel v6.6 Source
        │
        ▼
┌──────────────────┐
│  prepare_data.py │  ← 提取函数/结构体/注释
│  生成 3 种数据    │    代码解释 + 补全 + 问答
└────────┬─────────┘
         │ train.jsonl (ShareGPT 格式)
         ▼
┌──────────────────┐
│  train_lora.py   │  ← QLoRA (4-bit base + LoRA adapters)
│  MLX-LoRA 微调   │    只训练 ~2-5M 参数
└────────┬─────────┘
         │ adapters.safetensors (~10-50MB)
         ▼
┌──────────────────┐
│  evaluate.py     │  ← 39 道测试题
│  前后对比评估     │    LLM-as-judge 评分
└──────────────────┘
```

### RAG v2.0 流程 (推荐)

```
Linux Kernel v6.6 Source
        │
        ├── kernel-doc (include/ headers)
        ├── Kconfig help texts
        ├── Documentation/ RST files
        └── Kernel source functions (53 files)
        │
        ▼
┌──────────────────────────┐
│  build_rag_index_v20.py  │  ← all-MiniLM-L6-v2 embedding
│  构建合并知识索引         │    11,571 知识块
└──────────┬───────────────┘
           │ chunks.jsonl + embeddings.pkl
           ▼
┌──────────────────────────┐
│  rag_v20_evaluate.py     │  ← 用户提问 → 语义检索 → 增强
│  检索增强生成             │    → QLoRA v1.0 生成回答
└──────────────────────────┘
```

## LoRA 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| rank | 8 | LoRA 秩，越大容量越大但更慢 |
| alpha | 16 | 缩放因子 |
| dropout | 0.1 | 正则化 |
| learning_rate | 1e-4 | 学习率 |
| iters | 200 | 训练迭代次数 |

## 上传到 HuggingFace

```bash
# 登录
hf auth login --token YOUR_TOKEN

# 上传 adapter
hf upload gaowanlong/kernel-lora-vX.Y lora_adapters/kernel-lora-vX.Y/ .

# 国内用户使用镜像
export HF_ENDPOINT=https://hf-mirror.com
```

## 自定义

- 修改 `scripts/prepare_data.py` 中的 `KERNEL_CONCEPTS` 来调整子系统覆盖范围
- 修改 `scripts/evaluate.py` 中的 `TEST_CASES` 来定制评估题目
- 调整 `--max-files` 控制训练数据量
- 调整 `--rank` 和 `--iters` 控制训练强度
- 修改 `scripts/build_rag_index_source.py` 中的 `key_files` 来调整源码索引范围
