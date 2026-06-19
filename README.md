# Linux Kernel Post-Training

用 Linux Kernel 源码对端侧 LLM 做 QLoRA 后训练，让模型天然拥有内核知识。

## HuggingFace Model

The fine-tuned model is available for upload to HuggingFace. To upload:

1. Login to HuggingFace: `huggingface-cli login`
2. Upload the fused model: `python -m mlx_lm fuse --model models/qwen2.5-7b --adapter-path lora_adapters/kernel-lora --save-path models/qwen2.5-7b-fused --upload-repo gaowanlong/kernel-lora-v0.4`

The fused model is at `models/qwen2.5-7b-fused/` (~4.3GB).


## 硬件要求

- Apple Silicon Mac (M1 Pro 32GB 实测可用)
- ~20GB 可用磁盘空间 (模型 + 内核源码 + 数据)

## 项目结构

```
kernel-training/
├── scripts/
│   ├── download_model.py   # 模型选型与下载
│   ├── prepare_data.py     # 内核源码 → 训练数据
│   ├── train_lora.py       # QLoRA 微调
│   └── evaluate.py         # 训练前后对比评估
├── data/
│   ├── raw/                # 原始内核源码
│   ├── processed/          # 训练数据 (train.jsonl, valid.jsonl)
│   └── eval/               # 评估测试用例
├── models/                 # 下载的模型
├── lora_adapters/          # 训练出的 LoRA 权重
├── results/                # 评估报告
└── requirements.txt
```

## 快速开始

### 1. 安装依赖

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
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

### 3. 准备训练数据

```bash
# 克隆 Linux Kernel v6.6 并生成训练数据
python scripts/prepare_data.py --clone --version v6.6 --max-files 500
```

这一步会：
- 浅克隆 Linux Kernel v6.6 源码
- 从 500 个源文件中提取函数和结构体
- 生成三种训练数据：代码解释、代码补全、问答对
- 输出 `data/processed/train.jsonl` 和 `valid.jsonl`

### 4. 训练

```bash
# QLoRA 微调 (约 30-60 分钟)
python scripts/train_lora.py \
    --model models/qwen2.5-7b \
    --data data/processed \
    --output lora_adapters/kernel-lora \
    --iters 200 \
    --rank 8
```

训练过程会清晰打印每一步：
- Step 0: 加载基础模型
- Step 1: 加载训练数据
- Step 2: Tokenize 数据集
- Step 3: 配置 LoRA 参数
- Step 4: 训练循环 (每 10 步打印 loss)
- Step 5: 快速测试

### 5. 评估对比

```bash
# 同时评估基础模型和微调模型，生成对比报告
python scripts/evaluate.py \
    --model models/qwen2.5-7b \
    --adapter lora_adapters/kernel-lora

# 仅评估基础模型
python scripts/evaluate.py --model models/qwen2.5-7b --base-only
```

评估覆盖四个维度：
- **Kernel Concepts** — 内核基础概念 (10 题)
- **Code Understanding** — 内核函数理解 (5 题)
- **Chinese Knowledge** — 中文内核知识 (5 题)
- **Code Completion** — 内核代码补全 (3 题)

## 模型选型理由

| 模型 | 参数量 | 4-bit 内存 | 中文能力 | 推荐度 |
|------|--------|-----------|---------|--------|
| Qwen2.5-7B-Instruct | 7B | ~5GB | 优秀 | ★★★★★ |
| Mistral-7B-Instruct-v0.3 | 7B | ~4.5GB | 一般 | ★★★ |
| Llama-3.1-8B-Instruct | 8B | ~5GB | 一般 | ★★★ |

选择 Qwen2.5-7B-Instruct 的原因：
1. 当前 7B 级别综合能力最强的开源模型
2. 中英文双语能力强，适合学习内核的中英文资料
3. MLX 社区支持完善，转换和推理稳定
4. 4-bit 量化后在 M1 Pro 32GB 上运行流畅

## 技术架构

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
│  evaluate.py     │  ← 23 道测试题
│  前后对比评估     │    关键词匹配 + 分类统计
└──────────────────┘
```

## LoRA 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| rank | 8 | LoRA 秩，越大容量越大但更慢 |
| alpha | 16 | 缩放因子 |
| dropout | 0.1 | 正则化 |
| learning_rate | 1e-4 | 学习率 |
| iters | 200 | 训练迭代次数 |

## 自定义

- 修改 `scripts/prepare_data.py` 中的 `KERNEL_CONCEPTS` 来调整子系统覆盖范围
- 修改 `scripts/evaluate.py` 中的 `TEST_CASES` 来定制评估题目
- 调整 `--max-files` 控制训练数据量
- 调整 `--rank` 和 `--iters` 控制训练强度
