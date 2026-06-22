# Linux Kernel Post-Training

用 Linux Kernel 源码对端侧 LLM 做 QLoRA 后训练，让模型天然拥有内核知识。

## HuggingFace Model

Fine-tuned models are available on HuggingFace:

| Version | Model | Description |
|---------|-------|-------------|
| **v1.2** | **[gaowanlong/kernel-lora-v1.2](https://huggingface.co/gaowanlong/kernel-lora-v1.2)** | Kconfig + Documentation expansion (v1.0 data + 1,684 Kconfig + 2,575 doc Q&A) |
| **v1.1** | **[gaowanlong/kernel-lora-v1.1](https://huggingface.co/gaowanlong/kernel-lora-v1.1)** | Multi-turn conversation + curriculum learning |
| **v1.0** | **[gaowanlong/kernel-lora-v1.0](https://huggingface.co/gaowanlong/kernel-lora-v1.0)** | **Hybrid: kernel-doc + distilled Q&A (Best!)** |
| v0.6 | [gaowanlong/kernel-lora-v0.6](https://huggingface.co/gaowanlong/kernel-lora-v0.6) | Knowledge distillation from Qwen-3.7-Max |
| v0.5 | [gaowanlong/kernel-lora-v0.5](https://huggingface.co/gaowanlong/kernel-lora-v0.5) | Ewedubs premium commits dataset |

### Upload to HuggingFace

If you are in China and cannot access huggingface.co directly, use the mirror site [hf-mirror.com](https://hf-mirror.com):

```bash
# Set the mirror endpoint (no proxy needed)
export HF_ENDPOINT=https://hf-mirror.com

# Login with your HuggingFace token
hf auth login --token YOUR_TOKEN

# Upload the fused model
hf upload gaowanlong/kernel-lora-v0.X models/qwen2.5-7b-fused/ .
```

Note: hf-mirror.com is a read-only mirror of huggingface.co. Uploads go through the mirror endpoint and sync to the main site.

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

### 6. RAG (Retrieval-Augmented Generation)

**RAG 是本项目表现最好的方法** — 无需微调，直接使用基础模型 + 检索即可达到 **+7.4%** 的提升（vs 基础模型），超越所有 QLoRA 版本。

RAG 从内核源码、Kconfig 帮助文本和 Documentation/ 中提取知识块，构建 TF-IDF 索引，在回答问题时检索最相关的上下文注入 prompt。

#### 构建 RAG 索引

```bash
# 从 kernel-doc + Kconfig + Documentation 构建索引
python scripts/build_rag_index.py
```

索引文件保存在 `data/rag_index/`：
- `chunks.jsonl` — 9,919 个知识块（kernel-doc / Kconfig / Documentation）
- `vectorizer.pkl` — TF-IDF 向量化器
- `tfidf_matrix.pkl` — 预计算的 TF-IDF 矩阵

#### 交互式对话

```bash
# 交互模式
python scripts/rag_chat.py --interactive

# 单次提问
python scripts/rag_chat.py --question "What is a spinlock?"

# 显示检索到的上下文
python scripts/rag_chat.py --question "Explain RCU" --show-context
```

交互模式支持的命令：
- `exit` / `quit` — 退出
- `--context` — 切换上下文显示开关

#### 评估 RAG

```bash
# 对 39 道测试题进行 RAG 评估
python scripts/rag_evaluate.py
```

#### RAG vs QLoRA 对比

| 方法 | 总体得分 | vs 基础模型 |
|------|---------|------------|
| **RAG (TF-IDF + base model)** | **76.7%** | **+7.4%** |
| v1.0 QLoRA (最佳微调版本) | 66.4% | -2.8% |
| 基础模型 (无增强) | 69.2% | — |

RAG 在所有类别上都优于基础模型和 QLoRA 版本：

| 类别 | RAG | v1.0 QLoRA | 基础模型 |
|------|-----|-----------|---------|
| Basic Concepts | **80.0%** | 72.5% | 67.5% |
| Kernel Mechanisms | **75.0%** | 67.5% | 65.0% |
| Advanced Internals | **73.3%** | 65.0% | 66.7% |
| Code Understanding | **73.3%** | 43.3% | 70.0% |
| Chinese Knowledge | **71.7%** | 73.3% | 73.3% |
| Code Completion | **88.0%** | 88.0% | 86.0% |


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

### RAG 流程 (推荐)

```
Linux Kernel v6.6 Source
        │
        ├── kernel-doc (include/ headers)
        ├── Kconfig help texts
        └── Documentation/ RST files
        │
        ▼
┌──────────────────────┐
│  build_rag_index.py  │  ← TF-IDF 向量化
│  构建知识索引         │    9,919 知识块
└──────────┬───────────┘
           │ chunks.jsonl + vectorizer.pkl + tfidf_matrix.pkl
           ▼
┌──────────────────────┐
│  rag_chat.py         │  ← 用户提问 → 检索 → 增强 → 生成
│  检索增强生成         │    TF-IDF 相似度匹配 top-3
└──────────────────────┘
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
