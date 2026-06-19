#!/usr/bin/env python3
"""
模型选型与下载脚本

选型分析 (M1 Pro 32GB):
- Qwen2.5-7B-Instruct: 当前最强 7B 开源模型，MLX 原生支持，4-bit 量化后约 4-5GB 显存
- 备选: Mistral-7B, Llama-3.1-8B, DeepSeek-R1-Distill-Qwen-7B

本脚本下载 Qwen2.5-7B-Instruct 并转换为 MLX 格式，同时做 4-bit 量化以节省内存。
"""

import argparse
import subprocess
import sys
from pathlib import Path

MODEL_CHOICES = {
    "qwen2.5-7b": {
        "hf_id": "Qwen/Qwen2.5-7B-Instruct",
        "description": "Qwen2.5-7B-Instruct — 当前最强 7B 模型，中文能力强，推荐首选",
        "estimated_memory_4bit": "~5GB",
    },
    "mistral-7b": {
        "hf_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "description": "Mistral-7B-Instruct-v0.3 — 英文强，轻量高效",
        "estimated_memory_4bit": "~4.5GB",
    },
    "llama-3.1-8b": {
        "hf_id": "meta-llama/Llama-3.1-8B-Instruct",
        "description": "Llama-3.1-8B-Instruct — Meta 官方，需申请访问权限",
        "estimated_memory_4bit": "~5GB",
    },
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


def list_models():
    """列出所有可选模型及选型理由。"""
    print("=" * 70)
    print("M1 Pro 32GB 端侧模型选型分析")
    print("=" * 70)
    print()
    print("约束条件: M1 Pro, 32GB 统一内存, MLX 框架")
    print("策略: 4-bit 量化 + QLoRA 微调")
    print()
    for key, info in MODEL_CHOICES.items():
        print(f"  [{key}]")
        print(f"    HuggingFace: {info['hf_id']}")
        print(f"    说明: {info['description']}")
        print(f"    4-bit 预估内存: {info['estimated_memory_4bit']}")
        print()
    print("推荐: qwen2.5-7b — 综合能力最强，中文支持好，MLX 社区活跃")
    print("=" * 70)


def download_model(model_key: str, quantize: bool = True):
    """下载模型并转换为 MLX 格式。"""
    if model_key not in MODEL_CHOICES:
        print(f"未知模型: {model_key}")
        print(f"可选: {', '.join(MODEL_CHOICES.keys())}")
        sys.exit(1)

    info = MODEL_CHOICES[model_key]
    hf_id = info["hf_id"]
    model_dir = MODELS_DIR / model_key

    print(f"📥 下载模型: {hf_id}")
    print(f"📁 目标目录: {model_dir}")
    print()

    # Step 1: 用 mlx-lm 下载并转换
    cmd_convert = [
        sys.executable, "-m", "mlx_lm.convert",
        "--hf-path", hf_id,
        "--mlx-path", str(model_dir),
    ]
    if quantize:
        cmd_convert.extend(["-q", "--q-bits", "4"])

    print(f"[1/2] 转换模型为 MLX 格式 (4-bit 量化)...")
    print(f"  命令: {' '.join(cmd_convert)}")
    print()
    subprocess.run(cmd_convert, check=True)

    # Step 2: 验证
    print(f"\n[2/2] 验证模型文件...")
    expected_files = ["model.safetensors", "config.json", "tokenizer.json"]
    for fname in expected_files:
        fpath = model_dir / fname
        if fpath.exists():
            size_mb = fpath.stat().st_size / (1024 * 1024)
            print(f"  ✅ {fname} ({size_mb:.1f} MB)")
        else:
            print(f"  ⚠️  {fname} 未找到")

    print(f"\n✅ 模型下载完成: {model_dir}")
    print(f"   总大小: {sum(f.stat().st_size for f in model_dir.rglob('*') if f.is_file()) / (1024**3):.2f} GB")


def test_model(model_key: str):
    """快速测试模型是否可正常推理。"""
    model_dir = MODELS_DIR / model_key
    if not model_dir.exists():
        print(f"模型目录不存在: {model_dir}")
        print("请先运行: python scripts/download_model.py --download {model_key}")
        sys.exit(1)

    from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

    print(f"Testing model: {model_key}")
    model, tokenizer = load(str(model_dir))

    test_prompts = [
        "What is the Linux kernel?",
        "Explain what a page fault is in operating systems.",
        "什么是内核态和用户态？",
    ]

    for prompt in test_prompts:
        print(f"\n{'='*60}")
        print(f"Q: {prompt}")
        response = generate(
            model, tokenizer,
            prompt=prompt,
            max_tokens=256,
            sampler=make_sampler(temp=0.7),
        )
        print(f"A: {response}")


def main():
    parser = argparse.ArgumentParser(description="模型选型与下载工具")
    parser.add_argument("--list", action="store_true", help="列出可选模型及选型分析")
    parser.add_argument("--download", type=str, metavar="MODEL_KEY",
                        help="下载指定模型 (如 qwen2.5-7b)")
    parser.add_argument("--no-quantize", action="store_true",
                        help="不做量化 (不推荐，内存占用大)")
    parser.add_argument("--test", type=str, metavar="MODEL_KEY",
                        help="测试已下载的模型")
    args = parser.parse_args()

    if args.list:
        list_models()
    elif args.download:
        download_model(args.download, quantize=not args.no_quantize)
    elif args.test:
        test_model(args.test)
    else:
        parser.print_help()
        print("\n提示: 使用 --list 查看模型选型分析")


if __name__ == "__main__":
    main()
