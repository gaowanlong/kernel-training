#!/usr/bin/env python3
"""
Linux Kernel source -> post-training data preparation pipeline.

Data strategy:
1. Clone Linux Kernel source (specified version)
2. Parse source to extract functions/structs/comments
3. Generate three types of training data:
   a) Code explanation: given function -> explain its purpose
   b) Code completion: given context+signature -> complete implementation
   c) Q&A pairs: kernel concept Q&A
4. Output in ShareGPT format (MLX-LoRA compatible)
"""

import argparse
import json
import os
import random
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MAX_FILE_SIZE = 500_000  # Skip files larger than 500KB to avoid regex hangs

KERNEL_CONCEPTS = {
    "process_management": {
        "cn": "进程管理",
        "keywords": ["task_struct", "fork", "exec", "sched", "pid", "signal"],
        "top_dirs": ["kernel"],
    },
    "memory_management": {
        "cn": "内存管理",
        "keywords": ["mm_struct", "vm_area_struct", "page", "alloc", "kmalloc", "vmalloc", "mmap"],
        "top_dirs": ["mm"],
    },
    "file_system": {
        "cn": "文件系统",
        "keywords": ["inode", "dentry", "file", "vfs", "ext4", "btrfs", "xfs"],
        "top_dirs": ["fs"],
    },
    "network_stack": {
        "cn": "网络协议栈",
        "keywords": ["sk_buff", "socket", "tcp", "ip", "netdev", "netfilter"],
        "top_dirs": ["net"],
    },
    "device_drivers": {
        "cn": "设备驱动",
        "keywords": ["pci", "usb", "driver", "device", "probe", "remove"],
        "top_dirs": ["drivers"],
    },
    "interrupts": {
        "cn": "中断处理",
        "keywords": ["irq", "interrupt", "softirq", "tasklet", "handler"],
        "top_dirs": ["kernel"],
    },
    "locking": {
        "cn": "锁机制",
        "keywords": ["spinlock", "mutex", "semaphore", "rcu", "rwlock", "atomic"],
        "top_dirs": ["kernel"],
    },
    "system_calls": {
        "cn": "系统调用",
        "keywords": ["SYSCALL_DEFINE", "sys_", "__x64_sys_"],
        "top_dirs": ["kernel", "fs", "mm"],
    },
}


def clone_kernel(version: str = "v6.6", depth: int = 1):
    """Shallow clone Linux Kernel source."""
    kernel_dir = RAW_DIR / "linux"
    if kernel_dir.exists():
        print(f"Kernel source already exists: {kernel_dir}")
        return kernel_dir

    print(f"Cloning Linux Kernel {version} (--depth={depth})...")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        "git", "clone",
        "--depth", str(depth),
        "--branch", version,
        "https://github.com/torvalds/linux.git",
        str(kernel_dir),
    ]
    subprocess.run(cmd, check=True)
    print(f"Clone done: {kernel_dir}")
    return kernel_dir


def extract_functions(filepath: Path) -> list[dict]:
    """Extract function definitions and their comments from a C source file."""
    if filepath.stat().st_size > MAX_FILE_SIZE:
        return []
    try:
        content = filepath.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []

    functions = []
    lines = content.split("\n")

    func_pattern = re.compile(
        r'^(?:static\s+)?(?:inline\s+)?(?:__\w+\s+)*'
        r'([\w\s*]+?)\s+(\w+)\s*\(([^)]*)\)\s*$'
    )

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        match = func_pattern.match(line)

        if match and not line.startswith(("if", "while", "for", "switch", "return")):
            return_type, func_name, params = match.groups()

            if func_name in ("if", "while", "for", "switch", "return", "sizeof",
                             "typeof", "defined", "case", "default", "else", "do"):
                i += 1
                continue

            # Collect preceding comments
            comment_lines = []
            j = i - 1
            while j >= 0:
                prev = lines[j].strip()
                if prev.startswith("/*") or prev.startswith("*") or prev.startswith("//"):
                    comment_lines.insert(0, prev)
                    j -= 1
                elif prev == "":
                    j -= 1
                else:
                    break

            # Collect function body (max 300 lines)
            body_start = i
            brace_count = 0
            body_end = i
            found_open = False

            for k in range(i, min(i + 300, len(lines))):
                brace_count += lines[k].count("{") - lines[k].count("}")
                if "{" in lines[k]:
                    found_open = True
                if found_open and brace_count == 0:
                    body_end = k + 1
                    break

            if body_end > body_start + 2:
                body = "\n".join(lines[body_start:body_end])
                functions.append({
                    "file": str(filepath.relative_to(RAW_DIR / "linux")),
                    "name": func_name,
                    "return_type": return_type.strip(),
                    "params": params.strip(),
                    "comment": "\n".join(comment_lines) if comment_lines else "",
                    "body": body,
                    "line_start": body_start + 1,
                    "line_end": body_end,
                })

            i = body_end
        else:
            i += 1

    return functions


def extract_structs(filepath: Path) -> list[dict]:
    """Extract struct definitions."""
    if filepath.stat().st_size > MAX_FILE_SIZE:
        return []
    try:
        content = filepath.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []

    structs = []
    lines = content.split("\n")

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("struct ") and "{" in line:
            name_match = re.match(r"struct\s+(\w+)\s*\{", line)
            if not name_match:
                i += 1
                continue

            struct_name = name_match.group(1)
            brace_count = line.count("{") - line.count("}")
            end = i + 1
            for k in range(i + 1, min(i + 200, len(lines))):
                brace_count += lines[k].count("{") - lines[k].count("}")
                if brace_count == 0:
                    end = k + 1
                    break

            body = "\n".join(lines[i:end])
            structs.append({
                "file": str(filepath.relative_to(RAW_DIR / "linux")),
                "name": struct_name,
                "body": body,
            })
            i = end
        else:
            i += 1

    return structs


def detect_subsystem(filepath: str) -> str:
    """Detect which subsystem a file belongs to based on its path."""
    for name, info in KERNEL_CONCEPTS.items():
        for top_dir in info["top_dirs"]:
            if filepath.startswith(top_dir + "/") or filepath == top_dir:
                return name
    return "kernel_core"


def generate_code_explanation(func: dict) -> dict:
    """Generate a code-explanation training sample."""
    subsystem = detect_subsystem(func["file"])
    comment_text = func["comment"].replace("/*", "").replace("*/", "").replace("*", "").strip()

    if comment_text:
        prompt = (
            f"Explain the following Linux kernel function in detail. "
            f"Describe what it does, its parameters, return value, and any important side effects.\n\n"
            f"Function (from {func['file']}):\n```c\n{func['body'][:2000]}\n```\n\n"
            f"Hint from comments: {comment_text[:500]}"
        )
    else:
        prompt = (
            f"Explain the following Linux kernel function in detail. "
            f"Describe what it does, its parameters, return value, and any important side effects.\n\n"
            f"Function (from {func['file']}):\n```c\n{func['body'][:2000]}\n```"
        )

    response = f"This is the `{func['name']}` function from the Linux kernel's {subsystem} subsystem. "
    if comment_text:
        response += f"\n\n{comment_text}"
    response += f"\n\nIt returns `{func['return_type']}` and takes parameters: `{func['params']}`."

    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ],
        "type": "code_explanation",
        "subsystem": subsystem,
        "source_file": func["file"],
    }


def generate_code_completion(func: dict) -> Optional[dict]:
    """Generate a code-completion training sample."""
    body_lines = func["body"].split("\n")
    if len(body_lines) < 5:
        return None

    context = "\n".join(body_lines[:2])
    completion = "\n".join(body_lines[2:])

    if len(completion) < 50:
        return None

    prompt = (
        f"Complete the following Linux kernel function implementation:\n\n"
        f"```c\n{context}\n```"
    )

    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": f"```c\n{completion}\n```"},
        ],
        "type": "code_completion",
        "source_file": func["file"],
    }


def generate_qa_pairs(struct: dict) -> Optional[dict]:
    """Generate Q&A pairs based on struct definitions."""
    subsystem = detect_subsystem(struct["file"])

    questions = [
        f"What is the `{struct['name']}` structure in the Linux kernel? Explain its fields and purpose.",
        f"Describe the `{struct['name']}` data structure used in the Linux kernel's {subsystem} subsystem.",
    ]

    response = (
        f"The `{struct['name']}` structure is defined in `{struct['file']}` "
        f"and is part of the Linux kernel's {subsystem} subsystem. "
        f"It is defined as:\n\n```c\n{struct['body'][:1500]}\n```"
    )

    return {
        "messages": [
            {"role": "user", "content": random.choice(questions)},
            {"role": "assistant", "content": response},
        ],
        "type": "qa_pair",
        "subsystem": subsystem,
        "source_file": struct["file"],
    }


def process_kernel_source(kernel_dir: Path, max_files: int = 500):
    """Process kernel source and generate training data."""
    print("Scanning kernel source...")

    c_files = []
    for ext in ["*.c", "*.h"]:
        c_files.extend(kernel_dir.rglob(ext))

    exclude_dirs = {"tools", "scripts", "samples", "Documentation", "usr"}
    c_files = [
        f for f in c_files
        if not any(ex in f.parts for ex in exclude_dirs)
    ]

    print(f"  Found {len(c_files)} C source files")

    if max_files:
        # Pre-group files by top-level directory for O(n) sampling
        by_topdir = {}
        for f in c_files:
            top = f.relative_to(kernel_dir).parts[0]
            by_topdir.setdefault(top, []).append(f)

        sampled = []
        per_subsystem = max_files // len(KERNEL_CONCEPTS)
        for name, info in KERNEL_CONCEPTS.items():
            subsystem_files = []
            for td in info["top_dirs"]:
                if td in by_topdir:
                    subsystem_files.extend(by_topdir[td])
            n = min(len(subsystem_files), per_subsystem)
            if n > 0:
                sampled.extend(random.sample(subsystem_files, n))
        c_files = sampled
        print(f"  Sampled {len(c_files)} files (~{per_subsystem} per subsystem)")

    all_functions = []
    all_structs = []

    for i, fpath in enumerate(c_files):
        if i % 50 == 0:
            print(f"  Processing: {i}/{len(c_files)}")

        funcs = extract_functions(fpath)
        structs = extract_structs(fpath)

        all_functions.extend(funcs)
        all_structs.extend(structs)

    print(f"\nExtraction results:")
    print(f"  Functions: {len(all_functions)}")
    print(f"  Structs: {len(all_structs)}")

    samples = []

    for func in all_functions:
        if len(func["body"]) > 100:
            samples.append(generate_code_explanation(func))

    for func in all_functions:
        sample = generate_code_completion(func)
        if sample:
            samples.append(sample)

    for struct in all_structs:
        sample = generate_qa_pairs(struct)
        if sample:
            samples.append(sample)

    print(f"\nTotal samples: {len(samples)}")

    type_counts = Counter(s["type"] for s in samples)
    subsystem_counts = Counter(s.get("subsystem", "unknown") for s in samples)

    print("\n  Sample type distribution:")
    for t, c in type_counts.most_common():
        print(f"    {t}: {c}")

    print("\n  Subsystem distribution:")
    for s, c in subsystem_counts.most_common():
        print(f"    {s}: {c}")

    random.shuffle(samples)
    split_idx = int(len(samples) * 0.9)
    train = samples[:split_idx]
    valid = samples[split_idx:]

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    with open(PROCESSED_DIR / "train.jsonl", "w") as f:
        for s in train:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    with open(PROCESSED_DIR / "valid.jsonl", "w") as f:
        for s in valid:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    stats = {
        "total_samples": len(samples),
        "train_samples": len(train),
        "valid_samples": len(valid),
        "type_distribution": dict(type_counts),
        "subsystem_distribution": dict(subsystem_counts),
        "total_functions": len(all_functions),
        "total_structs": len(all_structs),
    }

    with open(PROCESSED_DIR / "stats.json", "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {PROCESSED_DIR}:")
    print(f"  train.jsonl: {len(train)} samples")
    print(f"  valid.jsonl: {len(valid)} samples")
    print(f"  stats.json")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Linux Kernel training data preparation")
    parser.add_argument("--clone", action="store_true",
                        help="Clone Linux Kernel source first")
    parser.add_argument("--version", type=str, default="v6.6",
                        help="Kernel version to clone (default: v6.6)")
    parser.add_argument("--max-files", type=int, default=500,
                        help="Max source files to process (default: 500)")
    parser.add_argument("--kernel-dir", type=str, default=None,
                        help="Path to existing kernel source (skip clone)")
    args = parser.parse_args()

    if args.clone:
        clone_kernel(args.version)

    kernel_dir = Path(args.kernel_dir) if args.kernel_dir else (RAW_DIR / "linux")
    if not kernel_dir.exists():
        print(f"Kernel source not found at {kernel_dir}")
        print("Run with --clone to download it first, or specify --kernel-dir")
        sys.exit(1)

    process_kernel_source(kernel_dir, max_files=args.max_files)


if __name__ == "__main__":
    main()
