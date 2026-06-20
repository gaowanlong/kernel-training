#!/usr/bin/env python3
"""
v0.7 Data Extraction Pipeline — Mine kernel-doc and Documentation/ for high-quality Q&A.

Strategy (learned from v0.1-v0.6 failures):
1. Kernel-doc comments (/** */) are the gold standard — written by kernel devs.
2. Struct field comments provide detailed field-level explanations.
3. Documentation/ RST files provide subsystem-level knowledge.
4. Answers are derived from actual source docs, not templates.
5. Cover all 10 subsystems with both English and Chinese Q&A.
"""

import argparse
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

SUBSYSTEM_MAP = {
    "kernel": "process_management",
    "mm": "memory_management",
    "fs": "file_system",
    "net": "network_stack",
    "drivers": "device_drivers",
    "block": "file_system",
    "arch": "arch_security",
    "security": "arch_security",
    "ipc": "process_management",
    "init": "kernel_core",
    "lib": "kernel_core",
    "crypto": "kernel_core",
    "sound": "device_drivers",
    "virt": "kernel_core",
}

SUBSYSTEM_NAMES = {
    "process_management": "进程管理",
    "memory_management": "内存管理",
    "file_system": "文件系统",
    "network_stack": "网络协议栈",
    "device_drivers": "设备驱动",
    "interrupts": "中断处理",
    "locking": "锁机制",
    "system_calls": "系统调用",
    "debug": "调试与追踪",
    "arch_security": "架构与安全",
    "kernel_core": "内核核心",
}

random.seed(42)


# ============================================================
# Step 1: Extract kernel-doc blocks from header/source files
# ============================================================

def extract_kerneldoc_blocks(filepath: Path) -> list[dict]:
    """Extract /** ... */ kernel-doc blocks with their associated declarations."""
    try:
        content = filepath.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []

    blocks = []
    lines = content.split("\n")
    i = 0

    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("/**") and not line.startswith("/**/"):
            comment_lines = []
            j = i
            while j < len(lines):
                cl = lines[j]
                comment_lines.append(cl)
                if "*/" in cl and j > i:
                    break
                j += 1

            comment_text = "\n".join(comment_lines)

            # Look ahead for the declaration
            decl_start = j + 1
            while decl_start < len(lines) and (
                not lines[decl_start].strip()
                or lines[decl_start].strip().startswith("#")
                or lines[decl_start].strip().startswith("__")
                or lines[decl_start].strip().startswith("DEFINE_")
                or lines[decl_start].strip().startswith("EXPORT_SYMBOL")
            ):
                decl_start += 1

            if decl_start < len(lines):
                decl_line = lines[decl_start].strip()
                block_type = "unknown"
                name = ""

                # Struct
                struct_match = re.match(r"struct\s+(\w+)\s*\{", decl_line)
                if struct_match:
                    block_type = "struct"
                    name = struct_match.group(1)

                # Function
                if not name:
                    func_match = re.match(
                        r"(?:static\s+)?(?:inline\s+)?(?:__\w+\s+)*"
                        r"(?:const\s+)?(?:unsigned\s+)?(?:struct\s+)?"
                        r"([\w\s*]+?)\s+(\**\w+)\s*\(([^)]*)\)",
                        decl_line,
                    )
                    if func_match:
                        block_type = "function"
                        name = func_match.group(2).lstrip("*")

                # Macro
                if not name and decl_line.startswith("#define"):
                    macro_match = re.match(r"#define\s+(\w+)", decl_line)
                    if macro_match:
                        block_type = "macro"
                        name = macro_match.group(1)

                # Enum
                if not name:
                    enum_match = re.match(r"enum\s+(\w+)\s*\{", decl_line)
                    if enum_match:
                        block_type = "enum"
                        name = enum_match.group(1)

                if name and block_type != "unknown":
                    clean_comment = []
                    for cl in comment_lines:
                        cl = cl.strip()
                        cl = re.sub(r"^/\*\*\s*", "", cl)
                        cl = re.sub(r"\s*\*/\s*$", "", cl)
                        cl = re.sub(r"^\s*\*\s?", "", cl)
                        clean_comment.append(cl)

                    doc_text = "\n".join(clean_comment).strip()

                    body = ""
                    if block_type == "struct":
                        body_start = decl_start
                        brace_count = 0
                        body_end = decl_start
                        for k in range(decl_start, min(decl_start + 200, len(lines))):
                            brace_count += lines[k].count("{") - lines[k].count("}")
                            if brace_count == 0 and k > decl_start:
                                body_end = k + 1
                                break
                        body = "\n".join(lines[body_start:body_end])

                    blocks.append({
                        "type": block_type,
                        "name": name,
                        "doc": doc_text,
                        "body": body,
                        "file": str(filepath.relative_to(RAW_DIR / "linux")),
                        "declaration": decl_line,
                    })

            i = j + 1
        else:
            i += 1

    return blocks


def extract_struct_field_comments(body: str) -> list[dict]:
    """Extract individual field comments from a struct body."""
    fields = []
    lines = body.split("\n")
    prev_comment = ""
    for line in lines:
        stripped = line.strip()
        comment_match = re.search(r"/\*\s*(.+?)\s*\*/", stripped)
        if comment_match:
            prev_comment = comment_match.group(1)
        field_match = re.match(
            r"\s*([\w]+\s+[\w\s*]+)\s+(\w+)\s*;",
            stripped,
        )
        if field_match:
            field_type = field_match.group(1).strip()
            field_name = field_match.group(2)
            if field_name not in ("struct", "union", "enum", "unsigned", "const", "volatile"):
                fields.append({
                    "name": field_name,
                    "type": field_type,
                    "comment": prev_comment,
                })
                prev_comment = ""
    return fields


# ============================================================
# Step 2: Parse Documentation/ RST files
# ============================================================

def extract_doc_sections(doc_path: Path) -> list[dict]:
    """Extract titled sections from RST documentation files."""
    try:
        content = doc_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []

    sections = []
    lines = content.split("\n")
    i = 0

    while i < len(lines) - 2:
        line = lines[i].strip()
        if line and i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            if next_line and all(c in "=-~^\"'`*+#" for c in next_line) and len(next_line) >= 3:
                title = line
                body_lines = []
                j = i + 2
                while j < len(lines):
                    if j + 1 < len(lines):
                        nl = lines[j + 1].strip()
                        if nl and all(c in "=-~^\"'`*+#" for c in nl) and len(nl) >= 3:
                            break
                    body_lines.append(lines[j])
                    j += 1

                body = "\n".join(body_lines).strip()
                if len(body) > 200:
                    sections.append({
                        "title": title,
                        "body": body,
                        "file": str(doc_path.relative_to(RAW_DIR / "linux")),
                    })
                i = j
                continue
        i += 1

    return sections


def detect_subsystem_from_path(filepath: str) -> str:
    """Detect subsystem from file path."""
    top = filepath.split("/")[0]
    return SUBSYSTEM_MAP.get(top, "kernel_core")


def detect_subsystem_from_doc_path(filepath: str) -> str:
    """Detect subsystem from Documentation/ path."""
    parts = filepath.split("/")
    if len(parts) >= 2:
        doc_dir = parts[1]  # e.g. Documentation/scheduler/...
        mapping = {
            "scheduler": "process_management",
            "process": "process_management",
            "memory": "memory_management",
            "vm": "memory_management",
            "filesystems": "file_system",
            "fs": "file_system",
            "block": "file_system",
            "networking": "network_stack",
            "net": "network_stack",
            "driver": "device_drivers",
            "pci": "device_drivers",
            "usb": "device_drivers",
            "locking": "locking",
            "rcu": "locking",
            "interrupt": "interrupts",
            "irq": "interrupts",
            "security": "arch_security",
            "crypto": "arch_security",
            "trace": "debug",
            "debug": "debug",
            "admin-guide": "kernel_core",
            "core-api": "kernel_core",
            "kernel-hacking": "kernel_core",
        }
        return mapping.get(doc_dir, "kernel_core")
    return "kernel_core"


# ============================================================
# Step 3: Generate Q&A pairs from extracted data
# ============================================================

def generate_function_qa(block: dict, subsystem: str) -> Optional[dict]:
    """Generate Q&A from a kernel-doc function block."""
    doc = block["doc"]
    if len(doc) < 50:
        return None

    name = block["name"]
    file = block["file"]

    params = re.findall(r"@(\w+):\s*(.+?)(?=\n\s*@|\n\s*Return|\n\s*Context|\n\s*Note|\Z)", doc, re.DOTALL)
    return_match = re.search(r"Return:\s*(.+?)(?=\n\s*\n|\n\s*@|\Z)", doc, re.DOTALL)
    context_match = re.search(r"Context:\s*(.+?)(?=\n\s*\n|\Z)", doc, re.DOTALL)

    answer_parts = []
    first_para = doc.split("\n\n")[0] if "\n\n" in doc else doc.split("\n")[0]
    first_para = first_para.replace(f"{name}() - ", "").replace(f"{name} - ", "").strip()
    answer_parts.append(f"`{name}()` is defined in `{file}`. {first_para}")

    if params:
        answer_parts.append("\nParameters:")
        for pname, pdesc in params[:6]:
            pdesc_clean = " ".join(pdesc.split())[:200]
            answer_parts.append(f"- `{pname}`: {pdesc_clean}")

    if return_match:
        ret_text = " ".join(return_match.group(1).split())[:300]
        answer_parts.append(f"\nReturns: {ret_text}")

    if context_match:
        ctx_text = " ".join(context_match.group(1).split())[:200]
        answer_parts.append(f"\nContext: {ctx_text}")

    answer = "\n".join(answer_parts)

    questions = [
        f"What does the `{name}()` function do in the Linux kernel? Explain its purpose and parameters.",
        f"Explain the `{name}()` function in the Linux kernel. What are its key parameters and return value?",
    ]

    return {
        "messages": [
            {"role": "user", "content": random.choice(questions)},
            {"role": "assistant", "content": answer},
        ],
        "type": "function_qa",
        "subsystem": subsystem,
        "source_file": file,
        "source": "kernel_doc",
        "difficulty": "L2",
    }


def generate_struct_qa(block: dict, subsystem: str) -> Optional[dict]:
    """Generate Q&A from a kernel-doc struct block."""
    doc = block["doc"]
    if len(doc) < 50:
        return None

    name = block["name"]
    file = block["file"]

    first_para = doc.split("\n\n")[0] if "\n\n" in doc else doc.split("\n")[0]
    first_para = first_para.replace(f"struct {name} - ", "").replace(f"struct {name} \u2014 ", "").strip()

    field_descriptions = []
    if block["body"]:
        fields = extract_struct_field_comments(block["body"])
        for f in fields:
            if f["comment"]:
                field_descriptions.append(f"- `{f['name']}` ({f['type']}): {f['comment'][:150]}")

    answer = f"`struct {name}` is defined in `{file}`. {first_para}\n\n"

    if field_descriptions:
        answer += "Key fields:\n" + "\n".join(field_descriptions[:8])

    remaining = doc[len(first_para):].strip()
    if remaining and len(remaining) > 20:
        answer += f"\n\n{remaining[:500]}"

    questions = [
        f"What is the `struct {name}` in the Linux kernel? Describe its purpose and key fields.",
        f"Explain the `struct {name}` data structure in the Linux kernel. What information does it store?",
    ]

    return {
        "messages": [
            {"role": "user", "content": random.choice(questions)},
            {"role": "assistant", "content": answer},
        ],
        "type": "struct_qa",
        "subsystem": subsystem,
        "source_file": file,
        "source": "kernel_doc",
        "difficulty": "L2",
    }


def generate_doc_section_qa(section: dict, subsystem: str) -> Optional[dict]:
    """Generate Q&A from a Documentation/ RST section."""
    title = section["title"]
    body = section["body"]

    body_clean = re.sub(r"\.\.\s+\w+::.*", "", body)
    body_clean = re.sub(r":\w+:`([^`]+)`", r"\1", body_clean)
    body_clean = re.sub(r"``([^`]+)``", r"\1", body_clean)
    body_clean = re.sub(r"\*\*([^*]+)\*\*", r"\1", body_clean)
    body_clean = re.sub(r"\n{3,}", "\n\n", body_clean).strip()

    if len(body_clean) < 100:
        return None

    answer = body_clean[:800]

    questions = [
        f"Explain the following Linux kernel concept: {title}. What are the key points?",
        f"What does the Linux kernel documentation say about {title}? Summarize the key concepts.",
    ]

    return {
        "messages": [
            {"role": "user", "content": random.choice(questions)},
            {"role": "assistant", "content": answer},
        ],
        "type": "doc_qa",
        "subsystem": subsystem,
        "source_file": section["file"],
        "source": "kernel_documentation",
        "difficulty": "L1",
    }


def generate_chinese_qa(block: dict, subsystem: str) -> Optional[dict]:
    """Generate Chinese Q&A from a kernel-doc block."""
    doc = block["doc"]
    if len(doc) < 80:
        return None

    name = block["name"]
    file = block["file"]
    cn_subsystem = SUBSYSTEM_NAMES.get(subsystem, "内核")

    first_para = doc.split("\n\n")[0] if "\n\n" in doc else doc.split("\n")[0]
    first_para = first_para.replace(f"{name}() - ", "").replace(f"{name} - ", "").strip()

    answer = f"`{name}` 定义在 `{file}` 中，属于 Linux 内核的 {cn_subsystem} 子系统。\n\n"
    answer += f"{first_para}\n\n"

    params = re.findall(r"@(\w+):\s*(.+?)(?=\n\s*@|\n\s*Return|\n\s*Context|\n\s*Note|\Z)", doc, re.DOTALL)
    if params:
        answer += "参数说明：\n"
        for pname, pdesc in params[:5]:
            pdesc_clean = " ".join(pdesc.split())[:150]
            answer += f"- `{pname}`: {pdesc_clean}\n"

    return_match = re.search(r"Return:\s*(.+?)(?=\n\s*\n|\n\s*@|\Z)", doc, re.DOTALL)
    if return_match:
        ret_text = " ".join(return_match.group(1).split())[:200]
        answer += f"\n返回值: {ret_text}"

    questions = [
        f"请解释 Linux 内核中 `{name}` 的作用和实现原理。",
        f"Linux 内核的 `{name}` 函数/结构体是做什么的？请详细说明。",
    ]

    return {
        "messages": [
            {"role": "user", "content": random.choice(questions)},
            {"role": "assistant", "content": answer},
        ],
        "type": "chinese_qa",
        "subsystem": subsystem,
        "source_file": file,
        "source": "kernel_doc",
        "difficulty": "L2",
        "language": "zh",
    }


# ============================================================
# Step 4: Generate advanced/code-understanding Q&A
# ============================================================

def generate_advanced_qa(block: dict, subsystem: str) -> Optional[dict]:
    """Generate advanced L3 questions from kernel-doc with deep internals focus."""
    doc = block["doc"]
    if len(doc) < 120:
        return None

    name = block["name"]
    file = block["file"]

    # Look for keywords indicating advanced topics
    advanced_keywords = ["race", "atomic", "barrier", "lock", "RCU", "rcu",
                         "preempt", "interrupt", "context", "sleep", "deadlock",
                         "cache", "DMA", "mmap", "page fault", "TLB", "swap",
                         "namespace", "cgroup", "signal", "clone", "fork",
                         "COW", "copy-on-write", "OOM", "memory barrier"]

    is_advanced = any(kw.lower() in doc.lower() for kw in advanced_keywords)
    if not is_advanced:
        return None

    answer_parts = []
    first_para = doc.split("\n\n")[0] if "\n\n" in doc else doc.split("\n")[0]
    first_para = first_para.replace(f"{name}() - ", "").replace(f"{name} - ", "").strip()
    answer_parts.append(f"`{name}()` is defined in `{file}`. {first_para}")

    # Include more detail for advanced questions
    if len(doc) > 200:
        answer_parts.append(f"\n\n{doc[:600]}")

    answer = "\n".join(answer_parts)

    questions = [
        f"Explain the implementation details of `{name}()` in the Linux kernel. What concurrency or memory ordering considerations does it handle?",
        f"How does `{name}()` work internally in the Linux kernel? Discuss any locking, atomic operations, or context requirements.",
    ]

    return {
        "messages": [
            {"role": "user", "content": random.choice(questions)},
            {"role": "assistant", "content": answer},
        ],
        "type": "advanced_qa",
        "subsystem": subsystem,
        "source_file": file,
        "source": "kernel_doc",
        "difficulty": "L3",
    }


def generate_code_understanding_qa(block: dict, subsystem: str) -> Optional[dict]:
    """Generate code-understanding Q&A from function kernel-doc."""
    doc = block["doc"]
    if len(doc) < 80:
        return None

    name = block["name"]
    file = block["file"]

    answer_parts = []
    first_para = doc.split("\n\n")[0] if "\n\n" in doc else doc.split("\n")[0]
    first_para = first_para.replace(f"{name}() - ", "").replace(f"{name} - ", "").strip()
    answer_parts.append(f"`{name}()` is defined in `{file}`. {first_para}")

    params = re.findall(r"@(\w+):\s*(.+?)(?=\n\s*@|\n\s*Return|\n\s*Context|\n\s*Note|\Z)", doc, re.DOTALL)
    if params:
        answer_parts.append("\nParameters:")
        for pname, pdesc in params[:5]:
            pdesc_clean = " ".join(pdesc.split())[:150]
            answer_parts.append(f"- `{pname}`: {pdesc_clean}")

    return_match = re.search(r"Return:\s*(.+?)(?=\n\s*\n|\n\s*@|\Z)", doc, re.DOTALL)
    if return_match:
        ret_text = " ".join(return_match.group(1).split())[:200]
        answer_parts.append(f"\nReturns: {ret_text}")

    answer = "\n".join(answer_parts)

    questions = [
        f"What does the `{name}()` function do in the Linux kernel? What are the common GFP flags and when would you use each?",
        f"Explain the purpose and usage of `{name}()` in the Linux kernel. What are its parameters and return values?",
    ]

    return {
        "messages": [
            {"role": "user", "content": random.choice(questions)},
            {"role": "assistant", "content": answer},
        ],
        "type": "code_understanding",
        "subsystem": subsystem,
        "source_file": file,
        "source": "kernel_doc",
        "difficulty": "L2",
    }


# ============================================================
# Step 5: Main pipeline
# ============================================================

def collect_kerneldoc_files(kernel_dir: Path, max_per_subsystem: int = 30) -> list[Path]:
    """Collect key header and source files with kernel-doc across subsystems."""
    # Priority: include/ headers first (richest kernel-doc), then core C files
    priority_dirs = {
        "process_management": [
            "include/linux/sched.h", "include/linux/sched/signal.h",
            "include/linux/pid.h", "kernel/sched/core.c", "kernel/sched/fair.c",
            "kernel/fork.c", "kernel/exit.c", "kernel/signal.c",
        ],
        "memory_management": [
            "include/linux/mm.h", "include/linux/mm_types.h",
            "include/linux/slab.h", "include/linux/vmalloc.h",
            "include/linux/swap.h", "mm/mmap.c", "mm/page_alloc.c",
            "mm/slab.c", "mm/vmalloc.c", "mm/memory.c",
        ],
        "file_system": [
            "include/linux/fs.h", "include/linux/dcache.h",
            "include/linux/namei.h", "include/linux/file.h",
            "fs/namei.c", "fs/open.c", "fs/read_write.c",
            "fs/file_table.c", "fs/dcache.c", "fs/inode.c",
        ],
        "network_stack": [
            "include/linux/netdevice.h", "include/linux/skbuff.h",
            "include/net/sock.h", "include/linux/socket.h",
            "net/core/dev.c", "net/core/skbuff.c",
            "net/ipv4/tcp.c", "net/ipv4/ip_output.c",
        ],
        "device_drivers": [
            "include/linux/device.h", "include/linux/platform_device.h",
            "include/linux/pci.h", "include/linux/usb.h",
            "drivers/base/driver.c", "drivers/base/platform.c",
            "drivers/pci/pci.c",
        ],
        "interrupts": [
            "include/linux/interrupt.h", "include/linux/irq.h",
            "include/linux/irqdesc.h", "kernel/irq/handle.c",
            "kernel/irq/manage.c", "kernel/softirq.c",
        ],
        "locking": [
            "include/linux/spinlock.h", "include/linux/mutex.h",
            "include/linux/rwsem.h", "include/linux/rcupdate.h",
            "include/linux/completion.h", "kernel/locking/mutex.c",
            "kernel/locking/spinlock.c", "kernel/rcu/update.c",
        ],
        "system_calls": [
            "include/linux/syscalls.h", "kernel/sys.c",
            "fs/open.c", "fs/read_write.c", "mm/mmap.c",
        ],
        "debug": [
            "include/linux/printk.h", "include/linux/ftrace.h",
            "include/linux/kprobes.h", "kernel/trace/trace.c",
            "kernel/trace/trace_output.c",
        ],
        "arch_security": [
            "include/linux/security.h", "security/security.c",
            "include/linux/lsm_hooks.h",
        ],
    }

    files = []
    for subsystem, paths in priority_dirs.items():
        for p in paths:
            full = kernel_dir / p
            if full.exists():
                files.append(full)

    # Also scan include/linux/ for additional kernel-doc rich headers
    include_dir = kernel_dir / "include" / "linux"
    if include_dir.exists():
        extra = sorted(include_dir.glob("*.h"))
        # Take a sample from include/linux/
        random.shuffle(extra)
        files.extend(extra[:50])

    # Deduplicate
    seen = set()
    unique = []
    for f in files:
        if str(f) not in seen:
            seen.add(str(f))
            unique.append(f)

    return unique


def collect_doc_files(kernel_dir: Path, max_files: int = 100) -> list[Path]:
    """Collect key Documentation/ RST files."""
    doc_dir = kernel_dir / "Documentation"
    if not doc_dir.exists():
        return []

    # Priority doc directories
    priority_dirs = [
        "scheduler", "core-api", "filesystems", "networking",
        "driver-api", "locking", "RCU", "admin-guide", "mm",
        "security", "trace", "block", "cgroup-v1", "cgroup-v2",
        "power", "timers", "interrupt",
    ]

    files = []
    for pd in priority_dirs:
        subdir = doc_dir / pd
        if subdir.exists():
            for rst in subdir.rglob("*.rst"):
                files.append(rst)

    # If we need more, scan broadly
    if len(files) < max_files:
        all_rst = list(doc_dir.rglob("*.rst"))
        random.shuffle(all_rst)
        for rst in all_rst:
            if rst not in files:
                files.append(rst)
            if len(files) >= max_files:
                break

    return files[:max_files]


def main():
    parser = argparse.ArgumentParser(description="v0.7: Extract high-quality Q&A from kernel source docs")
    parser.add_argument("--kernel-dir", type=str, default=None,
                        help="Path to Linux kernel source")
    parser.add_argument("--max-samples", type=int, default=5000,
                        help="Max total Q&A samples to generate")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for train/valid JSONL")
    args = parser.parse_args()

    kernel_dir = Path(args.kernel_dir) if args.kernel_dir else (RAW_DIR / "linux")
    if not kernel_dir.exists():
        print(f"Kernel source not found at {kernel_dir}")
        print("Clone with: python scripts/prepare_data.py --clone --version v6.6")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else PROCESSED_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("v0.7 Data Extraction Pipeline")
    print("=" * 60)
    print(f"Kernel source: {kernel_dir}")
    print(f"Target samples: {args.max_samples}")
    print()

    all_samples = []

    # Phase 1: Extract kernel-doc from header/source files
    print("[Phase 1] Extracting kernel-doc blocks...")
    source_files = collect_kerneldoc_files(kernel_dir)
    print(f"  Processing {len(source_files)} source files...")

    total_blocks = 0
    func_blocks = []
    struct_blocks = []

    for i, fpath in enumerate(source_files):
        if i % 20 == 0:
            print(f"  [{i}/{len(source_files)}] {fpath.relative_to(kernel_dir)}")

        blocks = extract_kerneldoc_blocks(fpath)
        total_blocks += len(blocks)

        subsystem = detect_subsystem_from_path(
            str(fpath.relative_to(kernel_dir))
        )

        for block in blocks:
            if block["type"] == "function":
                func_blocks.append((block, subsystem))
            elif block["type"] == "struct":
                struct_blocks.append((block, subsystem))

    print(f"  Total kernel-doc blocks: {total_blocks}")
    print(f"  Functions: {len(func_blocks)}")
    print(f"  Structs: {len(struct_blocks)}")

    # Phase 2: Generate Q&A from kernel-doc
    print()
    print("[Phase 2] Generating Q&A from kernel-doc...")

    # Generate function Q&A (target: ~1500)
    random.shuffle(func_blocks)
    for block, subsystem in func_blocks[:2000]:
        qa = generate_function_qa(block, subsystem)
        if qa:
            all_samples.append(qa)

    print(f"  Function Q&A: {len([s for s in all_samples if s['type'] == 'function_qa'])}")

    # Generate struct Q&A (target: ~500)
    random.shuffle(struct_blocks)
    for block, subsystem in struct_blocks[:800]:
        qa = generate_struct_qa(block, subsystem)
        if qa:
            all_samples.append(qa)

    print(f"  Struct Q&A: {len([s for s in all_samples if s['type'] == 'struct_qa'])}")

    # Generate advanced Q&A (target: ~500)
    random.shuffle(func_blocks)
    for block, subsystem in func_blocks[:2000]:
        qa = generate_advanced_qa(block, subsystem)
        if qa:
            all_samples.append(qa)

    print(f"  Advanced Q&A: {len([s for s in all_samples if s['type'] == 'advanced_qa'])}")

    # Generate code understanding Q&A (target: ~500)
    random.shuffle(func_blocks)
    for block, subsystem in func_blocks[:2000]:
        qa = generate_code_understanding_qa(block, subsystem)
        if qa:
            all_samples.append(qa)

    print(f"  Code Understanding Q&A: {len([s for s in all_samples if s['type'] == 'code_understanding'])}")

    # Generate Chinese Q&A (target: ~800)
    all_blocks = func_blocks + struct_blocks
    random.shuffle(all_blocks)
    for block, subsystem in all_blocks[:1500]:
        qa = generate_chinese_qa(block, subsystem)
        if qa:
            all_samples.append(qa)

    print(f"  Chinese Q&A: {len([s for s in all_samples if s['type'] == 'chinese_qa'])}")

    # Phase 3: Extract from Documentation/
    print()
    print("[Phase 3] Extracting from Documentation/...")
    doc_files = collect_doc_files(kernel_dir, max_files=80)
    print(f"  Processing {len(doc_files)} doc files...")

    doc_sections = []
    for i, fpath in enumerate(doc_files):
        if i % 20 == 0:
            print(f"  [{i}/{len(doc_files)}] {fpath.relative_to(kernel_dir)}")
        sections = extract_doc_sections(fpath)
        subsystem = detect_subsystem_from_doc_path(
            str(fpath.relative_to(kernel_dir))
        )
        for sec in sections:
            doc_sections.append((sec, subsystem))

    print(f"  Total doc sections: {len(doc_sections)}")

    # Generate doc Q&A (target: ~500)
    random.shuffle(doc_sections)
    for section, subsystem in doc_sections[:800]:
        qa = generate_doc_section_qa(section, subsystem)
        if qa:
            all_samples.append(qa)

    print(f"  Doc Q&A: {len([s for s in all_samples if s['type'] == 'doc_qa'])}")


    # Phase 4: Balance, shuffle, split
    print()
    print("[Phase 4] Balancing and splitting...")

    # Limit total samples
    if len(all_samples) > args.max_samples:
        random.shuffle(all_samples)
        all_samples = all_samples[:args.max_samples]

    random.shuffle(all_samples)
    split_idx = int(len(all_samples) * 0.9)
    train = all_samples[:split_idx]
    valid = all_samples[split_idx:]

    # Save
    train_path = output_dir / "train.jsonl"
    valid_path = output_dir / "valid.jsonl"

    with open(train_path, "w") as f:
        for s in train:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    with open(valid_path, "w") as f:
        for s in valid:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Stats
    print()
    print("=" * 60)
    print("Dataset Summary")
    print("=" * 60)
    print(f"Total samples: {len(all_samples)}")
    print(f"Train: {len(train)}")
    print(f"Valid: {len(valid)}")

    type_counts = Counter(s["type"] for s in all_samples)
    print(f"\nType distribution:")
    for t, c in type_counts.most_common():
        print(f"  {t}: {c}")

    subsystem_counts = Counter(s.get("subsystem", "unknown") for s in all_samples)
    print(f"\nSubsystem distribution:")
    for s, c in subsystem_counts.most_common():
        print(f"  {s}: {c}")

    source_counts = Counter(s.get("source", "unknown") for s in all_samples)
    print(f"\nSource distribution:")
    for s, c in source_counts.most_common():
        print(f"  {s}: {c}")

    diff_counts = Counter(s.get("difficulty", "unknown") for s in all_samples)
    print(f"\nDifficulty distribution:")
    for d, c in diff_counts.most_common():
        print(f"  {d}: {c}")

    # Count Chinese samples
    zh_count = len([s for s in all_samples if s.get("language") == "zh"])
    print(f"\nChinese samples: {zh_count}")
    print(f"English samples: {len(all_samples) - zh_count}")

    print(f"\nSaved to:")
    print(f"  {train_path}")
    print(f"  {valid_path}")

    # Show a few samples
    print(f"\nSample previews:")
    for i, s in enumerate(all_samples[:3]):
        q = s["messages"][0]["content"][:100]
        a = s["messages"][1]["content"][:100]
        print(f"\n  [{s['type']}] Q: {q}...")
        print(f"  A: {a}...")


if __name__ == "__main__":
    main()
