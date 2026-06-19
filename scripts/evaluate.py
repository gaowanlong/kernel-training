#!/usr/bin/env python3
"""
Evaluation framework for comparing base model vs fine-tuned model.

This script runs a comprehensive set of Linux Kernel knowledge tests
on both the base model and the LoRA-fine-tuned model, producing a
side-by-side comparison report.

Test categories:
1. Kernel Concepts - basic kernel knowledge questions
2. Code Understanding - explain specific kernel functions
3. Code Completion - complete kernel code snippets
4. Chinese Knowledge - kernel concepts in Chinese
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress

console = Console()
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"


# Comprehensive Linux Kernel test cases

# Comprehensive Linux Kernel test cases — v0.2
# Categories:
#   L1 (Basic):  General OS/kernel concepts any engineer should know
#   L2 (Intermediate): Linux-specific kernel mechanisms
#   L3 (Advanced):  Deep kernel internals, code-level understanding
#   Code:          Kernel code understanding & completion
#   Chinese:       Kernel concepts in Chinese

TEST_CASES = [
    # ===== L1: Basic Kernel Concepts (8 questions) =====
    {
        "id": "l1_01",
        "category": "basic_concepts",
        "difficulty": "L1",
        "question": "What is the Linux kernel? Explain its role in an operating system.",
        "reference_keywords": ["monolithic", "process", "memory", "hardware", "system calls"],
    },
    {
        "id": "l1_02",
        "category": "basic_concepts",
        "difficulty": "L1",
        "question": "Explain the difference between user space and kernel space in Linux.",
        "reference_keywords": ["ring 0", "ring 3", "system call", "protection", "privilege"],
    },
    {
        "id": "l1_03",
        "category": "basic_concepts",
        "difficulty": "L1",
        "question": "What is a system call? Give three examples of common Linux system calls and explain what they do.",
        "reference_keywords": ["fork", "exec", "open", "read", "write", "mmap"],
    },
    {
        "id": "l1_04",
        "category": "basic_concepts",
        "difficulty": "L1",
        "question": "What is a process? How does it differ from a thread in the Linux kernel?",
        "reference_keywords": ["task_struct", "PID", "TGID", "clone", "shared address space"],
    },
    {
        "id": "l1_05",
        "category": "basic_concepts",
        "difficulty": "L1",
        "question": "What is virtual memory? Why does the Linux kernel use it?",
        "reference_keywords": ["MMU", "page table", "TLB", "address space", "isolation"],
    },
    {
        "id": "l1_06",
        "category": "basic_concepts",
        "difficulty": "L1",
        "question": "What is a device driver in Linux? How does it interact with the kernel?",
        "reference_keywords": ["module", "probe", "file_operations", "platform_driver", "init"],
    },
    {
        "id": "l1_07",
        "category": "basic_concepts",
        "difficulty": "L1",
        "question": "What is a kernel panic? What causes it and how is it handled?",
        "reference_keywords": ["oops", "BUG", "die", "reboot", "kdump"],
    },
    {
        "id": "l1_08",
        "category": "basic_concepts",
        "difficulty": "L1",
        "question": "What is the Linux kernel's role in file systems? Explain the Virtual File System (VFS) layer.",
        "reference_keywords": ["inode", "dentry", "super_block", "file", "vfsmount"],
    },

    # ===== L2: Intermediate Kernel Mechanisms (8 questions) =====
    {
        "id": "l2_01",
        "category": "kernel_mechanisms",
        "difficulty": "L2",
        "question": "What is a page fault? Describe the steps the kernel takes when a page fault occurs.",
        "reference_keywords": ["MMU", "page table", "swap", "demand paging", "major fault", "minor fault"],
    },
    {
        "id": "l2_02",
        "category": "kernel_mechanisms",
        "difficulty": "L2",
        "question": "What is the purpose of the task_struct in the Linux kernel? What key information does it contain?",
        "reference_keywords": ["PID", "state", "priority", "mm_struct", "files_struct", "signal"],
    },
    {
        "id": "l2_03",
        "category": "kernel_mechanisms",
        "difficulty": "L2",
        "question": "Explain how Linux handles interrupts. What is the difference between top halves and bottom halves?",
        "reference_keywords": ["IRQ", "softirq", "tasklet", "workqueue", "handler", "request_irq"],
    },
    {
        "id": "l2_04",
        "category": "kernel_mechanisms",
        "difficulty": "L2",
        "question": "What is a spinlock? When should you use a spinlock vs a mutex in kernel code?",
        "reference_keywords": ["spin", "sleep", "atomic", "SMP", "preemption", "context"],
    },
    {
        "id": "l2_05",
        "category": "kernel_mechanisms",
        "difficulty": "L2",
        "question": "Describe the Linux kernel's scheduler (CFS). How does it decide which process to run next?",
        "reference_keywords": ["CFS", "vruntime", "red-black tree", "nice", "time slice", "fairness"],
    },
    {
        "id": "l2_06",
        "category": "kernel_mechanisms",
        "difficulty": "L2",
        "question": "How does the Linux kernel manage physical memory? Explain the buddy allocator and slab allocator.",
        "reference_keywords": ["page", "order", "buddy", "kmalloc", "kmem_cache", "fragmentation"],
    },
    {
        "id": "l2_07",
        "category": "kernel_mechanisms",
        "difficulty": "L2",
        "question": "What is RCU (Read-Copy-Update)? Explain its use case in the Linux kernel.",
        "reference_keywords": ["lock-free", "grace period", "synchronize_rcu", "call_rcu", "read-side"],
    },
    {
        "id": "l2_08",
        "category": "kernel_mechanisms",
        "difficulty": "L2",
        "question": "Explain the Linux kernel's approach to concurrency. What mechanisms exist beyond spinlocks and mutexes?",
        "reference_keywords": ["atomic", "RCU", "semaphore", "completion", "percpu", "seqcount"],

    },

    # ===== L3: Advanced Kernel Internals (6 questions) =====
    {
        "id": "l3_01",
        "category": "advanced_internals",
        "difficulty": "L3",
        "question": "Explain how copy_from_user() and copy_to_user() work. Why can't the kernel just dereference user-space pointers directly?",
        "reference_keywords": ["access_ok", "page fault", "probe_kernel", "SMAP", "SMEP", "KASLR"],
    },
    {
        "id": "l3_02",
        "category": "advanced_internals",
        "difficulty": "L3",
        "question": "What is the Linux kernel's approach to memory barriers? When would you use smp_mb(), smp_wmb(), or smp_rmb()?",
        "reference_keywords": ["ordering", "store buffer", "cache coherency", "acquire", "release", "DMA"],
    },
    {
        "id": "l3_03",
        "category": "advanced_internals",
        "difficulty": "L3",
        "question": "Explain the Linux kernel's approach to device driver model. What are the roles of struct device, struct bus_type, and struct device_driver?",
        "reference_keywords": ["probe", "remove", "match", "uevent", "sysfs", "devicetree"],
    },
    {
        "id": "l3_04",
        "category": "advanced_internals",
        "difficulty": "L3",
        "question": "How does the Linux kernel handle out-of-memory (OOM) situations? Explain the OOM killer.",
        "reference_keywords": ["oom_score", "badness", "overcommit", "panic_on_oom", "cgroup"],
    },
    {
        "id": "l3_05",
        "category": "advanced_internals",
        "difficulty": "L3",
        "question": "Explain the Linux kernel's approach to tracing and debugging. What are ftrace, perf, and eBPF?",
        "reference_keywords": ["tracepoint", "kprobe", "uprobe", "ring buffer", "bpftrace"],
    },
    {
        "id": "l3_06",
        "category": "advanced_internals",
        "difficulty": "L3",
        "question": "How does the Linux kernel handle namespaces and cgroups? What role do they play in containerization?",
        "reference_keywords": ["pid_namespace", "net_namespace", "cgroupfs", "resource limit", "isolation"],
    },

    # ===== Code Understanding (6 questions) =====
    {
        "id": "code_01",
        "category": "code_understanding",
        "difficulty": "L2",
        "question": "What does the kmalloc() function do in the Linux kernel? What are the common GFP flags and when would you use each?",
        "reference_keywords": ["allocate", "GFP_KERNEL", "GFP_ATOMIC", "GFP_DMA", "slab", "bytes"],
    },
    {
        "id": "code_02",
        "category": "code_understanding",
        "difficulty": "L2",
        "question": "What does the schedule() function do in the Linux kernel? When is it called?",
        "reference_keywords": ["context switch", "runqueue", "preempt", "yield", "TASK_RUNNING", "scheduler"],
    },
    {
        "id": "code_03",
        "category": "code_understanding",
        "difficulty": "L2",
        "question": "Explain the Linux kernel module init and exit mechanism. What are module_init() and module_exit()?",
        "reference_keywords": ["insmod", "rmmod", "MODULE_LICENSE", "__init", "__exit", "GPL"],
    },
    {
        "id": "code_04",
        "category": "code_understanding",
        "difficulty": "L2",
        "question": "What is the purpose of printk() in the kernel? How does it differ from printf()? What are the log levels?",
        "reference_keywords": ["KERN_EMERG", "KERN_INFO", "dmesg", "console", "ring buffer", "log level"],
    },
    {
        "id": "code_05",
        "category": "code_understanding",
        "difficulty": "L3",
        "question": "Explain the Linux kernel's linked list implementation (struct list_head). How does container_of() work?",
        "reference_keywords": ["list_add", "list_for_each", "offsetof", "type safety", "embedded"],
    },
    {
        "id": "code_06",
        "category": "code_understanding",
        "difficulty": "L3",
        "question": "What is a waitqueue in the Linux kernel? How do wait_event_interruptible() and wake_up() work?",
        "reference_keywords": ["wait_queue_head", "TASK_INTERRUPTIBLE", "condition", "blocking", "wake_up"],
    },

    # ===== Chinese Knowledge (6 questions) =====
    {
        "id": "cn_01",
        "category": "chinese_knowledge",
        "difficulty": "L1",
        "question": "请解释Linux内核中的进程调度器(scheduler)是如何工作的？CFS调度器的主要特点是什么？",
        "reference_keywords": ["CFS", "vruntime", "红黑树", "nice", "时间片", "公平"],
    },
    {
        "id": "cn_02",
        "category": "chinese_knowledge",
        "difficulty": "L1",
        "question": "什么是内核态和用户态？为什么需要区分这两种状态？",
        "reference_keywords": ["特权级", "系统调用", "保护", "Ring 0", "Ring 3", "安全"],
    },
    {
        "id": "cn_03",
        "category": "chinese_knowledge",
        "difficulty": "L2",
        "question": "请解释Linux中的虚拟内存管理机制，包括页表、TLB和缺页异常处理。",
        "reference_keywords": ["页表", "TLB", "缺页", "MMU", "交换", "多级页表"],
    },
    {
        "id": "cn_04",
        "category": "chinese_knowledge",
        "difficulty": "L2",
        "question": "Linux内核中的并发控制机制有哪些？请比较自旋锁、互斥锁和信号量的使用场景。",
        "reference_keywords": ["自旋锁", "互斥锁", "信号量", "RCU", "原子操作", "上下文"],
    },
    {
        "id": "cn_05",
        "category": "chinese_knowledge",
        "difficulty": "L2",
        "question": "请描述Linux网络协议栈的基本架构，从网卡接收到数据到应用层经历了哪些步骤？",
        "reference_keywords": ["NAPI", "sk_buff", "netfilter", "socket", "TCP/IP", "协议栈"],
    },
    {
        "id": "cn_06",
        "category": "chinese_knowledge",
        "difficulty": "L3",
        "question": "请解释Linux内核的OOM killer机制。它是如何选择要杀死的进程的？",
        "reference_keywords": ["OOM", "badness", "oom_score", "overcommit", "cgroup", "panic"],
    },
]

# Code completion test cases
CODE_COMPLETION_TESTS = [
    {
        "id": "complete_01",
        "category": "code_completion",
        "difficulty": "L1",
        "prompt": """Complete the following Linux kernel module initialization function:

```c
#include <linux/module.h>
#include <linux/kernel.h>

static int __init my_module_init(void)
{
    printk(KERN_INFO "My module loaded\\n");
""",
        "reference_keywords": ["return 0", "module_init", "module_exit", "MODULE_LICENSE", "GPL"],
    },
    {
        "id": "complete_02",
        "category": "code_completion",
        "difficulty": "L1",
        "prompt": """Complete the following Linux kernel function that allocates memory:

```c
void *allocate_buffer(size_t size)
{
    void *buf;
""",
        "reference_keywords": ["kmalloc", "GFP_KERNEL", "kfree", "NULL", "return"],
    },
    {
        "id": "complete_03",
        "category": "code_completion",
        "difficulty": "L2",
        "prompt": """Complete the following character device driver read function:

```c
static ssize_t my_read(struct file *filp, char __user *buf, size_t count, loff_t *f_pos)
{
    char *kernel_buf = "Hello from kernel\\n";
    size_t len = strlen(kernel_buf);
""",
        "reference_keywords": ["copy_to_user", "return", "bytes", "f_pos", "EFAULT"],
    },
    {
        "id": "complete_04",
        "category": "code_completion",
        "difficulty": "L2",
        "prompt": """Complete the following Linux kernel function that uses a spinlock:

```c
#include <linux/spinlock.h>

static DEFINE_SPINLOCK(my_lock);
static int shared_counter = 0;

void increment_counter(void)
{
""",
        "reference_keywords": ["spin_lock", "spin_unlock", "shared_counter", "irqsave", "flags"],
    },
    {
        "id": "complete_05",
        "category": "code_completion",
        "difficulty": "L3",
        "prompt": """Complete the following Linux kernel function that registers a platform driver:

```c
#include <linux/platform_device.h>

static struct platform_driver my_driver = {
    .probe = my_probe,
    .remove = my_remove,
    .driver = {
        .name = "my_device",
        .of_match_table = my_of_match,
    },
};

static int __init my_driver_init(void)
{
""",
        "reference_keywords": ["platform_driver_register", "return", "module_platform_driver", "module_init"],
    },
]


def run_evaluation(model, tokenizer, test_cases, code_tests, model_name: str, judge_model=None, judge_tokenizer=None) -> list[dict]:
    """Run all test cases and return results."""
    # Validate no duplicate IDs
    all_ids = [t["id"] for t in test_cases] + [t["id"] for t in code_tests]
    from collections import Counter
    dupes = {k:v for k,v in Counter(all_ids).items() if v > 1}
    if dupes:
        console.print(f"[yellow]Warning: {len(dupes)} duplicate test IDs found![/yellow]")
        console.print(f"  Duplicates: {list(dupes.keys())}")
    
    results = []
    total = len(test_cases) + len(code_tests)

    console.print(f"\n[bold]Running {total} tests for {model_name}...[/bold]\n")

    for i, test in enumerate(test_cases):
        console.print(f"  [{i+1}/{total}] {test['id']}: {test['question'][:60]}...")
        try:
            start_time = time.time()
            response = generate(
                model, tokenizer,
                prompt=test["question"],
                max_tokens=300,
                sampler=make_sampler(temp=0.7),
            )
            elapsed = time.time() - start_time

            # LLM-as-judge scoring (v0.3)
            # Use the model itself to evaluate answer quality semantically
            jm = judge_model or model
            jt = judge_tokenizer or tokenizer
            judge_prompt = (
                f"You are an expert Linux kernel evaluator. "
                f"Rate the following answer on a scale of 0-10 based on correctness, completeness, and precision.\n\n"
                f"Question: {test['question']}\n\n"
                f"Answer: {response[:1000]}\n\n"
                f"Reference keywords that should be mentioned: {', '.join(test['reference_keywords'])}\n\n"
                f"Output ONLY a number 0-10, nothing else."
            )
            try:
                judge_response = generate(
                    jm, jt,
                    prompt=judge_prompt,
                    max_tokens=10,
                    sampler=make_sampler(temp=0.1),
                )
                # Parse the numeric score
                import re
                score_match = re.search(r'\b(\d+)(?:/10)?\b', judge_response.strip())
                if score_match:
                    judge_score = int(score_match.group(1))
                    judge_score = max(0, min(10, judge_score))
                else:
                    judge_score = 5  # default
            except Exception:
                judge_score = 5

            normalized_score = judge_score / 10.0
            found_keywords = [kw for kw in test["reference_keywords"] if kw.lower() in response.lower()]

            results.append({
                "id": test["id"],
                "category": test["category"],
                "question": test["question"],
                "response": response,
                "score": normalized_score,
                "keywords_matched": found_keywords,
                "keywords_total": test["reference_keywords"],
                "elapsed_sec": elapsed,
            })

            console.print(f"    Score: {normalized_score:.0%} | {elapsed:.1f}s")

        except Exception as e:
            console.print(f"    [red]Error: {e}[/red]")
            results.append({
                "id": test["id"],
                "category": test["category"],
                "question": test["question"],
                "response": f"ERROR: {e}",
                "score": 0,
                "keywords_matched": [],
                "keywords_total": test["reference_keywords"],
                "elapsed_sec": 0,
            })

    # Code completion tests
    for test in code_tests:
        console.print(f"  [{len(results)+1}/{total}] {test['id']}: code completion...")
        try:
            start_time = time.time()
            response = generate(
                model, tokenizer,
                prompt=test["prompt"],
                max_tokens=200,
                sampler=make_sampler(temp=0.3),
            )
            elapsed = time.time() - start_time

            # LLM-as-judge for code completion
            jm = judge_model or model
            jt = judge_tokenizer or tokenizer
            judge_prompt = (
                f"You are an expert Linux kernel C programmer. "
                f"Rate the following code completion on a scale of 0-10 based on correctness and completeness.\n\n"
                f"Code context:\n{test['prompt'][:800]}\n\n"
                f"Completion:\n{response[:800]}\n\n"
                f"Output ONLY a number 0-10, nothing else."
            )
            try:
                judge_response = generate(
                    jm, jt,
                    prompt=judge_prompt,
                    max_tokens=10,
                    sampler=make_sampler(temp=0.1),
                )
                import re
                score_match = re.search(r'\b(\d+)(?:/10)?\b', judge_response.strip())
                judge_score = int(score_match.group(1)) if score_match else 5
                judge_score = max(0, min(10, judge_score))
            except Exception:
                judge_score = 5

            normalized_score = judge_score / 10.0
            found_keywords = [kw for kw in test["reference_keywords"] if kw.lower() in response.lower()]

            results.append({
                "id": test["id"],
                "category": test["category"],
                "question": test["prompt"][:100],
                "response": response,
                "score": normalized_score,
                "keywords_matched": found_keywords,
                "keywords_total": test["reference_keywords"],
                "elapsed_sec": elapsed,
            })

            console.print(f"    Score: {normalized_score:.0%} | {elapsed:.1f}s")

        except Exception as e:
            console.print(f"    [red]Error: {e}[/red]")
            results.append({
                "id": test["id"],
                "category": test["category"],
                "question": test["prompt"][:100],
                "response": f"ERROR: {e}",
                "score": 0,
                "keywords_matched": [],
                "keywords_total": test["reference_keywords"],
                "elapsed_sec": 0,
            })

    return results


def compute_statistics(results: list[dict]) -> dict:
    """Compute aggregate statistics from results."""
    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"scores": [], "times": []}
        categories[cat]["scores"].append(r["score"])
        categories[cat]["times"].append(r["elapsed_sec"])

    stats = {
        "overall_avg_score": sum(r["score"] for r in results) / len(results) if results else 0,
        "total_tests": len(results),
        "total_time_sec": sum(r["elapsed_sec"] for r in results),
    }

    for cat, data in categories.items():
        stats[cat] = {
            "avg_score": sum(data["scores"]) / len(data["scores"]) if data["scores"] else 0,
            "avg_time": sum(data["times"]) / len(data["times"]) if data["times"] else 0,
            "count": len(data["scores"]),
        }

    return stats


def print_comparison(base_results: list[dict], lora_results: list[dict],
                     base_stats: dict, lora_stats: dict):
    """Print a side-by-side comparison."""
    console.rule("[bold green]Comparison: Base Model vs Fine-tuned Model[/bold green]")
    console.print()

    # Overall comparison table
    table = Table(title="Overall Performance Comparison")
    table.add_column("Metric", style="cyan")
    table.add_column("Base Model", style="yellow")
    table.add_column("Fine-tuned", style="green")
    table.add_column("Delta", style="bold")

    base_avg = base_stats["overall_avg_score"]
    lora_avg = lora_stats["overall_avg_score"]
    delta = lora_avg - base_avg

    delta_style = "[green]" if delta > 0 else "[red]"
    table.add_row(
        "Avg Score (all tests)",
        f"{base_avg:.1%}",
        f"{lora_avg:.1%}",
        f"{delta_style}{delta:+.1%}[/{delta_style.split('[')[1]}",
    )

    table.add_row(
        "Total Time",
        f"{base_stats['total_time_sec']:.1f}s",
        f"{lora_stats['total_time_sec']:.1f}s",
        f"{lora_stats['total_time_sec'] - base_stats['total_time_sec']:+.1f}s",
    )

    console.print(table)
    console.print()

    # Per-category comparison
    cat_table = Table(title="Per-Category Comparison")
    cat_table.add_column("Category", style="cyan")
    cat_table.add_column("Base", style="yellow")
    cat_table.add_column("Fine-tuned", style="green")
    cat_table.add_column("Delta", style="bold")

    all_categories = set()
    for k in base_stats:
        if k not in ("overall_avg_score", "total_tests", "total_time_sec"):
            all_categories.add(k)
    for k in lora_stats:
        if k not in ("overall_avg_score", "total_tests", "total_time_sec"):
            all_categories.add(k)

    for cat in sorted(all_categories):
        base_cat = base_stats.get(cat, {"avg_score": 0})
        lora_cat = lora_stats.get(cat, {"avg_score": 0})
        base_s = base_cat["avg_score"]
        lora_s = lora_cat["avg_score"]
        d = lora_s - base_s
        ds = "[green]" if d > 0 else "[red]" if d < 0 else "[dim]"
        cat_table.add_row(
            cat.replace("_", " ").title(),
            f"{base_s:.1%}",
            f"{lora_s:.1%}",
            f"{ds}{d:+.1%}[/{ds.split('[')[1]}",
        )

    console.print(cat_table)
    console.print()

    # Per-question comparison
    console.print("[bold]Per-Question Score Comparison:[/bold]")
    q_table = Table()
    q_table.add_column("ID", style="dim")
    q_table.add_column("Question", style="cyan", max_width=40)
    q_table.add_column("Base", style="yellow")
    q_table.add_column("FT", style="green")
    q_table.add_column("Delta", style="bold")

    for br, lr in zip(base_results, lora_results):
        d = lr["score"] - br["score"]
        ds = "[green]" if d > 0 else "[red]" if d < 0 else "[dim]"
        q_table.add_row(
            br["id"],
            br["question"][:40],
            f"{br['score']:.0%}",
            f"{lr['score']:.0%}",
            f"{ds}{d:+.0%}[/{ds.split('[')[1]}",
        )

    console.print(q_table)


def save_results(base_results, lora_results, base_stats, lora_stats, output_dir: Path):
    """Save all results to JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    report = {
        "timestamp": timestamp,
        "base_model": {
            "results": base_results,
            "statistics": base_stats,
        },
        "fine_tuned_model": {
            "results": lora_results,
            "statistics": lora_stats,
        },
        "comparison": {
            "overall_delta": lora_stats["overall_avg_score"] - base_stats["overall_avg_score"],
        },
    }

    report_path = output_dir / f"eval_report_{timestamp}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    console.print(f"\n[green]Report saved to: {report_path}[/green]")

    # Dedup check: warn if duplicate IDs found
    base_ids = [r["id"] for r in base_results]
    from collections import Counter
    dupes = {k:v for k,v in Counter(base_ids).items() if v > 1}
    if dupes:
        console.print(f"[yellow]Warning: {len(dupes)} duplicate question IDs found![/yellow]")
        console.print(f"  Duplicates: {list(dupes.keys())[:5]}...")

    # Also save a human-readable summary
    summary_path = output_dir / f"eval_summary_{timestamp}.txt"
    with open(summary_path, "w") as f:
        f.write(f"Linux Kernel Knowledge Evaluation Report\\n")
        f.write(f"{'='*60}\\n")
        f.write(f"Timestamp: {timestamp}\\n\\n")
        f.write(f"Overall Scores:\\n")
        f.write(f"  Base Model:     {base_stats['overall_avg_score']:.1%}\\n")
        f.write(f"  Fine-tuned:     {lora_stats['overall_avg_score']:.1%}\\n")
        f.write(f"  Improvement:    {lora_stats['overall_avg_score'] - base_stats['overall_avg_score']:+.1%}\\n\\n")
        f.write(f"Per-Question Results:\\n")
        f.write(f"{'-'*60}\\n")
        for br, lr in zip(base_results, lora_results):
            f.write(f"\\n[{br['id']}] {br['question'][:80]}\\n")
            f.write(f"  Base: {br['score']:.0%} | FT: {lr['score']:.0%} | Delta: {lr['score']-br['score']:+.0%}\\n")

    console.print(f"[green]Summary saved to: {summary_path}[/green]")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Linux Kernel knowledge before/after fine-tuning")
    parser.add_argument("--model", type=str, default="models/qwen2.5-7b",
                        help="Path to base model")
    parser.add_argument("--adapter", type=str, default=None,
                        help="Path to LoRA adapter (if None, only evaluate base model)")
    parser.add_argument("--output", type=str, default="results",
                        help="Output directory for reports")
    parser.add_argument("--base-only", action="store_true",
                        help="Only evaluate base model (skip fine-tuned)")
    args = parser.parse_args()

    model_path = PROJECT_ROOT / args.model
    output_dir = PROJECT_ROOT / args.output

    console.rule("[bold]Linux Kernel Knowledge Evaluation[/bold]")
    console.print(f"  Model: {model_path}")
    console.print(f"  Adapter: {args.adapter or 'None (base model only)'}")
    console.print()

    # Load base model
    console.print("[bold]Loading base model...[/bold]")
    model, tokenizer = load(str(model_path))
    console.print("  [green]Base model loaded[/green]")

    # Evaluate base model
    console.rule("[bold yellow]Evaluating Base Model[/bold yellow]")
    base_results = run_evaluation(model, tokenizer, TEST_CASES, CODE_COMPLETION_TESTS, "Base Model", judge_model=model, judge_tokenizer=tokenizer)
    base_stats = compute_statistics(base_results)

    console.print(f"\\n[bold yellow]Base Model Overall Score: {base_stats['overall_avg_score']:.1%}[/bold yellow]")

    if args.base_only or not args.adapter:
        # Save base-only results
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report = {
            "timestamp": timestamp,
            "base_model": {
                "results": base_results,
                "statistics": base_stats,
            },
        }
        report_path = output_dir / f"base_eval_{timestamp}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        console.print(f"\\n[green]Base evaluation saved to: {report_path}[/green]")
        return

    # Load fine-tuned model with adapter
    console.print("\\n[bold]Loading fine-tuned model with LoRA adapter...[/bold]")
    adapter_path = PROJECT_ROOT / args.adapter
    if not adapter_path.exists() or not (adapter_path / "adapters.safetensors").exists():
        console.print(f"[red]Adapter not found at {adapter_path}[/red]")
        console.print("Skipping fine-tuned evaluation.")
        return

    model_ft, tokenizer_ft = load(str(model_path), adapter_path=str(adapter_path))  # adapter_path is the directory
    console.print("  [green]Fine-tuned model loaded[/green]")

    # Evaluate fine-tuned model
    console.rule("[bold green]Evaluating Fine-tuned Model[/bold green]")
    lora_results = run_evaluation(model_ft, tokenizer_ft, TEST_CASES, CODE_COMPLETION_TESTS, "Fine-tuned Model", judge_model=model, judge_tokenizer=tokenizer)
    lora_stats = compute_statistics(lora_results)

    console.print(f"\\n[bold green]Fine-tuned Model Overall Score: {lora_stats['overall_avg_score']:.1%}[/bold green]")

    # Print comparison
    print_comparison(base_results, lora_results, base_stats, lora_stats)

    # Save results
    save_results(base_results, lora_results, base_stats, lora_stats, output_dir)


if __name__ == "__main__":
    main()
