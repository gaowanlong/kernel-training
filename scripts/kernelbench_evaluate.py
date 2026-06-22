#!/usr/bin/env python3
"""
KernelBench: Linux kernel code generation benchmark.
Evaluates the model's ability to write correct kernel code.

Tests include:
1. Kernel module creation (init/exit, licensing)
2. Character device driver (open/read/write/ioctl)
3. Platform driver (probe/remove)
4. Memory allocation (kmalloc/kfree patterns)
5. Spinlock usage (critical sections)
6. Workqueue usage
7. Timer usage
8. Interrupt handler (request_irq/free_irq)
9. Procfs / sysfs file creation
10. Linked list operations (list_head)

Each test is scored by:
- Compilation check (does it compile with kernel headers?)
- Keyword check (does it use the right APIs?)
- LLM-as-judge (correctness and completeness)

Usage:
    python scripts/kernelbench_evaluate.py
    python scripts/kernelbench_evaluate.py --model models/qwen2.5-7b
    python scripts/kernelbench_evaluate.py --adapter lora_adapters/kernel-lora-v1.0
"""

import argparse, json, re, subprocess, sys, tempfile, time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

KERNEL_TESTS = [
    {
        "id": "kb_01",
        "category": "module_init",
        "difficulty": "L1",
        "prompt": "Write a complete Linux kernel module that prints 'Hello, Kernel!' on load and 'Goodbye, Kernel!' on unload. Include proper licensing (GPL), module_init, and module_exit.",
        "reference_keywords": ["module_init", "module_exit", "MODULE_LICENSE", "printk", "GPL"],
        "check_compile": True,
    },
    {
        "id": "kb_02",
        "category": "char_driver",
        "difficulty": "L2",
        "prompt": "Write a complete Linux character device driver that implements open, release, read, and write file operations. Use a single static buffer of 1024 bytes. Register the device with alloc_chrdev_region and cdev_init.",
        "reference_keywords": ["alloc_chrdev_region", "cdev_init", "cdev_add", "file_operations", "copy_to_user", "copy_from_user", "unregister_chrdev_region"],
        "check_compile": True,
    },
    {
        "id": "kb_03",
        "category": "platform_driver",
        "difficulty": "L2",
        "prompt": "Write a complete Linux platform driver with probe and remove functions. The probe should allocate memory and print a message. Use module_platform_driver macro.",
        "reference_keywords": ["platform_driver", "platform_driver_register", "probe", "remove", "of_match_table", "module_platform_driver"],
        "check_compile": True,
    },
    {
        "id": "kb_04",
        "category": "memory_allocation",
        "difficulty": "L1",
        "prompt": "Write a Linux kernel function that allocates an array of 100 integers using kmalloc, initializes them to zero, uses them, and then frees with kfree. Include proper error handling for allocation failure.",
        "reference_keywords": ["kmalloc", "GFP_KERNEL", "kfree", "NULL", "return -ENOMEM"],
        "check_compile": True,
    },
    {
        "id": "kb_05",
        "category": "spinlock",
        "difficulty": "L1",
        "prompt": "Write a Linux kernel function that uses a spinlock to protect a shared counter. Include spin_lock_init, spin_lock, spin_unlock, and proper irqsave variants.",
        "reference_keywords": ["spinlock_t", "spin_lock_init", "spin_lock", "spin_unlock", "spin_lock_irqsave", "spin_unlock_irqrestore", "DEFINE_SPINLOCK"],
        "check_compile": True,
    },
    {
        "id": "kb_06",
        "category": "workqueue",
        "difficulty": "L2",
        "prompt": "Write a complete Linux kernel module that uses a workqueue. On module load, schedule a work item that prints a message. Use alloc_workqueue, INIT_WORK, schedule_work, and destroy_workqueue.",
        "reference_keywords": ["workqueue", "INIT_WORK", "schedule_work", "alloc_workqueue", "destroy_workqueue", "struct work_struct"],
        "check_compile": True,
    },
    {
        "id": "kb_07",
        "category": "timer",
        "difficulty": "L2",
        "prompt": "Write a Linux kernel function that sets up a timer that fires after 5 seconds and prints a message. Use timer_setup, mod_timer, and del_timer.",
        "reference_keywords": ["timer_setup", "mod_timer", "del_timer", "struct timer_list", "jiffies", "HZ"],
        "check_compile": True,
    },
    {
        "id": "kb_08",
        "category": "interrupt_handler",
        "difficulty": "L2",
        "prompt": "Write a Linux kernel module that registers an interrupt handler for a shared IRQ. The handler should increment a counter and return IRQ_HANDLED or IRQ_NONE. Include proper cleanup on module exit.",
        "reference_keywords": ["request_irq", "free_irq", "IRQ_HANDLED", "irq_handler_t", "dev_id", "IRQF_SHARED"],
        "check_compile": True,
    },
    {
        "id": "kb_09",
        "category": "procfs",
        "difficulty": "L2",
        "prompt": "Write a Linux kernel module that creates a /proc/hello entry. When read, it should return 'Hello from kernel!'. Use proc_create, proc_ops, and remove_proc_entry.",
        "reference_keywords": ["proc_create", "proc_ops", "remove_proc_entry", "struct proc_dir_entry", "seq_file"],
        "check_compile": True,
    },
    {
        "id": "kb_10",
        "category": "linked_list",
        "difficulty": "L2",
        "prompt": "Write a Linux kernel function that creates a linked list of 5 nodes using struct list_head. Each node should contain an integer value. Traverse the list and print each value using list_for_each_entry.",
        "reference_keywords": ["list_head", "list_add", "list_for_each_entry", "INIT_LIST_HEAD", "container_of"],
        "check_compile": True,
    },
    {
        "id": "kb_11",
        "category": "kobject_sysfs",
        "difficulty": "L3",
        "prompt": "Write a Linux kernel module that creates a kobject under /sys/kernel with a sysfs file. The file should be readable and writable, storing an integer value. Use kobject_create_and_add, sysfs_create_file, and proper attribute handling.",
        "reference_keywords": ["kobject", "kobject_create_and_add", "sysfs_create_file", "kobj_attribute", "show", "store", "kobject_put"],
        "check_compile": True,
    },
    {
        "id": "kb_12",
        "category": "waitqueue",
        "difficulty": "L3",
        "prompt": "Write a Linux kernel module that demonstrates a waitqueue. Create a thread that waits on a condition, and another function that wakes it up. Use wait_event_interruptible, wake_up, and DECLARE_WAIT_QUEUE_HEAD.",
        "reference_keywords": ["wait_queue_head_t", "DECLARE_WAIT_QUEUE_HEAD", "wait_event_interruptible", "wake_up", "kthread_run", "kthread_stop"],
        "check_compile": True,
    },
]


def extract_code_block(response: str) -> str:
    """Extract C code block from model response."""
    # Try ```c ... ``` first
    m = re.search(r'```c\n(.*?)```', response, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Try ``` ... ```
    m = re.search(r'```\n?(.*?)```', response, re.DOTALL)
    if m:
        return m.group(1).strip()
    return response.strip()


def check_compilation(code: str, test_id: str) -> tuple[bool, str]:
    """Try to compile the kernel code using local kernel source."""
    kernel_dir = PROJECT_ROOT / "data" / "raw" / "linux"
    if not kernel_dir.exists():
        return False, "Kernel source not found at data/raw/linux/"
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        src_file = tmp_path / "test.c"
        src_file.write_text(code)
        
        # Create a minimal Makefile
        makefile = tmp_path / "Makefile"
        makefile.write_text(f"obj-m += test.o\n")
        
        try:
            result = subprocess.run(
                ["make", f"-C{str(kernel_dir)}", f"M={tmpdir}", "modules"],
                capture_output=True, text=True, timeout=30,
                env={**__import__('os').environ, "ARCH": "arm64"}
            )
            if result.returncode == 0:
                return True, "Compilation successful"
            else:
                # Extract relevant error
                errors = result.stderr[-500:] if result.stderr else result.stdout[-500:]
                return False, f"Compilation failed: {errors[:200]}"
        except subprocess.TimeoutExpired:
            return False, "Compilation timed out"
        except FileNotFoundError:
            return False, "make not found (need kernel build system)"
        except Exception as e:
            return False, f"Error: {e}"


def run_evaluation(model_path, adapter_path=None):
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler
    
    print(f"Loading model: {model_path}", flush=True)
    if adapter_path:
        print(f"  with adapter: {adapter_path}", flush=True)
        model, tokenizer = load(model_path, adapter_path=adapter_path)
        method_name = f"QLoRA ({Path(adapter_path).name})"
    else:
        model, tokenizer = load(model_path)
        method_name = "Base Model"
    
    sampler = make_sampler(temp=0.3)
    print("  Model loaded\n", flush=True)
    
    results = []
    for test in KERNEL_TESTS:
        tid = test["id"]
        prompt = test["prompt"]
        kws = test["reference_keywords"]
        
        print(f"  [{tid}] {test['category']}...", end=" ", flush=True)
        
        # Generate code
        full_prompt = f"""You are a Linux kernel C programmer. Write correct, compilable kernel code.

{prompt}

Write ONLY the C code, no explanation."""
        
        start = time.time()
        response = generate(model, tokenizer, prompt=full_prompt, max_tokens=800, sampler=sampler)
        elapsed = time.time() - start
        
        # Extract code
        code = extract_code_block(response)
        
        # Keyword check
        found_kws = [kw for kw in kws if kw.lower() in code.lower()]
        kw_score = len(found_kws) / len(kws) if kws else 0
        
        # Compilation check
        compile_ok, compile_msg = check_compilation(code, tid)
        
        # LLM-as-judge
        judge_prompt = (
            f"You are an expert Linux kernel C programmer evaluator. "
            f"Rate the following kernel code on a scale of 0-10 based on correctness, "
            f"completeness, and adherence to kernel coding style.\n\n"
            f"Task: {prompt}\n\n"
            f"Code:\n```c\n{code[:1500]}\n```\n\n"
            f"Output ONLY a number 0-10, nothing else."
        )
        try:
            judge_resp = generate(model, tokenizer, prompt=judge_prompt, max_tokens=10, sampler=make_sampler(temp=0.1))
            score_match = re.search(r'\b(\d+)(?:/10)?\b', judge_resp.strip())
            judge_score = int(score_match.group(1)) if score_match else 5
            judge_score = max(0, min(10, judge_score))
        except:
            judge_score = 5
        
        normalized_score = judge_score / 10.0
        
        result = {
            "id": tid,
            "category": test["category"],
            "difficulty": test["difficulty"],
            "keyword_score": kw_score,
            "compile_ok": compile_ok,
            "compile_msg": compile_msg,
            "llm_score": normalized_score,
            "overall_score": (kw_score + normalized_score + (1.0 if compile_ok else 0.0)) / 3.0,
            "elapsed_sec": round(elapsed, 1),
        }
        results.append(result)
        
        status = f"KW:{kw_score:.0%} | Judge:{normalized_score:.0%} | Compile:{'OK' if compile_ok else 'FAIL'}"
        print(f"{status} ({elapsed:.1f}s)", flush=True)
    
    # Summary
    print("\n" + "=" * 60)
    print(f"KernelBench Results: {method_name}")
    print("=" * 60)
    
    overall = sum(r["overall_score"] for r in results) / len(results)
    compile_rate = sum(1 for r in results if r["compile_ok"]) / len(results)
    avg_kw = sum(r["keyword_score"] for r in results) / len(results)
    avg_judge = sum(r["llm_score"] for r in results) / len(results)
    
    print(f"\nOverall: {overall:.1%}")
    print(f"Compilation rate: {compile_rate:.1%} ({sum(1 for r in results if r['compile_ok'])}/{len(results)})")
    print(f"Avg keyword score: {avg_kw:.1%}")
    print(f"Avg LLM judge score: {avg_judge:.1%}")
    
    print(f"\nPer-test breakdown:")
    for r in results:
        print(f"  {r['id']} ({r['category']:20s}): overall={r['overall_score']:.0%} kw={r['keyword_score']:.0%} judge={r['llm_score']:.0%} compile={'OK' if r['compile_ok'] else 'FAIL'}")
    
    # Save
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output = {
        "timestamp": timestamp,
        "method": method_name,
        "model": model_path,
        "adapter": adapter_path,
        "overall_score": overall,
        "compile_rate": compile_rate,
        "results": results,
    }
    output_path = PROJECT_ROOT / "results" / f"kernelbench_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KernelBench: Kernel code generation benchmark")
    parser.add_argument("--model", default=str(PROJECT_ROOT / "models" / "qwen2.5-7b"))
    parser.add_argument("--adapter", default=None)
    args = parser.parse_args()
    run_evaluation(args.model, args.adapter)
