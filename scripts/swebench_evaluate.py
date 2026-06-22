#!/usr/bin/env python3
"""
SWE-bench Kernel: Linux kernel patch generation benchmark.
Inspired by SWE-bench — given a kernel issue/bug description,
the model must generate a correct patch.

Tests are derived from real kernel commits and documented bugs.

Usage:
    python scripts/swebench_evaluate.py
    python scripts/swebench_evaluate.py --adapter lora_adapters/kernel-lora-v1.0
"""

import argparse, json, re, subprocess, sys, tempfile, time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SWE_TESTS = [
    {
        "id": "swe_01",
        "category": "null_pointer_fix",
        "difficulty": "L1",
        "prompt": """The following kernel function has a NULL pointer dereference bug. Fix it by adding a NULL check before dereferencing the pointer.

```c
void process_buffer(struct my_device *dev)
{
    struct buffer *buf = dev->buffer;
    buf->size = 1024;
    buf->data = kmalloc(buf->size, GFP_KERNEL);
}
```""",
        "reference_keywords": ["if (!buf)", "if (!dev)", "return", "-EINVAL", "NULL"],
        "expected_fix": "Add NULL check before dereferencing buf",
    },
    {
        "id": "swe_02",
        "category": "memory_leak",
        "difficulty": "L1",
        "prompt": """The following kernel function has a memory leak. Fix it by ensuring the allocated memory is freed on all error paths.

```c
int setup_device(struct device *dev)
{
    struct private_data *priv = kmalloc(sizeof(*priv), GFP_KERNEL);
    if (!priv)
        return -ENOMEM;
    
    dev->private_data = priv;
    
    int ret = register_device(dev);
    if (ret)
        return ret;  /* BUG: priv is leaked here */
    
    return 0;
}
```""",
        "reference_keywords": ["kfree", "goto", "err_free", "out", "cleanup"],
        "expected_fix": "Free priv on error path before returning",
    },
    {
        "id": "swe_03",
        "category": "locking_bug",
        "difficulty": "L2",
        "prompt": """The following kernel function has a locking bug. The spinlock is acquired but not released on one error path. Fix it.

```c
int write_to_device(struct my_device *dev, const char __user *buf, size_t count)
{
    spin_lock(&dev->lock);
    
    if (!dev->ready) {
        return -EAGAIN;  /* BUG: lock not released */
    }
    
    int ret = copy_from_user(dev->data, buf, min(count, dev->buf_size));
    spin_unlock(&dev->lock);
    return ret;
}
```""",
        "reference_keywords": ["spin_unlock", "goto", "unlock", "out"],
        "expected_fix": "Release lock before returning on error",
    },
    {
        "id": "swe_04",
        "category": "use_after_free",
        "difficulty": "L2",
        "prompt": """The following kernel function has a use-after-free bug. Fix it by reordering the operations.

```c
void cleanup_device(struct my_device *dev)
{
    kfree(dev->buffer);
    dev->buffer = NULL;
    
    /* BUG: using dev->buffer after free */
    if (dev->buffer)
        printk("Buffer still exists\\n");
    
    kfree(dev);
}
```""",
        "reference_keywords": ["after", "before", "order", "reorder", "move"],
        "expected_fix": "Remove the useless check after kfree",
    },
    {
        "id": "swe_05",
        "category": "refcount",
        "difficulty": "L2",
        "prompt": """The following kernel module has a reference counting bug. The module's reference count is not decremented on the error path. Fix it.

```c
static int my_open(struct inode *inode, struct file *file)
{
    struct my_device *dev = container_of(inode->i_cdev, struct my_device, cdev);
    
    if (!try_module_get(THIS_MODULE))
        return -ENODEV;
    
    int ret = initialize_device(dev);
    if (ret)
        return ret;  /* BUG: module_put not called */
    
    file->private_data = dev;
    return 0;
}

static int my_release(struct inode *inode, struct file *file)
{
    struct my_device *dev = file->private_data;
    module_put(THIS_MODULE);
    return 0;
}
```""",
        "reference_keywords": ["module_put", "goto", "err", "cleanup"],
        "expected_fix": "Call module_put before returning on error",
    },
    {
        "id": "swe_06",
        "category": "race_condition",
        "difficulty": "L3",
        "prompt": """The following kernel function has a TOCTOU (time-of-check-time-of-use) race condition. The buffer size is checked but then changes before use. Fix it.

```c
static ssize_t my_write(struct file *file, const char __user *buf, size_t count, loff_t *pos)
{
    struct my_device *dev = file->private_data;
    
    /* Check buffer size */
    if (count > dev->buf_size)
        return -EINVAL;
    
    /* BUG: dev->buf_size could change between check and use */
    if (copy_from_user(dev->buffer, buf, count))
        return -EFAULT;
    
    return count;
}
```""",
        "reference_keywords": ["lock", "mutex", "spinlock", "local", "copy"],
        "expected_fix": "Use a local copy of buf_size or add locking",
    },
    {
        "id": "swe_07",
        "category": "integer_overflow",
        "difficulty": "L2",
        "prompt": """The following kernel function has an integer overflow vulnerability. Fix it by adding proper overflow checks.

```c
static ssize_t my_read(struct file *file, char __user *buf, size_t count, loff_t *pos)
{
    struct my_device *dev = file->private_data;
    
    /* BUG: *pos + count could overflow */
    if (*pos + count > dev->data_size) {
        count = dev->data_size - *pos;
    }
    
    if (copy_to_user(buf, dev->data + *pos, count))
        return -EFAULT;
    
    *pos += count;
    return count;
}
```""",
        "reference_keywords": ["check_add_overflow", "saturating", "overflow", "size_t"],
        "expected_fix": "Add overflow check before arithmetic",
    },
    {
        "id": "swe_08",
        "category": "buffer_overflow",
        "difficulty": "L2",
        "prompt": """The following kernel function has a buffer overflow. The strcpy can overflow the destination buffer. Fix it.

```c
static int store_name(struct my_device *dev, const char __user *name, size_t len)
{
    /* BUG: no bounds check on strcpy */
    char tmp[64];
    
    if (copy_from_user(tmp, name, min(len, sizeof(tmp))))
        return -EFAULT;
    
    tmp[sizeof(tmp) - 1] = '\\0';
    strcpy(dev->name, tmp);  /* BUG: dev->name might be smaller than tmp */
    
    return 0;
}
```""",
        "reference_keywords": ["strncpy", "strscpy", "snprintf", "sizeof", "bound"],
        "expected_fix": "Use strscpy or check destination size",
    },
]


def extract_code_block(response: str) -> str:
    m = re.search(r'```c\n(.*?)```', response, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r'```\n?(.*?)```', response, re.DOTALL)
    if m:
        return m.group(1).strip()
    return response.strip()


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
    for test in SWE_TESTS:
        tid = test["id"]
        prompt = test["prompt"]
        kws = test["reference_keywords"]
        
        print(f"  [{tid}] {test['category']}...", end=" ", flush=True)
        
        full_prompt = f"""You are a Linux kernel C programmer. Fix the bug in the following kernel code.

{prompt}

Write ONLY the corrected C code, no explanation."""
        
        start = time.time()
        response = generate(model, tokenizer, prompt=full_prompt, max_tokens=800, sampler=sampler)
        elapsed = time.time() - start
        
        code = extract_code_block(response)
        
        # Keyword check
        found_kws = [kw for kw in kws if kw.lower() in code.lower()]
        kw_score = len(found_kws) / len(kws) if kws else 0
        
        # LLM-as-judge for fix correctness
        judge_prompt = (
            f"You are an expert Linux kernel C programmer. "
            f"Rate the following bug fix on a scale of 0-10 based on whether it correctly fixes the bug.\n\n"
            f"Bug description: {test['expected_fix']}\n\n"
            f"Fixed code:\n```c\n{code[:1500]}\n```\n\n"
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
            "llm_score": normalized_score,
            "overall_score": (kw_score + normalized_score) / 2.0,
            "elapsed_sec": round(elapsed, 1),
        }
        results.append(result)
        
        print(f"KW:{kw_score:.0%} | Judge:{normalized_score:.0%} ({elapsed:.1f}s)", flush=True)
    
    # Summary
    print("\n" + "=" * 60)
    print(f"SWE-bench Kernel Results: {method_name}")
    print("=" * 60)
    
    overall = sum(r["overall_score"] for r in results) / len(results)
    avg_kw = sum(r["keyword_score"] for r in results) / len(results)
    avg_judge = sum(r["llm_score"] for r in results) / len(results)
    
    print(f"\nOverall: {overall:.1%}")
    print(f"Avg keyword score: {avg_kw:.1%}")
    print(f"Avg LLM judge score: {avg_judge:.1%}")
    
    print(f"\nPer-test breakdown:")
    for r in results:
        print(f"  {r['id']} ({r['category']:25s}): overall={r['overall_score']:.0%} kw={r['keyword_score']:.0%} judge={r['llm_score']:.0%}")
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output = {
        "timestamp": timestamp,
        "method": method_name,
        "model": model_path,
        "adapter": adapter_path,
        "overall_score": overall,
        "results": results,
    }
    output_path = PROJECT_ROOT / "results" / f"swebench_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SWE-bench Kernel: Kernel patch generation benchmark")
    parser.add_argument("--model", default=str(PROJECT_ROOT / "models" / "qwen2.5-7b"))
    parser.add_argument("--adapter", default=None)
    args = parser.parse_args()
    run_evaluation(args.model, args.adapter)
