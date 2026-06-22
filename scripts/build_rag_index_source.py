#!/usr/bin/env python3
"""
Build RAG index with kernel source code function implementations.
Extracts key function implementations from kernel source and adds them
as retrievable chunks for code understanding questions.

Usage: python scripts/build_rag_index_source.py
"""

import json, pickle, re, time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAG_INDEX_DIR = PROJECT_ROOT / "data" / "rag_index_source"


def extract_functions(source_path: Path, max_functions: int = 500) -> list[dict]:
    """Extract function implementations from a C source file."""
    chunks = []
    try:
        text = source_path.read_text(encoding="utf-8", errors="ignore")
    except:
        return chunks
    
    # Match function definitions: return_type function_name(params) { ... }
    # This is a simplified regex — catches most kernel functions
    func_pattern = re.compile(
        r'(?:static\s+)?(?:inline\s+)?(?:unsigned\s+)?(?:struct\s+\w+\s*\*?|'
        r'void|int|char|long|size_t|ssize_t|bool|u\d+|s\d+|__init|__exit|'
        r'atomic_t|wait_queue_head_t|spinlock_t|struct\s+\w+)\s*\*?\n?'
        r'(\w+)\s*\(([^)]*)\)\s*\n?\{'
        r'(.*?)\n\}',
        re.DOTALL
    )
    
    for match in func_pattern.finditer(text):
        func_name = match.group(1)
        params = match.group(2)[:100]
        body = match.group(3)[:2000]  # Limit body length
        
        # Skip very short functions
        if len(body) < 50:
            continue
        
        # Build the full function
        func_text = f"{func_name}({params}) {{\n{body}\n}}"
        
        # Skip if it looks like a macro or inline asm
        if func_text.count('{') > 20:
            continue
        
        chunks.append({
            "content": func_text,
            "question": f"Explain the {func_name}() function in the Linux kernel",
            "answer": func_text,
            "type": "kernel_source",
            "subsystem": "/".join(source_path.relative_to(PROJECT_ROOT / "data" / "raw" / "linux").parts[:-1]),
            "source": str(source_path.relative_to(PROJECT_ROOT / "data" / "raw" / "linux")),
            "function": func_name,
        })
    
    return chunks


def collect_source_chunks() -> list[dict]:
    """Extract function implementations from key kernel source files."""
    kernel_dir = PROJECT_ROOT / "data" / "raw" / "linux"
    
    # Key source files with important kernel functions
    key_files = [
        "kernel/sched/core.c",
        "kernel/sched/fair.c",
        "kernel/fork.c",
        "kernel/exit.c",
        "kernel/printk/printk.c",
        "kernel/locking/spinlock.c",
        "kernel/locking/mutex.c",
        "kernel/locking/rtmutex.c",
        "kernel/rcu/tree.c",
        "kernel/rcu/rcu.h",
        "kernel/rcu/update.c",
        "kernel/irq/manage.c",
        "kernel/irq/chip.c",
        "kernel/time/timer.c",
        "kernel/time/hrtimer.c",
        "kernel/softirq.c",
        "kernel/workqueue.c",
        "kernel/kmod.c",
        "kernel/module.c",
        "kernel/kallsyms.c",
        "mm/slab.c",
        "mm/slub.c",
        "mm/page_alloc.c",
        "mm/vmalloc.c",
        "mm/oom_kill.c",
        "mm/memory.c",
        "mm/mmap.c",
        "fs/exec.c",
        "fs/open.c",
        "fs/read_write.c",
        "fs/file_table.c",
        "fs/namespace.c",
        "fs/super.c",
        "fs/buffer.c",
        "include/linux/sched.h",
        "include/linux/mm.h",
        "include/linux/fs.h",
        "include/linux/slab.h",
        "include/linux/list.h",
        "include/linux/spinlock.h",
        "include/linux/mutex.h",
        "include/linux/rcupdate.h",
        "include/linux/interrupt.h",
        "include/linux/wait.h",
        "include/linux/workqueue.h",
        "include/linux/timer.h",
        "include/linux/skbuff.h",
        "include/linux/netdevice.h",
        "include/linux/cgroup.h",
        "include/linux/nsproxy.h",
        "drivers/char/mem.c",
        "drivers/base/dd.c",
        "drivers/base/driver.c",
        "drivers/base/platform.c",
    ]
    
    all_chunks = []
    for rel_path in key_files:
        path = kernel_dir / rel_path
        if path.exists():
            chunks = extract_functions(path)
            all_chunks.extend(chunks)
            print(f"  {rel_path}: {len(chunks)} functions")
        else:
            print(f"  {rel_path}: NOT FOUND")
    
    print(f"\nTotal source chunks: {len(all_chunks)}")
    return all_chunks


def build_index(chunks):
    """Build TF-IDF index."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    texts = [c["content"] for c in chunks]
    
    print("Building TF-IDF index...")
    vectorizer = TfidfVectorizer(
        max_features=50000,
        stop_words="english",
        ngram_range=(1, 3),
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    print(f"  Vocabulary size: {len(vectorizer.get_feature_names_out())}")
    print(f"  Matrix shape: {tfidf_matrix.shape}")
    
    return vectorizer, tfidf_matrix


def save_index(chunks, vectorizer, tfidf_matrix, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "chunks.jsonl", "w") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    
    with open(output_dir / "vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    
    with open(output_dir / "tfidf_matrix.pkl", "wb") as f:
        pickle.dump(tfidf_matrix, f)
    
    print(f"\nIndex saved to {output_dir}/")
    print(f"  Total chunks: {len(chunks)}")


def main():
    print("Extracting kernel source function implementations...")
    chunks = collect_source_chunks()
    
    if not chunks:
        print("No source chunks found. Check kernel source path.")
        return
    
    vectorizer, tfidf_matrix = build_index(chunks)
    save_index(chunks, vectorizer, tfidf_matrix, RAG_INDEX_DIR)


if __name__ == "__main__":
    main()
