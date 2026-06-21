#!/usr/bin/env python3
"""
Build RAG index from kernel-doc + Kconfig + Documentation.
Indexes all knowledge sources for retrieval-augmented generation.
"""

import json, re, pickle, sys, os
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Collect all knowledge chunks from various sources
def collect_knowledge_chunks() -> list[dict]:
    chunks = []
    
    # 1) Kernel-doc from v0.7 extraction
    print("Loading kernel-doc data...")
    for split in ["train.jsonl", "valid.jsonl"]:
        path = PROJECT_ROOT / "data" / "processed" / split
        if path.exists():
            with open(path) as f:
                for line in f:
                    s = json.loads(line)
                    if s.get("source") in ("kernel_doc", "kernel_documentation", "kconfig"):
                        q = s["messages"][0]["content"]
                        a = s["messages"][1]["content"]
                        chunks.append({
                            "content": f"Q: {q}\nA: {a}",
                            "question": q,
                            "answer": a,
                            "type": s.get("type", "unknown"),
                            "subsystem": s.get("subsystem", "unknown"),
                            "source": s.get("source", "unknown"),
                        })
    
    # 2) Kconfig help texts (raw)
    print("Loading Kconfig data...")
    kconfig_path = PROJECT_ROOT / "data" / "processed" / "kconfig_samples.jsonl"
    if kconfig_path.exists():
        with open(kconfig_path) as f:
            for line in f:
                s = json.loads(line)
                q = s["messages"][0]["content"]
                a = s["messages"][1]["content"]
                chunks.append({
                    "content": f"Q: {q}\nA: {a}",
                    "question": q,
                    "answer": a,
                    "type": "kconfig_qa",
                    "subsystem": s.get("subsystem", "unknown"),
                    "source": "kconfig",
                })
    
    # 3) Raw kernel-doc from source files (re-extract key ones)
    print("Loading raw kernel source docs...")
    # We'll add key include/ headers as plain text
    kernel_dir = PROJECT_ROOT / "data" / "raw" / "linux"
    key_headers = [
        "include/linux/sched.h", "include/linux/mm.h", "include/linux/fs.h",
        "include/linux/netdevice.h", "include/linux/skbuff.h",
        "include/linux/spinlock.h", "include/linux/mutex.h",
        "include/linux/interrupt.h", "include/linux/rcupdate.h",
        "include/linux/slab.h", "include/linux/list.h",
    ]
    for hdr in key_headers:
        path = kernel_dir / hdr
        if path.exists():
            content = path.read_text(encoding="utf-8", errors="ignore")
            # Extract kernel-doc blocks
            blocks = re.findall(r'/\*\*(.*?)\*/', content, re.DOTALL)
            for block in blocks[:20]:  # Max 20 per file
                clean = re.sub(r'^\s*\*\s?', '', block, flags=re.MULTILINE).strip()
                if len(clean) > 100:
                    chunks.append({
                        "content": clean,
                        "question": "",
                        "answer": clean,
                        "type": "raw_kernel_doc",
                        "subsystem": hdr.split("/")[1],
                        "source": hdr,
                    })
    
    print(f"Total chunks: {len(chunks)}")
    return chunks


def build_index(chunks):
    """Build TF-IDF index from chunks."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    
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
    """Save index to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "chunks.jsonl", "w") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    
    with open(output_dir / "vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    
    with open(output_dir / "tfidf_matrix.pkl", "wb") as f:
        pickle.dump(tfidf_matrix, f)
    
    print(f"Index saved to {output_dir}/")


def main():
    chunks = collect_knowledge_chunks()
    vectorizer, tfidf_matrix = build_index(chunks)
    output_dir = PROJECT_ROOT / "data" / "rag_index"
    save_index(chunks, vectorizer, tfidf_matrix, output_dir)


if __name__ == "__main__":
    main()
