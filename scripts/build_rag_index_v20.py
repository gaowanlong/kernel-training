#!/usr/bin/env python3
"""
Build RAG index v2.0 — merged index combining:
1. v1 embedding index (9,919 doc/kconfig/qa chunks)
2. Source code function implementations (3,894 functions)
3. Documentation/ RST files (1,633 chunks)

All with sentence embeddings (all-MiniLM-L6-v2).

Usage: python scripts/build_rag_index_v20.py
"""

import json, pickle, time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAG_INDEX_DIR = PROJECT_ROOT / "data" / "rag_index_v20"


def main():
    chunks = []
    
    # 1) v1 embedding index (best performing so far)
    v1_path = PROJECT_ROOT / "data" / "rag_index_emb" / "chunks.jsonl"
    if v1_path.exists():
        with open(v1_path) as f:
            for line in f:
                c = json.loads(line)
                c["_source_group"] = "v1_doc"
                chunks.append(c)
        print(f"v1 doc chunks: {sum(1 for c in chunks if c['_source_group']=='v1_doc')}")
    
    # 2) Source code functions
    src_path = PROJECT_ROOT / "data" / "rag_index_source_emb" / "chunks.jsonl"
    if src_path.exists():
        with open(src_path) as f:
            for line in f:
                c = json.loads(line)
                c["_source_group"] = "source_code"
                chunks.append(c)
        print(f"Source code chunks: {sum(1 for c in chunks if c['_source_group']=='source_code')}")
    
    print(f"\nTotal chunks: {len(chunks)}")
    
    # Deduplicate by content hash
    seen = set()
    unique_chunks = []
    for c in chunks:
        h = hash(c.get("content", "")[:300])
        if h not in seen:
            seen.add(h)
            unique_chunks.append(c)
    print(f"After dedup: {len(unique_chunks)} chunks")
    
    # Build embeddings
    print("\nLoading embedding model (all-MiniLM-L6-v2)...", flush=True)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("  Model loaded", flush=True)
    
    texts = [c["content"] for c in unique_chunks]
    
    print(f"Encoding {len(texts)} chunks...", flush=True)
    start = time.time()
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
    elapsed = time.time() - start
    print(f"  Encoded {len(embeddings)} embeddings in {elapsed:.1f}s")
    print(f"  Embedding dimension: {embeddings.shape[1]}")
    
    # Save
    RAG_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(RAG_INDEX_DIR / "chunks.jsonl", "w") as f:
        for c in unique_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    
    with open(RAG_INDEX_DIR / "embeddings.pkl", "wb") as f:
        pickle.dump(embeddings, f)
    
    with open(RAG_INDEX_DIR / "model_name.txt", "w") as f:
        f.write("all-MiniLM-L6-v2")
    
    # Stats
    from collections import Counter
    type_counts = Counter(c.get("type", "unknown") for c in unique_chunks)
    source_counts = Counter(c.get("_source_group", "unknown") for c in unique_chunks)
    
    print(f"\nIndex saved to {RAG_INDEX_DIR}/")
    print(f"  chunks.jsonl: {len(unique_chunks)} chunks")
    print(f"  embeddings.pkl: {embeddings.shape}")
    print(f"\nBy source group:")
    for k, v in source_counts.most_common():
        print(f"  {k}: {v}")
    print(f"\nBy type:")
    for k, v in type_counts.most_common():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
