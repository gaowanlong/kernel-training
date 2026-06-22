#!/usr/bin/env python3
"""
Build embedding RAG index v2 — expanded with Documentation/ RST files.
Uses all-MiniLM-L6-v2 for semantic retrieval.

Usage: python scripts/build_rag_index_emb_v2.py
"""

import json, pickle, time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAG_INDEX_DIR = PROJECT_ROOT / "data" / "rag_index_emb_v2"


def main():
    # Load chunks from v2 TF-IDF index (which already has expanded sources)
    v2_chunks_path = PROJECT_ROOT / "data" / "rag_index_v2" / "chunks.jsonl"
    chunks = []
    with open(v2_chunks_path) as f:
        for line in f:
            chunks.append(json.loads(line))
    print(f"Loaded {len(chunks)} chunks from v2 index")
    
    # Build embeddings
    print("Loading embedding model (all-MiniLM-L6-v2)...", flush=True)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("  Model loaded", flush=True)
    
    texts = [c["content"] for c in chunks]
    
    print(f"Encoding {len(texts)} chunks...", flush=True)
    start = time.time()
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
    elapsed = time.time() - start
    print(f"  Encoded {len(embeddings)} embeddings in {elapsed:.1f}s")
    print(f"  Embedding dimension: {embeddings.shape[1]}")
    
    # Save
    RAG_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(RAG_INDEX_DIR / "chunks.jsonl", "w") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    
    with open(RAG_INDEX_DIR / "embeddings.pkl", "wb") as f:
        pickle.dump(embeddings, f)
    
    with open(RAG_INDEX_DIR / "model_name.txt", "w") as f:
        f.write("all-MiniLM-L6-v2")
    
    print(f"Index saved to {RAG_INDEX_DIR}/")
    print(f"  chunks.jsonl: {len(chunks)} chunks")
    print(f"  embeddings.pkl: {embeddings.shape}")


if __name__ == "__main__":
    main()
