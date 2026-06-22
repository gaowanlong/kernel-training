#!/usr/bin/env python3
"""
Build RAG index with sentence embeddings (replaces TF-IDF).
Uses a lightweight embedding model for better semantic retrieval.

Usage: python scripts/build_rag_index_emb.py
"""

import json, pickle, sys, time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAG_INDEX_DIR = PROJECT_ROOT / "data" / "rag_index_emb"


def collect_knowledge_chunks() -> list[dict]:
    """Reuse existing chunks from TF-IDF index."""
    tfidf_chunks_path = PROJECT_ROOT / "data" / "rag_index" / "chunks.jsonl"
    chunks = []
    if tfidf_chunks_path.exists():
        with open(tfidf_chunks_path) as f:
            for line in f:
                chunks.append(json.loads(line))
    print(f"Loaded {len(chunks)} chunks from existing index")
    return chunks


def build_embeddings(chunks):
    """Build sentence embeddings for all chunks."""
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
    
    return model, embeddings


def save_index(chunks, model, embeddings, output_dir):
    """Save index to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "chunks.jsonl", "w") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    
    with open(output_dir / "embeddings.pkl", "wb") as f:
        pickle.dump(embeddings, f)
    
    # Save model name for retrieval
    with open(output_dir / "model_name.txt", "w") as f:
        f.write("all-MiniLM-L6-v2")
    
    print(f"Index saved to {output_dir}/")
    print(f"  chunks.jsonl: {len(chunks)} chunks")
    print(f"  embeddings.pkl: {embeddings.shape}")


def main():
    chunks = collect_knowledge_chunks()
    model, embeddings = build_embeddings(chunks)
    save_index(chunks, model, embeddings, RAG_INDEX_DIR)


if __name__ == "__main__":
    main()
