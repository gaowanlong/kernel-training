#!/usr/bin/env python3
"""
Build RAG index v2 — expanded with Documentation/ RST files.
Indexes kernel-doc + Kconfig + Documentation/ RST + raw kernel-doc from headers.

Usage: python scripts/build_rag_index_v2.py
"""

import json, re, pickle, sys, os
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAG_INDEX_DIR = PROJECT_ROOT / "data" / "rag_index_v2"


def parse_rst_text(text: str, source_path: str, max_chunk_size: int = 1500) -> list[dict]:
    """Parse an RST file into chunks."""
    chunks = []
    
    # Remove RST directives and formatting
    text = re.sub(r'\.\. \w+::.*?(?:\n\n|\Z)', '', text, flags=re.DOTALL)
    text = re.sub(r'\|.*?\|', '', text)
    text = re.sub(r'`[^`]+`', '', text)
    
    # Split by sections (headings)
    sections = re.split(r'\n[=~\-^"]+\n', text)
    
    for section in sections:
        section = section.strip()
        if len(section) < 200:
            continue
        
        # Further split large sections
        if len(section) > max_chunk_size:
            paragraphs = re.split(r'\n\n+', section)
            current = ""
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                if len(current) + len(para) > max_chunk_size and current:
                    chunks.append(current.strip())
                    current = para
                else:
                    current += "\n\n" + para if current else para
            if current:
                chunks.append(current.strip())
        else:
            chunks.append(section)
    
    return chunks


def collect_documentation_chunks() -> list[dict]:
    """Extract chunks from Documentation/ RST files."""
    chunks = []
    kernel_dir = PROJECT_ROOT / "data" / "raw" / "linux"
    doc_dir = kernel_dir / "Documentation"
    
    # Key directories to index (focused on kernel internals)
    key_dirs = [
        "scheduler", "locking", "mm", "core-api", "RCU",
        "filesystems", "networking", "process", "memory-barriers.txt",
    ]
    
    for entry in key_dirs:
        path = doc_dir / entry
        if path.is_dir():
            rst_files = sorted(path.rglob("*.rst"))
        elif path.is_file():
            rst_files = [path]
        else:
            continue
        
        for rst_path in rst_files:
            # Skip index files
            if rst_path.name == "index.rst":
                continue
            # Skip very large files (>200KB)
            size = rst_path.stat().st_size
            if size > 200000:
                continue
            
            try:
                text = rst_path.read_text(encoding="utf-8", errors="ignore")
                subdir = "/".join(rst_path.relative_to(doc_dir).parts[:-1]) or entry
                
                rst_chunks = parse_rst_text(text, str(rst_path))
                for i, chunk_text in enumerate(rst_chunks):
                    if len(chunk_text) >= 200:
                        chunks.append({
                            "content": chunk_text,
                            "question": "",
                            "answer": chunk_text,
                            "type": "doc_rst",
                            "subsystem": subdir,
                            "source": str(rst_path.relative_to(kernel_dir)),
                        })
            except Exception as e:
                print(f"  Warning: {rst_path}: {e}")
    
    print(f"  Documentation chunks: {len(chunks)}")
    return chunks


def collect_existing_chunks() -> list[dict]:
    """Reuse existing chunks from v1 index."""
    chunks = []
    v1_path = PROJECT_ROOT / "data" / "rag_index" / "chunks.jsonl"
    if v1_path.exists():
        with open(v1_path) as f:
            for line in f:
                chunks.append(json.loads(line))
    print(f"  Existing chunks: {len(chunks)}")
    return chunks


def build_index(chunks):
    """Build TF-IDF index from chunks."""
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
    """Save index to disk."""
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
    print("Collecting knowledge chunks...")
    
    # 1) Existing chunks from v1
    existing = collect_existing_chunks()
    
    # 2) Documentation RST files
    print("\nExtracting Documentation/ RST files...")
    doc_chunks = collect_documentation_chunks()
    
    # Combine
    all_chunks = existing + doc_chunks
    print(f"\nTotal chunks: {len(all_chunks)}")
    
    # Deduplicate by content hash
    seen = set()
    unique_chunks = []
    for c in all_chunks:
        h = hash(c["content"][:200])
        if h not in seen:
            seen.add(h)
            unique_chunks.append(c)
    print(f"After dedup: {len(unique_chunks)} chunks")
    
    # Build index
    vectorizer, tfidf_matrix = build_index(unique_chunks)
    save_index(unique_chunks, vectorizer, tfidf_matrix, RAG_INDEX_DIR)


if __name__ == "__main__":
    main()
