#!/usr/bin/env python3
"""
RAG Evaluation v2.0 — merged index (documentation + source code).
Uses embedding retrieval + QLoRA v1.0.

Usage: python scripts/rag_v20_evaluate.py
       python scripts/rag_v20_evaluate.py --base  # use base model
"""

import json, pickle, sys, time
from pathlib import Path
import numpy as np
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAG_INDEX_DIR = PROJECT_ROOT / "data" / "rag_index_v20"

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import importlib.util
spec = importlib.util.spec_from_file_location("evaluate_module", PROJECT_ROOT / "scripts" / "evaluate.py")
eval_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_mod)

TEST_CASES = eval_mod.TEST_CASES
CODE_COMPLETION_TESTS = eval_mod.CODE_COMPLETION_TESTS

_emb_model = None

def get_emb_model():
    global _emb_model
    if _emb_model is None:
        from sentence_transformers import SentenceTransformer
        _emb_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _emb_model


def load_index():
    with open(RAG_INDEX_DIR / "chunks.jsonl") as f:
        chunks = [json.loads(line) for line in f]
    with open(RAG_INDEX_DIR / "embeddings.pkl", "rb") as f:
        embeddings = pickle.load(f)
    return chunks, embeddings


def retrieve(query, chunks, embeddings, top_k=5):
    model = get_emb_model()
    query_vec = model.encode([query])[0]
    similarities = np.dot(embeddings, query_vec) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_vec)
    )
    top_indices = similarities.argsort()[-top_k:][::-1]
    results = []
    for idx in top_indices:
        if similarities[idx] > 0.1:
            results.append({"chunk": chunks[idx], "score": float(similarities[idx])})
    return results


def build_rag_prompt(query, retrieved):
    """Build prompt with mixed doc + source context."""
    context_parts = []
    for r in retrieved[:3]:
        chunk = r["chunk"]
        source_group = chunk.get("_source_group", "")
        if source_group == "source_code":
            func_name = chunk.get("function", "")
            src_file = chunk.get("source", "")
            context_parts.append(f"From kernel source ({src_file}):\n```c\n{chunk['answer'][:500]}\n```")
        else:
            context_parts.append(f"From kernel documentation:\n{chunk['answer'][:500]}")
    context = "\n\n".join(context_parts)
    return f"""You are a Linux kernel expert. Use the following kernel documentation and source code to answer the question.

Context:
{context}

Question: {query}

Answer the question thoroughly based on the context above. If the context doesn't contain enough information, use your own knowledge of the Linux kernel."""


def run_evaluation(use_base=False):
    print("Loading RAG v2.0 index (doc + source merged)...", flush=True)
    chunks, embeddings = load_index()
    print(f"  Index: {len(chunks)} chunks, embedding dim: {embeddings.shape[1]}", flush=True)

    if use_base:
        print("Loading base model...", flush=True)
        model, tokenizer = load(str(PROJECT_ROOT / "models" / "qwen2.5-7b"))
        method_name = "RAG v2.0 + Base Model"
    else:
        print("Loading fine-tuned model (v1.0)...", flush=True)
        model, tokenizer = load(
            str(PROJECT_ROOT / "models" / "qwen2.5-7b"),
            adapter_path=str(PROJECT_ROOT / "lora_adapters" / "kernel-lora-v1.0")
        )
        method_name = "RAG v2.0 + QLoRA (v1.0)"

    sampler = make_sampler(temp=0.7)
    print("  Model loaded\n", flush=True)

    all_tests = TEST_CASES + CODE_COMPLETION_TESTS
    print(f"Running {len(all_tests)} tests with {method_name}...\n", flush=True)

    results = []
    for test in all_tests:
        qid = test["id"]
        question = test.get("question", test.get("prompt", ""))
        kws = test.get("reference_keywords", [])

        print(f"  [{qid}] ", end="", flush=True)

        retrieved = retrieve(question, chunks, embeddings)
        rag_prompt = build_rag_prompt(question, retrieved)

        start = time.time()
        response = generate(model, tokenizer, prompt=rag_prompt[:3000], max_tokens=300, sampler=sampler)
        elapsed = time.time() - start

        judge_prompt = (
            f"You are an expert Linux kernel evaluator. "
            f"Rate the following answer on a scale of 0-10 based on correctness, completeness, and precision.\n\n"
            f"Question: {question}\n\n"
            f"Answer: {response[:1000]}\n\n"
            f"Output ONLY a number 0-10, nothing else."
        )
        try:
            judge_resp = generate(model, tokenizer, prompt=judge_prompt, max_tokens=10, sampler=make_sampler(temp=0.1))
            import re
            score_match = re.search(r'\b(\d+)(?:/10)?\b', judge_resp.strip())
            judge_score = int(score_match.group(1)) if score_match else 5
            judge_score = max(0, min(10, judge_score))
        except:
            judge_score = 5

        normalized_score = judge_score / 10.0
        found_keywords = [kw for kw in kws if kw.lower() in response.lower()]

        results.append({
            "id": qid,
            "score": normalized_score,
            "keywords_matched": len(found_keywords),
            "keywords_total": len(kws),
            "retrieved_chunks": len(retrieved),
            "elapsed_sec": round(elapsed, 1),
        })

        print(f"Score: {normalized_score:.0%} | {elapsed:.1f}s | {len(retrieved)} chunks", flush=True)

    categories = {}
    for r in results:
        for test in all_tests:
            if test["id"] == r["id"]:
                cat = test.get("category", "unknown")
                categories.setdefault(cat, []).append(r["score"])
                break

    print("\n" + "=" * 60)
    print(f"RAG v2.0 Evaluation: {method_name}")
    print("=" * 60)

    all_scores = [r["score"] for r in results]
    overall = sum(all_scores) / len(all_scores)
    print(f"\nOverall: {overall:.1%}")

    for cat, scores in sorted(categories.items()):
        print(f"  {cat}: {sum(scores)/len(scores):.1%}")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output = {
        "timestamp": timestamp,
        "method": method_name,
        "index_size": len(chunks),
        "embedding_dim": embeddings.shape[1],
        "overall_score": overall,
        "results": results,
        "categories": {cat: sum(scores)/len(scores) for cat, scores in categories.items()},
    }

    output_path = PROJECT_ROOT / "results" / f"rag_v20_eval_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {output_path}")
    return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="RAG v2.0 Evaluation (merged index)")
    parser.add_argument("--base", action="store_true", help="Use base model instead of QLoRA")
    args = parser.parse_args()
    run_evaluation(use_base=args.base)
