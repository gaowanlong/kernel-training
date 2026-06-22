#!/usr/bin/env python3
"""
Hybrid RAG + QLoRA Evaluation:
Retrieve context via RAG, then generate answer using fine-tuned model (v1.0).

Usage: python scripts/rag_hybrid_evaluate.py
       python scripts/rag_hybrid_evaluate.py --adapter lora_adapters/kernel-lora-v1.0
"""

import json, re, pickle, sys, time
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAG_INDEX_DIR = PROJECT_ROOT / "data" / "rag_index"

# Load test cases from evaluate.py
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import importlib.util
spec = importlib.util.spec_from_file_location("evaluate_module", PROJECT_ROOT / "scripts" / "evaluate.py")
eval_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_mod)

TEST_CASES = eval_mod.TEST_CASES
CODE_COMPLETION_TESTS = eval_mod.CODE_COMPLETION_TESTS


def load_index():
    with open(RAG_INDEX_DIR / "chunks.jsonl") as f:
        chunks = [json.loads(line) for line in f]
    with open(RAG_INDEX_DIR / "vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open(RAG_INDEX_DIR / "tfidf_matrix.pkl", "rb") as f:
        tfidf_matrix = pickle.load(f)
    return chunks, vectorizer, tfidf_matrix


def retrieve(query, chunks, vectorizer, tfidf_matrix, top_k=5):
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    results = []
    for idx in top_indices:
        if similarities[idx] > 0.01:
            results.append({"chunk": chunks[idx], "score": float(similarities[idx])})
    return results


def build_rag_prompt(query, retrieved):
    context_parts = []
    for r in retrieved[:3]:
        chunk = r["chunk"]
        context_parts.append(f"From kernel documentation:\n{chunk['answer'][:500]}")
    context = "\n\n".join(context_parts)
    return f"""You are a Linux kernel expert. Use the following kernel documentation to answer the question.

Context:
{context}

Question: {query}

Answer the question thoroughly based on the context above. If the context doesn't contain enough information, use your own knowledge of the Linux kernel."""


def run_evaluation(adapter_path=None):
    print("Loading RAG index...", flush=True)
    chunks, vectorizer, tfidf_matrix = load_index()
    print(f"  Index: {len(chunks)} chunks", flush=True)

    if adapter_path:
        print(f"Loading fine-tuned model with adapter: {adapter_path}...", flush=True)
        model, tokenizer = load(str(PROJECT_ROOT / "models" / "qwen2.5-7b"), adapter_path=str(adapter_path))
        method_name = f"RAG + QLoRA ({adapter_path.name})"
    else:
        print("Loading base model...", flush=True)
        model, tokenizer = load(str(PROJECT_ROOT / "models" / "qwen2.5-7b"))
        method_name = "RAG + Base Model"

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

        retrieved = retrieve(question, chunks, vectorizer, tfidf_matrix)
        rag_prompt = build_rag_prompt(question, retrieved)

        start = time.time()
        response = generate(model, tokenizer, prompt=rag_prompt[:3000], max_tokens=300, sampler=sampler)
        elapsed = time.time() - start

        # LLM-as-judge scoring
        judge_prompt = (
            f"You are an expert Linux kernel evaluator. "
            f"Rate the following answer on a scale of 0-10 based on correctness, completeness, and precision.\n\n"
            f"Question: {question}\n\n"
            f"Answer: {response[:1000]}\n\n"
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

    # Stats by category
    categories = {}
    for r in results:
        for test in all_tests:
            if test["id"] == r["id"]:
                cat = test.get("category", "unknown")
                categories.setdefault(cat, []).append(r["score"])
                break

    print("\n" + "=" * 60)
    print(f"Hybrid RAG Evaluation: {method_name}")
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
        "adapter": str(adapter_path) if adapter_path else None,
        "index_size": len(chunks),
        "overall_score": overall,
        "results": results,
        "categories": {cat: sum(scores)/len(scores) for cat, scores in categories.items()},
    }

    output_path = PROJECT_ROOT / "results" / f"rag_hybrid_eval_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {output_path}")
    return output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Hybrid RAG + QLoRA Evaluation")
    parser.add_argument("--adapter", type=str, default=None,
                        help="Path to LoRA adapter (e.g. lora_adapters/kernel-lora-v1.0)")
    args = parser.parse_args()

    adapter_path = None
    if args.adapter:
        adapter_path = PROJECT_ROOT / args.adapter
        if not adapter_path.exists():
            print(f"Adapter not found: {adapter_path}")
            sys.exit(1)

    run_evaluation(adapter_path)
