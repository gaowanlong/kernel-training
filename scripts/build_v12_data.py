#!/usr/bin/env python3
"""
Build v1.2 dataset: v1.0 + Kconfig + more Documentation/
"""

import json, random
from pathlib import Path
from collections import Counter

random.seed(42)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 1) Load v1.0 data (best version - single turn only)
print("Loading v1.0 data...")
v10_train = [json.loads(l) for l in open(PROJECT_ROOT / "data" / "processed" / "train.jsonl")]
v10_valid = [json.loads(l) for l in open(PROJECT_ROOT / "data" / "processed" / "valid.jsonl")]

# Filter out multi-turn (from v1.1) - keep only single-turn
v10_train = [s for s in v10_train if len(s["messages"]) == 2]
v10_valid = [s for s in v10_valid if len(s["messages"]) == 2]
print(f"  v1.0 single-turn: {len(v10_train)} train, {len(v10_valid)} valid")

# 2) Load Kconfig data
print("Loading Kconfig data...")
kconfig = [json.loads(l) for l in open(PROJECT_ROOT / "data" / "processed" / "kconfig_samples.jsonl")]
print(f"  Kconfig: {len(kconfig)}")

# 3) Load more Documentation/ sections
print("Loading more Documentation/ sections...")
from scripts.extract_v07_data import extract_doc_sections, detect_subsystem_from_doc_path

KERNEL_DIR = PROJECT_ROOT / "data" / "raw" / "linux"
doc_dir = KERNEL_DIR / "Documentation"

# Priority directories that cover eval weak areas
priority_dirs = [
    "admin-guide/mm", "admin-guide/kernel-parameters",
    "scheduler", "locking", "RCU", "interrupt",
    "filesystems", "networking", "mm", "core-api",
    "driver-api", "trace", "block", "power",
]

doc_sections = []
for pd in priority_dirs:
    subdir = doc_dir / pd
    if subdir.exists():
        for rst in sorted(subdir.rglob("*.rst"))[:30]:  # 30 per dir
            sections = extract_doc_sections(rst)
            subsystem = detect_subsystem_from_doc_path(str(rst.relative_to(KERNEL_DIR)))
            for sec in sections:
                sec["subsystem"] = subsystem
            doc_sections.extend(sections)

print(f"  Doc sections: {len(doc_sections)}")

# Generate Q&A from doc sections
from scripts.extract_v07_data import generate_doc_section_qa
doc_qa = []
for sec in doc_sections:
    qa = generate_doc_section_qa(sec, sec["subsystem"])
    if qa:
        doc_qa.append(qa)

print(f"  Doc Q&A: {len(doc_qa)}")

# 4) Combine
all_train = v10_train + kconfig + doc_qa
all_valid = v10_valid + kconfig[:len(kconfig)//10] + doc_qa[:len(doc_qa)//10]

random.shuffle(all_train)
random.shuffle(all_valid)

print(f"\nTotal: {len(all_train)} train, {len(all_valid)} valid")

# Stats
types = Counter(s.get("type", "unknown") for s in all_train)
print(f"\nType distribution (train):")
for t, c in types.most_common():
    print(f"  {t}: {c}")

zh = sum(1 for s in all_train if s.get("language") == "zh" or 
         any("\u4e00" <= c <= "\u9fff" for c in s["messages"][0]["content"]))
print(f"\nChinese: {zh}/{len(all_train)} ({zh/len(all_train)*100:.0f}%)")

# Save
output_dir = PROJECT_ROOT / "data" / "processed"
with open(output_dir / "train.jsonl", "w") as f:
    for s in all_train:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")
with open(output_dir / "valid.jsonl", "w") as f:
    for s in all_valid:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

print(f"\nSaved to {output_dir}/")
