#!/usr/bin/env python3
"""
Build multi-turn conversation data from kernel-doc blocks.

Strategy:
1. Take existing kernel-doc function/struct blocks
2. Generate 2-3 turn conversation chains:
   Turn 1: "What does X do?" → kernel-doc description
   Turn 2: "What are the parameters/fields?" → parameter details
   Turn 3: "What are the caveats/use cases?" → context/return info
3. Mix with existing single-turn data for v1.1
"""

import json, random
from pathlib import Path
from collections import Counter

random.seed(42)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load v1.0 data
train = [json.loads(l) for l in open(PROJECT_ROOT / "data" / "processed" / "train.jsonl")]
valid = [json.loads(l) for l in open(PROJECT_ROOT / "data" / "processed" / "valid.jsonl")]
print(f"Loaded v1.0 data: {len(train)} train, {len(valid)} valid")

# Collect kernel-doc blocks from v0.7 extraction
# We'll use the function_qa and struct_qa samples as source material
kerneldoc_samples = [s for s in train + valid if s.get("source") in ("kernel_doc", "kernel_documentation")]
print(f"Kernel-doc source samples: {len(kerneldoc_samples)}")

# Generate multi-turn conversations
multiturn = []

for s in kerneldoc_samples:
    q1 = s["messages"][0]["content"]
    a1 = s["messages"][1]["content"]
    source_file = s.get("source_file", "")
    subsystem = s.get("subsystem", "kernel_core")
    difficulty = s.get("difficulty", "L2")
    
    # Extract function/structure name from the answer
    # Look for `name` pattern
    import re
    name_match = re.search(r'`([^`]+)`', a1)
    name = name_match.group(1) if name_match else ""
    
    # Only process samples with substantive content
    if len(a1) < 150 or not name:
        continue
    
    # Turn 2: Follow-up about parameters or implementation details
    # Look for "Parameters:" or "Key fields:" in the answer
    has_params = "Parameters:" in a1 or "parameters" in a1.lower()[:300]
    has_fields = "Key fields:" in a1 or "fields" in a1.lower()[:300]
    has_return = "Returns:" in a1 or "return" in a1.lower()
    
    # Generate follow-up questions based on content
    follow_ups = []
    
    if has_params:
        q2 = f"What are the parameters of {name} and what does each one do?"
        # Extract the parameter section from a1
        param_section = ""
        if "Parameters:" in a1:
            param_section = a1.split("Parameters:")[1].split("\n\n")[0] if "\n\n" in a1.split("Parameters:")[1] else a1.split("Parameters:")[1]
        elif "参数说明" in a1:
            param_section = a1.split("参数说明")[1].split("\n\n")[0] if "\n\n" in a1.split("参数说明")[1] else a1.split("参数说明")[1]
        
        a2 = f"The parameters of {name} are as follows:\n{param_section[:500]}"
        if len(a2) > 50:
            follow_ups.append((q2, a2))
    
    if has_return and not has_params:
        q2 = f"What does {name} return and what is the context in which it should be used?"
        ret_section = ""
        if "Returns:" in a1:
            ret_section = a1.split("Returns:")[1].split("\n\n")[0] if "\n\n" in a1.split("Returns:")[1] else a1.split("Returns:")[1]
        a2 = f"{name} returns: {ret_section[:300]}"
        if len(a2) > 50:
            follow_ups.append((q2, a2))
    
    if has_fields:
        q2 = f"Explain the key fields of {name} and their purposes."
        fields_section = ""
        if "Key fields:" in a1:
            fields_section = a1.split("Key fields:")[1].split("\n\n")[0] if "\n\n" in a1.split("Key fields:")[1] else a1.split("Key fields:")[1]
        a2 = f"The key fields of {name} are:\n{fields_section[:500]}"
        if len(a2) > 50:
            follow_ups.append((q2, a2))
    
    # If no structured sections found, generate a generic follow-up
    if not follow_ups:
        # Use the remaining doc content
        remaining = a1[len(a1.split("\n\n")[0]):].strip() if "\n\n" in a1 else ""
        if len(remaining) > 100:
            q2 = f"Can you elaborate more on the implementation details of {name}?"
            a2 = remaining[:500]
            follow_ups.append((q2, a2))
    
    # Create multi-turn sample (2 turns)
    for q2, a2 in follow_ups[:1]:  # At most 1 follow-up per source
        multiturn.append({
            "messages": [
                {"role": "user", "content": q1},
                {"role": "assistant", "content": a1},
                {"role": "user", "content": q2},
                {"role": "assistant", "content": a2},
            ],
            "type": "multiturn_qa",
            "subsystem": subsystem,
            "source_file": source_file,
            "source": "kernel_doc_multiturn",
            "difficulty": difficulty,
        })

print(f"Generated {len(multiturn)} multi-turn samples")

# Combine with v1.0 data
all_train = train + multiturn
all_valid = valid + multiturn[:len(multiturn)//10]  # 10% to valid

random.shuffle(all_train)
random.shuffle(all_valid)

# Save
output_dir = PROJECT_ROOT / "data" / "processed"
with open(output_dir / "train.jsonl", "w") as f:
    for s in all_train:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")
with open(output_dir / "valid.jsonl", "w") as f:
    for s in all_valid:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

print(f"\nSaved v1.1 dataset:")
print(f"  train.jsonl: {len(all_train)}")
print(f"  valid.jsonl: {len(all_valid)}")

# Stats
types = Counter(s.get("type", "unknown") for s in all_train)
print(f"\nType distribution (train):")
for t, c in types.most_common():
    print(f"  {t}: {c}")

# Multi-turn count
mt = sum(1 for s in all_train if len(s["messages"]) > 2)
print(f"\nMulti-turn samples in train: {mt}")

