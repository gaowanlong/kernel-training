#!/usr/bin/env python3
"""
Extract Kconfig help texts from Linux kernel source and convert to Q&A.
"""

import json, re, random
from pathlib import Path
from collections import Counter

random.seed(42)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
KERNEL_DIR = PROJECT_ROOT / "data" / "raw" / "linux"

SUBSYSTEM_MAP = {
    "init": "kernel_core", "kernel": "process_management", "mm": "memory_management",
    "fs": "file_system", "net": "network_stack", "drivers": "device_drivers",
    "block": "file_system", "arch": "arch_security", "security": "arch_security",
    "ipc": "process_management", "lib": "kernel_core", "crypto": "arch_security",
    "sound": "device_drivers", "virt": "kernel_core", "samples": "kernel_core",
    "usr": "kernel_core", "io_uring": "file_system",
}

DIFFICULTY_MAP = {"bool": "L1", "tristate": "L1", "def_bool": "L1", "def_tristate": "L1",
                   "int": "L2", "hex": "L2", "string": "L2"}


def extract_kconfig_help(filepath: Path) -> list[dict]:
    try:
        content = filepath.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []

    entries = []
    lines = content.split("\n")
    i = 0
    current_config = None
    current_type = None
    current_prompt = None
    current_help = []
    in_help = False

    while i < len(lines):
        raw = lines[i]
        line = raw.strip()

        # Detect new config entry
        config_match = re.match(r"config\s+(\w+)", line)
        if config_match:
            if current_config and current_help:
                help_text = "\n".join(current_help).strip()
                if len(help_text) > 80:
                    entries.append({
                        "config": current_config, "type": current_type or "bool",
                        "prompt": current_prompt or "", "help": help_text,
                    })
            current_config = config_match.group(1)
            current_type = None; current_prompt = None; current_help = []; in_help = False
            i += 1; continue

        # Detect type (bool, tristate, def_bool, int, etc.)
        if current_config and not current_type:
            type_match = re.match(r"(def_)?(bool|tristate|int|hex|string)\b", line)
            if type_match:
                current_type = type_match.group(0)
                prompt_match = re.search(r'"([^"]+)"', line)
                if prompt_match:
                    current_prompt = prompt_match.group(1)
                i += 1; continue

        # Detect prompt on its own line
        if current_config and current_type and not current_prompt:
            prompt_match = re.search(r'prompt\s+"([^"]+)"', line)
            if prompt_match:
                current_prompt = prompt_match.group(1)
                i += 1; continue

        # Detect "help" keyword
        if line == "help" or line.startswith("help "):
            in_help = True
            i += 1; continue

        # Collect help text lines
        if in_help:
            # Help lines are indented (tab or spaces). Detect by checking if line starts with whitespace
            if raw.startswith("\t") or raw.startswith(" "):
                current_help.append(line)
                i += 1; continue
            else:
                in_help = False

        # End of config block
        if line.startswith("config ") or line.startswith("menu") or line.startswith("endmenu") or \
           line.startswith("if ") or line.startswith("endif") or line.startswith("source ") or \
           line.startswith("comment") or line.startswith("choice") or line.startswith("endchoice"):
            if current_config and current_help:
                help_text = "\n".join(current_help).strip()
                if len(help_text) > 80:
                    entries.append({
                        "config": current_config, "type": current_type or "bool",
                        "prompt": current_prompt or "", "help": help_text,
                    })
            if line.startswith("config "):
                config_match = re.match(r"config\s+(\w+)", line)
                if config_match:
                    current_config = config_match.group(1)
                    current_type = None; current_prompt = None; current_help = []; in_help = False
                    i += 1; continue
            current_config = None; current_help = []; in_help = False

        i += 1

    if current_config and current_help:
        help_text = "\n".join(current_help).strip()
        if len(help_text) > 80:
            entries.append({
                "config": current_config, "type": current_type or "bool",
                "prompt": current_prompt or "", "help": help_text,
            })

    return entries


def detect_subsystem(filepath: str) -> str:
    parts = filepath.split("/")
    return SUBSYSTEM_MAP.get(parts[0], "kernel_core") if parts else "kernel_core"


def generate_kconfig_qa(entry: dict, subsystem: str) -> dict:
    config = entry["config"]
    prompt_text = entry["prompt"]
    help_text = entry["help"]
    diff = DIFFICULTY_MAP.get(entry["type"], "L2")
    help_clean = re.sub(r"\n{3,}", "\n\n", help_text).strip()

    if prompt_text:
        question = f"Explain the Linux kernel configuration option {config} ({prompt_text}). What does it do and when should it be used?"
    else:
        question = f"Explain the Linux kernel configuration option {config}. What does it control?"

    answer = f"The `{config}` kernel configuration option is part of the {subsystem} subsystem.\n\n{help_clean[:800]}"

    return {"messages": [{"role": "user", "content": question}, {"role": "assistant", "content": answer}],
            "type": "kconfig_qa", "subsystem": subsystem, "source": "kconfig", "difficulty": diff}


def generate_kconfig_chinese_qa(entry: dict, subsystem: str) -> dict:
    config = entry["config"]
    prompt_text = entry["prompt"]
    help_text = entry["help"]
    diff = DIFFICULTY_MAP.get(entry["type"], "L2")
    help_clean = re.sub(r"\n{3,}", "\n\n", help_text).strip()
    cn_map = {"process_management":"进程管理","memory_management":"内存管理","file_system":"文件系统",
              "network_stack":"网络协议栈","device_drivers":"设备驱动","interrupts":"中断处理",
              "locking":"锁机制","system_calls":"系统调用","debug":"调试与追踪",
              "arch_security":"架构与安全","kernel_core":"内核核心"}
    cn = cn_map.get(subsystem, "内核")
    question = f"请解释 Linux 内核配置选项 {config} 的作用。{'(' + prompt_text + ')' if prompt_text else ''}"
    answer = f"`{config}` 是 Linux 内核中 {cn} 子系统的一个配置选项。\n\n{help_clean[:600]}"
    return {"messages": [{"role": "user", "content": question}, {"role": "assistant", "content": answer}],
            "type": "kconfig_qa", "subsystem": subsystem, "source": "kconfig", "difficulty": diff, "language": "zh"}


def main():
    if not KERNEL_DIR.exists():
        print(f"Kernel source not found at {KERNEL_DIR}"); return

    print("Scanning Kconfig files...")
    kconfig_files = list(KERNEL_DIR.rglob("Kconfig*"))
    kconfig_files = [f for f in kconfig_files if not any(
        p in f.parts for p in ("tools", "scripts", "samples", "Documentation", "usr"))]
    print(f"  Found {len(kconfig_files)} Kconfig files")

    all_entries = []
    for i, fpath in enumerate(kconfig_files):
        if i % 200 == 0:
            print(f"  Processing: {i}/{len(kconfig_files)}")
        entries = extract_kconfig_help(fpath)
        subsystem = detect_subsystem(str(fpath.relative_to(KERNEL_DIR)))
        for e in entries:
            e["subsystem"] = subsystem
            e["file"] = str(fpath.relative_to(KERNEL_DIR))
        all_entries.extend(entries)

    print(f"\n  Total Kconfig entries with help text: {len(all_entries)}")
    good_entries = [e for e in all_entries if len(e["help"]) > 100]
    print(f"  Entries with >100 chars help: {len(good_entries)}")

    # Sample for diversity: max 200 per subsystem
    by_subsystem = {}
    for e in good_entries:
        by_subsystem.setdefault(e["subsystem"], []).append(e)

    sampled = []
    for sub, entries in by_subsystem.items():
        random.shuffle(entries)
        sampled.extend(entries[:200])

    print(f"  Sampled (max 200/subsystem): {len(sampled)}")

    # Generate Q&A
    samples = []
    for e in sampled:
        samples.append(generate_kconfig_qa(e, e["subsystem"]))
        if random.random() < 0.35:
            samples.append(generate_kconfig_chinese_qa(e, e["subsystem"]))

    print(f"\n  Generated Q&A samples: {len(samples)}")

    subsystems = Counter(s["subsystem"] for s in samples)
    print(f"\n  Subsystem distribution:")
    for s, c in subsystems.most_common():
        print(f"    {s}: {c}")

    zh_count = len([s for s in samples if s.get("language") == "zh"])
    print(f"\n  Chinese samples: {zh_count} ({zh_count/len(samples)*100:.0f}%)")

    output_path = PROJECT_ROOT / "data" / "processed" / "kconfig_samples.jsonl"
    with open(output_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"\n  Saved to {output_path}")


if __name__ == "__main__":
    main()
