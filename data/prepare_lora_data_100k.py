from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from datasets import load_dataset
from datasets.exceptions import DataFilesNotFoundError
from datasets.utils.info_utils import NonMatchingSplitsSizesError
from requests.exceptions import ReadTimeout

SYSTEM_CONTEXT_PREFIX = (
    "Use ONLY the provided context to answer. "
    "Do not use outside knowledge."
)

@dataclass
class SourceSpec:
    name: str
    hf_path: str
    split: str
    target_count: int

# We want 100K total instruction pairs.
# Let's use:
# 30K ShareGPT
# 20K Alpaca
# 10K Dolly
# 10K OASST1
# 15K FLAN
# 15K Tool Calling
INSTRUCTION_SOURCES = [
    # Primary entry for ShareGPT — resolved via SHAREGPT_FALLBACK_PATHS below.
    # hf_path here is only a label; actual loading goes through the fallback list.
    SourceSpec("sharegpt", "Aeala/ShareGPT_Vicuna_unfiltered", "train", 30_000),
    SourceSpec("alpaca", "tatsu-lab/alpaca", "train", 20_000),
    SourceSpec("dolly", "databricks/databricks-dolly-15k", "train", 10_000),
    SourceSpec("oasst1", "OpenAssistant/oasst1", "train", 10_000),
    SourceSpec("flan", "Muennighoff/flan", "train", 15_000),
]

# Ordered from most-preferred to least-preferred.
# anon8231489123 has been removed — that repo no longer exists on HuggingFace.
SHAREGPT_FALLBACK_PATHS = [
    ("Aeala/ShareGPT_Vicuna_unfiltered", "ShareGPT_V4.3_unfiltered_cleaned_split.json"),
    ("Aeala/ShareGPT_Vicuna_unfiltered", "ShareGPT_2023.05.04v0_Wasteland_Edition.json"),
    # RyokoAI mirror — 90K conversations split across two files
    ("RyokoAI/ShareGPT52K", "sg_90k_part1.json"),
    ("RyokoAI/ShareGPT52K", "sg_90k_part2.json"),
    # Older 52K version kept as last resort
    ("RyokoAI/ShareGPT52K", "old/sg_52k.json"),
]

def _norm_text(value: str) -> str:
    return " ".join((value or "").replace("\r", "\n").split()).strip()

def _format_pair(prompt: str, response: str, source: str) -> dict:
    return {
        "prompt": _norm_text(prompt),
        "response": _norm_text(response),
        "source": source,
    }

def _safe_sample(items: list[dict], n: int, rng: random.Random) -> list[dict]:
    if len(items) <= n:
        return items
    return rng.sample(items, n)

def _extract_instruction_pairs(dataset_name: str, rows: Iterable[dict], max_pairs: int | None = None) -> list[dict]:
    out: list[dict] = []
    for row in rows:
        if dataset_name == "alpaca":
            inst = _norm_text(row.get("instruction", ""))
            inp = _norm_text(row.get("input", ""))
            resp = _norm_text(row.get("output", ""))
            if not inst or not resp:
                continue
            prompt = f"{inst}\n{inp}" if inp else inst
            out.append(_format_pair(prompt, resp, dataset_name))
        elif dataset_name == "dolly":
            inst = _norm_text(row.get("instruction", ""))
            ctx = _norm_text(row.get("context", ""))
            resp = _norm_text(row.get("response", ""))
            if not inst or not resp:
                continue
            prompt = f"{inst}\n{ctx}" if ctx else inst
            out.append(_format_pair(prompt, resp, dataset_name))
        elif dataset_name == "sharegpt":
            conv = row.get("conversations") or []
            if len(conv) < 2:
                continue
            user = conv[0].get("value", "") if isinstance(conv[0], dict) else ""
            assistant = conv[1].get("value", "") if isinstance(conv[1], dict) else ""
            if user and assistant:
                out.append(_format_pair(user, assistant, dataset_name))
        elif dataset_name == "oasst1":
            text = _norm_text(row.get("text", ""))
            role = _norm_text(row.get("role", ""))
            parent = row.get("parent_id")
            if text and role == "assistant" and parent is not None:
                out.append(_format_pair("Continue the conversation.", text, dataset_name))
        elif dataset_name == "flan":
            inp = _norm_text(row.get("inputs", ""))
            targ = _norm_text(row.get("targets", ""))
            if inp and targ:
                out.append(_format_pair(inp, targ, dataset_name))
        if max_pairs is not None and len(out) >= max_pairs:
            break
    return out

def _build_tool_math_synthetic(n: int, rng: random.Random) -> list[dict]:
    ops = ["+", "-", "*", "/"]
    samples: list[dict] = []
    for _ in range(n):
        a = rng.randint(1, 9999)
        b = rng.randint(1, 9999)
        op = rng.choice(ops)
        expr = f"{a}{op}{b}"
        prompt = f"User: What is {expr}?"
        # Force strict tool calling format for math
        response = f"<tool_call>{{\"name\":\"calculator\",\"arguments\":{{\"expression\":\"{expr}\"}}}}</tool_call>"
        samples.append(_format_pair(prompt, response, "synthetic_tool_math"))
    return samples

def _build_context_synthetic(n: int, rng: random.Random) -> list[dict]:
    entities = [
        ("Sahara Desert", "North Africa", "9.2 million square kilometers"),
        ("Amazon River", "Brazil, Peru, and Colombia", "approximately 6,400 kilometers"),
        ("Pacific Ocean", "165.25 million square kilometers", "largest ocean"),
        ("DNA", "double helix", "Watson and Crick"),
        ("Mount Everest", "Himalayas", "8,848 meters"),
        ("Eiffel Tower", "Paris", "330 meters tall"),
    ]
    out: list[dict] = []
    for _ in range(n):
        name, fact_a, fact_b = rng.choice(entities)
        prompt = (
            f"{SYSTEM_CONTEXT_PREFIX}\n\n"
            f"Context:\n{name}: {fact_a}; {fact_b}.\n\n"
            f"Question: What does the context say about {name}?"
        )
        response = f"The context says {name} is associated with {fact_a} and {fact_b}."
        out.append(_format_pair(prompt, response, "synthetic_context"))
    return out

def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def _hf_dataset_file_url(repo_id: str, filename: str) -> str:
    return f"https://huggingface.co/datasets/{repo_id}/resolve/main/{filename}"


def _load_instruction_dataset(spec: SourceSpec):
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

    def _load_with_recovery(path: str, split: str, data_files: str | None = None):
        last_exc: Exception | None = None
        for attempt in range(4):
            try:
                try:
                    if data_files is not None:
                        return load_dataset("json", data_files=data_files, split=split, streaming=True)
                    return load_dataset(path, split=split, streaming=True)
                except TypeError:
                    if data_files is not None:
                        return load_dataset("json", data_files=data_files, split=split)
                    return load_dataset(path, split=split)
                except NonMatchingSplitsSizesError:
                    if data_files is not None:
                        return load_dataset(
                            "json",
                            data_files=data_files,
                            split=split,
                            verification_mode="no_checks",
                        )
                    return load_dataset(path, split=split, verification_mode="no_checks")
            except Exception as exc:
                last_exc = exc
                if attempt == 3:
                    break
                sleep_s = 2 ** attempt
                print(
                    f"[prepare_lora_data] transient load error for {path} "
                    f"(attempt {attempt + 1}/4): {exc}. Retrying in {sleep_s}s..."
                )
                time.sleep(sleep_s)
                continue
        if last_exc is not None:
            raise last_exc
        raise RuntimeError(f"Dataset load failed unexpectedly for {path}")

    if spec.name != "sharegpt":
        return _load_with_recovery(spec.hf_path, spec.split), spec.hf_path

    last_error: Exception | None = None
    for candidate, filename in SHAREGPT_FALLBACK_PATHS:
        try:
            data_file = _hf_dataset_file_url(candidate, filename)
            print(f"[prepare_lora_data] Trying ShareGPT source: {candidate}/{filename}")
            ds = _load_with_recovery(candidate, spec.split, data_files=data_file)
            return ds, f"{candidate}/{filename}"
        except Exception as exc:  # noqa: BLE001 — broad catch is intentional for fallback chain
            last_error = exc
            print(f"[prepare_lora_data] ShareGPT source unavailable: {candidate}/{filename} ({exc})")
            continue

    raise RuntimeError(
        "Unable to load any ShareGPT fallback dataset. "
        f"Tried: {SHAREGPT_FALLBACK_PATHS}"
    ) from last_error

def build_dataset(
    output_path: str,
    seed: int = 42,
    context_target: int = 15_000,
    tool_target: int = 15_000,
) -> None:
    rng = random.Random(seed)
    instruction_rows: list[dict] = []
    
    # 1. Load HuggingFace Datasets (target ~85k)
    for spec in INSTRUCTION_SOURCES:
        ds, resolved_path = _load_instruction_dataset(spec)
        print(f"[prepare_lora_data] Loading {spec.name} from {resolved_path}")
        extracted = _extract_instruction_pairs(spec.name, ds, max_pairs=spec.target_count * 3)
        sampled = _safe_sample(extracted, spec.target_count, rng)
        instruction_rows.extend(sampled)

    # 2. Add Synthetics
    context_rows = _build_context_synthetic(context_target, rng)
    tool_rows = _build_tool_math_synthetic(tool_target, rng)

    full = instruction_rows + context_rows + tool_rows
    
    # Shuffle and trim to exact 100K if it went over
    rng.shuffle(full)
    if len(full) > 100_000:
        full = full[:100_000]

    _write_jsonl(Path(output_path), full)

    print(
        f"[prepare_lora_data] wrote {len(full)} rows "
        f"(instruction={len(instruction_rows)}, context={len(context_rows)}, tool={len(tool_rows)}) "
        f"to {output_path}"
    )

def main() -> None:
    parser = argparse.ArgumentParser(description="Build 100K LoRA SFT dataset")
    parser.add_argument("--output", default="data/lora_sft_100k.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--context-target", type=int, default=15_000)
    parser.add_argument("--tool-target", type=int, default=15_000)
    args = parser.parse_args()
    build_dataset(
        output_path=args.output,
        seed=args.seed,
        context_target=args.context_target,
        tool_target=args.tool_target,
    )

if __name__ == "__main__":
    main()
