"""
Data Mixing & Loading
=====================

Streaming data pipeline for KT_GPT pretraining.

Two-phase data mix:
  Phase 1 (1B tokens)  — 60% FineWeb-Edu + 40% DCLM  (fluency + stability)
  Phase 2 (20B tokens) — 40% Stack v2 + 30% FineWeb + 20% DCLM + 10% OpenWebMath

Key design decisions:
  - Full streaming: never loads entire dataset into RAM
  - Document packing: concatenates documents into fixed-length sequences
    with BOS/EOS boundaries to maximize GPU utilization
  - Deterministic: seeded shuffling for reproducible ablation comparisons
"""

from __future__ import annotations

import math
import itertools
from dataclasses import dataclass, field
from typing import Iterator, Optional

import torch
from torch.utils.data import IterableDataset, DataLoader

from datasets import load_dataset, interleave_datasets
from transformers import AutoTokenizer


# ═══════════════════════════════════════════════════════════════════════
#  Data Source Configuration
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class DataSourceConfig:
    """Configuration for a single HuggingFace data source.

    Encapsulates the dataset path, any subset/split, the text column name,
    streaming filter criteria, and the sampling weight for mixing.
    """
    name: str
    hf_path: str
    weight: float                          # sampling probability in the mix
    text_column: str = "text"
    hf_subset: Optional[str] = None
    split: str = "train"
    filter_column: Optional[str] = None    # e.g. "score" for FineWeb-Edu
    filter_min: Optional[float] = None     # keep rows >= this value


# ── Phase 1: Foundation (stability + fluency) ────────────────────────

PHASE_1_SOURCES: list[DataSourceConfig] = [
    DataSourceConfig(
        name="fineweb-edu",
        hf_path="HuggingFaceFW/fineweb-edu",
        hf_subset="sample-10BT",           # 10B-token sample, enough for 1B
        weight=0.60,
        text_column="text",
        filter_column="score",
        filter_min=4.0,                     # keep only high-quality (≥4/5)
    ),
    DataSourceConfig(
        name="dclm-baseline",
        hf_path="mlfoundations/dclm-baseline-1.0",
        weight=0.40,
        text_column="text",
    ),
]


# ── Phase 2: Structured intelligence ─────────────────────────────────

PHASE_2_SOURCES: list[DataSourceConfig] = [
    DataSourceConfig(
        name="stack-v2-python",
        hf_path="bigcode/the-stack-v2-train-smol-ids",
        weight=0.40,
        text_column="content",
    ),
    DataSourceConfig(
        name="fineweb-edu",
        hf_path="HuggingFaceFW/fineweb-edu",
        hf_subset="sample-100BT",
        weight=0.30,
        text_column="text",
        filter_column="score",
        filter_min=4.0,
    ),
    DataSourceConfig(
        name="dclm-baseline",
        hf_path="mlfoundations/dclm-baseline-1.0",
        weight=0.20,
        text_column="text",
    ),
    DataSourceConfig(
        name="openwebmath",
        hf_path="open-web-math/open-web-math",
        weight=0.10,
        text_column="text",
    ),
]


# ═══════════════════════════════════════════════════════════════════════
#  Tokenizer
# ═══════════════════════════════════════════════════════════════════════

def get_tokenizer(model_name: str = "mistralai/Mistral-7B-v0.1") -> AutoTokenizer:
    """Load the Mistral tokenizer (32k vocab, open access).

    Sets pad_token = eos_token since Mistral doesn't define one.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


# ═══════════════════════════════════════════════════════════════════════
#  Packed Dataset (Streaming)
# ═══════════════════════════════════════════════════════════════════════

class PackedDataset(IterableDataset):
    """Streaming dataset that tokenizes + packs documents into fixed-length sequences.

    Why packing?  Naive padding wastes 30-60% of compute on pad tokens.
    Instead, we concatenate tokenized documents end-to-end (separated by
    EOS tokens) and slice into fixed-length chunks.  Every token in every
    batch is a real training token.

    Flow:
      HF streaming dataset → tokenize each document → append EOS →
      accumulate into a token buffer → yield seq_len chunks → discard
      any remainder shorter than seq_len.
    """

    def __init__(
        self,
        sources: list[DataSourceConfig],
        tokenizer: AutoTokenizer,
        seq_len: int = 4096,
        total_tokens: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.sources = sources
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.total_tokens = total_tokens
        self.seed = seed
        self.eos_id = tokenizer.eos_token_id
        self.bos_id = tokenizer.bos_token_id

    def _load_and_interleave(self) -> Iterator[dict]:
        """Load all sources as streaming datasets, interleave by weight."""
        datasets = []
        weights = []

        for src in self.sources:
            ds = load_dataset(
                src.hf_path,
                name=src.hf_subset,
                split=src.split,
                streaming=True,
            )
            # Apply quality filter if specified
            if src.filter_column and src.filter_min is not None:
                col = src.filter_column
                threshold = src.filter_min
                ds = ds.filter(lambda row: row.get(col, 0) >= threshold)

            # Rename text column to "text" for uniformity
            if src.text_column != "text":
                ds = ds.rename_column(src.text_column, "text")

            datasets.append(ds)
            weights.append(src.weight)

        # Normalize weights (interleave_datasets requires probabilities)
        total_w = sum(weights)
        probs = [w / total_w for w in weights]

        return iter(interleave_datasets(
            datasets,
            probabilities=probs,
            seed=self.seed,
            stopping_strategy="all_exhausted",
        ))

    def _tokenize_stream(self, doc_stream: Iterator[dict]) -> Iterator[int]:
        """Yield individual token IDs from a stream of documents.

        Each document is tokenized and terminated with EOS.
        BOS is prepended at the start of each document.
        """
        for doc in doc_stream:
            text = doc.get("text", "")
            if not text or len(text.strip()) < 20:
                continue
            # Tokenize without special tokens — we add them manually
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
            if not token_ids:
                continue
            # BOS + content + EOS
            yield self.bos_id
            yield from token_ids
            yield self.eos_id

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        """Yield packed sequences of exactly seq_len tokens.

        Returns dicts with:
          - input_ids: (seq_len,) — the token sequence
          - labels:    (seq_len,) — same as input_ids (shifted internally by loss fn)
        """
        doc_stream = self._load_and_interleave()
        token_stream = self._tokenize_stream(doc_stream)

        buffer: list[int] = []
        tokens_yielded = 0

        for token_id in token_stream:
            buffer.append(token_id)

            if len(buffer) >= self.seq_len:
                # Slice exactly seq_len tokens
                chunk = buffer[:self.seq_len]
                buffer = buffer[self.seq_len:]

                ids = torch.tensor(chunk, dtype=torch.long)
                yield {"input_ids": ids, "labels": ids.clone()}

                tokens_yielded += self.seq_len
                if self.total_tokens and tokens_yielded >= self.total_tokens:
                    return

        # Discard any remainder shorter than seq_len


# ═══════════════════════════════════════════════════════════════════════
#  DataLoader Factory
# ═══════════════════════════════════════════════════════════════════════

def create_dataloader(
    phase: int,
    batch_size: int,
    seq_len: int = 4096,
    total_tokens: Optional[int] = None,
    seed: int = 42,
    num_workers: int = 2,
    tokenizer_name: str = "mistralai/Mistral-7B-v0.1",
) -> DataLoader:
    """Create a streaming DataLoader for the specified training phase.

    Args:
        phase:       1 or 2 — selects the data mix
        batch_size:  micro batch size (per GPU step, before grad accum)
        seq_len:     sequence length for packing (default 4096)
        total_tokens: stop after this many tokens (None = exhaust dataset)
        seed:        random seed for reproducible interleaving
        num_workers:  DataLoader workers (keep low for streaming)
        tokenizer_name: HF tokenizer to use

    Returns:
        DataLoader yielding batches of {input_ids, labels} tensors
    """
    sources = PHASE_1_SOURCES if phase == 1 else PHASE_2_SOURCES
    tokenizer = get_tokenizer(tokenizer_name)

    dataset = PackedDataset(
        sources=sources,
        tokenizer=tokenizer,
        seq_len=seq_len,
        total_tokens=total_tokens,
        seed=seed,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,          # avoid partial batches at end
        prefetch_factor=4 if num_workers > 0 else None,
    )
