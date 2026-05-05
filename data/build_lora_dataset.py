"""
build_lora_dataset.py
=====================
Builds the final LoRA fine-tuning dataset.

Target composition (midpoints used for derivation):
  Alpaca (COMPLETE dataset, ~52 K)   55 %   50–60 %
  RAG grounding                      17.5%  15–20 %
  Concise answers                    12.5%  10–15 %
  No-info refusal                     7.5%   5–10 %
  Tool-result format                 12.5%  10–15 %
  ─────────────────────────────────────────────────
  Estimated total                    ~94 K

Deriving custom counts:
  The script loads the COMPLETE Alpaca split first (no cap / no sampling).
  It then treats the actual Alpaca row count as 55 % of the target total
  and computes each custom category count from that anchor:

    total   = round(alpaca_count / 0.55)
    rag     = round(total * 0.175)
    concise = round(total * 0.125)
    refusal = round(total * 0.075)
    tool    = total − alpaca_count − rag − concise − refusal  (exact remainder)

  This keeps the ratios exact even if Alpaca's upstream row count changes.

Output format — Alpaca 4-field format (one JSON object per line):
  {
    "instruction": "The task description / system behaviour",
    "input":       "The user data / context / question  (empty string if none)",
    "output":      "The expected model response",
    "text":        "Full formatted prompt the model sees at train AND inference time",
    "source":      "alpaca | rag | concise | refusal | tool_fmt"
  }

WHY this format instead of flat prompt/response?
  - Keeps instruction separate from data — model learns the difference
  - The `text` field is the EXACT string tokenised at training time
  - Inference must produce the same `text` prefix → train/inference aligned
  - Matches original Alpaca template so pre-trained knowledge transfers cleanly

Alpaca prompt template (used for both training & inference):
  Below is an instruction that describes a task, paired with an input that
  provides further context. Write a response that appropriately completes
  the request.

  ### Instruction:
  {instruction}

  ### Input:
  {input}

  ### Response:

  (when input is empty the "paired with an input..." sentence is omitted)

Usage:
  python data/build_lora_dataset.py                         # full dataset (default)
  python data/build_lora_dataset.py --output data/my.jsonl  # custom output path
  python data/build_lora_dataset.py --custom-only           # skip Alpaca download
  python data/build_lora_dataset.py --alpaca-only           # write Alpaca only
  python data/build_lora_dataset.py --seed 123              # reproducibility
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

# ── optional HF import (only needed for Alpaca) ─────────────────────────────
try:
    from datasets import load_dataset as _hf_load
    HF_AVAILABLE = True
except ImportError:  # pragma: no cover
    HF_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════
#  Alpaca prompt template  (single source of truth)
#  MUST match what the inference backend constructs at query time.
# ═══════════════════════════════════════════════════════════════════════════

_PROMPT_WITH_INPUT = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)

_PROMPT_NO_INPUT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n"
)


def _make_text(instruction: str, input_: str, output: str) -> str:
    """Build the full formatted string (prompt + response) stored in `text`."""
    if input_.strip():
        return _PROMPT_WITH_INPUT.format(instruction=instruction, input=input_) + output
    return _PROMPT_NO_INPUT.format(instruction=instruction) + output


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _norm(text: str) -> str:
    return " ".join((text or "").replace("\r", "\n").split()).strip()


def _row(instruction: str, input_: str, output: str, source: str) -> dict:
    instruction = _norm(instruction)
    input_      = _norm(input_)
    output      = _norm(output)
    return {
        "instruction": instruction,
        "input":       input_,
        "output":      output,
        "text":        _make_text(instruction, input_, output),
        "source":      source,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Alpaca transformation pipeline
#
#  Philosophy: fix + transform first — delete only when necessary.
#
#  Pipeline per row:
#    1. Hard-delete: subjective / open-ended / creative  (no useful signal)
#    2. Math detection:
#         Explicit arithmetic  → convert to tool_fmt (stronger signal)
#         Implicit math + nums → drop  (can't auto-convert reliably)
#    3. Template-phrase strip: remove "according to the context" etc.
#    4. Output truncation:    long outputs → first 2 sentences (not deleted)
#
# ═══════════════════════════════════════════════════════════════════════════

# ── 1. Hard-delete: subjective / low-signal ───────────────────────────────
_LOW_SIGNAL_RE = re.compile(
    r"(?:"
    r"write\s+a\s+(?:short\s+)?(?:story|poem|essay|letter|song|speech|blog|article|paragraph|narrative|dialogue|script|cover\s+letter)"
    r"|compose\s+a\s+(?:poem|song|letter|essay|story)"
    r"|tell\s+(?:me\s+)?a\s+(?:story|joke|tale|anecdote)"
    r"|describe\s+a\s+time\s+(?:when|you)"
    r"|share\s+(?:your\s+)?(?:thoughts|feelings|opinion|experience)"
    r"|give\s+(?:your\s+)?(?:opinion|thoughts)\s+on"
    r"|what\s+(?:do\s+you\s+think|are\s+your\s+thoughts)\s+about"
    r"|make\s+up\s+a|imagine\s+(?:you\s+are|a\s+world)"
    r"|\bcreative\s+writing\b|\bfiction\b|roleplay|role-play"
    r"|pretend\s+(?:you\s+are|to\s+be)|write\s+from\s+the\s+perspective"
    r"|personal\s+(?:opinion|experience|story)"
    r"|benefits\s+of\s+(?:meditation|yoga|exercise|journaling|mindfulness|gratitude)"
    r"|tips\s+for\s+(?:better\s+sleep|stress|anxiety|motivation|productivity|self\s*-?care)"
    r"|how\s+to\s+(?:be\s+happy|find\s+peace|overcome\s+fear|be\s+more\s+confident|deal\s+with\s+stress)"
    r")",
    re.IGNORECASE,
)


def _is_low_signal(row: dict) -> bool:
    return bool(_LOW_SIGNAL_RE.search(row["instruction"]))


# ── 2. Math detection ─────────────────────────────────────────────────────
# Explicit: clear arithmetic operation the model would compute by itself.
# Implicit: estimation/ratio/average keyword + numbers present.

_EXPLICIT_MATH_RE = re.compile(
    r"(?:"
    r"^\s*(?:calculat|comput|evaluat|simplif|integrat|differentiat|factori[sz])\w*"
    r"|solve\s+(?:the\s+)?(?:equation|expression|following|for\s+[a-z])"
    r"|find\s+(?:the\s+)?(?:derivative|integral|root|value\s+of\s+[a-z])"
    r"|what\s+is\s+\d[\d\s]*[\+\-\xd7\xf7\*\/\^]"
    r"|compute\s+the\s+(?:sum|product|difference|quotient)"
    r")",
    re.IGNORECASE | re.MULTILINE,
)
_IMPLICIT_MATH_RE = re.compile(
    r"\b(?:estimat|how\s+many|how\s+much|how\s+long|how\s+far|averag|ratio|percentag|probabilit)\w*",
    re.IGNORECASE,
)
_HAS_DIGITS_RE = re.compile(r"\b\d+\b")

# Matches "a OP b" patterns for explicit conversion to tool_fmt
_ARITH_EXPR_RE = re.compile(
    r"(\d[\d,]*(?:\.\d+)?)\s*([\+\-\*\xd7\xf7\/])\s*(\d[\d,]*(?:\.\d+)?)",
)
_OP_MAP = {"+": lambda a, b: a + b, "-": lambda a, b: a - b,
           "*": lambda a, b: a * b, "\xd7": lambda a, b: a * b,
           "/": lambda a, b: a / b if b else None, "\xf7": lambda a, b: a / b if b else None}


def _try_convert_to_tool_fmt(row: dict, rng: "random.Random") -> "dict | None":
    """If the instruction contains a simple a OP b pattern, convert to tool_fmt."""
    text = row["instruction"] + " " + row.get("input", "")
    m = _ARITH_EXPR_RE.search(text)
    if not m:
        return None
    try:
        a = float(m.group(1).replace(",", ""))
        op = m.group(2)
        b = float(m.group(3).replace(",", ""))
        result_f = _OP_MAP.get(op, lambda *_: None)(a, b)
        if result_f is None:
            return None
        result = int(result_f) if result_f == int(result_f) else round(result_f, 4)
    except (ValueError, ZeroDivisionError):
        return None

    question = text.strip().rstrip(".").rstrip("?") + "?"
    # Import here to avoid forward-reference — _TOOL_INSTR defined later in file.
    # We build a minimal tool_fmt row directly.
    instr  = "The Python calculator returned a result. Use it to answer the question."
    input_ = f"Question: {question}\nPython result: {result}"
    ans    = f"The result is {result:,}." if isinstance(result, int) else f"The result is {result}."
    return _row(instr, input_, ans, "tool_fmt")


def _is_math(row: dict) -> bool:
    combined = row["instruction"] + " " + row.get("input", "")
    if _EXPLICIT_MATH_RE.search(combined):
        return True
    if _IMPLICIT_MATH_RE.search(combined) and _HAS_DIGITS_RE.search(combined):
        return True
    return False


# ── 3. Template-phrase removal ────────────────────────────────────────────
_TEMPLATE_PHRASE_RE = re.compile(
    r"(?:according\s+to\s+(?:the\s+)?(?:context|provided\s+(?:information|context|text))"
    r"|the\s+context\s+(?:states|says|tells\s+us|indicates|shows)\s+that"
    r"|based\s+on\s+(?:the\s+)?(?:provided\s+)?(?:context|information)"
    r"|as\s+(?:stated|mentioned)\s+in\s+(?:the\s+)?context)"
    r"\s*[,.]?\s*",
    re.IGNORECASE,
)


def _strip_template_phrases(text: str) -> str:
    cleaned = _TEMPLATE_PHRASE_RE.sub("", text).strip()
    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned or text  # fallback to original if result is empty


# ── 4. Output truncation ──────────────────────────────────────────────────
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"\'])")
_MAX_OUTPUT_CHARS  = 350  # only truncate beyond this; don't delete


def _truncate_output(text: str, max_sentences: int = 2) -> str:
    sentences = _SENTENCE_SPLIT_RE.split(text.strip())
    result = " ".join(sentences[:max_sentences]).strip()
    if len(result) > _MAX_OUTPUT_CHARS:
        result = result[:_MAX_OUTPUT_CHARS].rsplit(" ", 1)[0] + "."
    return result


# ── Master transformation function ────────────────────────────────────────
def _clean_alpaca_row(row: dict, rng: "random.Random") -> "dict | None":
    """Transform one Alpaca row through the pipeline.

    Returns:
        None  → discard the row
        dict  → cleaned row (source may change to 'tool_fmt' if converted)
    """
    # Step 1: hard-delete subjective / creative / open-ended
    if _is_low_signal(row):
        return None

    # Step 2: math handling
    if _is_math(row):
        converted = _try_convert_to_tool_fmt(row, rng)
        return converted  # None if can't convert → discarded

    # Step 3: strip template phrases from output
    output = _strip_template_phrases(row["output"])

    # Step 4: truncate long outputs (don't delete)
    if len(output) > _MAX_OUTPUT_CHARS:
        output = _truncate_output(output)

    # Rebuild with cleaned output
    return _row(row["instruction"], row.get("input", ""), output, "alpaca")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[build_lora_dataset] wrote {len(rows):,} rows \u2192 {path}")


# ═══════════════════════════════════════════════════════════════════════════
#  1. Alpaca loader
# ═══════════════════════════════════════════════════════════════════════════

def load_alpaca(rng: random.Random) -> list[dict]:
    """Load and clean the complete Alpaca dataset.

    Transformation pipeline (in order):
      1. Hard-delete  — subjective / creative / open-ended
      2. Math convert — explicit arithmetic  → tool_fmt
                        implicit math + nums → discard
      3. Strip        — template phrases from outputs
      4. Truncate     — long outputs to ≤ 2 sentences  (NOT deleted)
    """
    if not HF_AVAILABLE:
        raise ImportError("datasets package is required: pip install datasets")

    print("[build_lora_dataset] Loading COMPLETE Alpaca dataset...")
    ds = _hf_load("tatsu-lab/alpaca", split="train")

    raw: list[dict] = []
    for row in ds:
        inst = _norm(row.get("instruction", ""))
        inp  = _norm(row.get("input",       ""))
        resp = _norm(row.get("output",      ""))
        if not inst or not resp:
            continue
        raw.append(_row(inst, inp, resp, "alpaca"))

    kept_alpaca: list[dict] = []
    converted_tool: list[dict] = []
    discarded = 0

    for row in raw:
        result = _clean_alpaca_row(row, rng)
        if result is None:
            discarded += 1
        elif result["source"] == "tool_fmt":
            converted_tool.append(result)
        else:
            kept_alpaca.append(result)

    print(f"[build_lora_dataset] Alpaca raw            : {len(raw):,}")
    print(f"[build_lora_dataset] Discarded             : {discarded:,}")
    print(f"[build_lora_dataset] Converted → tool_fmt  : {len(converted_tool):,}")
    print(f"[build_lora_dataset] Alpaca kept           : {len(kept_alpaca):,}")

    return kept_alpaca, converted_tool


# ═══════════════════════════════════════════════════════════════════════════
#  2. Custom data generators
#
#  Each generator produces rows in the same 4-field format so the
#  training loop sees ONE consistent structure regardless of source.
#
#  The split between `instruction` and `input` is intentional:
#    instruction = WHAT to do  (stable system-level directive)
#    input       = DATA to work with  (changes per query at inference)
# ═══════════════════════════════════════════════════════════════════════════

# ── 2a. RAG grounding (~17.5% of total) ─────────────────────────────────
#
#  instruction = "Answer using the provided context." (grounding directive)
#  input       = "Context: <factual snippet>\nQuestion: <question>"
#  output      = clean, natural-language answer  ← NO "according to context"
#
#  The output must read like normal knowledge, not a template sentence.
#  The model learns: "use the context, but write naturally."
#
#  Pool format: (name, context_text, clean_answer)
#    name         — used in the question
#    context_text — the snippet injected at inference time
#    clean_answer — pre-written natural answer (grammar-checked, complete)

_RAG_POOL: list[tuple[str, str, str]] = [
    # Technology / AI
    ("ForgeTube",
     "ForgeTube generates videos from text prompts using diffusion models.",
     "ForgeTube generates videos from text prompts using diffusion models."),
    ("Stable Diffusion",
     "Stable Diffusion is an open-source image generation model developed by Stability AI.",
     "Stable Diffusion is an open-source image generation model developed by Stability AI."),
    ("GPT-4",
     "GPT-4 is a large multimodal language model created by OpenAI.",
     "GPT-4 is a large multimodal language model created by OpenAI."),
    ("KTGPT",
     "KTGPT is a 1B-parameter Mixture-of-Experts model built for constrained-device deployment.",
     "KTGPT is a 1B-parameter Mixture-of-Experts model designed to run on constrained devices."),
    ("LoRA",
     "LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique for large models.",
     "LoRA is a parameter-efficient fine-tuning technique that trains small rank-decomposition matrices instead of full model weights."),
    ("RAG",
     "RAG (Retrieval-Augmented Generation) combines document retrieval with language model generation.",
     "RAG combines document retrieval with language model generation to ground answers in retrieved text."),
    ("PyTorch",
     "PyTorch is an open-source machine learning framework developed primarily by Meta AI.",
     "PyTorch is an open-source machine learning framework developed primarily by Meta AI."),
    ("Hugging Face Transformers",
     "The Hugging Face Transformers library provides thousands of pretrained models for NLP and beyond.",
     "The Hugging Face Transformers library provides thousands of pretrained models for NLP and related tasks."),
    # Science
    ("the Sahara Desert",
     "The Sahara Desert is the largest hot desert on Earth, covering 9.2 million square kilometres in North Africa.",
     "The Sahara Desert is the largest hot desert on Earth, covering 9.2 million square kilometres across North Africa."),
    ("the Amazon River",
     "The Amazon River is the largest river by discharge volume, flowing approximately 6,400 km through South America.",
     "The Amazon River is the largest river by discharge volume, flowing approximately 6,400 km through South America."),
    ("the Pacific Ocean",
     "The Pacific Ocean is the largest and deepest ocean, covering 165.25 million square kilometres.",
     "The Pacific Ocean is the largest and deepest ocean on Earth, covering about 165 million square kilometres."),
    ("DNA",
     "DNA is a double-helix molecule that carries the genetic instructions for life.",
     "DNA is a double-helix molecule that carries the genetic instructions for all living organisms."),
    ("Mount Everest",
     "Mount Everest is the highest mountain above sea level, standing 8,848 metres in the Himalayas.",
     "Mount Everest is the highest mountain above sea level, standing 8,848 metres tall in the Himalayas."),
    ("black holes",
     "Black holes are regions of spacetime where gravity is so strong that nothing, not even light, can escape.",
     "Black holes are regions of spacetime where gravity is so strong that nothing, not even light, can escape."),
    ("photosynthesis",
     "Photosynthesis is the process plants use to convert sunlight into glucose, releasing oxygen as a byproduct.",
     "Photosynthesis is the process by which plants convert sunlight, water, and CO\u2082 into glucose, releasing oxygen as a byproduct."),
    ("quantum entanglement",
     "Quantum entanglement is a phenomenon where two particles remain correlated regardless of the distance separating them.",
     "Quantum entanglement is a phenomenon where two particles remain correlated regardless of how far apart they are."),
    # History / Culture
    ("the Eiffel Tower",
     "The Eiffel Tower is an iron lattice tower in Paris, standing 330 metres tall, completed in 1889.",
     "The Eiffel Tower is an iron lattice tower in Paris standing 330 metres tall. It was completed in 1889."),
    ("the Great Wall of China",
     "The Great Wall of China is a series of fortifications stretching over 21,000 kilometres across China.",
     "The Great Wall of China is a series of fortifications stretching over 21,000 kilometres across northern China."),
    ("World War II",
     "World War II was the deadliest global conflict in history, fought between 1939 and 1945.",
     "World War II was the deadliest global conflict in history, lasting from 1939 to 1945."),
    ("the Renaissance",
     "The Renaissance was a cultural and intellectual movement that emerged in Italy in the 14th century.",
     "The Renaissance was a cultural and intellectual movement that began in Italy in the 14th century and spread across Europe."),
    ("Shakespeare",
     "Shakespeare was an English playwright and poet widely regarded as the greatest writer in the English language.",
     "Shakespeare was an English playwright and poet, widely regarded as the greatest writer in the English language."),
    # Computing
    ("binary search",
     "Binary search is an efficient algorithm that halves the search space at each step.",
     "Binary search is an efficient algorithm that finds a target value by repeatedly halving the search space."),
    ("gradient descent",
     "Gradient descent is an optimisation algorithm that iteratively moves parameters toward a function's minimum.",
     "Gradient descent is an optimisation algorithm that iteratively adjusts parameters to minimise a loss function."),
    ("backpropagation",
     "Backpropagation is the algorithm used to train neural networks by computing gradients via the chain rule.",
     "Backpropagation trains neural networks by computing gradients of the loss with respect to each parameter using the chain rule."),
    ("a hash table",
     "A hash table is a data structure that maps keys to values using a hash function for O(1) average lookups.",
     "A hash table is a data structure that maps keys to values using a hash function, giving average O(1) lookup time."),
    ("Big-O notation",
     "Big-O notation describes algorithm complexity as a function of input size, focusing on worst-case growth.",
     "Big-O notation describes how an algorithm's runtime or space requirements grow as a function of input size."),
    # Health / Biology
    ("insulin",
     "Insulin is a hormone produced by the pancreas that regulates blood glucose levels.",
     "Insulin is a hormone produced by the pancreas that regulates blood glucose levels."),
    ("mitochondria",
     "Mitochondria are membrane-bound organelles in eukaryotic cells responsible for producing ATP.",
     "Mitochondria are organelles found in eukaryotic cells that produce ATP, the cell's primary energy source."),
    ("antibiotics",
     "Antibiotics are medications that kill or inhibit bacteria. They are ineffective against viruses.",
     "Antibiotics are medications that kill or inhibit bacteria. They are ineffective against viral infections."),
    ("mRNA vaccines",
     "mRNA vaccines use messenger RNA to instruct cells to produce a protein that triggers an immune response.",
     "mRNA vaccines deliver messenger RNA that instructs cells to produce a protein, triggering an immune response without using live virus."),
    # Economics
    ("supply and demand",
     "Supply and demand is the economic model that determines prices in a free market based on buyer and seller interactions.",
     "Supply and demand is the model that explains how prices are determined by the interaction of buyers and sellers in a free market."),
    ("compound interest",
     "Compound interest is interest calculated on the initial principal and also on accumulated interest from previous periods.",
     "Compound interest is calculated on both the initial principal and the interest already accumulated, causing exponential growth over time."),
    ("venture capital",
     "Venture capital is a form of private equity financing provided to early-stage companies with high growth potential.",
     "Venture capital is private equity financing given to early-stage startups with high growth potential in exchange for equity."),
    ("GDP",
     "GDP (Gross Domestic Product) is the total monetary value of all goods and services produced in a country within a period.",
     "GDP (Gross Domestic Product) measures the total value of all goods and services produced within a country in a given time period."),
]

_RAG_INSTRUCTIONS = [
    "Answer using the provided context.",
    "Use the context below to answer the question.",
    "Answer based on the information in the context.",
    "Using only the provided context, answer the question.",
    "Read the context and answer the question concisely.",
]

_RAG_QUESTIONS = [
    "What is {name}?",
    "What does the context say about {name}?",
    "Describe {name} based on the context.",
    "What information is given about {name}?",
    "What is {name}, according to the context?",
    "Explain {name} using the provided context.",
]


def _build_rag_grounding(n: int, rng: random.Random) -> list[dict]:
    out: list[dict] = []
    for _ in range(n):
        name, ctx_text, clean_answer = rng.choice(_RAG_POOL)
        instruction = rng.choice(_RAG_INSTRUCTIONS)
        question    = rng.choice(_RAG_QUESTIONS).format(name=name)
        # input = exactly what the retriever injects at inference time
        input_ = f"Context: {ctx_text}\nQuestion: {question}"
        # output = clean natural language — no template phrases, no grammar breaks
        out.append(_row(instruction, input_, clean_answer, "rag"))
    return out


# ── 2b. No-info refusal (~20%) ───────────────────────────────────────────
#
#  instruction = "Answer the question. If you don't have the info, say so."
#  input       = "Question: <unknowable question>"
#  output      = "I don't have that information."
#
#  Teaches the model to refuse gracefully when RAG returns nothing.

_REFUSAL_QUESTIONS = [
    "What is the internal architecture of my private project?",
    "What are the secret API keys used in our production system?",
    "What did my team discuss in yesterday's meeting?",
    "What is the password for the admin database?",
    "What are the Q3 revenue figures for our company?",
    "What is the content of the unreleased report?",
    "What does my colleague's private code do?",
    "What is the salary of the engineering team?",
    "What are the unannounced product features for next quarter?",
    "What decisions were made in the board meeting last week?",
    "What will the stock market do tomorrow?",
    "What is the exact cure for cancer?",
    "What will the weather be like in 10 years?",
    "Who will win the next election?",
    "What are someone's private thoughts?",
    "What is the best investment for guaranteed returns?",
    "How many people are thinking about pizza right now?",
    "What happened in the news last hour?",
    "What is in the document you haven't seen yet?",
    "What did I say in my last conversation with you?",
    "What is on the server you have no access to?",
    "What will my boss say about this proposal?",
    "What are the details of the classified report?",
    "What is in the file I haven't uploaded?",
]

_REFUSAL_INSTRUCTIONS = [
    "Answer the question. If you don't have the information, say so clearly.",
    "Answer concisely. If the answer is not available to you, say you don't know.",
    "Respond to the question. Admit when you lack the necessary information.",
    "Answer only if you have reliable information. Otherwise, acknowledge the limitation.",
]

_REFUSAL_RESPONSES = [
    "I don't have that information.",
    "I'm not able to access that information.",
    "That information isn't available to me.",
    "I don't have access to that data.",
    "I can't provide that — I don't have the relevant information.",
    "That's outside what I can access or know.",
    "I don't have enough information to answer that accurately.",
    "I'm unable to answer that — the required information isn't available to me.",
]


def _build_no_info_refusal(n: int, rng: random.Random) -> list[dict]:
    out: list[dict] = []
    for _ in range(n):
        instruction = rng.choice(_REFUSAL_INSTRUCTIONS)
        question    = rng.choice(_REFUSAL_QUESTIONS)
        response    = rng.choice(_REFUSAL_RESPONSES)
        # `input` = the actual question being asked
        input_ = f"Question: {question}"
        out.append(_row(instruction, input_, response, "refusal"))
    return out


# ── 2c. Concise explanations (~25%) ─────────────────────────────────────
#
#  instruction = "Answer concisely."
#  input       = "Question: <factual question>"
#  output      = short, accurate answer
#
#  Teaches the model NOT to ramble when it knows the answer.

_CONCISE_QA: list[tuple[str, str]] = [
    # ML / AI
    ("What is overfitting?",           "Overfitting is when a model memorises training data and fails to generalise to new inputs."),
    ("What is a transformer?",         "A transformer is a neural network architecture based on self-attention, widely used in NLP."),
    ("What is a learning rate?",       "The learning rate controls how much model weights are adjusted during each training step."),
    ("What is gradient descent?",      "Gradient descent is an optimisation algorithm that iteratively moves parameters toward a loss function's minimum."),
    ("What is tokenisation?",          "Tokenisation splits raw text into smaller units (tokens) that a model can process."),
    ("What is fine-tuning?",           "Fine-tuning adapts a pretrained model to a specific task by training on a smaller, task-specific dataset."),
    ("What is LoRA?",                  "LoRA (Low-Rank Adaptation) fine-tunes small rank-decomposition matrices instead of full weight updates."),
    ("What is a context window?",      "A context window is the maximum number of tokens a model can process in a single forward pass."),
    ("What are embeddings?",           "Embeddings are dense vector representations that capture semantic meaning in a continuous space."),
    ("What is batch size?",            "Batch size is the number of training samples processed together before updating model weights."),
    ("What is dropout?",               "Dropout randomly sets neuron outputs to zero during training to prevent overfitting."),
    ("What is layer normalisation?",   "Layer normalisation normalises inputs across features per sample to stabilise training."),
    ("What is perplexity in NLP?",     "Perplexity measures how well a language model predicts text; lower is better."),
    ("What is attention?",             "Attention lets each token weigh the relevance of all other tokens when building its representation."),
    ("What is RLHF?",                  "RLHF trains language models to align with human preferences using reward signals from human ratings."),
    # CS / Programming
    ("What is recursion?",             "Recursion is when a function calls itself to break a problem into smaller sub-problems of the same type."),
    ("What is Big-O notation?",        "Big-O notation describes algorithm complexity as a function of input size, focusing on worst-case growth."),
    ("What is a hash table?",          "A hash table maps keys to values using a hash function, giving average O(1) lookup, insert, and delete."),
    ("What is polymorphism?",          "Polymorphism allows different classes to be treated as the same interface, enabling flexible reusable code."),
    ("What is a REST API?",            "A REST API uses HTTP methods (GET, POST, PUT, DELETE) for stateless communication between systems."),
    ("What is a binary search tree?",  "A BST stores values so each node's left subtree has smaller values and right subtree has larger values."),
    ("What is garbage collection?",    "Garbage collection automatically reclaims memory that is no longer referenced by a program."),
    ("What is memoisation?",           "Memoisation caches expensive function call results so repeated identical calls return instantly."),
    ("What is a deadlock?",            "A deadlock occurs when two or more threads wait indefinitely for resources held by each other."),
    ("What is a pointer?",             "A pointer is a variable that stores the memory address of another variable."),
    # Math / Science
    ("What is the Pythagorean theorem?","In a right triangle, a² + b² = c², where c is the hypotenuse."),
    ("What is entropy?",               "Entropy measures the degree of disorder in a system; it tends to increase over time."),
    ("What is photosynthesis?",        "Photosynthesis is the process by which plants use sunlight, water, and CO₂ to produce glucose and oxygen."),
    ("What is the speed of light?",    "The speed of light in a vacuum is approximately 299,792,458 metres per second."),
    ("What is a prime number?",        "A prime number is a natural number greater than 1 with no divisors other than 1 and itself."),
    # General knowledge
    ("What is inflation?",             "Inflation is the rate at which the general price level rises, reducing purchasing power."),
    ("What is democracy?",             "Democracy is a system where power is held by the people, exercised directly or through elected representatives."),
    ("What is GDP?",                   "GDP is the total monetary value of all goods and services produced in a country within a given period."),
    ("What is compound interest?",     "Compound interest is interest calculated on both the principal and accumulated interest from prior periods."),
    ("What is evolution?",             "Evolution is change in heritable traits of populations over generations, driven by natural selection."),
]

_CONCISE_INSTRUCTIONS = [
    "Answer concisely.",
    "Give a short, clear answer.",
    "Respond briefly and accurately.",
    "Answer in one or two sentences.",
    "Keep your answer concise.",
    "Provide a concise definition.",
]


def _build_concise_answers(n: int, rng: random.Random) -> list[dict]:
    out: list[dict] = []
    pool = _CONCISE_QA * (n // len(_CONCISE_QA) + 2)
    chosen = rng.sample(pool, n) if n <= len(pool) else pool[:n]
    for question, answer in chosen:
        instruction = rng.choice(_CONCISE_INSTRUCTIONS)
        input_      = f"Question: {question}"
        out.append(_row(instruction, input_, answer, "concise"))
    return out


# ── 2d. Tool-result formatting ───────────────────────────────────────────
#
#  These examples mirror what the model sees AFTER the hard-router has:
#    1. Detected a math / computation query
#    2. Executed Python and obtained the result
#    3. Injected the result back into the LLM context as:
#         instruction = "The Python calculator returned a result. Use it to answer."
#         input       = "Question: <user question>\nPython result: <value>"
#
#  The model's ONLY job: format the pre-computed result into a clear answer.
#  It must NOT recompute — the Python tool already did that.
#
#  Problem categories (generated programmatically so values are always correct):
#    arithmetic   — a OP b  (large numbers, all four ops)
#    percentage   — X% of Y
#    area         — rectangle / square / triangle
#    average      — mean of N numbers
#    interest     — simple interest  P × R × T / 100
#    speed        — distance = speed × time  (or solve for time/speed)
#    power        — a ** b  (small exponents)

_TOOL_INSTR = [
    "The Python calculator returned a result. Use it to answer the question.",
    "A Python script computed the following. Use the result to respond.",
    "The calculator tool ran and returned the value below. Answer using it.",
    "Python was used to compute this. Present the answer clearly.",
    "Use the Python tool output to answer the question concisely.",
    "The system ran a calculation and got the result shown. Answer accordingly.",
    "Python returned the following output. Use it to respond to the question.",
]


# ── individual problem generators ────────────────────────────────────────
# Each returns (question: str, python_result: str, answer: str)

def _tg_arithmetic(rng: random.Random):
    ops = [
        (lambda a, b: a + b,   "{a:,} + {b:,}",   "{a:,} + {b:,} = {r:,}."),
        (lambda a, b: a - b,   "{a:,} - {b:,}",   "{a:,} - {b:,} = {r:,}."),
        (lambda a, b: a * b,   "{a:,} × {b:,}",   "{a:,} × {b:,} = {r:,}."),
        (lambda a, b: a // b,  "{a:,} ÷ {b:,}",   "{a:,} ÷ {b:,} = {r:,} (integer division)."),
    ]
    fn, q_t, a_t = rng.choice(ops)
    a = rng.randint(10, 9_999)
    b = rng.randint(2, min(a, 9_999))
    r = fn(a, b)
    q   = f"What is {q_t.format(a=a, b=b)}?"
    ans = a_t.format(a=a, b=b, r=r)
    return q, str(r), ans


def _tg_percentage(rng: random.Random):
    pct  = rng.choice([5, 10, 12.5, 15, 20, 25, 30, 40, 50, 60, 75])
    base = rng.randint(100, 20_000)
    r    = round(pct / 100 * base, 2)
    r_s  = str(int(r)) if r == int(r) else str(r)
    q    = rng.choice([
        f"What is {pct}% of {base:,}?",
        f"How much is {pct} percent of {base:,}?",
        f"Calculate {pct}% of {base:,}.",
    ])
    ans  = rng.choice([
        f"{pct}% of {base:,} is {r_s}.",
        f"The answer is {r_s}.",
        f"{pct} percent of {base:,} equals {r_s}.",
    ])
    return q, r_s, ans


def _tg_area(rng: random.Random):
    shape = rng.choice(["rectangle", "square", "triangle"])
    if shape == "rectangle":
        w, h = rng.randint(2, 500), rng.randint(2, 500)
        r    = w * h
        q    = f"What is the area of a rectangle with width {w} and height {h}?"
        ans  = f"The area of the rectangle is {r:,} square units."
    elif shape == "square":
        s = rng.randint(2, 500)
        r = s * s
        q   = f"What is the area of a square with side length {s}?"
        ans = f"The area of the square is {r:,} square units."
    else:
        b, h = rng.randint(2, 300), rng.randint(2, 300)
        r    = round(0.5 * b * h, 2)
        q    = f"What is the area of a triangle with base {b} and height {h}?"
        ans  = f"The area of the triangle is {r} square units."
    return q, str(r), ans


def _tg_average(rng: random.Random):
    nums    = [rng.randint(1, 1_000) for _ in range(rng.randint(3, 8))]
    r       = round(sum(nums) / len(nums), 2)
    nums_s  = ", ".join(str(x) for x in nums)
    q   = rng.choice([
        f"What is the average of {nums_s}?",
        f"Find the mean of {nums_s}.",
        f"What is the mean of these numbers: {nums_s}?",
    ])
    ans = rng.choice([
        f"The average is {r}.",
        f"The mean of these numbers is {r}.",
        f"The average of {nums_s} is {r}.",
    ])
    return q, str(r), ans


def _tg_simple_interest(rng: random.Random):
    P = rng.randint(500, 50_000)
    R = rng.choice([3, 4, 5, 6, 7, 8, 10, 12])
    T = rng.randint(1, 10)
    r = round(P * R * T / 100, 2)
    q   = f"What is the simple interest on ${P:,} at {R}% per year for {T} year(s)?"
    ans = f"The simple interest is ${r:,.2f}."
    return q, str(r), ans


def _tg_speed(rng: random.Random):
    mode = rng.choice(["distance", "time"])
    if mode == "distance":
        speed = rng.randint(20, 200)
        time  = rng.randint(1, 12)
        r     = speed * time
        q     = f"How far does an object travel at {speed} km/h for {time} hour(s)?"
        ans   = f"The object travels {r:,} km."
    else:
        dist  = rng.randint(100, 5_000)
        speed = rng.randint(20, 200)
        r     = round(dist / speed, 2)
        q     = f"How long does it take to travel {dist:,} km at {speed} km/h?"
        ans   = f"It takes {r} hours."
    return q, str(r), ans


def _tg_power(rng: random.Random):
    base = rng.randint(2, 20)
    exp  = rng.randint(2, 5)
    r    = base ** exp
    q    = rng.choice([
        f"What is {base} to the power of {exp}?",
        f"What is {base}^{exp}?",
        f"Calculate {base} raised to the power {exp}.",
    ])
    ans  = rng.choice([
        f"{base}^{exp} = {r:,}.",
        f"{base} to the power of {exp} is {r:,}.",
        f"The result is {r:,}.",
    ])
    return q, str(r), ans


_TOOL_GENERATORS = [
    _tg_arithmetic,
    _tg_arithmetic,   # weighted 2× — most common query type
    _tg_percentage,
    _tg_area,
    _tg_average,
    _tg_simple_interest,
    _tg_speed,
    _tg_power,
]


def _build_tool_result_format(n: int, rng: random.Random) -> list[dict]:
    out: list[dict] = []
    for _ in range(n):
        gen  = rng.choice(_TOOL_GENERATORS)
        q, result_str, ans = gen(rng)
        instr  = rng.choice(_TOOL_INSTR)
        input_ = f"Question: {q}\nPython result: {result_str}"
        out.append(_row(instr, input_, ans, "tool_fmt"))
    return out


# ═══════════════════════════════════════════════════════════════════════════
#  3. Composition & CLI
# ═══════════════════════════════════════════════════════════════════════════

# Target share of the full dataset that Alpaca occupies (midpoint of 50–60 %).
# All custom category counts are derived from this anchor.
_ALPACA_SHARE = 0.55

# Custom category shares of the TOTAL dataset (midpoints of requested ranges).
_RAG_SHARE     = 0.175  # 15–20 %
_CONCISE_SHARE = 0.125  # 10–15 %
_REFUSAL_SHARE = 0.075  #  5–10 %
# Tool gets the exact remainder so percentages always sum to 100 %.


def build_dataset(
    output_path: str  = "data/lora_final.jsonl",
    seed: int         = 42,
    alpaca_only: bool = False,
    custom_only: bool = False,
) -> None:
    """Build the full LoRA dataset.

    Alpaca is loaded, cleaned, and transformed.  Any Alpaca math rows that
    contain a recognisable arithmetic expression are converted to tool_fmt
    examples and folded into the tool pool.  Custom category counts are
    derived from the remaining clean Alpaca count so percentages stay exact.
    """
    rng = random.Random(seed)

    # ── Step 1: load + clean the complete Alpaca split ────────────────────
    alpaca_rows: list[dict] = []
    converted_tool: list[dict] = []
    if not custom_only:
        alpaca_rows, converted_tool = load_alpaca(rng)

    alpaca_count = len(alpaca_rows)

    # ── Step 2: derive custom counts from Alpaca anchor ───────────────────
    if custom_only:
        total     = 10_000
        rag_n     = round(total * _RAG_SHARE     / (1 - _ALPACA_SHARE))
        concise_n = round(total * _CONCISE_SHARE / (1 - _ALPACA_SHARE))
        refuse_n  = round(total * _REFUSAL_SHARE / (1 - _ALPACA_SHARE))
        tool_n    = total - rag_n - concise_n - refuse_n
    else:
        total     = round(alpaca_count / _ALPACA_SHARE)
        rag_n     = round(total * _RAG_SHARE)
        concise_n = round(total * _CONCISE_SHARE)
        refuse_n  = round(total * _REFUSAL_SHARE)
        # Subtract already-converted tool rows so we don't over-generate
        raw_tool_n = total - alpaca_count - rag_n - concise_n - refuse_n
        tool_n     = max(0, raw_tool_n - len(converted_tool))

    total_tool = tool_n + len(converted_tool)
    custom_total = rag_n + concise_n + refuse_n + total_tool

    print(f"\n[build_lora_dataset] Dataset composition plan")
    print(f"  Alpaca (cleaned)   : {alpaca_count:>7,}  ({alpaca_count / (alpaca_count + custom_total) * 100:.1f}%)")
    print(f"  Custom total       : {custom_total:>7,}  ({custom_total / (alpaca_count + custom_total) * 100:.1f}%)")
    print(f"    RAG grounding    : {rag_n:>7,}  ({rag_n / (alpaca_count + custom_total) * 100:.1f}%)")
    print(f"    Concise          : {concise_n:>7,}  ({concise_n / (alpaca_count + custom_total) * 100:.1f}%)")
    print(f"    Refusal          : {refuse_n:>7,}  ({refuse_n / (alpaca_count + custom_total) * 100:.1f}%)")
    print(f"    Tool (generated) : {tool_n:>7,}  ({tool_n / (alpaca_count + custom_total) * 100:.1f}%)")
    print(f"    Tool (converted) : {len(converted_tool):>7,}  ({len(converted_tool) / (alpaca_count + custom_total) * 100:.1f}%)")
    print(f"  TOTAL              : {alpaca_count + custom_total:>7,}\n")

    all_rows: list[dict] = list(alpaca_rows)
    all_rows.extend(converted_tool)   # fold in converted math→tool rows

    if not alpaca_only:
        print("[build_lora_dataset] Building RAG grounding examples...")
        all_rows.extend(_build_rag_grounding(rag_n, rng))

        print("[build_lora_dataset] Building concise answer examples...")
        all_rows.extend(_build_concise_answers(concise_n, rng))

        print("[build_lora_dataset] Building no-info refusal examples...")
        all_rows.extend(_build_no_info_refusal(refuse_n, rng))

        print("[build_lora_dataset] Building tool-result formatting examples...")
        all_rows.extend(_build_tool_result_format(tool_n, rng))

    rng.shuffle(all_rows)
    _write_jsonl(Path(output_path), all_rows)

    from collections import Counter
    counts = Counter(r["source"] for r in all_rows)
    total_written = len(all_rows)
    print(f"\n[build_lora_dataset] Source breakdown:")
    for src, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {src:<18} {cnt:>7,}  ({cnt / total_written * 100:.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build LoRA fine-tuning dataset (Alpaca 4-field format)"
    )
    parser.add_argument("--output",      default="data/lora_final.jsonl",
                        help="Output JSONL path")
    parser.add_argument("--seed",        type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--alpaca-only", action="store_true",
                        help="Write only the Alpaca rows")
    parser.add_argument("--custom-only", action="store_true",
                        help="Skip Alpaca download, write ~10K custom rows only")
    args = parser.parse_args()

    build_dataset(
        output_path=args.output,
        seed=args.seed,
        alpaca_only=args.alpaca_only,
        custom_only=args.custom_only,
    )


if __name__ == "__main__":
    main()
