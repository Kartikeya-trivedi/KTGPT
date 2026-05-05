"""
Synthetic Dataset Generator — Multi-Stage Post-Training Pipeline
=================================================================
Generates JSONL datasets for KT-GPT post-training with strict constraints:

  Stage 0A:  Basic arithmetic (add/mul only, NO CoT, NO word problems)
  Stage 0B:  Expanded arithmetic (sub/div, commutativity, edge cases)
  Stage 0C:  Multi-step arithmetic + short word problems
  Stage 1:   Instruction following (format, conciseness, refusal)
  Stage 1.5: Pure context grounding (RAG-style: answer ONLY from context)
  Stage 2:   Function calling (synthetic tool-use, includes no-tool-needed cases)
  Stage 3:   GRPO prompts (correctness + tool-decision rewards ONLY, no format reward)

Usage:
  python data/generate_sft_pipeline.py --stage 0a
  python data/generate_sft_pipeline.py --stage 1.5
  python data/generate_sft_pipeline.py --stage all
  python data/generate_sft_pipeline.py --stage 2 --count 5000
"""
from __future__ import annotations
import argparse, json, random, math
from pathlib import Path
from dataclasses import dataclass

SEED = 42

def fmt(prompt: str, response: str) -> dict:
    return {"prompt": prompt.strip(), "response": response.strip()}

# =====================================================================
#  Per-Stage Anchors — high-frequency correctness locks
# =====================================================================
# Each stage gets ~5% anchors repeated to prevent drift on core skills.

MATH_ANCHORS = [
    ("What is 2+2?", "4"), ("What is 3+3?", "6"), ("What is 5+5?", "10"),
    ("What is 7*8?", "56"), ("What is 9*9?", "81"), ("What is 12*12?", "144"),
    ("What is 6*7?", "42"), ("What is 8*9?", "72"), ("What is 10-3?", "7"),
    ("What is 100/10?", "10"), ("What is 15-7?", "8"), ("What is 24/6?", "4"),
]

INSTRUCTION_ANCHORS = [
    ("Respond with one word: the opposite of cold.", "Hot"),
    ("What is 2+2? Respond with just the number.", "4"),
    ("Say hello in French.", "Bonjour"),
    ("Is the sky blue? Answer yes or no.", "Yes"),
    ("Name a color.", "Blue"),
]

CONTEXT_ANCHORS = [
    ("Answer ONLY using the context.\n\nContext:\nThe capital of France is Paris.\n\nQuestion:\nWhat is the capital of France?", "Paris"),
    ("Answer ONLY using the context.\n\nContext:\nWater boils at 100 degrees Celsius.\n\nQuestion:\nWhat is the boiling point of water?", "100 degrees Celsius"),
    ("Answer ONLY using the context.\n\nContext:\nThe Earth orbits the Sun.\n\nQuestion:\nWhat does the Earth orbit?", "The Sun"),
    ("Answer ONLY using the context.\n\nContext:\nPython was created by Guido van Rossum.\n\nQuestion:\nWhat is the latest Python version?",
     "The context does not mention the latest Python version."),
]

TOOL_ANCHORS_USE = [
    ("Calculate 999*999", "calculator"),   # must use tool
    ("What is 847+293?", "calculator"),
]

TOOL_ANCHORS_NOUSE = [
    ("What is 2+2?", "4"),     # must NOT use tool
    ("Say hello.", "Hello!"),
]

def inject_anchors(samples: list[dict], anchors: list[tuple[str,str]], ratio: float = 0.05) -> list[dict]:
    """Inject high-frequency anchor samples into a dataset."""
    n = max(1, int(len(samples) * ratio))
    anchor_samples = [fmt(q, a) for q, a in random.choices(anchors, k=n)]
    combined = samples + anchor_samples
    random.shuffle(combined)
    print(f"  [ANCHORS] Injected {n} anchor samples ({ratio:.0%})")
    return combined

def save_jsonl(samples: list[dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"  Saved {len(samples)} samples -> {path}")

def add_replay_buffer(current: list[dict], prev_paths: list[str], ratio: float = 0.15) -> list[dict]:
    """MANDATORY replay buffer from previous stages to prevent catastrophic forgetting."""
    replay = []
    for p in prev_paths:
        if Path(p).exists():
            with open(p, "r") as f:
                replay.extend([json.loads(l) for l in f if l.strip()])
    if not replay:
        return current
    n = max(1, int(len(current) * ratio))
    picked = random.choices(replay, k=min(n, len(replay)))
    combined = current + picked
    random.shuffle(combined)
    print(f"  [REPLAY] Mixed in {len(picked)} samples from previous stages ({ratio:.0%})")
    return combined

# =====================================================================
#  STAGE 0A — Basic Arithmetic ONLY (add + mul, no CoT, no word problems)
# =====================================================================
def generate_stage0a(count: int = 3000) -> list[dict]:
    print("\n[Stage 0A] Basic arithmetic (add/mul only)")
    random.seed(SEED)
    samples = []

    # Addition grid 1-20 (both orderings for commutativity)
    for a in range(1, 21):
        for b in range(a, 21):  # avoid full duplication
            samples.append(fmt(f"What is {a}+{b}?", str(a + b)))
            if a != b:
                samples.append(fmt(f"What is {b}+{a}?", str(a + b)))

    # Multiplication grid 1-12 (both orderings)
    for a in range(1, 13):
        for b in range(a, 13):
            samples.append(fmt(f"What is {a}*{b}?", str(a * b)))
            if a != b:
                samples.append(fmt(f"What is {b}*{a}?", str(a * b)))

    # Edge cases: identity and zero
    for x in range(0, 25):
        samples.append(fmt(f"What is {x}+0?", str(x)))
        samples.append(fmt(f"What is 0+{x}?", str(x)))
        samples.append(fmt(f"What is {x}*1?", str(x)))
        samples.append(fmt(f"What is 1*{x}?", str(x)))
        samples.append(fmt(f"What is {x}*0?", "0"))
        samples.append(fmt(f"What is 0*{x}?", "0"))

    # High-frequency anchors (repeated)
    anchors = [
        ("What is 2+2?", "4"), ("What is 3+3?", "6"), ("What is 5+5?", "10"),
        ("What is 7*8?", "56"), ("What is 9*9?", "81"), ("What is 12*12?", "144"),
        ("What is 6*7?", "42"), ("What is 8*9?", "72"),
    ]
    for _ in range(300):
        q, a = random.choice(anchors)
        samples.append(fmt(q, a))

    # Random add/mul for coverage
    for _ in range(500):
        if random.random() < 0.5:
            a, b = random.randint(1, 50), random.randint(1, 50)
            samples.append(fmt(f"What is {a}+{b}?", str(a + b)))
        else:
            a, b = random.randint(1, 12), random.randint(1, 12)
            samples.append(fmt(f"What is {a}*{b}?", str(a * b)))

    random.shuffle(samples)
    samples = samples[:count]
    samples = inject_anchors(samples, MATH_ANCHORS)
    print(f"  Generated {len(samples)} samples (add/mul, no CoT)")
    return samples

# =====================================================================
#  STAGE 0B — Expanded Arithmetic (sub, div, carries, commutativity)
# =====================================================================
def generate_stage0b(count: int = 3000) -> list[dict]:
    print("\n[Stage 0B] Expanded arithmetic (sub/div, edge cases)")
    random.seed(SEED + 1)
    samples = []

    # Subtraction grid (a >= b to keep non-negative)
    for a in range(1, 26):
        for b in range(0, a + 1):
            samples.append(fmt(f"What is {a}-{b}?", str(a - b)))

    # Division (clean results only)
    for a in range(1, 13):
        for b in range(1, 13):
            product = a * b
            samples.append(fmt(f"What is {product}/{a}?", str(b)))
            if b != a:
                samples.append(fmt(f"What is {product}/{b}?", str(a)))

    # Edge cases
    for x in range(1, 20):
        samples.append(fmt(f"What is {x}-0?", str(x)))
        samples.append(fmt(f"What is {x}-{x}?", "0"))
        samples.append(fmt(f"What is {x}/{x}?", "1"))
        samples.append(fmt(f"What is {x}/1?", str(x)))

    # Carry cases (e.g. 18+7=25, 99+1=100)
    carry_pairs = [
        (18, 7), (99, 1), (45, 55), (28, 14), (37, 26),
        (199, 1), (55, 45), (67, 33), (88, 12), (75, 25),
    ]
    for a, b in carry_pairs:
        samples.append(fmt(f"What is {a}+{b}?", str(a + b)))
        samples.append(fmt(f"What is {b}+{a}?", str(a + b)))

    # Random mix of all 4 ops
    for _ in range(500):
        op = random.choice(["+", "-", "*", "/"])
        if op == "+":
            a, b = random.randint(1, 100), random.randint(1, 100)
            samples.append(fmt(f"What is {a}+{b}?", str(a + b)))
        elif op == "-":
            a = random.randint(1, 100)
            b = random.randint(0, a)
            samples.append(fmt(f"What is {a}-{b}?", str(a - b)))
        elif op == "*":
            a, b = random.randint(1, 15), random.randint(1, 15)
            samples.append(fmt(f"What is {a}*{b}?", str(a * b)))
        else:
            b = random.randint(1, 12)
            a = b * random.randint(1, 12)
            samples.append(fmt(f"What is {a}/{b}?", str(a // b)))

    random.shuffle(samples)
    samples = samples[:count]
    samples = inject_anchors(samples, MATH_ANCHORS)
    print(f"  Generated {len(samples)} samples (all 4 ops, edge cases)")
    return samples

# =====================================================================
#  STAGE 0C — Multi-step + Short Word Problems (still no CoT)
# =====================================================================
def generate_stage0c(count: int = 2000) -> list[dict]:
    print("\n[Stage 0C] Multi-step + word problems (no CoT)")
    random.seed(SEED + 2)
    samples = []

    # Multi-step: order of operations
    for _ in range(count // 2):
        t = random.choice(["add_mul", "sub_add", "chain_add"])
        if t == "add_mul":
            a, b, c = random.randint(1, 20), random.randint(1, 10), random.randint(1, 10)
            ans = a + b * c
            samples.append(fmt(f"What is {a}+{b}*{c}?", str(ans)))
        elif t == "sub_add":
            a = random.randint(10, 50)
            b = random.randint(1, a)
            c = random.randint(1, 20)
            ans = a - b + c
            samples.append(fmt(f"What is {a}-{b}+{c}?", str(ans)))
        else:
            a, b, c = random.randint(1, 30), random.randint(1, 30), random.randint(1, 30)
            ans = a + b + c
            samples.append(fmt(f"What is {a}+{b}+{c}?", str(ans)))

    # Short word problems (answer is just the number, no CoT)
    templates = [
        ("I have {a} apples and buy {b} more. How many do I have?",
         lambda a, b: a + b),
        ("A box has {a} items. {b} are removed. How many remain?",
         lambda a, b: a - b),
        ("There are {a} rows with {b} seats each. Total seats?",
         lambda a, b: a * b),
        ("{a} cookies split equally among {b} kids. How many each?",
         lambda a, b: a // b),
    ]
    for _ in range(count // 2):
        tmpl, fn = random.choice(templates)
        if "split" in tmpl:
            b = random.randint(2, 10)
            a = b * random.randint(1, 12)
        elif "removed" in tmpl:
            a = random.randint(5, 50)
            b = random.randint(1, a)
        else:
            a, b = random.randint(1, 30), random.randint(1, 30)
        ans = fn(a, b)
        samples.append(fmt(tmpl.format(a=a, b=b), str(ans)))

    random.shuffle(samples)
    samples = samples[:count]
    samples = inject_anchors(samples, MATH_ANCHORS)
    print(f"  Generated {len(samples)} samples (multi-step + word)")
    return samples

# =====================================================================
#  STAGE 1 — Instruction Following + Context Grounding
# =====================================================================

KNOWLEDGE_FACTS = [
    ("Python was created by Guido van Rossum in 1991.", "Who created Python?", "Guido van Rossum"),
    ("The speed of light is approximately 299,792,458 m/s.", "What is the speed of light?", "Approximately 299,792,458 meters per second"),
    ("Machine learning enables systems to learn from data without explicit programming.", "What is machine learning?", "A field that enables systems to learn from data without explicit programming."),
    ("The Great Wall of China stretches over 13,000 miles.", "How long is the Great Wall?", "Over 13,000 miles"),
    ("HTTP stands for HyperText Transfer Protocol.", "What does HTTP stand for?", "HyperText Transfer Protocol"),
    ("DNA stands for deoxyribonucleic acid.", "What does DNA stand for?", "Deoxyribonucleic acid"),
    ("Water boils at 100 degrees Celsius at standard pressure.", "What is the boiling point of water?", "100 degrees Celsius"),
    ("JavaScript was created by Brendan Eich in 1995.", "Who created JavaScript?", "Brendan Eich"),
    ("An adult human body has 206 bones.", "How many bones does an adult have?", "206"),
    ("Git was created by Linus Torvalds in 2005.", "Who created Git?", "Linus Torvalds"),
    ("Photosynthesis produces glucose and oxygen from CO2 and water.", "What does photosynthesis produce?", "Glucose and oxygen"),
    ("The capital of Japan is Tokyo.", "What is the capital of Japan?", "Tokyo"),
]

INSTRUCTION_TASKS = [
    ("List 3 primary colors.", "Red, blue, yellow."),
    ("Respond with only 'yes' or 'no': Is the sky blue?", "Yes"),
    ("Say hello in French.", "Bonjour"),
    ("What is 2+2? Respond with just the number.", "4"),
    ("Name the 4 seasons.", "Spring, summer, fall, winter."),
    ("Translate 'thank you' to Spanish.", "Gracias"),
    ("Respond with one word: the opposite of cold.", "Hot"),
    ("Is Python compiled or interpreted? One word.", "Interpreted"),
    ("What color do you get mixing red and blue?", "Purple"),
    ("Complete: The sun rises in the ___.", "East"),
    ("What is the chemical symbol for water?", "H2O"),
    ("What is the plural of 'mouse'?", "Mice"),
]

FORMAT_TASKS = [
    ("List the planets as a JSON array.",
     '["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]'),
    ("Return as JSON: name is Alice, age is 30.",
     '{"name": "Alice", "age": 30}'),
    ("Format as numbered list: apple, banana, cherry.",
     "1. Apple\n2. Banana\n3. Cherry"),
    ("Write a Python function that returns the sum of two numbers.",
     "def add(a, b):\n    return a + b"),
    ("Write a Python function to check if a number is even.",
     "def is_even(n):\n    return n % 2 == 0"),
]

REFUSAL_TASKS = [
    ("Based on the context, what is the population of Mars?\n\nContext: Mars is the fourth planet from the Sun.",
     "The provided context does not contain information about the population of Mars."),
    ("Using only the context, what is the CEO's name?\n\nContext: The company was founded in 2020.",
     "The context does not mention the CEO's name."),
    ("From the passage, what year did the event happen?\n\nContext: The building is 50 meters tall.",
     "I don't know. The context does not mention any event or year."),
]

def generate_stage1(count: int = 15000) -> list[dict]:
    print("\n[Stage 1] Instruction + Context Grounding")
    random.seed(SEED + 10)
    samples = []

    # Context-grounded QA
    templates = [
        "Use the provided context to answer.\n\nContext:\n{ctx}\n\nQuestion: {q}",
        "Answer based ONLY on the context below.\n\nContext: {ctx}\n\nQ: {q}",
        "Read the passage and answer.\n\n{ctx}\n\nQuestion: {q}\nAnswer concisely.",
    ]
    for _ in range(count // 3):
        ctx, question, answer = random.choice(KNOWLEDGE_FACTS)
        prompt = random.choice(templates).format(ctx=ctx, q=question)
        samples.append(fmt(prompt, answer))

    # Instruction following
    for _ in range(count // 3):
        q, a = random.choice(INSTRUCTION_TASKS)
        samples.append(fmt(q, a))

    # Format tasks
    for _ in range(count // 6):
        q, a = random.choice(FORMAT_TASKS)
        samples.append(fmt(q, a))

    # Refusal (context-insufficient)
    for _ in range(count // 10):
        q, a = random.choice(REFUSAL_TASKS)
        samples.append(fmt(q, a))

    while len(samples) < count:
        q, a = random.choice(INSTRUCTION_TASKS)
        samples.append(fmt(q, a))

    random.shuffle(samples)
    samples = samples[:count]
    samples = inject_anchors(samples, INSTRUCTION_ANCHORS + MATH_ANCHORS[:4])
    return samples

# =====================================================================
#  STAGE 1.5 — Pure Context Grounding (RAG-style)
# =====================================================================

# Diverse knowledge passages for context-grounded QA
RAG_PASSAGES = [
    # Science
    {"context": "The mitochondria are membrane-bound organelles found in the cytoplasm of eukaryotic cells. They generate most of the cell's supply of adenosine triphosphate (ATP), used as a source of chemical energy.",
     "questions": [("What do mitochondria produce?", "Adenosine triphosphate (ATP)"),
                   ("Where are mitochondria found?", "In the cytoplasm of eukaryotic cells")]},
    {"context": "The Earth's atmosphere is composed of 78% nitrogen, 21% oxygen, and 1% other gases including argon and carbon dioxide.",
     "questions": [("What is the most abundant gas in Earth's atmosphere?", "Nitrogen"),
                   ("What percentage of the atmosphere is oxygen?", "21%")]},
    {"context": "The periodic table was first published by Dmitri Mendeleev in 1869. He arranged elements by atomic weight and noticed periodic patterns in their properties.",
     "questions": [("Who published the periodic table?", "Dmitri Mendeleev"),
                   ("When was the periodic table first published?", "1869")]},
    {"context": "Sound travels at approximately 343 meters per second in air at 20 degrees Celsius. It travels faster in water at about 1,480 meters per second.",
     "questions": [("How fast does sound travel in air?", "Approximately 343 meters per second"),
                   ("Does sound travel faster in air or water?", "Water")]},
    # Technology
    {"context": "TCP/IP stands for Transmission Control Protocol/Internet Protocol. It is the fundamental communication protocol of the Internet, first standardized in 1983.",
     "questions": [("What does TCP/IP stand for?", "Transmission Control Protocol/Internet Protocol"),
                   ("When was TCP/IP standardized?", "1983")]},
    {"context": "The first iPhone was released by Apple on June 29, 2007. It featured a 3.5-inch touchscreen and ran iPhone OS 1.0.",
     "questions": [("When was the first iPhone released?", "June 29, 2007"),
                   ("What size was the first iPhone's screen?", "3.5 inches")]},
    {"context": "Linux is a family of open-source Unix-like operating systems based on the Linux kernel, first released by Linus Torvalds on September 17, 1991.",
     "questions": [("Who released the Linux kernel?", "Linus Torvalds"),
                   ("When was Linux first released?", "September 17, 1991")]},
    # Geography
    {"context": "The Amazon River is approximately 6,400 kilometers long, making it the second longest river in the world after the Nile. It flows through Brazil, Peru, and Colombia.",
     "questions": [("How long is the Amazon River?", "Approximately 6,400 kilometers"),
                   ("Which countries does the Amazon flow through?", "Brazil, Peru, and Colombia")]},
    {"context": "Mount Everest is 8,849 meters tall and is located on the border between Nepal and Tibet. It was first summited by Edmund Hillary and Tenzing Norgay in 1953.",
     "questions": [("How tall is Mount Everest?", "8,849 meters"),
                   ("Who first summited Mount Everest?", "Edmund Hillary and Tenzing Norgay")]},
    {"context": "The Sahara Desert covers approximately 9.2 million square kilometers across North Africa. It is the largest hot desert in the world.",
     "questions": [("How large is the Sahara Desert?", "Approximately 9.2 million square kilometers"),
                   ("Where is the Sahara Desert located?", "North Africa")]},
    # History
    {"context": "The Berlin Wall fell on November 9, 1989. It had divided East and West Berlin for 28 years since its construction in 1961.",
     "questions": [("When did the Berlin Wall fall?", "November 9, 1989"),
                   ("How long did the Berlin Wall stand?", "28 years")]},
    {"context": "The printing press was invented by Johannes Gutenberg around 1440. His first major work was the Gutenberg Bible, printed around 1455.",
     "questions": [("Who invented the printing press?", "Johannes Gutenberg"),
                   ("When was the Gutenberg Bible printed?", "Around 1455")]},
    # Math/CS concepts
    {"context": "A binary search algorithm works by repeatedly dividing a sorted array in half. Its time complexity is O(log n), making it much faster than linear search for large datasets.",
     "questions": [("What is the time complexity of binary search?", "O(log n)"),
                   ("What does binary search require the array to be?", "Sorted")]},
    {"context": "Python uses indentation to define code blocks instead of curly braces. The standard indentation is 4 spaces per level, as recommended by PEP 8.",
     "questions": [("How does Python define code blocks?", "Using indentation"),
                   ("How many spaces does PEP 8 recommend?", "4 spaces per level")]},
]

# Contexts where the answer is NOT present (refusal training)
RAG_REFUSAL = [
    ("The capital of Germany is Berlin. It has a population of about 3.7 million.",
     "What is the GDP of Germany?",
     "The context does not contain information about Germany's GDP."),
    ("Python was created in 1991 by Guido van Rossum.",
     "What is the latest version of Python?",
     "The context does not mention the latest version of Python."),
    ("The Eiffel Tower is 330 meters tall and located in Paris.",
     "How many visitors does the Eiffel Tower get per year?",
     "I don't know. The context does not provide visitor statistics."),
    ("HTTP uses port 80 by default. HTTPS uses port 443.",
     "Who invented HTTP?",
     "The context does not mention who invented HTTP."),
    ("Mars is the fourth planet from the Sun. It has two moons.",
     "What is the temperature on Mars?",
     "The context does not contain information about Mars's temperature."),
    ("The Great Wall of China is over 13,000 miles long.",
     "When was the Great Wall built?",
     "The context does not specify when the Great Wall was built."),
    ("JavaScript runs in web browsers. It was created in 1995.",
     "What is TypeScript?",
     "The context does not mention TypeScript."),
    ("DNA has a double helix structure discovered by Watson and Crick.",
     "How many chromosomes do humans have?",
     "The context does not provide information about human chromosomes."),
]

# Distractor contexts — answer is in one paragraph, distractor is irrelevant
RAG_DISTRACTOR = [
    ("Bananas are a popular fruit rich in potassium.\n\nThe speed of light is 299,792,458 meters per second.",
     "What is the speed of light?", "299,792,458 meters per second"),
    ("Football is the most popular sport globally.\n\nPython was created by Guido van Rossum in 1991.",
     "Who created Python?", "Guido van Rossum"),
    ("The Mona Lisa hangs in the Louvre Museum.\n\nThe boiling point of water is 100 degrees Celsius.",
     "What is the boiling point of water?", "100 degrees Celsius"),
    ("Cats are obligate carnivores.\n\nThe Amazon River is approximately 6,400 km long.",
     "How long is the Amazon River?", "Approximately 6,400 km"),
]


def generate_stage1_5(count: int = 10000) -> list[dict]:
    """Pure context grounding: answer ONLY from provided context."""
    print("\n[Stage 1.5] Pure Context Grounding (RAG-style)")
    random.seed(SEED + 15)
    samples = []

    # Prompt templates — all enforce context-only answering
    templates = [
        "Answer ONLY using the context.\n\nContext:\n{ctx}\n\nQuestion:\n{q}",
        "Use ONLY the provided context to answer. Do not use outside knowledge.\n\nContext:\n{ctx}\n\nQuestion: {q}",
        "Read the context below and answer the question. If the answer is not in the context, say you don't know.\n\nContext:\n{ctx}\n\nQuestion: {q}",
        "Based strictly on the following passage, answer the question.\n\nPassage:\n{ctx}\n\nQ: {q}",
    ]

    # --- Standard context QA (60%) ---
    for _ in range(int(count * 0.6)):
        passage = random.choice(RAG_PASSAGES)
        q, a = random.choice(passage["questions"])
        tmpl = random.choice(templates)
        prompt = tmpl.format(ctx=passage["context"], q=q)
        samples.append(fmt(prompt, a))

    # --- Refusal: answer NOT in context (25%) ---
    for _ in range(int(count * 0.25)):
        ctx, q, a = random.choice(RAG_REFUSAL)
        tmpl = random.choice(templates)
        prompt = tmpl.format(ctx=ctx, q=q)
        samples.append(fmt(prompt, a))

    # --- Distractor contexts (15%) ---
    for _ in range(int(count * 0.15)):
        ctx, q, a = random.choice(RAG_DISTRACTOR)
        tmpl = random.choice(templates)
        prompt = tmpl.format(ctx=ctx, q=q)
        samples.append(fmt(prompt, a))

    random.shuffle(samples)
    samples = samples[:count]
    samples = inject_anchors(samples, CONTEXT_ANCHORS + MATH_ANCHORS[:4])
    print(f"  Generated {len(samples)} context-grounding samples")
    print(f"    - QA: {int(count*0.6)}, Refusal: {int(count*0.25)}, Distractor: {int(count*0.15)}")
    return samples


# =====================================================================
#  STAGE 2 — Function Calling (with no-tool-needed constraint)
# =====================================================================

TOOL_DEFS = [
    {"name": "calculator", "description": "Evaluate a math expression",
     "parameters": {"expression": {"type": "string", "description": "Math expression"}}},
    {"name": "search", "description": "Search for information",
     "parameters": {"query": {"type": "string", "description": "Search query"}}},
    {"name": "get_weather", "description": "Get weather for a location",
     "parameters": {"location": {"type": "string"}, "unit": {"type": "string", "default": "celsius"}}},
]

TOOL_SCENARIOS = [
    ("What is 15% of 200?", "calculator", '{"expression": "200 * 0.15"}', "30.0"),
    ("Calculate 45 * 23", "calculator", '{"expression": "45 * 23"}', "1035"),
    ("Square root of 144?", "calculator", '{"expression": "144 ** 0.5"}', "12.0"),
    ("What is 2^10?", "calculator", '{"expression": "2 ** 10"}', "1024"),
    ("999 - 456?", "calculator", '{"expression": "999 - 456"}', "543"),
    ("Who is the president of France?", "search", '{"query": "president of France"}', "Emmanuel Macron"),
    ("Tallest building in the world?", "search", '{"query": "tallest building"}', "Burj Khalifa (828m)"),
    ("When was the Eiffel Tower built?", "search", '{"query": "Eiffel Tower built"}', "1887-1889"),
    ("Weather in Tokyo?", "get_weather", '{"location": "Tokyo", "unit": "celsius"}', "22C, partly cloudy"),
    ("Weather in London?", "get_weather", '{"location": "London", "unit": "celsius"}', "15C, light rain"),
]

# Explicit NO-TOOL samples: obvious answers that should NOT trigger tool calls
NO_TOOL_SCENARIOS = [
    ("Say hello.", "Hello!"),
    ("What is 2+2?", "4"),  # trivial math = no tool needed
    ("What is 1+1?", "2"),
    ("Name a color.", "Blue"),
    ("What is the opposite of hot?", "Cold"),
    ("Is 10 greater than 5?", "Yes"),
    ("What is 3*1?", "3"),
    ("Respond with one word: sky color.", "Blue"),
]

def _format_tools_system(tools: list[dict]) -> str:
    lines = ["You have access to the following tools:\n"]
    for t in tools:
        params = ", ".join(f'{k}: {v["type"]}' for k, v in t["parameters"].items())
        lines.append(f'- {t["name"]}({params}): {t["description"]}')
    lines.append("\nTo use a tool, respond with:")
    lines.append('<tool_call>{"name": "tool_name", "arguments": {...}}</tool_call>')
    lines.append("\nALWAYS use the calculator tool for any mathematical expression, no matter how simple.")
    return "\n".join(lines)

def generate_stage2(count: int = 25000) -> list[dict]:
    print("\n[Stage 2] Function Calling")
    random.seed(SEED + 20)
    samples = []

    # Tool-use samples (60%)
    for _ in range(int(count * 0.6)):
        n_tools = random.randint(2, 3)
        tools = random.sample(TOOL_DEFS, k=n_tools)
        scenario = random.choice(TOOL_SCENARIOS)
        q, tool_name, args_json, result = scenario

        # Ensure needed tool is available
        needed = next((t for t in TOOL_DEFS if t["name"] == tool_name), None)
        if needed and needed not in tools:
            tools[-1] = needed

        system = _format_tools_system(tools)
        prompt = f"{system}\n\nUser: {q}"
        response = (
            f'<tool_call>{{"name": "{tool_name}", "arguments": {args_json}}}</tool_call>\n\n'
            f'Tool returned: {result}\n\n{result}'
        )
        samples.append(fmt(prompt, response))

    # NO-TOOL samples (40%) — model must answer directly
    for _ in range(int(count * 0.4)):
        n_tools = random.randint(2, 3)
        tools = random.sample(TOOL_DEFS, k=n_tools)
        system = _format_tools_system(tools)
        q, a = random.choice(NO_TOOL_SCENARIOS)
        prompt = f"{system}\n\nUser: {q}"
        samples.append(fmt(prompt, a))

    random.shuffle(samples)
    samples = samples[:count]
    samples = inject_anchors(samples, TOOL_ANCHORS_NOUSE + MATH_ANCHORS[:4])
    return samples

# =====================================================================
#  STAGE 3 — GRPO Prompts (correctness + tool-decision rewards ONLY)
# =====================================================================
def generate_stage3(count: int = 2000) -> list[dict]:
    """GRPO prompts. NO format reward. Only correctness + tool-decision."""
    print("\n[Stage 3] GRPO Prompts (correctness-only rewards)")
    random.seed(SEED + 30)
    samples = []
    tools_sys = _format_tools_system(TOOL_DEFS)

    # Math requiring calculator (non-trivial)
    for _ in range(count // 3):
        a, b = random.randint(10, 999), random.randint(10, 999)
        op = random.choice(["+", "-", "*"])
        if op == "-" and b > a:
            a, b = b, a
        ans = eval(f"{a}{op}{b}")
        samples.append({
            "prompt": f"{tools_sys}\n\nUser: What is {a}{op}{b}?",
            "answer": str(ans),
            "requires_tool": True,
            "tool_name": "calculator",
            "reward_type": "correctness",
        })

    # Trivial math — should NOT use tool
    for _ in range(count // 6):
        a, b = random.randint(1, 10), random.randint(1, 10)
        ans = a + b
        samples.append({
            "prompt": f"{tools_sys}\n\nUser: What is {a}+{b}?",
            "answer": str(ans),
            "requires_tool": False,
            "tool_name": None,
            "reward_type": "correctness",
        })

    # Direct-answer (no tool)
    direct = [("Say hello", "Hello!"), ("What color is the sky?", "Blue"),
              ("Is water wet?", "Yes"), ("Name a fruit", "Apple")]
    for _ in range(count // 6):
        q, a = random.choice(direct)
        samples.append({
            "prompt": f"{tools_sys}\n\nUser: {q}",
            "answer": a, "requires_tool": False, "tool_name": None,
            "reward_type": "correctness",
        })

    # Knowledge search
    search_qs = [("Who invented the telephone?", "Alexander Graham Bell"),
                 ("What year did WW2 end?", "1945"),
                 ("Largest ocean?", "Pacific Ocean"),
                 ("Chemical symbol for gold?", "Au")]
    for _ in range(count // 3):
        q, a = random.choice(search_qs)
        samples.append({
            "prompt": f"{tools_sys}\n\nUser: {q}",
            "answer": a, "requires_tool": True, "tool_name": "search",
            "reward_type": "correctness",
        })

    random.shuffle(samples)
    return samples[:count]

# =====================================================================
#  Pipeline Runner
# =====================================================================
@dataclass
class PipelineConfig:
    output_dir: str = "data/pipeline"
    stage0a_count: int = 8000      # basic add/mul — needs density for grid coverage
    stage0b_count: int = 6000      # sub/div/edges — slightly less (fewer combos)
    stage0c_count: int = 5000      # multi-step + word problems
    stage1_count: int = 20000      # instruction following
    stage1_5_count: int = 15000    # pure context grounding (RAG)
    stage2_count: int = 30000      # function calling
    stage3_count: int = 3000       # GRPO prompts
    replay_ratio: float = 0.15

VALID_STAGES = ["0a", "0b", "0c", "1", "1.5", "2", "3", "all"]

def run_pipeline(stage: str, cfg: PipelineConfig) -> None:
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if stage == "all":
        stages = ["0a", "0b", "0c", "1", "1.5", "2", "3"]
    else:
        stages = [stage]

    for s in stages:
        if s == "0a":
            data = generate_stage0a(cfg.stage0a_count)
            save_jsonl(data, str(out / "stage0a_basic_math.jsonl"))

        elif s == "0b":
            data = generate_stage0b(cfg.stage0b_count)
            data = add_replay_buffer(data, [str(out / "stage0a_basic_math.jsonl")], cfg.replay_ratio)
            save_jsonl(data, str(out / "stage0b_expanded_math.jsonl"))

        elif s == "0c":
            data = generate_stage0c(cfg.stage0c_count)
            data = add_replay_buffer(data, [
                str(out / "stage0a_basic_math.jsonl"),
                str(out / "stage0b_expanded_math.jsonl"),
            ], cfg.replay_ratio)
            save_jsonl(data, str(out / "stage0c_multistep_math.jsonl"))

        elif s == "1":
            data = generate_stage1(cfg.stage1_count)
            data = add_replay_buffer(data, [
                str(out / "stage0a_basic_math.jsonl"),
                str(out / "stage0b_expanded_math.jsonl"),
                str(out / "stage0c_multistep_math.jsonl"),
            ], cfg.replay_ratio)
            save_jsonl(data, str(out / "stage1_instruct.jsonl"))

        elif s == "1.5":
            data = generate_stage1_5(cfg.stage1_5_count)
            data = add_replay_buffer(data, [
                str(out / "stage0c_multistep_math.jsonl"),
                str(out / "stage1_instruct.jsonl"),
            ], cfg.replay_ratio)
            save_jsonl(data, str(out / "stage1_5_context_grounding.jsonl"))

        elif s == "2":
            data = generate_stage2(cfg.stage2_count)
            data = add_replay_buffer(data, [
                str(out / "stage0c_multistep_math.jsonl"),
                str(out / "stage1_instruct.jsonl"),
                str(out / "stage1_5_context_grounding.jsonl"),
            ], cfg.replay_ratio)
            save_jsonl(data, str(out / "stage2_function_calling.jsonl"))

        elif s == "3":
            data = generate_stage3(cfg.stage3_count)
            save_jsonl(data, str(out / "stage3_grpo_prompts.jsonl"))

    # Summary
    print("\n" + "=" * 55)
    print("  Pipeline Summary")
    print("=" * 55)
    for f in sorted(out.glob("*.jsonl")):
        with open(f) as fh:
            n = sum(1 for _ in fh)
        print(f"  {f.name:<40} {n:>6}")
    print("=" * 55)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic SFT pipeline data")
    parser.add_argument("--stage", default="all", choices=VALID_STAGES)
    parser.add_argument("--output-dir", default="data/pipeline")
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--replay-ratio", type=float, default=0.15)
    args = parser.parse_args()

    cfg = PipelineConfig(output_dir=args.output_dir, replay_ratio=args.replay_ratio)
    if args.count and args.stage != "all":
        stage_key = f"stage{args.stage.replace('.', '_')}_count"
        if hasattr(cfg, stage_key):
            setattr(cfg, stage_key, args.count)

    run_pipeline(args.stage, cfg)
