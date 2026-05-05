import random
import json

SEED = 42
TOTAL_SAMPLES = 6000  # slightly larger for stronger signal

random.seed(SEED)

def format_sample(inst, resp):
    return {"prompt": inst.strip(), "response": resp.strip()}


# -------------------------
# 1. STRUCTURED ADDITION GRID (pattern learning)
# -------------------------
def addition_grid():
    data = []
    for a in range(1, 21):
        for b in range(1, 21):
            q = f"What is {a}+{b}?"
            data.append((q, str(a + b)))
    return data


# -------------------------
# 2. STRUCTURED MULTIPLICATION GRID
# -------------------------
def multiplication_grid():
    data = []
    for a in range(1, 13):
        for b in range(1, 13):
            q = f"What is {a}*{b}?"
            data.append((q, str(a * b)))
    return data


# -------------------------
# 3. HIGH-FREQUENCY ANCHORS (repeated → locks correctness)
# -------------------------
def anchors():
    base = [
        ("What is 2+2?", "4"),
        ("What is 3+3?", "6"),
        ("What is 4+4?", "8"),
        ("What is 5+5?", "10"),
        ("What is 6+6?", "12"),
        ("What is 6*7?", "42"),
        ("What is 7*8?", "56"),
        ("What is 8*9?", "72"),
    ]

    data = []
    for _ in range(150):  # repeat anchors
        q, a = random.choice(base)
        data.append((q, a))

    return data


# -------------------------
# 4. RANDOM MATH (light variation)
# -------------------------
def random_math():
    op = random.choice(["+", "*"])

    if op == "+":
        a = random.randint(1, 50)
        b = random.randint(1, 50)
        ans = a + b
    else:
        a = random.randint(1, 12)
        b = random.randint(1, 12)
        ans = a * b

    return f"What is {a}{op}{b}?", str(ans)


# -------------------------
# 5. FACTS (light grounding)
# -------------------------
facts = [
    ("What is AI?", "Artificial intelligence is the field of building machines that can perform tasks requiring human intelligence."),
    ("What is machine learning?", "Machine learning is a field of AI that enables systems to learn from data."),
    ("What is Python?", "Python is a programming language."),
    ("What is a computer?", "A computer processes data."),
]

def get_fact():
    return random.choice(facts)


# -------------------------
# 6. INSTRUCTIONS (keep alignment alive)
# -------------------------
instructions = [
    ("Say hello", "Hello!"),
    ("Respond with one word: color of sky", "Blue"),
    ("Respond with one word: opposite of hot", "Cold"),
]

def get_instruction():
    return random.choice(instructions)


# -------------------------
# BUILD DATASET
# -------------------------
samples = []

# ---- inject structured math (core signal) ----
structured = addition_grid() + multiplication_grid()
structured = random.choices(structured, k=2000)  # ~40%

# ---- inject anchors (high-frequency correction) ----
anchor_data = anchors()  # ~20%

# ---- convert to format ----
structured = [format_sample(q, a) for q, a in structured]
anchor_data = [format_sample(q, a) for q, a in anchor_data]

samples.extend(structured)
samples.extend(anchor_data)


# ---- remaining mix (random math + facts + instructions) ----
remaining = TOTAL_SAMPLES - len(samples)

for _ in range(remaining):
    r = random.random()

    if r < 0.5:
        q, a = random_math()      # ~20%
    elif r < 0.75:
        q, a = get_fact()         # ~10–15%
    else:
        q, a = get_instruction()  # ~10–15%

    samples.append(format_sample(q, a))


# ---- shuffle ----
random.shuffle(samples)


# -------------------------
# SAVE
# -------------------------
with open("data/phase1_sft_v3.jsonl", "w") as f:
    for s in samples:
        f.write(json.dumps(s) + "\n")

print("Dataset ready:", len(samples))
