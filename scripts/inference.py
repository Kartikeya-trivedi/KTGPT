"""
KT-GPT Inference
=================

Run inference on Modal with various checkpoint stages.

Usage:
    modal run scripts/inference.py            # default test (latest SFT)
    modal run scripts/inference.py::eval_math # math eval after Stage 0
    modal run scripts/inference.py::eval_rag  # context grounding eval after Stage 1.5
    modal run scripts/inference.py::eval_tool # tool-use eval after Stage 2
"""

import os
import sys
import torch
import modal
from typing import Optional

# Setup Modal
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1.0",
        "transformers>=4.36.0",
        "sentencepiece>=0.1.99",
        "protobuf>=3.20.0",
    )
    .add_local_dir(
        PROJECT_ROOT,
        remote_path="/root/kt-gpt",
        ignore=["**/__pycache__", "**/.venv", "**/checkpoints", "**/.git",
                "**/node_modules", "**/.next", "**/ktgpt_chat", "*.pt", "*.jsonl"],
    )
)

app = modal.App("kt-gpt-inference")
checkpoint_volume = modal.Volume.from_name("kt-gpt-checkpoints", create_if_missing=True)
CHECKPOINT_MOUNT = "/checkpoints"

# ═══════════════════════════════════════════════════════════════════════
#  Core Inference Function
# ═══════════════════════════════════════════════════════════════════════

@app.function(
    image=image,
    volumes={CHECKPOINT_MOUNT: checkpoint_volume},
    gpu="A100",
    timeout=3600,
)
def run_inference(
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.15,
    ckpt_path: str = "/checkpoints/sft_lora/phase3/final.pt",
    lora_path: Optional[str] = None,
    strict_tool_math: bool = True,
):
    sys.path.insert(0, "/root/kt-gpt")
    
    from model.config import KTGPTConfig
    from model.lora import LoRAConfig, inject_lora, load_lora_state_dict
    from model.model import KTGPT
    from transformers import AutoTokenizer
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading on {device}...")
    
    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 2. Load Model Config & Initialize Model
    config = KTGPTConfig()
    config.max_seq_len = 2048  # Match SFT training
    model = KTGPT(config).to(device)
    if lora_path:
        inject_lora(model, LoRAConfig())
    
    # 3. Load Checkpoint
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
        
    print(f"Loading checkpoint from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    if lora_path:
        lora_ckpt = torch.load(lora_path, map_location=device, weights_only=False)
        load_lora_state_dict(model, lora_ckpt["lora"], strict=False)
    
    # Restore router biases
    for i, layer in enumerate(model.layers):
        key = f"layer_{i}"
        if key in checkpoint.get("router_biases", {}):
            layer.ffn.router.expert_bias.data.copy_(checkpoint["router_biases"][key])
            
    model.eval()
    
    # 4. Prepare Prompt
    print(f"\n--- Prompt ---\n{prompt}\n--------------\n")
    input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True).to(device)
    
    print("Generating...")
    def _generate_once() -> str:
        generated_tokens_local = []
        local_input_ids = input_ids
        local_past_kv = None
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits, new_kv = model(local_input_ids, use_cache=True, past_kv_list=local_past_kv)
                next_token_logits = logits[0, -1, :] / max(temperature, 1e-6)
                if repetition_penalty > 1.0 and len(generated_tokens_local) > 0:
                    for token_id in set(generated_tokens_local):
                        if next_token_logits[token_id] < 0:
                            next_token_logits[token_id] *= repetition_penalty
                        else:
                            next_token_logits[token_id] /= repetition_penalty
                if top_k > 0:
                    top_k_val = min(top_k, next_token_logits.size(-1))
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k_val)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float("-inf")
                probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                if top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumsum = torch.cumsum(sorted_probs, dim=-1)
                    remove = cumsum > top_p
                    remove[..., 1:] = remove[..., :-1].clone()
                    remove[..., 0] = False
                    probs[sorted_indices[remove]] = 0
                    probs = probs / probs.sum()
                next_token = torch.multinomial(probs, num_samples=1)
                generated_tokens_local.append(next_token.item())
                local_input_ids = next_token.unsqueeze(0)
                local_past_kv = new_kv
                if next_token.item() == tokenizer.eos_token_id:
                    break
        return tokenizer.decode(generated_tokens_local, skip_special_tokens=True)

    full_response = _generate_once()
    if not full_response.strip() or (strict_tool_math and any(ch.isdigit() for ch in prompt) and "<tool_call>" not in full_response):
        full_response = _generate_once()

    print(full_response, end="", flush=True)

    # Check for tool calls in output
    if "<tool_call>" in full_response:
        print("\n\n[TOOL CALL DETECTED]")
        import json
        try:
            tc_start = full_response.index("<tool_call>") + len("<tool_call>")
            tc_end = full_response.index("</tool_call>")
            tool_json = json.loads(full_response[tc_start:tc_end])
            print(f"  Tool: {tool_json.get('name', 'unknown')}")
            print(f"  Args: {tool_json.get('arguments', {})}")
            
            # Simulate tool execution
            result = _execute_tool(tool_json)
            print(f"  Result: {result}")
        except (ValueError, json.JSONDecodeError) as e:
            print(f"  [Parse error: {e}]")
                
    print("\n\n✅ Generation complete.")
    full_output = prompt + full_response
    return full_output


def _execute_tool(tool_call: dict) -> str:
    """Simulate tool execution for testing."""
    name = tool_call.get("name", "")
    args = tool_call.get("arguments", {})
    
    if name == "calculator":
        try:
            # Safe eval for math expressions only
            expr = args.get("expression", "")
            allowed = set("0123456789+-*/.(). ")
            if all(c in allowed for c in expr):
                return str(eval(expr))
            return "Error: invalid expression"
        except Exception as e:
            return f"Error: {e}"
    elif name == "search":
        return f"[Mock search result for: {args.get('query', '')}]"
    elif name == "get_weather":
        return f"22°C, partly cloudy in {args.get('location', 'unknown')}"
    else:
        return f"Unknown tool: {name}"


# ═══════════════════════════════════════════════════════════════════════
#  Eval Entrypoints — Mix of SEEN (in training) and UNSEEN (generalization)
# ═══════════════════════════════════════════════════════════════════════

@app.local_entrypoint()
def eval_math(stage: str = "0c"):
    """Eval after Stage 0A/0B/0C — math correctness (seen + unseen)."""
    stage_map = {
        "0a": "/checkpoints/sft_stage0a/phase3/final.pt",
        "0b": "/checkpoints/sft_stage0b/phase3/final.pt",
        "0c": "/checkpoints/sft_stage0c/phase3/final.pt",
    }
    ckpt = stage_map.get(stage, stage_map["0c"])
    
    questions = [
        # --- SEEN (in training data / anchors) ---
        ("What is 7*8?", "56", "SEEN"),
        ("What is 9*9?", "81", "SEEN"),
        ("What is 12*12?", "144", "SEEN"),
        # --- UNSEEN (never in training data) ---
        ("What is 17+38?", "55", "UNSEEN"),         # random 2-digit add
        ("What is 11*13?", "143", "UNSEEN"),         # outside 1-12 grid
        ("What is 200-67?", "133", "UNSEEN"),        # larger subtraction
        ("What is 144/12?", "12", "UNSEEN"),         # reverse of anchor
        ("What is 7+3*6?", "25", "UNSEEN"),          # order of operations
        ("What is 99+99?", "198", "UNSEEN"),         # carry case, unseen pair
        ("What is 50-25+10?", "35", "UNSEEN"),       # multi-step unseen
    ]
    
    print(f"[EVAL] Math eval using checkpoint: {ckpt}")
    for q, expected, split in questions:
        prompt = f"[INST] {q} [/INST]\n"
        print(f"\n  [{split}] Q: {q} (expected: {expected})")
        run_inference.remote(
            prompt=prompt, ckpt_path=ckpt,
            max_new_tokens=20, temperature=0.1,
            top_k=50, repetition_penalty=1.0,
        )

@app.local_entrypoint()
def eval_instruct():
    """Eval after Stage 1 — instruction following (seen + unseen)."""
    ckpt = "/checkpoints/sft_stage1/phase3/final.pt"
    
    questions = [
        # --- SEEN ---
        ("Respond with one word: the opposite of cold.", "Hot", "SEEN"),
        ("Say hello in French.", "Bonjour", "SEEN"),
        # --- UNSEEN ---
        ("Respond with one word: the opposite of light.", "Dark/Heavy", "UNSEEN"),
        ("Say goodbye in Spanish.", "Adiós", "UNSEEN"),
        ("What is 5+3? Respond with just the number.", "8", "UNSEEN"),
        ("List the 3 states of matter.", "Solid, liquid, gas", "UNSEEN"),
        ("Return as JSON: city is London, country is UK.", '{"city":"London","country":"UK"}', "UNSEEN"),
        ("Write a Python function that checks if a string is a palindrome.", "def is_palindrome(s): ...", "UNSEEN"),
    ]
    
    print(f"[EVAL] Instruction eval using: {ckpt}")
    for q, expected, split in questions:
        prompt = f"[INST] {q} [/INST]\n"
        print(f"\n  [{split}] Q: {q[:60]}... (expected: {expected})")
        run_inference.remote(
            prompt=prompt, ckpt_path=ckpt,
            max_new_tokens=80, temperature=0.1,
            top_k=50, repetition_penalty=1.0,
        )

@app.local_entrypoint()
def eval_rag():
    """Eval after Stage 1.5 — context grounding (seen + unseen contexts)."""
    ckpt = "/checkpoints/sft_stage1_5/phase3/final.pt"
    
    tests = [
        # --- SEEN context ---
        ("[INST] Answer ONLY using the context.\n\nContext:\nThe periodic table was first published by Dmitri Mendeleev in 1869.\n\nQuestion:\nWho published the periodic table? [/INST]\n",
         "Dmitri Mendeleev", "SEEN"),
        # --- SEEN refusal pattern, but different context ---
        ("[INST] Answer ONLY using the context.\n\nContext:\nThe Eiffel Tower is 330 meters tall.\n\nQuestion:\nHow many visitors does the Eiffel Tower get per year? [/INST]\n",
         "SHOULD REFUSE", "SEEN"),
        # --- UNSEEN context (never in training) ---
        ("[INST] Answer ONLY using the context.\n\nContext:\nThe Pacific Ocean covers approximately 165.25 million square kilometers, making it larger than all land area combined.\n\nQuestion:\nHow large is the Pacific Ocean? [/INST]\n",
         "Approximately 165.25 million square kilometers", "UNSEEN"),
        # --- UNSEEN refusal (novel context + unanswerable question) ---
        ("[INST] Answer ONLY using the context.\n\nContext:\nRust is a systems programming language focused on safety and performance.\n\nQuestion:\nWho created Rust? [/INST]\n",
         "SHOULD REFUSE — creator not in context", "UNSEEN"),
        # --- UNSEEN distractor (novel paragraphs) ---
        ("[INST] Answer ONLY using the context.\n\nContext:\nElephants are the largest land animals.\n\nThe human brain contains approximately 86 billion neurons.\n\nQuestion:\nHow many neurons does the human brain have? [/INST]\n",
         "Approximately 86 billion", "UNSEEN"),
        # --- UNSEEN context with contradictory world knowledge ---
        ("[INST] Answer ONLY using the context.\n\nContext:\nThe capital of Australia is Canberra.\n\nQuestion:\nWhat is the capital of Australia? [/INST]\n",
         "Canberra (tests context > world knowledge)", "UNSEEN"),
    ]
    
    print(f"[EVAL] RAG grounding eval using: {ckpt}")
    for prompt, expected, split in tests:
        print(f"\n  [{split}] Expected: {expected}")
        run_inference.remote(
            prompt=prompt, ckpt_path=ckpt,
            max_new_tokens=50, temperature=0.1,
            top_k=50, repetition_penalty=1.0,
        )

@app.local_entrypoint()
def eval_tool():
    """Eval after Stage 2 — function calling (seen + unseen scenarios)."""
    ckpt = "/checkpoints/sft_stage2/phase3/final.pt"
    
    tools_sys = """You have access to the following tools:

- calculator(expression: string): Evaluate a math expression
- search(query: string): Search for information

To use a tool, respond with:
<tool_call>{"name": "tool_name", "arguments": {...}}</tool_call>

Use calculator for all math questions, including trivial arithmetic."""

    tests = [
        # --- SEEN patterns ---
        (f"[INST] {tools_sys}\n\nUser: What is 2+2? [/INST]\n",
         "SHOULD USE calculator", "SEEN"),
        (f"[INST] {tools_sys}\n\nUser: Say hello. [/INST]\n",
         "SHOULD ANSWER: Hello! (no tool)", "SEEN"),
        # --- UNSEEN scenarios ---
        (f"[INST] {tools_sys}\n\nUser: What is 347*891? [/INST]\n",
         "SHOULD USE calculator", "UNSEEN"),
        (f"[INST] {tools_sys}\n\nUser: What is the square root of 2025? [/INST]\n",
         "SHOULD USE calculator", "UNSEEN"),
        (f"[INST] {tools_sys}\n\nUser: Who won the Nobel Prize in Physics in 2023? [/INST]\n",
         "SHOULD USE search", "UNSEEN"),
        (f"[INST] {tools_sys}\n\nUser: What is 5+3? [/INST]\n",
         "SHOULD USE calculator", "UNSEEN"),
        (f"[INST] {tools_sys}\n\nUser: What is the meaning of life? [/INST]\n",
         "SHOULD ANSWER directly (philosophical, no tool)", "UNSEEN"),
    ]
    
    print(f"[EVAL] Tool-use eval using: {ckpt}")
    for prompt, expected, split in tests:
        print(f"\n  [{split}] Expected: {expected}")
        run_inference.remote(
            prompt=prompt, ckpt_path=ckpt,
            max_new_tokens=100, temperature=0.1,
            top_k=50, repetition_penalty=1.0,
        )

@app.local_entrypoint()
def eval_base():
    """Test the base pretrained model (phase2/final.pt)."""
    prompt = """[INST] Use the provided context to answer the question.

Context:
Machine learning learns patterns from data, while traditional programming relies on explicit rules written by humans.

Question:
How is machine learning different from traditional programming?

Answer concisely. [/INST]"""
    print("Sending Phase 2 pretrain inference job to Modal...")
    run_inference.remote(
        prompt=prompt,
        ckpt_path="/checkpoints/phase2/final.pt",
        max_new_tokens=300,
        temperature=0.6,
        top_k=50,
        repetition_penalty=1.15,
    )

@app.local_entrypoint()
def eval_lora():
    """Test the LoRA fine-tuned model (sft_lora/phase3/final.pt)."""
    ckpt = "/checkpoints/sft_lora/phase3/final.pt"
    
    tools_sys = "You have access to the following tools:\n- calculator(expression: string)\n- search(query: string)\nTo use a tool, respond with:\n<tool_call>{\"name\": \"tool_name\", \"arguments\": {...}}</tool_call>"

    tests = [
        # Instruction / Generic
        ("[INST] Define the theory of relativity concisely. [/INST]\n", "SHOULD ANSWER DIRECTLY", "GENERIC"),
        # Strict Tool Math
        (f"[INST] {tools_sys}\n\nUser: What is 7 + 8? [/INST]\n", "SHOULD USE calculator", "MATH"),
        (f"[INST] {tools_sys}\n\nUser: Compute 250 * 4 [/INST]\n", "SHOULD USE calculator", "MATH"),
        # RAG Grounding
        ("[INST] Use ONLY the provided context to answer.\n\nContext:\nThe capital of Atlantis is Poseidonis.\n\nQuestion: What is the capital of Atlantis? [/INST]\n", "Poseidonis", "RAG"),
    ]
    
    print(f"[EVAL] LoRA evaluation using: {ckpt}")
    for prompt, expected, split in tests:
        print(f"\n  [{split}] Expected: {expected}")
        run_inference.remote(
            prompt=prompt, ckpt_path=ckpt,
            max_new_tokens=100, temperature=0.7,
            top_p=0.9, top_k=50, repetition_penalty=1.15,
        )

@app.local_entrypoint()
def main():
    """Default: run latest SFT checkpoint with a simple test."""
    user_question = "What is 7*8?"
    prompt = f"[INST] {user_question} [/INST]\n"
    
    print("Sending job to Modal...")
    run_inference.remote(
        prompt=prompt,
        max_new_tokens=50,
        temperature=0.1,
        top_k=50,
        repetition_penalty=1.0,
    )

if __name__ == "__main__":
    main()
