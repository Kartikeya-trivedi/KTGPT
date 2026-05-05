"""
KT-GPT Chat Backend — Modal Cloud Deployment
=============================================

Deploy:   modal deploy ktgpt_chat/backend/service.py
"""

import os
import sys
import json
import torch
import modal

# ── Modal Setup ──────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1.0",
        "transformers>=4.36.0",
        "sentencepiece>=0.1.99",
        "protobuf>=3.20.0",
        "duckduckgo-search>=5.0",
        "fastapi[standard]",
    )
    .add_local_dir(
        PROJECT_ROOT,
        remote_path="/root/kt-gpt",
        ignore=["**/__pycache__", "**/.venv", "**/checkpoints", "**/.git",
                "**/node_modules", "**/.next", "**/ktgpt_chat", "*.pt", "*.jsonl"],
    )
)

app = modal.App("ktgpt-chat-service")
volume = modal.Volume.from_name("kt-gpt-checkpoints")

# ── Global model cache ───────────────────────────────────────────────
_model = None
_tokenizer = None
_device = None

TOOLS_SYSTEM_PROMPT = """You have access to the following tools:

- calculator(expression: string): Evaluate a math expression
- search(query: string): Search for information

To use a tool, respond with:
<tool_call>{"name": "tool_name", "arguments": {...}}</tool_call>

If the answer is obvious or trivial, respond directly WITHOUT calling a tool."""


def _load_model():
    global _model, _tokenizer, _device
    if _model is not None:
        return

    sys.path.insert(0, "/root/kt-gpt")
    from model.config import KTGPTConfig
    from model.model import KTGPT
    from transformers import AutoTokenizer

    _device = torch.device("cuda")
    print("Loading model on CUDA...")

    config = KTGPTConfig()
    _model = KTGPT(config).to(device=_device, dtype=torch.bfloat16)

    # Temporarily using Stage 1.5 checkpoint because Stage 2 suffered from
    # catastrophic forgetting (overfit to math/code) and spits out code for chat.
    ckpt_path = "/checkpoints/sft_stage1_5/phase3/final.pt"
    if not os.path.exists(ckpt_path):
        ckpt_path = "/checkpoints/sft_stage2/phase3/final.pt"

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=_device)
        _model.load_state_dict(ckpt["model"])
        print(f"✅ Loaded checkpoint from {ckpt_path}")
    else:
        print(f"❌ Checkpoint not found at {ckpt_path}")

    _model.eval()
    _tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", use_fast=True)
    _tokenizer.pad_token = _tokenizer.eos_token
    print("✅ Model ready")


# ── Web Search (DuckDuckGo — free, no API key) ──────────────────────
def fetch_web_context(query: str, max_results: int = 3) -> str:
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))

        if not results:
            return ""

        snippets = []
        for r in results:
            title = r.get("title", "")
            body = r.get("body", "")
            snippets.append(f"{title}: {body}")

        context_str = "\n\n".join(snippets)
        # Hard limit context size so it doesn't push out the [INST] instructions
        return context_str[:3000]
    except Exception as e:
        print(f"⚠ Search failed: {e}")
        return ""


# ── Calculator ───────────────────────────────────────────────────────
def _execute_calculator(expression: str) -> str:
    try:
        allowed = set("0123456789+-*/.(). ")
        if not all(c in allowed for c in expression):
            return "Error: Invalid characters"
        return str(eval(expression))
    except Exception as e:
        return f"Error: {str(e)}"


from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

web_app = FastAPI(title="KT-GPT Modal Service")

web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.function(
    gpu="A10G",
    volumes={"/checkpoints": volume},
    image=image,
    timeout=120,
    scaledown_window=300,
)
@modal.asgi_app()
def fastapi_app():
    return web_app

@web_app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    _load_model()

    user_msg = data.get("prompt", "")
    tool_mode = data.get("toolMode", False)

    if not user_msg:
        return {"error": "No prompt provided"}

    if tool_mode:
        # ── TOOL MODE ────────────────────────────────────────
        prompt = f"[INST] {TOOLS_SYSTEM_PROMPT}\n\nUser: {user_msg} [/INST]\n"
    else:
        # ── CHAT MODE: fetch web context → RAG prompt ────────
        print(f"🔍 Searching web for: {user_msg}")
        context = fetch_web_context(user_msg)

        if context:
            print(f"📄 Got {len(context)} chars of context")
            prompt = (
                f"[INST] Answer ONLY using the context.\n\n"
                f"Context:\n{context}\n\n"
                f"Question:\n{user_msg} [/INST]\n"
            )
            print(f"\n--- CONSTRUCTED RAG PROMPT ---\n{prompt}------------------------------\n")
        else:
            prompt = f"[INST] {user_msg} [/INST]\n"

    input_ids = torch.tensor(
        [_tokenizer.encode(prompt, add_special_tokens=False)], device=_device
    )

    # Truncate if too long
    max_prompt_len = 1800
    if input_ids.shape[1] > max_prompt_len:
        input_ids = input_ids[:, -max_prompt_len:]
        print(f"⚠ Prompt truncated to {max_prompt_len} tokens")

    # Use very low temperature for RAG to force it to stick to the facts
    generation_temp = 0.8 if tool_mode else 0.1

    with torch.no_grad():
        output_ids = _model.generate(
            input_ids,
            max_new_tokens=256,
            temperature=generation_temp,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.2,
            eos_token_id=_tokenizer.eos_token_id,
        )

    full_text = _tokenizer.decode(output_ids[0], skip_special_tokens=True)
    generated_ids = output_ids[0][input_ids.shape[1]:]
    response_part = _tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # ── Tool Handling ────────────────────────────────────────
    if tool_mode and "<tool_call>" in response_part and "</tool_call>" in response_part:
        try:
            start = response_part.find("<tool_call>") + len("<tool_call>")
            end = response_part.find("</tool_call>")
            call_json = json.loads(response_part[start:end])

            if call_json.get("name") == "calculator":
                expr = call_json.get("arguments", {}).get("expression", "")
                result = _execute_calculator(expr)

                followup_prompt = f"{full_text}\n\nTool returned: {result}\n\n"
                followup_ids = torch.tensor(
                    [_tokenizer.encode(followup_prompt, add_special_tokens=False)],
                    device=_device,
                )

                with torch.no_grad():
                    final_ids = _model.generate(
                        followup_ids,
                        max_new_tokens=128,
                        temperature=0.7,
                        repetition_penalty=1.2,
                        eos_token_id=_tokenizer.eos_token_id,
                    )

                final_text = _tokenizer.decode(final_ids[0], skip_special_tokens=True)
                answer = final_text[len(followup_prompt):].strip()

                return {
                    "raw_output": response_part,
                    "tool_call": call_json,
                    "tool_result": result,
                    "answer": answer if answer else f"The answer is {result}",
                }
        except Exception as e:
            return {"error": f"Tool execution failed: {e}", "raw_output": response_part}

    return {
        "raw_output": response_part,
        "answer": response_part,
    }
