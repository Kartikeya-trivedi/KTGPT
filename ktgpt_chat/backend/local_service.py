import os
import torch
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Project Imports
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from model.config import KTGPTConfig
from model.model import KTGPT
from data.mix import get_tokenizer

app = FastAPI(title="KT-GPT Local Inference Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Globals ──────────────────────────────────────────────────────────
model = None
tokenizer = None
device = None

TOOLS_SYSTEM_PROMPT = """You have access to the following tools:

- calculator(expression: string): Evaluate a math expression
- search(query: string): Search for information

To use a tool, respond with:
<tool_call>{"name": "tool_name", "arguments": {...}}</tool_call>

If the answer is obvious or trivial, respond directly WITHOUT calling a tool."""


class ChatRequest(BaseModel):
    prompt: str
    toolMode: bool = False


# ── Model Loading ────────────────────────────────────────────────────
def load_model():
    global model, tokenizer, device
    print("Loading model to local GPU...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = KTGPTConfig()
    model = KTGPT(config).to(device=device, dtype=torch.bfloat16)

    checkpoint_paths = [
        "../../final.pt",
        "../../final_sft_stage2.pt",
        "./final.pt",
        "./final_sft_stage2.pt",
    ]

    path_to_load = None
    for path in checkpoint_paths:
        if os.path.exists(path):
            path_to_load = path
            break

    if path_to_load:
        ckpt = torch.load(path_to_load, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"✅ Loaded model from {path_to_load}")
    else:
        print("❌ CRITICAL: No checkpoint found!")

    model.eval()
    tokenizer = get_tokenizer()
    print("✅ Model ready for inference")


# ── Web Search (DuckDuckGo — free, no API key) ──────────────────────
def fetch_web_context(query: str, max_results: int = 3) -> str:
    """Search the web and return top snippets as context."""
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

        return "\n\n".join(snippets)
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


@app.on_event("startup")
async def startup_event():
    load_model()


# ── Chat Endpoint ────────────────────────────────────────────────────
@app.post("/chat")
async def chat(request: ChatRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    user_msg = request.prompt

    if request.toolMode:
        # ── TOOL MODE: wrap with tool system prompt ──────────────
        prompt = f"[INST] {TOOLS_SYSTEM_PROMPT}\n\nUser: {user_msg} [/INST]\n"
    else:
        # ── CHAT MODE: fetch web context → RAG prompt ────────────
        print(f"🔍 Searching web for: {user_msg}")
        context = fetch_web_context(user_msg)

        if context:
            print(f"📄 Got {len(context)} chars of context")
            prompt = (
                f"[INST] Answer ONLY using the context.\n\n"
                f"Context:\n{context}\n\n"
                f"Question:\n{user_msg} [/INST]\n"
            )
        else:
            # Fallback: no context found, just ask directly
            prompt = f"[INST] {user_msg} [/INST]\n"

    input_ids = torch.tensor(
        [tokenizer.encode(prompt, add_special_tokens=False)], device=device
    )

    # Truncate if prompt is too long (max_seq_len - generation room)
    max_prompt_len = 1800  # leave 248 tokens for generation
    if input_ids.shape[1] > max_prompt_len:
        input_ids = input_ids[:, -max_prompt_len:]
        print(f"⚠ Prompt truncated to {max_prompt_len} tokens")

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=256,
            temperature=0.8,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.2,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = output_ids[0][input_ids.shape[1]:]
    response_part = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # ── Tool Handling (only in tool mode) ────────────────────────
    if request.toolMode and "<tool_call>" in response_part and "</tool_call>" in response_part:
        try:
            start = response_part.find("<tool_call>") + len("<tool_call>")
            end = response_part.find("</tool_call>")
            call_json = json.loads(response_part[start:end])

            if call_json.get("name") == "calculator":
                expr = call_json.get("arguments", {}).get("expression", "")
                result = _execute_calculator(expr)

                followup_prompt = f"{full_text}\n\nTool returned: {result}\n\n"
                followup_ids = torch.tensor(
                    [tokenizer.encode(followup_prompt, add_special_tokens=False)],
                    device=device,
                )

                with torch.no_grad():
                    final_ids = model.generate(
                        followup_ids,
                        max_new_tokens=128,
                        temperature=0.7,
                        repetition_penalty=1.2,
                        eos_token_id=tokenizer.eos_token_id,
                    )

                final_text = tokenizer.decode(final_ids[0], skip_special_tokens=True)
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


if __name__ == "__main__":
    print("\n🚀 Starting KT-GPT Local Inference Server...")
    print("   Frontend: open ktgpt_chat/frontend/index.html")
    print("   Backend:  http://localhost:8000\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
