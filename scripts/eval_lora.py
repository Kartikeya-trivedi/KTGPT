import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from model.config import KTGPTConfig
from model.model import KTGPT
from model.lora import LoRAConfig, inject_lora
from data.mix import get_tokenizer

def generate(model, tokenizer, prompt, device, max_new_tokens=128):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            top_p=0.9,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True).strip()

def run_eval(checkpoint_path, device="cuda"):
    print(f"--- LoRA Evaluation ---")
    print(f"Loading Checkpoint: {checkpoint_path}")
    
    config = KTGPTConfig()
    model = KTGPT(config).to(device=device, dtype=torch.bfloat16)
    
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    
    # Check if this is a LoRA-only checkpoint or a full model
    is_lora_only = any("lora_" in k for k in state_dict.keys())
    
    if is_lora_only:
        print("Detected LoRA-only weights. Injecting adapters...")
        inject_lora(model, LoRAConfig(r=8, alpha=16))
        model.to(device=device, dtype=torch.bfloat16)
        model.load_state_dict(state_dict, strict=False)
    else:
        print("Detected Full/Merged model. Loading directly...")
        # If the checkpoint is merged, it might have structural differences if it was saved
        # with LoRA wrappers. We try a clean load first.
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            print("Direct load failed (likely structure mismatch). Trying with LoRA injection...")
            inject_lora(model, LoRAConfig(r=8, alpha=16))
            model.to(device=device, dtype=torch.bfloat16)
            model.load_state_dict(state_dict, strict=False)
    
    model.eval()
    tokenizer = get_tokenizer()

    test_cases = [
        {
            "category": "RAG Grounding",
            "instruction": "Answer using the provided context.",
            "input": "Context: ForgeTube generates videos from text prompts using diffusion models.\nQuestion: What is ForgeTube?"
        },
        {
            "category": "Math Tool Format (Trigger)",
            "instruction": "The Python calculator returned a result. Use it to answer the question.",
            "input": "Question: What is 456 multiplied by 12?\nPython result: 5472"
        },
        {
            "category": "Conciseness",
            "instruction": "Answer concisely.",
            "input": "Question: What is a prime number?"
        },
        {
            "category": "Refusal / No-Info",
            "instruction": "Answer using the provided context.",
            "input": "Context: The document discusses quantum computing basics.\nQuestion: What is the capital of France?"
        }
    ]

    print(f"\n{'='*80}")
    for case in test_cases:
        prompt = f"### Instruction:\n{case['instruction']}\n\n### Input:\n{case['input']}\n\n### Response:\n"
        response = generate(model, tokenizer, prompt, device)
        
        display_input = case['input'].replace('\n', ' | ')
        print(f"Category: {case['category']}")
        print(f"Input:    {display_input}")
        print(f"Output:   {response}")
        print(f"{'-'*80}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model or lora .pt")
    # Base is now optional
    parser.add_argument("--base", type=str, help="Optional path to base model if using lora-only")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_eval(args.checkpoint, device)
