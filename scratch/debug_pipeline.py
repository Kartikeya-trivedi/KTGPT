import modal
import os
import json

app = modal.App("kt-gpt-debug")
checkpoint_volume = modal.Volume.from_name("kt-gpt-checkpoints", create_if_missing=True)
CHECKPOINT_MOUNT = "/checkpoints"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "transformers==4.44.2",
        "safetensors==0.4.5"
    )
)

@app.function(
    image=image,
    volumes={CHECKPOINT_MOUNT: checkpoint_volume},
)
def run_debug():
    from transformers import AutoTokenizer
    
    print("=== STEP 1: Tokenizer Sanity Check ===")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", use_fast=True)
    text = "hello world"
    tokens = tokenizer.encode(text)
    detokenized = tokenizer.decode(tokens)
    print(f"Original: '{text}'")
    print(f"Tokens: {tokens}")
    print(f"Detokenized: '{detokenized}'")

    print("\n=== STEP 2: Inspect Raw Training Samples ===")
    data_path = "/checkpoints/data/openhermes_sft_100k.jsonl"
    
    if not os.path.exists(data_path):
        print(f"File not found: {data_path}")
        # List contents of data dir
        print("Contents of /checkpoints/data/:")
        print(os.listdir("/checkpoints/data/"))
        return

    with open(data_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            sample = json.loads(line.strip())
            prompt = sample.get("prompt", "")
            response = sample.get("response", "")
            
            print(f"\nSample {i+1}:")
            print(f"--- Prompt (first 100 chars) ---\n{prompt[:100]!r}")
            print(f"--- Response (first 100 chars) ---\n{response[:100]!r}")
            
            # Show how it's tokenized in sft.py
            formatted_prompt = f"[INST] {prompt} [/INST]"
            formatted_response = f" {response}{tokenizer.eos_token}"
            
            p_ids = tokenizer.encode(formatted_prompt, add_special_tokens=True)
            r_ids = tokenizer.encode(formatted_response, add_special_tokens=False)
            
            print("Tokenized prompt (first 10):", p_ids[:10])
            print("Prompt detokenized (first 10):", repr(tokenizer.decode(p_ids[:10])))
            print("Tokenized response (last 5):", r_ids[-5:])
            print("Response detokenized (last 5):", repr(tokenizer.decode(r_ids[-5:])))

@app.local_entrypoint()
def main():
    run_debug.remote()

if __name__ == "__main__":
    main()
