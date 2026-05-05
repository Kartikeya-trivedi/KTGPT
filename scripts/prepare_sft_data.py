import json
import random
from datasets import load_dataset
from tqdm import tqdm
import os
import modal

image = modal.Image.debian_slim(python_version="3.12").pip_install("datasets", "tqdm")
app = modal.App("kt-gpt-prep-sft")
checkpoint_volume = modal.Volume.from_name("kt-gpt-checkpoints", create_if_missing=True)

def is_high_quality(response: str) -> bool:
    """Filters out low quality, excessively long/short, or repetitive examples."""
    if len(response) < 50:
        return False
    if len(response) > 3000:
        return False
    if response.count('\n') > 50:
        return False
    return True

@app.function(
    image=image,
    volumes={"/checkpoints": checkpoint_volume},
    timeout=7200
)
def prepare_sft_data():
    output_file = "/checkpoints/data/openhermes_sft_100k.jsonl"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    valid_samples = []
    
    # 1. OpenHermes-2.5 (40K)
    print("Loading teknium/OpenHermes-2.5...")
    ds_hermes = load_dataset("teknium/OpenHermes-2.5", split="train", streaming=True)
    hermes_count = 0
    for row in tqdm(ds_hermes, desc="OpenHermes"):
        conv = row.get("conversations", [])
        if len(conv) != 2 or conv[0]["from"] != "human" or conv[1]["from"] != "gpt":
            continue
        
        prompt, response = conv[0]["value"], conv[1]["value"]
        
        # Keep general knowledge (avoid heavy code here to let Magicoder handle it)
        if "```" in response:
            continue
            
        if is_high_quality(response):
            valid_samples.append({"prompt": prompt, "response": response})
            hermes_count += 1
            if hermes_count >= 40000:
                break
                
    # 2. Magicoder (35K)
    print("Loading ise-uiuc/Magicoder-Evol-Instruct-110K...")
    ds_magicoder = load_dataset("ise-uiuc/Magicoder-Evol-Instruct-110K", split="train", streaming=True)
    magicoder_count = 0
    for row in tqdm(ds_magicoder, desc="Magicoder"):
        prompt = row.get("instruction", "")
        response = row.get("response", "")
        
        if prompt and response and is_high_quality(response):
            valid_samples.append({"prompt": prompt, "response": response})
            magicoder_count += 1
            if magicoder_count >= 35000:
                break
                
    # 3. MetaMathQA (25K)
    print("Loading meta-math/MetaMathQA...")
    ds_math = load_dataset("meta-math/MetaMathQA", split="train", streaming=True)
    math_count = 0
    for row in tqdm(ds_math, desc="MetaMathQA"):
        prompt = row.get("query", "")
        response = row.get("response", "")
        
        if prompt and response and is_high_quality(response):
            valid_samples.append({"prompt": prompt, "response": response})
            math_count += 1
            if math_count >= 25000:
                break

    print(f"Collected: {hermes_count} Hermes, {magicoder_count} Magicoder, {math_count} Math.")
    
    print("Shuffling and saving 100K mixed dataset...")
    random.seed(42)
    random.shuffle(valid_samples)
    
    print(f"Saving to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in valid_samples:
            f.write(json.dumps(sample) + "\n")
            
    print("Done!")

@app.local_entrypoint()
def main():
    print("Launching SFT data prep on Modal...")
    prepare_sft_data.remote()

if __name__ == "__main__":
    main()
