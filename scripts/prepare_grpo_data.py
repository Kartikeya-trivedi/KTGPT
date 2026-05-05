"""
Pre-GRPO Data Preparation
=========================

Downloads the GSM8K dataset, applies the XML reasoning format,
and saves it as a JSONL file to the Modal volume. This prevents
on-the-fly downloading bottlenecks during GRPO training.
"""

import json
import modal
from typing import Optional

# Setup Modal App
app = modal.App("kt-gpt-prep-grpo")

# Volume for saving data
checkpoint_volume = modal.Volume.from_name("kt-gpt-checkpoints", create_if_missing=True)
CHECKPOINT_MOUNT = "/checkpoints"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "datasets>=2.16.0",
        "transformers>=4.36.0",
        "sentencepiece>=0.1.99",
        "protobuf>=3.20.0"
    )
    .add_local_dir(".", remote_path="/root/kt-gpt", ignore=["**/__pycache__", "**/.venv", "**/checkpoints", "**/.git"])
)


SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

def extract_hash_answer(text: str) -> Optional[str]:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

@app.local_entrypoint()
def main():
    print("Sending GRPO data prep job to Modal...")
    process_grpo_data.remote()

@app.function(
    image=image,
    volumes={CHECKPOINT_MOUNT: checkpoint_volume},
    timeout=3600,
    memory=4096,
)
def process_grpo_data():
    from datasets import load_dataset
    from data.mix import get_tokenizer
    import os

    tokenizer = get_tokenizer()
    output_dir = f"{CHECKPOINT_MOUNT}/data"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "gsm8k_grpo.jsonl")

    print(f"Downloading openai/gsm8k dataset...")
    data = load_dataset('openai/gsm8k', 'main', split='train')
    
    samples_saved = 0
    with open(output_file, 'w', encoding='utf-8') as f:
        for row in data:
            question = row['question']
            raw_answer = row['answer']
            
            answer = extract_hash_answer(raw_answer)
            if not answer:
                continue
                
            prompt_text = f"<s>[INST] <<SYS>>\n{SYSTEM_PROMPT}\n<</SYS>>\n\n{question} [/INST]"
            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
            
            sample = {
                "prompt_ids": prompt_ids,
                "prompt_text": prompt_text,
                "answer": answer
            }
            
            f.write(json.dumps(sample) + "\n")
            samples_saved += 1
            
    print(f"Successfully saved {samples_saved} samples to {output_file}.")
