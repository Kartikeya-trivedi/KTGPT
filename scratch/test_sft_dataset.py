import sys
sys.path.append(".")

from train.sft import SFTDataset
from data.mix import get_tokenizer

def main():
    tokenizer = get_tokenizer()
    dataset = SFTDataset("data/phase1_sft.jsonl", tokenizer, seq_len=1024)

    print("\n--- SANITY TEST: FIRST 5 SAMPLES ---")
    for i in range(5):
        sample = dataset[i]
        input_ids = sample["input_ids"]
        labels = sample["labels"]

        # find actual unpadded length
        valid_len = (labels != -100).nonzero(as_tuple=True)[0]
        if len(valid_len) > 0:
            last_valid = valid_len[-1].item()
        else:
            last_valid = -1
        
        # total sequence including prompt, up to the end of the response
        first_pad = (input_ids == tokenizer.pad_token_id).nonzero(as_tuple=True)[0]
        if len(first_pad) > 0:
            total_len = first_pad[0].item()
        else:
            total_len = len(input_ids)

        seq = input_ids[:total_len]
        seq_labels = labels[:total_len]

        print(f"\nSample {i+1}:")
        print(f"Total length before padding: {total_len}")
        print("Raw decoded input:", repr(tokenizer.decode(seq)))
        
        prompt_mask = (seq_labels == -100)
        resp_mask = (seq_labels != -100)
        
        prompt_ids = seq[prompt_mask]
        resp_ids = seq[resp_mask]
        
        print("Decoded prompt (masked -100):", repr(tokenizer.decode(prompt_ids)))
        print("Decoded response (learned):", repr(tokenizer.decode(resp_ids)))
        print("Response IDs:", resp_ids.tolist())
        
        if len(resp_ids) > 0:
            print("Last token ID in response:", resp_ids[-1].item())
            print("EOS token ID:", tokenizer.eos_token_id)
            print("Is last token EOS?", resp_ids[-1].item() == tokenizer.eos_token_id)

if __name__ == "__main__":
    main()
