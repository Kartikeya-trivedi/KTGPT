import time
from datasets import load_dataset

print("\nTesting fineweb-edu filter speed...")
try:
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-100BT", split="train", streaming=True)
    ds = ds.filter(lambda row: row.get("score", 0) >= 4.0)
    
    start = time.time()
    print("Waiting for the first fineweb-edu row to pass the filter...")
    row = next(iter(ds))
    elapsed = time.time() - start
    
    print(f"Found row in {elapsed:.2f} seconds!")
    print("score value:", row.get("score"))
    
except Exception as e:
    print("Error:", e)
