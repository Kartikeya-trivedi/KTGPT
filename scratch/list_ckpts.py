import modal
import os

app = modal.App("kt-gpt-list-checkpoints")
checkpoint_volume = modal.Volume.from_name("kt-gpt-checkpoints")

@app.local_entrypoint()
def main():
    list_files.remote()

@app.function(volumes={"/checkpoints": checkpoint_volume})
def list_files():
    if os.path.exists("/checkpoints/sft"):
        files = os.listdir("/checkpoints/sft")
        print("SFT Checkpoints:", files)
    else:
        print("No /checkpoints/sft directory found.")
