from datasets import load_dataset
import os

dataset = load_dataset(
    "SWE-bench/SWE-bench_Multilingual",
    split="test"
)

os.makedirs("gold_patches", exist_ok=True)

for item in dataset:
    instance_id = item["instance_id"]
    patch = item["patch"]

    with open(f"gold_patches/{instance_id}.patch", "w") as f:
        f.write(patch)

print(f"Saved {len(dataset)} gold patches")