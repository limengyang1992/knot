# Phase 6, Step 6.2: Upload to HuggingFace Hub

from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi
import jsonlines, os

HF_REPO = "limengyang1992/KNOT"  # Update with actual HF username

all_configs = {}
for task in ["entity", "concept", "skill"]:
    splits = {}
    for split in ["forget", "retain", "boundary"]:
        path = f"knot_data/knot_{task}/{split}_final.jsonl"
        if not os.path.exists(path):
            print(f"Warning: {path} not found, skipping")
            continue
        with jsonlines.open(path) as r:
            items = list(r)
        # Normalize: ensure all items have string values for HF compatibility
        normalized = []
        for item in items:
            norm = {}
            for k, v in item.items():
                if isinstance(v, (list, dict)):
                    import json
                    norm[k] = json.dumps(v)
                elif v is None:
                    norm[k] = ""
                else:
                    norm[k] = v
            normalized.append(norm)
        splits[split] = Dataset.from_list(normalized)
    if splits:
        all_configs[f"knot_{task}"] = DatasetDict(splits)

print(f"Uploading {len(all_configs)} configs to {HF_REPO}")
for config_name, ddict in all_configs.items():
    ddict.push_to_hub(HF_REPO, config_name=config_name)
    print(f"Uploaded {config_name}")

print("Upload complete.")
