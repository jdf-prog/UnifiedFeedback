import json
import random
import os
random.seed(42)
from pathlib import Path
from huggingface_hub import HfApi
from itertools import combinations
mix_datasets = ['hh_rlhf', "summary_from_feedback", "webgpt", "synthetic-instruct-gptj-pairwise", "chatbot_arena_conv", "ultra_feedback_clean", "nectar"]
data_dir = Path("./sub_datasets")
unified_data = {}
for set_name in ['train', 'val']:
    total_data = []
    print(f"Loading {set_name}")
    for dataset in mix_datasets:
        file_name = data_dir / dataset / f"{set_name}_release_data.json"
        if not file_name.exists():
            continue
        with open(file_name, "r") as f:
            data = json.load(f)
            for item in data:
                item['source'] = dataset
            print(f"Loaded #{len(data)} from {dataset}")
            total_data += data
    unified_data[set_name] = total_data
    
train_data = unified_data['train']
val_data = unified_data['val']

# upload 
api = HfApi()
for set_name, data in [("train", train_data), ("val", val_data)]:
    if set_name != "val":
        continue
    with open(f"{set_name}_release_data.json", 'w') as f:
        json.dump(data, f, indent=2)
    api.upload_file(
        path_or_fileobj=f"{set_name}_release_data.json",
        path_in_repo=f"datasets/unified/{set_name}_release_data.json",
        repo_id="llm-blender/Unified-Feedback",
        repo_type="dataset",
        token=os.environ.get("HUGGINGFACE_TOKEN")
    )