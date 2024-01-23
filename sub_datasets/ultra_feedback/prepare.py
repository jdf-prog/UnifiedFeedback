from datasets import load_dataset
import os
import json
from huggingface_hub import HfApi
dataset = load_dataset("openbmb/UltraFeedback")
data = [x for x in dataset['train']]

import json
from itertools import combinations
import random
random.seed(42)
new_data = []
for i, x in enumerate(data):
    # randomly sample 2 pairs
    all_combs = list(combinations(x['completions'], 2))
    random.shuffle(all_combs)
    
    for completion1, completion2 in all_combs[:2]:
        text1 = completion1['response']
        text2 = completion2['response']
        model1 = completion1['model']
        model2 = completion2['model']
        cand1_scores = {
            "human_preference": completion1['overall_score'],
        }
        cand1_scores.update({
            aspect: annot['Rating'] for aspect, annot in completion1['annotations'].items()
        })
        cand2_scores = {
            "human_preference": completion2['overall_score'],
        }
        cand2_scores.update({
            aspect: annot['Rating'] for aspect, annot in completion2['annotations'].items()
        })
        new_item = {
            "id": "ultrafeedback_{}_{}".format(x['source'], i),
            "instruction": x['instruction'],
            "input": "",
            "candidates": [
                {
                    "text": text1,
                    "model": model1,
                    "decoding_method": "unknown",
                    "scores": cand1_scores
                },
                {
                    "text": text2,
                    "model": model2,
                    "decoding_method": "unknown",
                    "scores": cand2_scores
                }
            ]
        }
        new_data.append(new_item)

print(len(new_data))
train_data = new_data[:-5000]
val_data = new_data[-5000:]

# upload to huggingface hub
api = HfApi()
source = "synthetic-instruct-gptj-pairwise"
for set_name, data in [("train", train_data), ("val", val_data)]:
    release_data = []
    for item in data:
        candidates = item["candidates"]
        cand1_text = candidates[0]["text"]
        cand2_text = candidates[1]["text"]
        cand1_rating = candidates[0]["scores"]["human_preference"]
        cand2_rating = candidates[1]["scores"]["human_preference"]
        if "model" not in candidates[0]:
            print(item)
            break
        cand1_model = candidates[0]["model"]
        cand2_model = candidates[1]["model"]
        if cand1_rating > cand2_rating:
            chosen_text = cand1_text
            chosen_model = cand1_model
            chosen_rating = cand1_rating
            rejected_text = cand2_text
            rejected_model = cand2_model
            rejected_rating = cand2_rating
        else:
            chosen_text = cand2_text
            chosen_model = cand2_model
            chosen_rating = cand2_rating
            rejected_text = cand1_text
            rejected_model = cand1_model
            rejected_rating = cand1_rating
        
        release_item = {
            "id": item["id"],
            "prompt": item["instruction"] + "\n" + item["input"],
            "chosen_text": chosen_text,
            "chosen_model": chosen_model,
            "chosen_rating": chosen_rating,
            "rejected_text": rejected_text,
            "rejected_model": rejected_model,
            "rejected_rating": rejected_rating,
            "source": source,
        }
        release_data.append(release_item)
    with open(f"{set_name}_release_data.json", 'w') as f:
        json.dump(release_data, f, indent=2)
    api.upload_file(
        path_or_fileobj=f"{set_name}_release_data.json",
        path_in_repo=f"datasets/{source}/{set_name}_release_data.json",
        repo_id="llm-blender/Unified-Feedback",
        repo_type="dataset",
        token=os.environ.get("HUGGINGFACE_TOKEN")
    )