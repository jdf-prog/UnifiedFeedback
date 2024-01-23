import os
import json
import random
from huggingface_hub import HfApi
from datasets import load_dataset

dataset = load_dataset("argilla/ultrafeedback-binarized-preferences-cleaned")

import json
new_data = []
for i, item in enumerate(dataset['train']):
    assert item['chosen'][0]['content'] == item['rejected'][0]['content'] == item['prompt']
    chosen_response = item['chosen'][1]['content']
    chosen_model = item['chosen-model']
    chosen_rating = item['chosen-rating']
    rejected_response = item['rejected'][1]['content']
    rejected_model = item['rejected-model']
    rejected_rating = item['rejected-rating']
    instruction = item['prompt']
    assert chosen_rating > rejected_rating, "Chosen response should be better than rejected response"
    new_data.append({
        "id": "ultrafeedback_binarized_preferences_cleaned_" + str(i),
        "instruction": instruction,
        "input": "",
        "candidates": [
            {
                "text": chosen_response,
                "model": chosen_model,
                "decoding_method": "unknown",
                "scores": {
                    "human_preference": chosen_rating,
                }
            },
            {
                "text": rejected_response,
                "model": rejected_model,
                "decoding_method": "unknown",
                "scores": {
                    "human_preference": rejected_rating,
                }
                
            }
        ]
    })

random.seed(42)
random_idx = random.sample(range(len(new_data)), 1000)
train_data = [new_data[i] for i in range(len(new_data)) if i not in random_idx]
val_data = [new_data[i] for i in range(len(new_data)) if i in random_idx]


# upload to huggingface hub
api = HfApi()
source = "ultrafeedback_binarized_preferences_cleaned"
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