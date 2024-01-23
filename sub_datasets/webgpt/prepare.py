import os
import json
from huggingface_hub import HfApi
from datasets import load_dataset


dataset = load_dataset("openai/webgpt_comparisons")

new_data = []
for i, item in enumerate(dataset["train"]):
    new_data.append({
        "id": "webgpt" + str(i),
        "instruction": "",
        "input": item['question']['full_text'],
        "candidates": [
            {
                "text": item['answer_0'],
                "model": "unknown",
                "decoding_method": "unknown",
                "scores": {
                    "human_preference": item['score_0']
                }
            },
            {
                "text": item['answer_1'],
                "model": "unknown",
                "decoding_method": "unknown",
                "scores": {
                    "human_preference": item['score_1']
                }
            }
        ]
    })
    
train_data = new_data[:-1000]
val_data = new_data[-1000:]
# upload to huggingface hub
api = HfApi()
source = "webgpt"
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