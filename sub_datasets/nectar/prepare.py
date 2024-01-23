from datasets import load_dataset
import os
import json
import json
import regex as re
import random
from huggingface_hub import HfApi
from collections import defaultdict

dataset = load_dataset("berkeley-nest/Nectar")


from llm_blender.blender.blender_utils import get_pair_from_conv

random.seed(42)
new_data = []
num_pair_samples_per_instance = 2
for i, item in enumerate(dataset['train']):
    human_instructions = re.findall(r'(?<=Human:)(?:.|\n)*?(?=Assistant:)', item['prompt'])
    assistant_responses = re.findall(r'(?<=Assistant:)(?:.|\n)*?(?=Human:)', item['prompt'])
    human_instructions = [x.strip(' \n') for x in human_instructions]
    assistant_responses = [x.strip(' \n') for x in assistant_responses]
    assert len(human_instructions) == len(assistant_responses) + 1, f"Error in {i}"
    assert len(human_instructions) > 0, f"Error in {i}"
    
    if len(human_instructions) == 1:
        instruction = human_instructions[0]
        input = ""
    elif len(human_instructions) > 1:
        instruction = "Respond to the latest instruction based on the conversation history."
        input = item['prompt'].strip(' \n')
    candidate_texts = [x['answer'] for x in item['answers']]
    candiate_scores = [-x['rank'] for x in item['answers']]
    candidate_models = [x['model'] for x in item['answers']]
    candidates = [
        {
            'text': candidate_texts[j],
            'model': candidate_models[j],
            'decoding_method': 'unknown',
            'scores': {
                'human_preference': candiate_scores[j]
            }
        }
        for j in range(len(candidate_texts))
    ]
    random.shuffle(candidates)
    
    assert num_pair_samples_per_instance * 2 <= len(candidate_texts), f"Error in {i}"
    for j in range(0, num_pair_samples_per_instance * 2, 2):
        new_item = {
            'id': "berkeley-nest-nectar-" + str(i) + "-pair_" + str(j),
            'instruction': instruction,
            'input': input,
            'candidates': candidates[j*num_pair_samples_per_instance:(j+1)*num_pair_samples_per_instance]
        }
        new_data.append(new_item)
    
random.seed(42)
random_idx = random.sample(range(len(new_data)), 1000)
train_data = [new_data[i] for i in range(len(new_data)) if i not in random_idx]
val_data = [new_data[i] for i in range(len(new_data)) if i in random_idx]

    
# upload to huggingface hub
api = HfApi()
source = "nectar"
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
