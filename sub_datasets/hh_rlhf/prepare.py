from datasets import load_dataset
from typing import List, Tuple
from datasets import load_dataset
import os
import json
from huggingface_hub import HfApi
from collections import defaultdict

dataset = load_dataset("Anthropic/hh-rlhf")

def get_pair_from_conv(convAs: List[str], convBs: List[str]):
    """Compare two conversations by takeing USER turns as inputs and ASSISTANT turns as candidates
        Multi-turn conversations comparison is also supportted.
        a conversation format is:
        ```python
        [
            {
                "content": "hello",
                "role": "USER"
            },
            {
                "content": "hi",
                "role": "ASSISTANT"
            },
            ...
        ]
        ```
    Args:
        convAs (List[List[dict]]): List of conversations
        convAs (List[List[dict]]): List of conversations
    """
    for c in convAs + convBs:
        assert len(c) % 2 == 0, "Each conversation must have even number of turns"
        assert all([c[i]['role'] == 'USER' for i in range(0, len(c), 2)]), "Each even turn must be USER"
        assert all([c[i]['role'] == 'ASSISTANT' for i in range(1, len(c), 2)]), "Each odd turn must be ASSISTANT"
    # check conversations correctness
    assert len(convAs) == len(convBs), "Number of conversations must be the same"
    for c_a, c_b in zip(convAs, convBs):
        assert len(c_a) == len(c_b), "Number of turns in each conversation must be the same"
        assert all([c_a[i]['content'] == c_b[i]['content'] for i in range(0, len(c_a), 2)]), "USER turns must be the same"
    
    instructions = ["Finish the following coversation in each i-th turn by filling in <Response i> with your response."] * len(convAs)
    inputs = [
        "\n".join([
            "USER: " + x[i]['content'] +
            f"\nAssistant: <Response {i//2+1}>" for i in range(0, len(x), 2)
        ]) for x in convAs
    ]
    cand1_texts = [
        "\n".join([
            f"<Response {i//2+1}>: " + x[i]['content'] for i in range(1, len(x), 2)
        ]) for x in convAs
    ]
    cand2_texts = [
        "\n".join([
            f"<Response {i//2+1}>: " + x[i]['content'] for i in range(1, len(x), 2)
        ]) for x in convBs
    ]
    return instructions, inputs, cand1_texts, cand2_texts


import regex as re
convAs = []
convBs = []
for item in dataset['train']:
    """Human: ... Assistant: ... Human: ... ..."""
    convs = []
    cur_text = item['chosen'] + ' '
    while True:
        human_start_idx = cur_text.find('Human: ')
        assistant_start_idx = cur_text.find('Assistant: ')
        next_human_start_idx = cur_text.find('Human: ', human_start_idx + 1)
        human_text = cur_text[human_start_idx + 7:assistant_start_idx].strip('\n ')
        assistant_text = cur_text[assistant_start_idx + 11:next_human_start_idx].strip('\n ')
        cur_text = cur_text[next_human_start_idx:]
        
        convs.extend([
            {
                "role": "USER",
                "content": human_text
            },
            {
                "role": "ASSISTANT",
                "content": assistant_text
            }
        ])
        if next_human_start_idx == -1:
            break
    convAs.append(convs)
    cur_text = item['rejected'] + ' '
    while True:
        human_start_idx = cur_text.find('Human: ')
        assistant_start_idx = cur_text.find('Assistant: ')
        next_human_start_idx = cur_text.find('Human: ', human_start_idx + 1)
        human_text = cur_text[human_start_idx + 7:assistant_start_idx].strip('\n ')
        assistant_text = cur_text[assistant_start_idx + 11:next_human_start_idx].strip('\n ')
        cur_text = cur_text[next_human_start_idx:]
        
        convs.extend([
            {
                "role": "USER",
                "content": human_text
            },
            {
                "role": "ASSISTANT",
                "content": assistant_text
            }
        ])
        if next_human_start_idx == -1:
            break
        
    convBs.append(convs)
    
    
instructions, inputs, cand1_texts, cand2_texts = get_pair_from_conv(convAs, convBs)
new_data = []
import random
random.seed(42)
for i in range(len(inputs)):
    new_data.append({
        "id": "hh_rlhf_" + str(i),
        "instruction": instructions[i],
        "input": inputs[i],
        "candidates": [
            {
                "text": cand1_texts[i],
                "model": "unknown",
                "decoding_method": "unknown",
                "scores": {
                    "human_preference": 1.0
                }
            },
            {
                "text": cand2_texts[i],
                "model": "unknown",
                "decoding_method": "unknown",
                "scores": {
                    "human_preference": 0.0
                }
            }
        ]
    })
    random.shuffle(new_data[-1]['candidates'])

val_size = min(1000, len(new_data) * 0.05)
train_data = new_data[:-val_size]
val_data = new_data[-val_size:]

# upload to huggingface hub
api = HfApi()
source = "hh_rlhf"
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