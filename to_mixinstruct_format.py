import datasets
import fire
import json
import os
from typing import List, Dict, Any

def get_pair_from_conv_for_single_turn(convAs: List[List[dict]], convBs: List[List[dict]]):
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
        assert all([c[i]['role'].upper() == 'USER' for i in range(0, len(c), 2)]), "Each even turn must be USER"
        assert all([c[i]['role'].upper() == 'ASSISTANT' for i in range(1, len(c), 2)]), "Each odd turn must be ASSISTANT"
    # check conversations correctness
    assert len(convAs) == len(convBs), "Number of conversations must be the same"
    for c_a, c_b in zip(convAs, convBs):
        assert len(c_a) == len(c_b), "Number of turns in each conversation must be the same"
        assert all([c_a[i]['content'] == c_b[i]['content'] for i in range(0, len(c_a), 2)]), "USER turns must be the same"
    
    inputs = [
        convAs[i][0]['content'] for i in range(len(convAs))
    ]
    cand1_texts = [
        convAs[i][1]['content'] for i in range(len(convAs))
    ]
    cand2_texts = [
        convBs[i][1]['content'] for i in range(len(convBs))
    ]
    return inputs, cand1_texts, cand2_texts


def get_pair_from_conv(convAs: List[List[dict]], convBs: List[List[dict]]):
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
        assert all([c[i]['role'].upper() == 'USER' for i in range(0, len(c), 2)]), "Each even turn must be USER"
        assert all([c[i]['role'].upper() == 'ASSISTANT' for i in range(1, len(c), 2)]), "Each odd turn must be ASSISTANT"
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
    inputs = [inst + inp for inst, inp in zip(instructions, inputs)]
    return inputs, cand1_texts, cand2_texts



def map_to_mixinstruct_format(item):
    if item['num_turns'] > 1:
        inputs, cand1_texts, cand2_texts = get_pair_from_conv([item['conv_A']], [item['conv_B']])
    else:
        inputs, cand1_texts, cand2_texts = get_pair_from_conv_for_single_turn([item['conv_A']], [item['conv_B']])

    input_text, cand1_text, cand2_text = inputs[0], cand1_texts[0], cand2_texts[0]
    return {
        "id": item['id'],
        "instruction": "",
        "input": input_text,
        "candidates": [
            {
                "text": cand1_text,
                "model": "unknown",
                "decoding_method": "unknown",
                "scores": {
                    "human_preference": item['conv_A_rating']
                }
            },
            {
                "text": cand2_text,
                "model": "unknown",
                "decoding_method": "unknown",
                "scores": {
                    "human_preference": item['conv_B_rating']
                }
            }
        ]
    }

def check_item(item):
    try:
        if item['num_turns'] > 1:
            inputs, cand1_texts, cand2_texts = get_pair_from_conv([item['conv_A']], [item['conv_B']])
        else:
            inputs, cand1_texts, cand2_texts = get_pair_from_conv_for_single_turn([item['conv_A']], [item['conv_B']])
        return True
    except:
        return False

def main(
    unified_feedback_dataset_name: str = "all",
    split='train',
    to_save_data_dir: str = "data/unified_feedback"
):
    
    output_file = f"{to_save_data_dir}/{unified_feedback_dataset_name}_{split}.json"
    if not os.path.exists(to_save_data_dir):
        os.makedirs(to_save_data_dir, exist_ok=True)
    dataset = datasets.load_dataset("llm-blender/Unified-Feedback", unified_feedback_dataset_name, split=split)
    filtered_dataset = dataset.filter(check_item)
    print(f"Filtered {len(dataset) - len(filtered_dataset)} from {len(dataset)}")
    dataset = filtered_dataset.map(map_to_mixinstruct_format)
    with open(output_file, "w") as f:
        json.dump([x for x in dataset], f, indent=2, ensure_ascii=False)
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    fire.Fire(main)


"""
python to_mixinstruct_format.py --unified_feedback_dataset_name "all" --split "train" --to_save_data_dir "data/unified_feedback"
python to_mixinstruct_format.py --unified_feedback_dataset_name "all" --split "val" --to_save_data_dir "data/unified_feedback"

"""
    