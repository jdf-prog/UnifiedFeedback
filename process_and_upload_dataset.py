import os
import json
import datasets
import random
import regex as re
import fire
from datasets import load_dataset
from typing import List, Tuple
from datasets import load_dataset
from huggingface_hub import HfApi
from collections import defaultdict
from itertools import combinations

existed_dataset_sources = [
    "lmsys/chatbot_arena_conversations",
    "Anthropic/hh-rlhf",
    "berkeley-nest/Nectar",
    "openai/summarize_from_feedback(comparisons)",
    "Dahoas/synthetic-instruct-gptj-pairwise",
    "openbmb/UltraFeedback",
    "argilla/ultrafeedback-binarized-preferences-cleaned",
    "openai/webgpt_comparisons",
]

def reformat_data_from_hf_dataset(dataset_source: str, dataset_name: str, seed: int, val_num: int = 1000):
    if dataset_source == "lmsys/chatbot_arena_conversations":
        dataset = load_dataset("lmsys/chatbot_arena_conversations")
        new_data = []
        for i, x in enumerate(dataset['train']):
            
            new_item = {
                "id": f"chatbot_arena_conversations_{x['question_id']}",
                "conv_A": x['conversation_a'],
                "conv_B": x['conversation_b'],
                "conv_A_rating": 1.0 if x['winner'] == 'model_a' else 0.5 if x['winner'] == 'tie' else 0.0,
                "conv_B_rating": 1.0 if x['winner'] == 'model_b' else 0.5 if x['winner'] == 'tie' else 0.0,
                "num_turns": len(x['conversation_a']) // 2,
                "source": "lmsys/chatbot_arena_conversations",
            }
            new_data.append(new_item)
    elif dataset_source == "Anthropic/hh-rlhf":
        dataset = load_dataset("Anthropic/hh-rlhf")
        new_data = []
        for i, item in enumerate(dataset['train']):
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
                        "role": "user",
                        "content": human_text
                    },
                    {
                        "role": "assistant",
                        "content": assistant_text
                    }
                ])
                if next_human_start_idx == -1:
                    break
            convA = convs
            
            convs = []
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
                        "role": "user",
                        "content": human_text
                    },
                    {
                        "role": "assistant",
                        "content": assistant_text
                    }
                ])
                if next_human_start_idx == -1:
                    break
            convB = convs
            
            if random.random() > 0.5:
                convA_rating = 1.0
                convB_rating = 0.0
            else:
                convA, convB = convB, convA
                convA_rating = 0.0
                convB_rating = 1.0
            
            new_item = {
                "id": f"hh-rlhf_{i}",
                "conv_A": convA,
                "conv_B": convB,
                "conv_A_rating": convA_rating,
                "conv_B_rating": convB_rating,
                "num_turns": len(convA) // 2,
                "source": "Anthropic/hh-rlhf",
            }
            try:
                assert len(convA) == len(convB), f"Error in {i}"
                assert new_item['num_turns'] > 0, f"Error in {i}"
                assert all([convA[i]['content'] == convB[i]['content'] for i in range(0, len(convA), 2)]), f"Error in {i}"
                new_data.append(new_item)
            except:
                continue
    elif dataset_source == "berkeley-nest/Nectar":
        dataset = load_dataset("berkeley-nest/Nectar")
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
                    "id": f"{dataset_name}_{i}_{j}",
                    "conv_A": [
                        {
                            "role": "user",
                            "content": (instruction+"\n"+input).strip('\n ')
                        },
                        {
                            "role": "assistant",
                            "content": candidates[j*num_pair_samples_per_instance]['text']
                        }
                    ],
                    "conv_B": [
                        {
                            "role": "user",
                            "content": (instruction+"\n"+input).strip('\n ')
                        },
                        {
                            "role": "assistant",
                            "content": candidates[(j+1)*num_pair_samples_per_instance]['text']
                        }
                    ],
                    "conv_A_rating": candidates[j*num_pair_samples_per_instance]['scores']['human_preference'],
                    "conv_B_rating": candidates[(j+1)*num_pair_samples_per_instance]['scores']['human_preference'],
                    "num_turns": 1,
                    "source": f"{dataset_source}",
                }
                new_data.append(new_item)
                
    elif dataset_source == "openai/summarize_from_feedback(comparisons)":
        dataset = load_dataset("openai/summarize_from_feedback", "comparisons")
        new_data = []
        for i, item in enumerate(dataset["train"]):
            new_data.append(
                {
                    "id": f"{dataset_name}_{item['info']['id']}",
                    "conv_A": [
                        {
                            "role": "user",
                            "content": item['info']['post']
                        },
                        {
                            "role": "assistant",
                            "content": item['summaries'][0]['text']
                        }
                    ],
                    "conv_B": [
                        {
                            "role": "user",
                            "content": item['info']['post']
                        },
                        {
                            "role": "assistant",
                            "content": item['summaries'][1]['text']
                        }
                    ],
                    "conv_A_rating": 1 if item['choice'] == 0 else 0,
                    "conv_B_rating": 1 if item['choice'] == 1 else 0,
                    "num_turns": 1,
                    "source": f"{dataset_source}",
                }
            )
    elif dataset_source == "Dahoas/synthetic-instruct-gptj-pairwise":
        dataset = load_dataset("Dahoas/synthetic-instruct-gptj-pairwise")
        new_data = []
        for i, item in enumerate(dataset['train']):
            conv_A = [
                {
                    "role": "user",
                    "content": item['prompt']
                },
                {
                    "role": "assistant",
                    "content": item['chosen']
                }
            ]
            conv_B = [
                {
                    "role": "user",
                    "content": item['prompt']
                },
                {
                    "role": "assistant",
                    "content": item['rejected']
                }
            ]
            if random.random() < 0.5:
                conv_A, conv_B = conv_B, conv_A
                conv_A_rating = 0
                conv_B_rating = 1
            else:
                conv_A_rating = 1
                conv_B_rating = 0
                
            new_data.append(
                {
                    "id": f"{dataset_name}_{i}",
                    "conv_A": conv_A,
                    "conv_B": conv_B,
                    "conv_A_rating": conv_A_rating,
                    "conv_B_rating": conv_B_rating,
                    "num_turns": 1,
                    "source": f"{dataset_source}",
                }
            )
    elif dataset_source == "openbmb/UltraFeedback":
        dataset = load_dataset("openbmb/UltraFeedback")
        new_data = []
        for i, x in enumerate(dataset['train']):
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
                    "id": f"{dataset_name}_{x['source']}_{i}",
                    "conv_A": [
                        {
                            "role": "user",
                            "content": x['instruction']
                        },
                        {
                            "role": "assistant",
                            "content": text1
                        }
                    ],
                    "conv_B": [
                        {
                            "role": "user",
                            "content": x['instruction']
                        },
                        {
                            "role": "assistant",
                            "content": text2
                        }
                    ],
                    "conv_A_rating": cand1_scores['human_preference'],
                    "conv_B_rating": cand2_scores['human_preference'],
                    "num_turns": 1,
                    "source": x['source'],
                }
                new_data.append(new_item)
    elif dataset_source == "argilla/ultrafeedback-binarized-preferences-cleaned":
        dataset = load_dataset("argilla/ultrafeedback-binarized-preferences-cleaned")
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
            
            if random.random() < 0.5:
                chosen_response, rejected_response = rejected_response, chosen_response
                chosen_rating, rejected_rating = rejected_rating, chosen_rating
            new_data.append(
                {
                    "id": f"{dataset_name}_{i}",
                    "conv_A": [
                        {
                            "role": "user",
                            "content": instruction
                        },
                        {
                            "role": "assistant",
                            "content": chosen_response
                        }
                    ],
                    "conv_B": [
                        {
                            "role": "user",
                            "content": instruction
                        },
                        {
                            "role": "assistant",
                            "content": rejected_response
                        }
                    ],
                    "conv_A_rating": chosen_rating,
                    "conv_B_rating": rejected_rating,
                    "num_turns": 1,
                    "source": f"{dataset_source}",
                }
            )
    elif dataset_source == "openai/webgpt_comparisons":
        dataset = load_dataset("openai/webgpt_comparisons")
        new_data = []
        for i, item in enumerate(dataset["train"]):
            new_data.append(
                {
                    "id": f"{dataset_name}_{i}",
                    "conv_A": [
                        {
                            "role": "user",
                            "content": item['question']['full_text']
                        },
                        {
                            "role": "assistant",
                            "content": item['answer_0']
                        }
                    ],
                    "conv_B": [
                        {
                            "role": "user",
                            "content": item['question']['full_text']
                        },
                        {
                            "role": "assistant",
                            "content": item['answer_1']
                        }
                    ],
                    "conv_A_rating": item['score_0'],
                    "conv_B_rating": item['score_1'],
                    "num_turns": 1,
                    "source": f"{dataset_source}",
                }
            )
    else:
        raise NotImplementedError(f"Unknown dataset source: {dataset_source}, please add the processing code in process_and_upload_dataset.py")

    for item in new_data:
        assert len(item['conv_A']) == len(item['conv_B']), f"Cov_A and conv_B should have the same number of turns in {item['id']}"
        assert item['num_turns'] > 0, f"num_turns should be greater than 0 in {item['id']}"
        assert all([item['conv_A'][i]['content'] == item['conv_B'][i]['content'] for i in range(0, len(item['conv_A']), 2)]), f"The user content in conv_A and conv_B should be the same for each turn in {item['id']}"
    random.seed(seed)
    random_idx = random.sample(range(len(new_data)), val_num)
    train_data = [new_data[i] for i in range(len(new_data)) if i not in random_idx]
    val_data = [new_data[i] for i in range(len(new_data)) if i in random_idx]
    
    return train_data, val_data


def main(
    dataset_source: str,
    dataset_name: str = None,
    to_upload_repo_id: str = "llm-blender/Unified-Feedback",
    seed: int = 42,
    val_num = 1000,
):
    random.seed(seed)
    if dataset_source == 'all':
        train_data = []
        val_data = []
        print("Processing all datasets")
        for dataset_source in existed_dataset_sources:
            print(f"Processing {dataset_source}")
            _train_data, _val_data = reformat_data_from_hf_dataset(dataset_source, dataset_name, seed, val_num)
            train_data.extend(_train_data)
            val_data.extend(_val_data)
    else:
        train_data, val_data = reformat_data_from_hf_dataset(dataset_source, dataset_name, seed)


    features = datasets.Features({
        "id": datasets.Value("string"),
        "conv_A": [
            {
                "role": datasets.Value("string"),
                "content": datasets.Value("string"),
            }
        ],
        "conv_B": [
            {
                "role": datasets.Value("string"),
                "content": datasets.Value("string"),
            }
        ],
        "conv_A_rating": datasets.Value("float32"),
        "conv_B_rating": datasets.Value("float32"),
        "num_turns": datasets.Value("int32"),
        "source": datasets.Value("string"),
    })

    train_hf_dataset = datasets.Dataset.from_list(
        train_data,
        features=features,
    )

    val_hf_dataset = datasets.Dataset.from_list(
        val_data,
        features=features,
    )
        
    train_hf_dataset.push_to_hub(
        repo_id=to_upload_repo_id,
        config_name=dataset_name,
        split="train",
        token=os.environ.get("HUGGINGFACE_TOKEN"),
        commit_message=f"Add {dataset_name} train dataset",
    )

    val_hf_dataset.push_to_hub(
        repo_id=to_upload_repo_id,
        config_name=dataset_name,
        split="val",
        token=os.environ.get("HUGGINGFACE_TOKEN"),
        commit_message=f"Add {dataset_name} val dataset",
    )
        
if __name__ == "__main__":
    fire.Fire(main)