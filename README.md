## Unified-Feedback
Codes to process existing feedback dataset to pairwise training format.

See `sub_datasets/{dataset_name}/prepare.ipynb` for details of processing.

to recreate the dataset, please check the `prepare.py` under each dataset folder in the `sub_datasets`. Running this file formatting the data and uploads to 🤗[Unified_feedback](https://huggingface.co/datasets/llm-blender/Unified-Feedback)

To interacte with the dataset during the processing, please check the `prepare.ipynb` under each dataset folder in the `sub_datasets`

Finally, after processing each subdataset, check `prepare_unified.py` aggregates the data of each sub datasets and upload to 🤗[Unified_feedback](https://huggingface.co/datasets/llm-blender/Unified-Feedback).

