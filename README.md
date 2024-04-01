## Unified-Feedback
Codes to process existing feedback dataset to pairwise training format.

See `process_and_upload_dataset.py` for details of processing.

to recreate the dataset, please check the `process_and_upload_datasets.sh`. After running this script, data is uploaded to 🤗[Unified_feedback](https://huggingface.co/datasets/llm-blender/Unified-Feedback).


## To mixinstruct format
```bash
python to_mixinstruct_format.py --unified_feedback_dataset_name "all" --split "train" --to_save_data_dir "data/unified_feedback"
python to_mixinstruct_format.py --unified_feedback_dataset_name "all" --split "val" --to_save_data_dir "data/unified_feedback"
```