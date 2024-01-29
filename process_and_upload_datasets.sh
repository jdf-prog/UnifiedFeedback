# The following are bash commands
dataset_sources=(
    "lmsys/chatbot_arena_conversations"
    "Anthropic/hh-rlhf"
    "berkeley-nest/Nectar"
    "openai/summarize_from_feedback(comparisons)"
    "Dahoas/synthetic-instruct-gptj-pairwise"
    "openbmb/UltraFeedback"
    "argilla/ultrafeedback-binarized-preferences-cleaned"
    "openai/webgpt_comparisons"
)
dataset_names=(
    "chatbot_arena_conversations"
    "hh-rlhf"
    "Nectar"
    "summarize_from_feedback"
    "synthetic-instruct-gptj-pairwise"
    "UltraFeedback"
    "ultrafeedback-binarized-preferences-cleaned"
    "webgpt_comparisons"
)
to_upload_repo_id="llm-blender/Unified-Feedback"
for i in {0..7}; do
    echo "Processing ${dataset_sources[$i]}"
    python process_and_upload_dataset.py \
        --dataset_source "${dataset_sources[$i]}" \
        --dataset_name "${dataset_names[$i]}" \
        --to_upload_repo_id "$to_upload_repo_id"
done
echo Done

python process_and_upload_dataset.py \
    --dataset_source "all" \
    --dataset_name "all" \
    --to_upload_repo_id "$to_upload_repo_id"