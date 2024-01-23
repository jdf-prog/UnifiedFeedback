SUB_DATASETS_DIR="./sub_datasets"

export HUGGINGFACE_TOKEN="..."

# Iterate over each sub-directory in the sub_datasets directory
for subdir in "$SUB_DATASETS_DIR"/*; do
    # Check if the directory contains python.py
    if [[ -f "$subdir/prepare.py" ]]; then
        echo "Executing prepare.py in $subdir"
        # Change directory to the sub-directory
        cd "$subdir"
        # Execute python.py
        python python.py
        # Change back to the original directory
        cd - > /dev/null
    else
        echo "python.py not found in $subdir"
    fi
done

echo "Executing prepare_unified.py"
python ./prepare_unified.py