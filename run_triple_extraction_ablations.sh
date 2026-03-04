#!/bin/bash

# Create logs directory
mkdir -p logs

# Define models
# Qwen3 models: 0.6B, 1.7B, 4B, 8B
# Gemma-3 models: 1B, 4B, 12B
MODELS=(
    "Qwen/Qwen3-0.6B"
    "Qwen/Qwen3-1.7B"
    "Qwen/Qwen3-4B"
    "Qwen/Qwen3-8B"
    "google/gemma-3-1b-it"
    "google/gemma-3-4b-it"
    "google/gemma-3-12b-it"
)

# Define shots order: 5, 0, 1, 3, 10
SHOTS=(5 0 1 3 10)

# Function to get folder name from model name
get_folder_name() {
    local model=$1
    if [[ "$model" == *"Qwen"* ]]; then
        echo "${model#*/}"
    elif [[ "$model" == *"gemma"* ]]; then
        # Extract size (e.g., 1b, 4b, 12b)
        local size=$(echo "$model" | sed -n 's/.*-\([0-9][0-9]*b\)-.*/\1/p')
        echo "Gemma-$(echo $size | tr '[:lower:]' '[:upper:]')"
    fi
}

# Function to run the ablation pipeline
run_ablation() {
    local model=$1
    local shot=$2
    local gpu_id=$3
    local folder_name=$(get_folder_name "$model")
    
    # Find the input JSON file
    local search_path="./data/extractedTriples/${folder_name}"
    local json_file=$(find "$search_path" -name "*augmented*${shot}shots.json" | head -n 1)
    
    if [ -z "$json_file" ]; then
        echo "Error: JSON file not found for Model: $model, Shot: $shot"
        return 1
    fi
    
    echo "Running ablation for Model: $model, Shot: $shot on GPU: $gpu_id"
    echo "Input File: $json_file"
    
    # 1. Run createMovieKGData.py from data folder
    # Pass relative path from data/ folder (../data/extractedTriples/...)
    cd data
    CUDA_VISIBLE_DEVICES=$gpu_id python createMovieKGData.py --extracted_triples "../$json_file" > "../logs/create_data_${folder_name}_${shot}shot.log" 2>&1
    cd ..
    
    # Determine generated dataset name
    # Regex in python script: augmented(.*)\.json
    local filename=$(basename "$json_file")
    local suffix=$(echo "$filename" | sed -n 's/.*augmented\(.*\)\.json/\1/p')
    local dataset_name="movieKnowledgeGraphDataset${suffix}"
    
    # 2. Run train_federated.py
    # Pass base_config.yaml, model.name, dataset.name
    local train_log="logs/train_${folder_name}_${shot}shot.log"
    CUDA_VISIBLE_DEVICES=$gpu_id python train_federated.py \
        --cfg base_config.yaml \
        model.name="$model" \
        dataset.name="$dataset_name" \
        > "$train_log" 2>&1
        
    # Extract the save path from the log
    local save_path=$(grep "Final model available at: " "$train_log" | awk '{print $NF}')
    
    if [ -z "$save_path" ]; then
        echo "Training failed for $model $shot shot. See $train_log"
        return 1
    fi
    
    local lora_path="${save_path}/peft-128"
    
    # 3. Run test.py
    CUDA_VISIBLE_DEVICES=$gpu_id python test.py \
        --base_model_path "$model" \
        --lora_path "$lora_path" \
        > "logs/test_${folder_name}_${shot}shot.log" 2>&1
        
    echo "Completed ablation for Model: $model, Shot: $shot"
}

# Initialize semaphore for 2 GPUs
PIPE_FILE=$(mktemp -u)
mkfifo "$PIPE_FILE"
exec 3<>"$PIPE_FILE"
rm "$PIPE_FILE"

echo "0" >&3
echo "1" >&3

# Main loop
for shot in "${SHOTS[@]}"; do
    echo "Starting Shot: $shot"
    
    for model in "${MODELS[@]}"; do
        read -u 3 gpu_id
        
        (
            run_ablation "$model" "$shot" "$gpu_id"
            echo "$gpu_id" >&3
        ) &
    done
    wait
done
exec 3>&-
