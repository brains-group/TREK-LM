#!/bin/bash

# Create logs directory
mkdir -p logs

# Define models
# Qwen3 models: 0.6B, 1.5B, 3B, 8B
# Gemma models: 2B, 9B (Assuming Gemma 2 "it" models for 1B-12B range)
MODELS=(
    "Qwen/Qwen3-0.6B"
    "Qwen/Qwen3-1.5B"
    "Qwen/Qwen3-3B"
    "Qwen/Qwen3-8B"
    "google/gemma-2-2b-it"
    "google/gemma-2-9b-it"
)

# Define shots order: 5, 0, 1, 3, 10
SHOTS=(5 0 1 3 10)

# Function to get folder name from model name
# Assumes folder name is the part after the slash (e.g., Qwen3-0.6B)
get_folder_name() {
    echo "${1#*/}"
}

# Function to run the ablation pipeline
run_ablation() {
    local model=$1
    local shot=$2
    local gpu_id=$3
    local folder_name=$(get_folder_name "$model")
    
    # Find the input JSON file
    # Pattern: *augmented*${shot}shot.json inside data/extractedTriples/${folder_name}
    local json_file=$(find ./data/extractedTriples/"$folder_name" -name "*augmented*${shot}shot.json" | head -n 1)
    
    if [ -z "$json_file" ]; then
        echo "Error: JSON file not found for Model: $model, Shot: $shot"
        return 1
    fi
    
    echo "Running ablation for Model: $model, Shot: $shot on GPU: $gpu_id"
    echo "Input File: $json_file"
    
    # 1. Run createMovieKGData.py
    # This generates the dataset in the current directory
    CUDA_VISIBLE_DEVICES=$gpu_id python data/createMovieKGData.py --extracted_triples "$json_file" > "logs/create_data_${folder_name}_${shot}shot.log" 2>&1
    
    # Determine generated dataset name
    # Regex in python script: augmented(.*)\.json
    local filename=$(basename "$json_file")
    local suffix=$(echo "$filename" | sed -n 's/.*augmented\(.*\)\.json/\1/p')
    local dataset_name="movieKnowledgeGraphDataset${suffix}"
    
    # Move generated datasets to data/ folder
    mv "${dataset_name}.json" "./data/" 2>/dev/null
    mv "nonFederated${dataset_name}.json" "./data/" 2>/dev/null
    mv "${dataset_name}WithSyntheticData.json" "./data/" 2>/dev/null
    mv "nonFederated${dataset_name}WithSyntheticData.json" "./data/" 2>/dev/null
    
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

# Main loop
for shot in "${SHOTS[@]}"; do
    echo "Starting Shot: $shot"
    
    # Run in batches of 2 (for 2 GPUs)
    for ((i=0; i<${#MODELS[@]}; i+=2)); do
        model1="${MODELS[i]}"
        model2="${MODELS[i+1]}"
        
        # Launch job 1 on GPU 0
        if [ -n "$model1" ]; then
            run_ablation "$model1" "$shot" 0 &
        fi
        
        # Launch job 2 on GPU 1
        if [ -n "$model2" ]; then
            run_ablation "$model2" "$shot" 1 &
        fi
        
        # Wait for this batch to finish
        wait
    done
done
