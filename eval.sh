#!/bin/bash

ENV_ID="Freeway"
AGENT_PATH="PATH_TO_AGENT"
CUDA_DEVICE=2
ARCHITECTURE="PPO"

MODIFICATIONS=(
    "stop_random_car"
    "stop_all_cars_edge"
    "stop_all_cars"
    "disable_cars"
    "all_black_cars"
)


run_evaluation() {
    local modif=$1
    local gpu_id=$2
    
    echo "Starting evaluation for modification: $modif on GPU $gpu_id"
    
    CUDA_VISIBLE_DEVICES=$gpu_id python eval.py \
        --env_id "$ENV_ID" \
        --modifs "$modif" \
        --agent_path "$AGENT_PATH" \
        --track \
        --capture_video \
        --architecture "$ARCHITECTURE"
    
    echo "Completed evaluation for modification: $modif"
}

export -f run_evaluation
export ENV_ID AGENT_PATH ARCHITECTURE

if command -v parallel &> /dev/null; then
    echo "Using GNU parallel for concurrent execution..."
    
    printf '%s\n' "${MODIFICATIONS[@]}" | parallel -j $(nproc) run_evaluation {} $CUDA_DEVICE
    
else
    echo "GNU parallel not found. Running evaluations sequentially in background..."
    
    # Alternative: Run in background processes
    for modif in "${MODIFICATIONS[@]}"; do
        run_evaluation "$modif" "$CUDA_DEVICE" &
    done
    
    wait
fi

echo "All evaluations completed!"