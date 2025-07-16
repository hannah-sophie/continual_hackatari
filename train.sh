#!/bin/bash
CUDA_DEVICE=2
RUN_CONFIG_PATH="run_config.json"
ENV_ID="Freeway"
SEEDS=(0 1 2)
AUTHOR="LR"

run_training() {
    local seed=$1
    local gpu_id=$2
    
    echo "Starting evaluation for modification: $modif on GPU $gpu_id"
    
    CUDA_VISIBLE_DEVICES=$gpu_id python continual_training_example.py \
        --env_id "$ENV_ID" \
        --seed "$seed" \
        --config "$RUN_CONFIG_PATH" \
        --track \
        --capture_video \
        --author "$AUTHOR" \
    
    echo "Completed evaluation for seed: $seed"
}

export -f run_training
export RUN_CONFIG_PATH AUTHOR ENV_ID

# Check if GNU parallel is available
if command -v parallel &> /dev/null; then
    echo "Using GNU parallel for concurrent execution..."
    
    printf '%s\n' "${SEEDS[@]}" | parallel -j $(nproc) run_training {} $CUDA_DEVICE
    
else
    echo "GNU parallel not found."
    for seed in "${SEEDS[@]}"; do
        run_training "$seed" "$CUDA_DEVICE" &
    done
    
    wait
fi

echo "All evaluations completed!"