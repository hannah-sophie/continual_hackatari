#!/bin/bash
RUN_CONFIG_PATH="run_config.json"
ENV_ID="Freeway"
SEEDS=(0 1 2)
AVAILABLE_GPUS=(1 2 3)
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
    
    echo "Completed evaluation for seed: $seed on GPU $gpu_id"
}

export -f run_training
export RUN_CONFIG_PATH AUTHOR ENV_ID
num_seeds=${#SEEDS[@]}
num_gpus=${#AVAILABLE_GPUS[@]}

pids=()

for i in "${!SEEDS[@]}"; do
    seed=${SEEDS[$i]}
    gpu_idx=$((i % num_gpus))
    gpu_id=${AVAILABLE_GPUS[$gpu_idx]}
    
    if [ ${#pids[@]} -ge $num_gpus ]; then
        wait ${pids[0]}
        pids=("${pids[@]:1}")  # Remove first element
    fi
    
    run_training "$seed" "$gpu_id" &
    pids+=($!)
    
    echo "Launched seed $seed on GPU $gpu_id (PID: $!)"
done

for pid in "${pids[@]}"; do
    wait $pid
done

echo "All evaluations completed!"