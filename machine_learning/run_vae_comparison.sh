#!/bin/bash

# =================================================================
# SCRIPT CONFIGURATION
# =================================================================
TARGET_SUBJECT="goto_1219_0.8s"
EPOCHS=100
LATENT_SIZE=4 # 潜在空間の次元数
TRAINING_DATA_ORIENTATION="flipped" # 9chデータセットの学習データの向き ("normal" or "flipped")

# 再構成損失のP波、R波、T波部分の重み
P_WEIGHT="1.0" 
R_WEIGHT="0.001"
T_WEIGHT="1.0"
# =================================================================

# Define the experiments in an array
# Each element is a string: "TestName;DatasetName;NumChannels"
declare -a experiments=(
    "15ch_baseline;15ch_arrange_direction;15"
    "9ch_virtual;virtual_electrode_dataset;9"
    "9ch_rotated;rotated_datasets;9"
)

# Common arguments for all runs
COMMON_ARGS="--TARGET_NAME $TARGET_SUBJECT --epochs $EPOCHS --train_batch_size 16 --latent_size $LATENT_SIZE --loss_pt_on_off_P_weight $P_WEIGHT --loss_pt_on_off_R_weight $R_WEIGHT --loss_pt_on_off_T_weight $T_WEIGHT"

# Loop through and run each experiment
for experiment in "${experiments[@]}"; do
    # Split the experiment string into parts
    IFS=';' read -r TestName DatasetName NumChannels <<< "$experiment"

    echo "===================================================================="
    echo "RUNNING EXPERIMENT: $TestName"
    echo "===================================================================="

    # Add orientation argument only for 9-channel datasets
    ORIENTATION_ARG=""
    if [ "$NumChannels" -eq 9 ]; then
        ORIENTATION_ARG="--orientation $TRAINING_DATA_ORIENTATION"
    fi

    python3 refactored/train_vae.py \
        $COMMON_ARGS \
        --Dataset_name "$DatasetName" \
        --num_channels "$NumChannels" \
        $ORIENTATION_ARG \
        --current_time "run_$TestName"
    
    # Check exit code
    if [ $? -ne 0 ]; then
        echo "Error during experiment: $TestName. Aborting."
        exit 1
    fi
done

echo "===================================================================="
echo "All experiments finished successfully."
echo "===================================================================="