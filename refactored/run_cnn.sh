#!/bin/bash

# Define the list of target names
TARGET_NAMES=(
    "patient4"
    "asano"
    "gosha"
    "goto"
    "ikejima"
    "nishio"
    "takahashi_jr"
    "kanda"
    "kawai"
    "matumoto"
    "nakashimizu"
    "noda"
    "taniguchi"
    "yoshikura"
    "patient6"
    "patient8"
    "patient9"
)

# Loop through each target name and run the training and testing scripts
for target_name in "${TARGET_NAMES[@]}"; do
    echo "Running SimpleCNN training for TARGET_NAME: ${target_name}"

    # Training
    python3 train_cnn.py \
        --mode train \
        --Dataset_name 15ch_arrange_direction \
        --TARGET_NAME "${target_name}" \
        --epochs 1000 \
        --train_batch_size 16 \
        --num_channels 15 \
        --ecg_ch_num 8 \
        --cnn_depth 4 \
        --cnn_init_filters 16

    echo "Running SimpleCNN testing for TARGET_NAME: ${target_name}"

    # Testing
    python3 train_cnn.py \
        --mode test \
        --Dataset_name 15ch_arrange_direction \
        --TARGET_NAME "${target_name}" \
        --num_channels 15 \
        --ecg_ch_num 8 \
        --cnn_depth 4 \
        --cnn_init_filters 16

done

echo "All SimpleCNN training and testing runs completed."
