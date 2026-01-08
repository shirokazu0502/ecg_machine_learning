#!/bin/bash

# Define the list of target names for GNN training and testing
TARGET_NAMES=(
    "arai"
    "asano"
    "ikejima"
    "gosha"
    "goto"
    "kanda"
    "kasahara"
    "kawai"
    "kinoshita"
    "matumoto"
    "nakanishi"
    "nakashimizu"
    "nishio"
    "noda"
    "patient1"
    "patient2"
    "patient3"
    "patient4"
    "patient5"
    "patient6"
    "patient7"
    "patient8"
    "patient9"
    "patient10"
    "takahashi_jr"
    "taniguchi"
    "tosue"
    "yoshikura"
)

# Loop through each target name and run the training and testing scripts for the GNN model
for target_name in "${TARGET_NAMES[@]}"; do
    echo "Running GNN training for TARGET_NAME: ${target_name}"

    # Training
    python3 train_gnn.py \
        --mode train \
        --Dataset_name reference_center_4ch \
        --TARGET_NAME "${target_name}" \
        --epochs 200 \
        --train_batch_size 64 \
        --learning_rate 0.001 \
        --num_channels 16 \
        --ecg_ch_num 8 \
        --rnn_hidden_size 64 \
        --gnn_hidden_size 64 \

    echo "Running GNN testing for TARGET_NAME: ${target_name}"

    # Testing
    python3 train_gnn.py \
        --mode test \
        --Dataset_name reference_center_4ch \
        --TARGET_NAME "${target_name}" \
        --num_channels 16 \
        --ecg_ch_num 8 \
        --rnn_hidden_size 64 \
        --gnn_hidden_size 64 \

done

echo "All GNN training and testing runs completed."