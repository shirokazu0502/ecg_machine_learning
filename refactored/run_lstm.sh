#!/bin/bash
# List of unique TARGET_NAME values
TARGET_NAMES=("asano" "gosha" "goto" "ikejima" "kanda" "kawai" "matumoto" "nakashimizu" "noda" "taniguchi" "yoshikura")

for name in "${TARGET_NAMES[@]}"; do
    echo "Running for TARGET_NAME: $name"

    # Training
    python3 train_lstm.py \
        --TARGET_NAME "$name" \
        --Dataset_name rotated_datasets \
        --mode train \
        --epochs 600 \
        --train_batch_size 16 \
        --num_channels 9 \
        --ave_data_flg 1

    # Testing
    python3 train_lstm.py \
        --TARGET_NAME "$name" \
        --Dataset_name rotated_datasets \
        --mode test \
        --num_channels 9 \
        --ave_data_flg 1

done