#!/bin/bash
# List of unique TARGET_NAME values
TARGET_NAMES=("patient4" "asano" "gosha" "goto" "ikejima" "nishio" "takahashi_jr" "kanda" "kawai" "matumoto" "nakashimizu" "noda" "taniguchi" "yoshikura"  "patient6" "patient8" "patient9")

for name in "${TARGET_NAMES[@]}"; do
    echo "Running for TARGET_NAME: $name"

    # Training
    python3 train_lstm.py \
        --TARGET_NAME "$name" \
        --Dataset_name rotated_datasets \
        --mode train \
        --epochs 1000 \
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