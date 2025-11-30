#!/bin/bash
# List of unique TARGET_NAME values
TARGET_NAMES=("patient4" "asano" "gosha" "goto" "ikejima" "nishio" "takahashi_jr" "kanda" "kawai" "matumoto" "nakashimizu" "noda" "taniguchi" "yoshikura"  "patient6" "patient8" "patient9")

for name in "${TARGET_NAMES[@]}"; do
    echo "Running for TARGET_NAME: $name"

    # Training
    python3 train_unet.py \
        --TARGET_NAME "$name" \
        --Dataset_name 15ch_arrange_direction \
        --mode train \
        --epochs 1000 \
        --train_batch_size 16 \
        --num_channels 15 \
        --ave_data_flg 1

    # Testing
    python3 train_unet.py \
        --TARGET_NAME "$name" \
        --Dataset_name 15ch_arrange_direction \
        --mode test \
        --num_channels 15 \
        --ave_data_flg 1

done