#!/bin/bash

cd src/refactored/

# List of unique TARGET_NAME values
TARGET_NAMES=("asano" "gosha" "goto" "ikejima" "kanda" "kawai" "matumoto" "nakashimizu" "nishio" "noda" "patient10" "patient2" "patient3" "patient4" "patient5" "patient6" "patient7" "patient8" "patient9" "takahashi_jr" "taniguchi" "yoshikura")

for name in "${TARGET_NAMES[@]}"; do
    echo "Running for TARGET_NAME: $name"

    # Training
    python3 train_lstm.py \
        --TARGET_NAME "$name" \
        --Dataset_name 15ch_arrange_direction \
        --mode train \
        --epochs 600 \
        --train_batch_size 16 \
        --num_channels 15

    # Testing
    python3 train_lstm.py \
        --TARGET_NAME "$name" \
        --Dataset_name 15ch_arrange_direction \
        --mode test \
        --num_channels 15

done