#!/bin/bash
# List of unique TARGET_NAME values
TARGET_NAMES=(
    "asano"
    "gosha"
    "goto"
    "ikejima"
    "kanda"
    "kawai"
    "matumoto"
    "nakashimizu"
    "nishio"
    "noda"
    "patient1"
    "patient10"
    "patient2"
    "patient3"
    "patient4"
    "patient5"
    "patient6"
    "patient7"
    "patient8"
    "patient9"
    "takahashi_jr"
    "taniguchi"
    "yoshikura"
)

for name in "${TARGET_NAMES[@]}"; do
    echo "Running for TARGET_NAME: $name"

    # Training
    python3 train_unet.py \
        --TARGET_NAME "$name" \
        --Dataset_name 15ch_arrange_direction \
        --mode train \
        --epochs 500 \
        --train_batch_size 32 \
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