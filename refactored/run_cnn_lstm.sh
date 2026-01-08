#!/bin/bash
# List of unique TARGET_NAME values
TARGET_NAMES=("patient4" "asano" "gosha" "goto" "ikejima" "nishio" "takahashi_jr" "kanda" "kawai" "matumoto" "nakashimizu" "noda" "taniguchi" "yoshikura"  "patient6" "patient8" "patient9")

for name in "${TARGET_NAMES[@]}"; do
    echo "Running for TARGET_NAME: $name"

    # Training
    python3 train_cnn_lstm.py \
        --TARGET_NAME "$name" \
        --Dataset_name 15ch_arrange_direction \
        --mode train \
        --epochs 1000 \
        --train_batch_size 16 \
        --num_channels 15 \
        --ave_data_flg 1 \
        --DataAugmentation "st_warp,t_height" \
        --cnn_filters_1 32 \
        --cnn_filters_2 64 \
        --cnn_kernel_size 5 \
        --lstm_hidden_size 128 \
        --lstm_num_layers 1

    # Testing
    python3 train_cnn_lstm.py \
        --TARGET_NAME "$name" \
        --Dataset_name 15ch_arrange_direction \
        --mode test \
        --num_channels 15 \
        --ave_data_flg 1 \
        --cnn_filters_1 32 \
        --cnn_filters_2 64 \
        --cnn_kernel_size 5 \
        --lstm_hidden_size 128 \
        --lstm_num_layers 1

done
