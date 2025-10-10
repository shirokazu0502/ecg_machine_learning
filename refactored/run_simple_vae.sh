#!/bin/bash

python train_simple_vae.py \
    --TARGET_NAME goto \
    --Dataset_name 15ch_arrange_direction \
    --mode train \
    --epochs 100 \
    --train_batch_size 16 \
    --num_channels 15
