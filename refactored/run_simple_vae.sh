#!/bin/bash

# --- 設定部分 (参考コードより適用) ---

# 対象被験者リスト
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
# measure_names="asano gosha taniguchi yoshikura goto matumoto kawai" # 短縮版が必要な場合

# 再構成項のP波，R波,T波部分の重み
P_WEIGHT="1.0" 
R_WEIGHT="0.001"
T_WEIGHT="1.0"

# データ拡張設定
p_augumentation=""
r_augumentation=""
t_augumentation=""

# その他フラグ
ave_data_flg=1


# --- ループ処理実行 ---

for target_name in "${TARGET_NAMES[@]}"; do
    echo "Running training for: $target_name"
    
    python3 train_simple_vae.py \
        --TARGET_NAME "$target_name" \
        --Dataset_name 15ch_arrange_direction \
        --mode train \
        --epochs 300 \
        --train_batch_size 16 \
        --num_channels 15 \
        --loss_pt_on_off off \
        --loss_pt_on_off_P_weight "$P_WEIGHT" \
        --loss_pt_on_off_R_weight "$R_WEIGHT" \
        --loss_pt_on_off_T_weight "$T_WEIGHT" \
        --ave_data_flg $ave_data_flg
    
    python3 train_simple_vae.py \
        --TARGET_NAME "$target_name" \
        --Dataset_name 15ch_arrange_direction \
        --mode test \
        --num_channels 15 \
        --ave_data_flg $ave_data_flg

done