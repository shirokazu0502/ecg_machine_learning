#!/bin/bash

# --- 設定部分 (参考コードより適用) ---

# 対象被験者リスト
measure_names=(
    "aoki"
    "arai"
    "asano"
    "gosha"
    "goto"
    "ikejima"
    "kanda"
    "kawai"
    "kinoshita"
    "matumoto"
    "miyano"
    "nakanishi"
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
    "tosue"
    "yoshikura"
)
# measure_names="asano gosha taniguchi yoshikura goto matumoto kawai" # 短縮版が必要な場合

# 再構成項のP波，R波,T波部分の重み
P_WEIGHT="1.0" 
R_WEIGHT="1.0"
T_WEIGHT="1.0"

# データ拡張設定
p_augumentation=""
r_augumentation=""
t_augumentation=""

# その他フラグ
ave_data_flg=1


# --- ループ処理実行 ---

for name in $measure_names; do
    echo "Running training for: $name"
    
    python3 train_simple_vae.py \
        --TARGET_NAME "$name" \
        --Dataset_name reference_center_4ch \
        --mode train \
        --epochs 500 \
        --train_batch_size 64 \
        --num_channels 16 \
        --loss_pt_on_off off \
        --loss_pt_on_off_P_weight "$P_WEIGHT" \
        --loss_pt_on_off_R_weight "$R_WEIGHT" \
        --loss_pt_on_off_T_weight "$T_WEIGHT" \
        --ave_data_flg $ave_data_flg
    
    # Testing
    python3 train_simple_vae.py \
        --TARGET_NAME "$target_name" \
        --Dataset_name reference_center_4ch \
        --mode test \
        --num_channels 16 \
        --ave_data_flg 1
done