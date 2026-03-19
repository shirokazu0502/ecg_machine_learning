#!/bin/bash

PATIENTS=(
    "asano 0714"
    "gosha 0807"
    "goto 1219"
    "ikejima 0714"
    "kanda 0807"
    "kawai 1115"
    "matumoto 1128"
    "nakashimizu 0512"
    "nishio 0513"
    "noda 0714"
    "taniguchi 1107"
    "takahashi_jr 0512"
    "yoshikura 1130"
    "patient4 1001"
    "patient6 1001"
    "patient8 1109"
    "patient9 1109"
)

for patient_info in "${PATIENTS[@]}"; do
    # 患者名と日付を分割
    NAME=$(echo $patient_info | cut -d' ' -f1)
    DATE=$(echo $patient_info | cut -d' ' -f2)

    # 出力先ディレクトリを作成
    OUTPUT_DIR="/mnt/ecg_project/data/processed_data/best_resample/${NAME}_${DATE}"

    # Pythonスクリプトを実行
    python3 /mnt/ecg_project/src_for_gemini_test/dataset_creation/Make_dataset_0120_16ch_synchro_per_heart.py --name $NAME --date $DATE --output_filepath $OUTPUT_DIR
done
