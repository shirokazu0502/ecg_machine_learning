#!/bin/bash
measure_names="matumoto yoshikura takahashi taniguchi kawai goto asano"
measure_dates="1128 1130 1102 1107 1115 1219 0710"
dataset_date="0427"
subject_group="1"
#再構成項のP波，R波,T波部分の重みを変更できる。
P_weights="1.0"
R_weights="0.1 1.0"
T_weights="1.0"
augumentation=""
for P_weight in $P_weights; do 
    for T_weight in $T_weights; do 
        for R_weight in $R_weights; do 
            python3 src/machine_learning/output_data_pqrst_v3.py --dataset_date "$dataset_date" --names "$measure_names" --measure_dates "$measure_dates" --subject_group "$subject_group" --P_weight "$P_weight" --R_weight "$R_weight" --T_weight "$T_weight" --augumentation "$augumentation"
        done
    done
done