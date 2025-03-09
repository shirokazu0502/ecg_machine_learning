#!/bin/bash
name="patient"
measure_dates="1128 1130 1102 1107 1115 1219"
dataset_date="0428"
subject_group="1"
#再構成項のP波，R波,T波部分の重みを変更できる。
P_weight="1.0"
R_weight="0.001"
T_weight="1.0"
augumentation=""
python3 vae_goto_val_pt.py --TARGET_NAME "$name" --latent_size 2 --beta 1 --mode test --transform_type normal --Dataset_name patient_data"$dataset_date" --loss_pt_on_off off --loss_pt_on_off_R_weight "$R_weight" --augumentation "$augumentation" --loss_pt_on_off_P_weight "$P_weight" --loss_pt_on_off_T_weight "$T_weight"