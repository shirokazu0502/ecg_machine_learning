measure_names="yoshikura taniguchi kawai goto gosha patient1 patient4 patient6 patient8 patient9"
measure_dates="1130 1107 1115 1219 0712 1001 1001 1001 1109 1109"
dataset_date="1226"
subject_group="1"
ave_data_flg=0
#再構成項のP波，R波,T波部分の重みを変更できる。
P_weights="0.5"
R_weights="0.01"
T_weights="0.5"
augumentation=""

python3 vae_goto_val_pt.py --TARGET_NAME "$name" --epochs 500 --latent_size 40 --beta 1 --mode 15ch_only --transform_type normal --Dataset_name pqrst_nkmodule_since"$dataset_date"_cwt --loss_pt_on_off off --loss_pt_on_off_R_weight "$R_weight" --augumentation "$augumentation" --loss_pt_on_off_P_weight "$P_weight" --loss_pt_on_off_T_weight "$T_weight" --ave_data_flg $ave_data_flg
"""
augumentation="st" #ST部分の延長短縮をするデータ拡張を行う。
for P_weight in $P_weights; do
    for T_weight in $T_weights; do
        for R_weight in $R_weights; do
            # for name in $names; do
            #     python3 vae_goto_val_pt.py --TARGET_NAME "$name" --epochs 500 --latent_size 2 --beta 1 --mode train --transform_type normal --Dataset_name pqrst_nkmodule_since"$dataset_date"_cwt --loss_pt_on_off off --loss_pt_on_off_R_weight "$R_weight" --augumentation "$augumentation" --loss_pt_on_off_P_weight "$P_weight" --loss_pt_on_off_T_weight "$T_weight"
            #     python3 vae_goto_val_pt.py --TARGET_NAME "$name" --epochs 500 --latent_size 2 --beta 1 --mode zplot --transform_type normal --Dataset_name pqrst_nkmodule_since"$dataset_date"_cwt --loss_pt_on_off off --loss_pt_on_off_R_weight "$R_weight" --augumentation "$augumentation" --loss_pt_on_off_P_weight "$P_weight" --loss_pt_on_off_T_weight "$T_weight"
            #     python3 vae_goto_val_pt.py --TARGET_NAME "$name" --epochs 500 g--latent_size 2 --beta 1 --mode test --transform_type normal --Dataset_name pqrst_nkmodule_since"$dataset_date"_cwt --loss_pt_on_off off --loss_pt_on_off_R_weight "$R_weight" --augumentation "$augumentation" --loss_pt_on_off_P_weight "$P_weight" --loss_pt_on_off_T_weight "$T_weight"
            # done
            python3 output_data_pqrst_v3.py --dataset_date $dataset_date --subject_group "$subject_group" --dataset_ver "$dataset" --P_weight "$P_weight" --R_weight "$R_weight" --T_weight "$T_weight" --augumentation "$augumentation"
        done
    done
done
"""
# # augumentation="st"
# # # augumentation="pq"
# for P_weight in $P_weights; do
#     for T_weight in $T_weights; do
#         for R_weight in $R_weights; do
#             # for name in $names; do
#             #     python3 vae_goto_val_pt.py --TARGET_NAME "$name" --epochs 500 --latent_size 2 --beta 1 --mode train --transform_type normal --Dataset_name pqrst_nkmodule_since"$dataset_date"_cwt_"$dataset" --loss_pt_on_off off --loss_pt_on_off_R_weight "$R_weight" --augumentation "$augumentation" --loss_pt_on_off_P_weight "$P_weight" --loss_pt_on_off_T_weight "$T_weight"
#             #     python3 vae_goto_val_pt.py --TARGET_NAME "$name" --epochs 500 --latent_size 2 --beta 1 --mode test --transform_type normal --Dataset_name pqrst_nkmodule_since"$dataset_date"_cwt_"$dataset" --loss_pt_on_off off --loss_pt_on_off_R_weight "$R_weight" --augumentation "$augumentation" --loss_pt_on_off_P_weight "$P_weight" --loss_pt_on_off_T_weight "$T_weight"
#             # done
#             python3 output_data_pqrst_v3.py --dataset_date $dataset_date --subject_group "$subject_group" --P_weight "$P_weight" --R_weight "$R_weight" --T_weight "$T_weight" --augumentation "$augumentation"
#         done
#     done
# done