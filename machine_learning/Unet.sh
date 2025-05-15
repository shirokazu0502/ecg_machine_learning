# #icceの時のデータ
# names="asano gosha matumoto mori sato taniguchi"
# dataset_date="icce0120"
# subject_group="2"
# #腹部につけた時のデータ
# measure_names="matumoto yoshikura taniguchi kawai goto gosha nakanishi kasahara patient2 patient4 patient6 patient8 patient9"
# measure_dates="1130 1107 1115 1219 0407 0407 0304 1001 1001 1001 1109 1109"
# measure_names="matumoto yoshikura takahashi taniguchi kawai goto"
# measure_dates="1128 1130 1220 1107 1115 1219"
measure_names="matumoto yoshikura taniguchi kawai goto gosha nakanishi kasahara"
measure_dates="1130 1107 1115 1219 0407 0407 0304"
dataset_date="0418"
subject_group="1"
#再構成項のP波，R波,T波部分の重みを変更できる。
# P_weights="1.0"
# R_weights="0.001"
# T_weights="1.0"
P_weight="0.100 1.000"
R_weight="0.005 1.000"
T_weight="0.100 1.000"

# augumentation="st" #ST部分の延長短縮をするデータ拡張を行う。
# p_augumentation=""
# r_argumentation=""
# t_augumentation=""
ave_data_flg=1
# for P_weight in $P_weights; do
#     for T_weight in $T_weights; do
#         for R_weight in $R_weights; do
#             for name in $measure_names; do
#                 python3 vae_goto_val_pt.py --TARGET_NAME "$name" --epochs 500 --latent_size 10 --beta 1 --mode train --transform_type normal --Dataset_name pqrst_nkmodule_since"$dataset_date"_cwt --loss_pt_on_off off --loss_pt_on_off_R_weight "$R_weight" --augumentation "$augumentation" --loss_pt_on_off_P_weight "$P_weight" --loss_pt_on_off_T_weight "$T_weight" --ave_data_flg $ave_data_flg
#                 python3 vae_goto_val_pt.py --TARGET_NAME "$name" --epochs 500 --latent_size 10 --beta 1 --mode zplot --transform_type normal --Dataset_name pqrst_nkmodule_since"$dataset_date"_cwt --loss_pt_on_off off --loss_pt_on_off_R_weight "$R_weight" --augumentation "$augumentation" --loss_pt_on_off_P_weight "$P_weight" --loss_pt_on_off_T_weight "$T_weight" --ave_data_flg $ave_data_flg
#                 python3 vae_goto_val_pt.py --TARGET_NAME "$name" --epochs 500 --latent_size 10 --beta 1 --mode test --transform_type normal --Dataset_name pqrst_nkmodule_since"$dataset_date"_cwt --loss_pt_on_off off --loss_pt_on_off_R_weight "$R_weight" --augumentation "$augumentation" --loss_pt_on_off_P_weight "$P_weight" --loss_pt_on_off_T_weight "$T_weight" --ave_data_flg $ave_data_flg
#             done
#             python3 output_data_pqrst_v3.py --dataset_date $dataset_date --names $measure_names --measure_dates $measure_dates ----subject_group "$subject_group" --dataset_ver "$dataset" --P_weight "$P_weight" --R_weight "$R_weight" --T_weight "$T_weight" --augumentation "$augumentation"
#         done
#     done
# done

for name in $measure_names; do
    python3 unet_goto_val.py --TARGET_NAME "$name" --epochs 400 --beta 1 --mode train --transform_type normal --Dataset_name pqrst_nkmodule_since"$dataset_date"_cwt --loss_pt_on_off off --ave_data_flg $ave_data_flg
    python3 unet_goto_val.py --TARGET_NAME "$name" --epochs 400 --beta 1 --mode test --transform_type normal --Dataset_name pqrst_nkmodule_since"$dataset_date"_cwt --loss_pt_on_off off  --ave_data_flg $ave_data_flg
done
python3 output_data_pqrst_v3.py --dataset_date $dataset_date --names $measure_names --measure_dates $measure_dates ----subject_group "$subject_group" --dataset_ver "$dataset"--augumentation "$augumentation"


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