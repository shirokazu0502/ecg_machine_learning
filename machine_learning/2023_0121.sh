# #icceの時のデータ
# names="asano gosha matumoto mori sato taniguchi"
# dataset_date="icce0120"
# subject_group="2"
# #腹部につけた時のデータ
measure_names="matumoto yoshikura takahashi taniguchi kawai goto asano nakanishi mori togo gobara patient4 patient5 patient6 patient8 patient9"
measure_dates="1128 1130 1102 1107 1115 1219 0710 0717 0723 0722 0729 1001 1001 1001 1001 1001 1109"
dataset_date="1001"
current_time="2025_0101_1640"
subject_group="1"
#再構成項のP波，R波,T波部分の重みを変更できる。
P_weights="1.0"
R_weights="0.1 1.0"
T_weights="1.0"
augumentation=""
for P_weight in $P_weights; do
    for T_weight in $T_weights; do
        for R_weight in $R_weights; do
            for name in $measure_names; do
                python3 vae_goto_val_pt.py --TARGET_NAME "$name" --epochs 500 --latent_size 64 --beta 1 --mode train --transform_type normal --Dataset_name pqrst_nkmodule_since"$dataset_date"_cwt --loss_pt_on_off off --loss_pt_on_off_R_weight "$R_weight" --augumentation "$augumentation" --loss_pt_on_off_P_weight "$P_weight" --loss_pt_on_off_T_weight "$T_weight"
                python3 vae_goto_val_pt.py --TARGET_NAME "$name" --epochs 500 --latent_size 64 --beta 1 --mode zplot --transform_type normal --Dataset_name pqrst_nkmodule_since"$dataset_date"_cwt --loss_pt_on_off off --loss_pt_on_off_R_weight "$R_weight" --augumentation "$augumentation" --loss_pt_on_off_P_weight "$P_weight" --loss_pt_on_off_T_weight "$T_weight"
                python3 vae_goto_val_pt.py --TARGET_NAME "$name" --epochs 500 --latent_size 64 --beta 1 --mode test --transform_type normal --Dataset_name pqrst_nkmodule_since"$dataset_date"_cwt --loss_pt_on_off off --loss_pt_on_off_R_weight "$R_weight" --augumentation "$augumentation" --loss_pt_on_off_P_weight "$P_weight" --loss_pt_on_off_T_weight "$T_weight"
            done
            python3 output_data_pqrst_v3.py --dataset_date $dataset_date --names $measure_names --measure_dates $measure_dates ----subject_group "$subject_group" --dataset_ver "$dataset" --P_weight "$P_weight" --R_weight "$R_weight" --T_weight "$T_weight" --augumentation "$augumentation"
        done
    done
done

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