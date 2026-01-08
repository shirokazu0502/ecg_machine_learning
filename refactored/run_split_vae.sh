
# measure_names="asano gosha nishio takahashi_jr taniguchi yoshikura goto ikejima kanda kawai matumoto nakashimizu noda patient1 patient2 patient3 patient4 patient5 patient6 patient7 patient8 patient9 patient10"
# measure_dates="0714 0807 0513 0512 1107 1130 1219 0714 0807 1115 1128 0512 0714 1001 1001 1001 1001 1001 1109 1109 1109 1109"
# dataset_date="0602"
# subject_group="1"

# P_weight="0.100 10.000"
# R_weight="0.001 0.500"
# T_weight="0.010 1.000"

# p_augumentation=""
# r_augumentation=""
# t_augumentation=""
# ave_data_flg=1

# augumentation=""
# for P_weight in $P_weights; do
#     for T_weight in $T_weights; do
#         for R_weight in $R_weights; do
#             for name in $measure_names; do
#                 python3 train_vae.py --TARGET_NAME "$name" --epochs 500 --latent_size 4 --beta 1 --mode train --transform_type normal --Dataset_name 15ch_arrange_direction --loss_pt_on_off off --loss_pt_on_off_R_weight "$R_weight" --augumentation "$augumentation" --loss_pt_on_off_P_weight "$P_weight" --loss_pt_on_off_T_weight "$T_weight" --ave_data_flg $ave_data_flg
#                 python3 train_vae.py --TARGET_NAME "$name" --epochs 500 --latent_size 4 --beta 1 --mode test --transform_type normal --Dataset_name 15ch_arrange_direction --loss_pt_on_off off --loss_pt_on_off_R_weight "$R_weight" --augumentation "$augumentation" --loss_pt_on_off_P_weight "$P_weight" --loss_pt_on_off_T_weight "$T_weight" --ave_data_flg $ave_data_flg
#             done
#         done
#     done
# done

# measure_dates="1128 1130 1220 1107 1115 1219"
measure_names="asano gosha taniguchi yoshikura goto ikejima kanda kawai matumoto nakashimizu noda"
dataset_date="0602"
subject_group="1"

# measure_names="asano gosha taniguchi yoshikura goto matumoto kawai"

#再構成項のP波，R波,T波部分の重みを変更できる。
# P_weights="1.0"
# R_weights="0.001"
# T_weights="1.0"

P_weight="0.100 10.000"
R_weight="0.001 0.500"
T_weight="0.010 1.000"

# augumentation="st" #ST部分の延長短縮をするデータ拡張を行う。
p_augumentation=""
r_augumentation="" #R波の高さを変えるデータ拡張を行う。
t_augumentation=""
ave_data_flg=1

for name in $measure_names; do
    python3 train_vae.py --TARGET_NAME "$name" --epochs 400 --latent_size 2 --beta 1 --mode train --transform_type normal --Dataset_name 15ch_arrange_direction --loss_pt_on_off off --loss_pt_on_off_R_weight "$R_weight" --p_augumentation "$p_augumentation" --r_augumentation "$r_augumentation" --t_augumentation "$t_augumentation" --loss_pt_on_off_P_weight "$P_weight" --loss_pt_on_off_T_weight "$T_weight" --ave_data_flg $ave_data_flg
    # python3 vae_goto_val_PRTmodel_sep.py --TARGET_NAME "$name" --epochs 10 --latent_size 8 --beta 1 --mode zplot --transform_type normal --Dataset_name pqrst_nkmodule_since"$dataset_date"_cwt --loss_pt_on_off off --loss_pt_on_off_R_weight "$R_weight" --p_augumentation "$p_augumentation" --r_augumentation "$r_augumentation" --t_augumentation "$t_augumentation" --loss_pt_on_off_P_weight "$P_weight" --loss_pt_on_off_T_weight "$T_weight" --ave_data_flg $ave_data_flg
    # python3 vae_goto_val_PRTmodel_sep.py --TARGET_NAME "$name" --epochs 10 --latent_size 8 --beta 1 --mode test --transform_type normal --Dataset_name pqrst_nkmodule_since"$dataset_date"_cwt --loss_pt_on_off off --loss_pt_on_off_R_weight "$R_weight" --p_augumentation "$p_augumentation" --r_augumentation "$r_augumentation" --t_augumentation "$t_augumentation" --loss_pt_on_off_P_weight "$P_weight" --loss_pt_on_off_T_weight "$T_weight" --ave_data_flg $ave_data_flg
done