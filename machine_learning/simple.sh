names="matumoto yoshikura takahashi taniguchi kawai goto" #データセットに用いた人名を入力，テストデータを選択の際に使用
dataset_date="0120"
subject_group="1"
P_weights="1.0"
R_weights="1.0"
T_weights="1.0"
augumentation=""
#各々のウェイトを変えたときのパターンを全部行うためのfor文，今は１回だけ。
for P_weight in $P_weights; do
    for T_weight in $T_weights; do
        for R_weight in $R_weights; do
            for name in $names; do
                python3 vae_goto_val_pt.py --TARGET_NAME "$name" --epochs 500 --latent_size 2 --beta 1 --mode train --transform_type normal --Dataset_name pqrst_nkmodule_since"$dataset_date"_cwt --loss_pt_on_off off --loss_pt_on_off_R_weight "$R_weight" --augumentation "$augumentation" --loss_pt_on_off_P_weight "$P_weight" --loss_pt_on_off_T_weight "$T_weight"
                python3 vae_goto_val_pt.py --TARGET_NAME "$name" --epochs 500 --latent_size 2 --beta 1 --mode zplot --transform_type normal --Dataset_name pqrst_nkmodule_since"$dataset_date"_cwt --loss_pt_on_off off --loss_pt_on_off_R_weight "$R_weight" --augumentation "$augumentation" --loss_pt_on_off_P_weight "$P_weight" --loss_pt_on_off_T_weight "$T_weight"
                python3 vae_goto_val_pt.py --TARGET_NAME "$name" --epochs 500 --latent_size 2 --beta 1 --mode test --transform_type normal --Dataset_name pqrst_nkmodule_since"$dataset_date"_cwt --loss_pt_on_off off --loss_pt_on_off_R_weight "$R_weight" --augumentation "$augumentation" --loss_pt_on_off_P_weight "$P_weight" --loss_pt_on_off_T_weight "$T_weight"
            done
            python3 output_data_pqrst_v3.py --dataset_date $dataset_date --subject_group "$subject_group" --dataset_ver "$dataset" --P_weight "$P_weight" --R_weight "$R_weight" --T_weight "$T_weight" --augumentation "$augumentation"
        done
    done
done
