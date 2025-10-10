
# #icceの時のデータ
# names="asano gosha matumoto mori sato taniguchi"
# dataset_date="icce0120"
# subject_group="2"
# #腹部につけた時のデータ
measure_names="patient1 asano gosha nishio takahashi_jr taniguchi yoshikura goto ikejima kanda kawai matumoto nakashimizu noda patient2 patient3 patient4 patient5 patient6 patient7 patient8 patient9 patient10"
measure_dates="1001 0714 0807 0513 0512 1107 1130 1219 0714 0807 1115 1128 0512 0714 1001 1001 1001 1001 1001 1109 1109 1109 1109"

# measure_names="matumoto yoshikura taniguchi kawai goto gosha nakanishi kasahara takahashi_jr nakashimizu gobara nishio patient4 patient6 patient8 patient9"
# measure_dates="1128 1130 1107 1115 1219 0407 0407 0304 0512 0512 0512 0512 0513 1001 1001 1109 1109"
dataset_date="0602"
subject_group="1"
augumentation="st" #ST部分の延長短縮をするデータ拡張を行う。
ave_data_flg=1

for name in $measure_names; do
    python3 train_unet.py --TARGET_NAME "$name" --epochs 500  --mode train --transform_type normal --Dataset_name 15ch_arrange_direction --loss_pt_on_off off --ave_data_flg $ave_data_flg --train_batch_size 16
    python3 train_unet.py --TARGET_NAME "$name" --epochs 500  --mode test --transform_type normal --Dataset_name 15ch_arrange_direction --loss_pt_on_off off  --ave_data_flg $ave_data_flg 
done
