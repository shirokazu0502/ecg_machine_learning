
# #icceの時のデータ
# names="asano gosha matumoto mori sato taniguchi"
# dataset_date="icce0120"
# subject_group="2"
# #腹部につけた時のデータ
measure_names="asano"

for name in $measure_names; do
    python3 train_unet.py --TARGET_NAME "$name" --epochs 300  --mode train --transform_type normal --Dataset_name 15ch_arrange_direction --loss_pt_on_off off --ave_data_flg 1 --train_batch_size 4
    python3 train_unet.py --TARGET_NAME "$name" --epochs 300  --mode test --transform_type normal --Dataset_name 15ch_arrange_direction --loss_pt_on_off off  --ave_data_flg 1
done
