import argparse
import os

from settings import OUTPUT_DIR, OUTPUT_MAE_DIR, RATE


import datetime


def get_args():
    current_time = datetime.date.today().strftime("LSTM_%Y%m%d%H_after_rotation")
    datalength = int(RATE * 0.8)

    parser = argparse.ArgumentParser()
    parser.add_argument("--p_augumentation", type=str, default="")
    parser.add_argument("--r_augumentation", type=str, default="")
    parser.add_argument("--t_augumentation", type=str, default="")
    parser.add_argument("--Dataset_name", type=str, default="")
    parser.add_argument("--loss_pt_on_off", type=str, default="off")
    parser.add_argument("--loss_pt_on_off_R_weight", type=str, default="")
    parser.add_argument("--loss_pt_on_off_P_weight", type=str, default="")
    parser.add_argument("--loss_pt_on_off_T_weight", type=str, default="")
    parser.add_argument("--dataset_num", type=int, default=100)
    parser.add_argument("--TARGET_NAME", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--test_batch_size", type=int, default=1)
    parser.add_argument("--val_batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=0.0005)
    parser.add_argument("--datalength", type=int, default=datalength)
    parser.add_argument(
        "--enc_convlayer_sizes",
        type=list,
        default=[[15, 1], [30, 2]],
    )
    parser.add_argument("--enc_fclayer_sizes", type=list, default=[6000, 500, 64])
    parser.add_argument("--dec_fclayer_sizes", type=list, default=[64, 512, 3200])
    parser.add_argument("--dec_convlayer_sizes", type=list, default=[[16, 2], [8, 1]])
    parser.add_argument("--latent_size", type=int, default=2)
    parser.add_argument("--print_every", type=int, default=2000)
    parser.add_argument(
        "--fig_root", type=str, default=OUTPUT_DIR + "/" + f"figs_newref/{current_time}"
    )
    parser.add_argument(
        "--mae_folder", type=str, default=OUTPUT_MAE_DIR + f"/{current_time}"
    )
    parser.add_argument("--conditional", action="store_true")
    parser.add_argument("--train_off", action="store_false")
    parser.add_argument("--pth", type=str, default=r"vae_prt_sep.pth")
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--loss_fn_type", type=str, default="mse")
    parser.add_argument("--beta", type=int, default=1)
    parser.add_argument("--alpha", type=int, default=1000)
    parser.add_argument("--transform_type", type=str, default="normal")
    parser.add_argument("--current_time", type=str, default=current_time)
    parser.add_argument("--ecg_ch_num", type=int, default=8)
    parser.add_argument("--num_channels", type=int, default=15)
    parser.add_argument("--orientation", type=str, default="normal")
    parser.add_argument("--ave_data_flg", type=int, default=0)
    parser.add_argument("--base_filters", type=int, default=32)
    parser.add_argument("--lstm_hidden_size", type=int, default=128)
    parser.add_argument("--lstm_num_layers", type=int, default=2)
    args = parser.parse_args()
    return args
