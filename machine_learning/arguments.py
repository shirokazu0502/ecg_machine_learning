import argparse
import os

from settings import OUTPUT_DIR, OUTPUT_MAE_DIR, RATE


import datetime


def get_args():
    current_time = datetime.date.today().strftime(
        "u-net_confirm_train_curve%Y%m%d%H_reference_center_4ch_for_thesis"
    )
    datalength = int(RATE * 0.8)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--DataAugmentation",
        type=str,
        default="",
        help="Comma-separated string of augmentations to apply (e.g., 'pq_warp,st_warp,t_height')",
    )
    parser.add_argument("--Dataset_name", type=str, default="")
    parser.add_argument("--loss_pt_on_off", type=str, default="off")
    parser.add_argument("--loss_pt_on_off_R_weight", type=str, default="")
    parser.add_argument("--loss_pt_on_off_P_weight", type=str, default="")
    parser.add_argument("--loss_pt_on_off_T_weight", type=str, default="")
    parser.add_argument("--dataset_num", type=int, default=100)
    parser.add_argument("--TARGET_NAME", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--test_batch_size", type=int, default=1)
    parser.add_argument("--val_batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--datalength", type=int, default=datalength)
    parser.add_argument(
        "--enc_convlayer_sizes",
        type=list,
        default=[[16, 1], [30, 2]],
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
    parser.add_argument("--loss_fn_type", type=str, default="mse_corr")
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--alpha", type=int, default=1000)
    parser.add_argument("--transform_type", type=str, default="normal")
    parser.add_argument("--current_time", type=str, default=current_time)
    parser.add_argument("--ecg_ch_num", type=int, default=8)
    parser.add_argument("--num_channels", type=int, default=16)
    parser.add_argument("--orientation", type=str, default="normal")
    parser.add_argument("--ave_data_flg", type=int, default=1)

    # Model-specific hyperparameters
    parser.add_argument("--base_filters", type=int, default=32)
    parser.add_argument(
        "--unet_depth", type=int, default=4, help="Depth of the UNet model."
    )
    parser.add_argument("--lstm_hidden_size", type=int, default=128)
    parser.add_argument("--lstm_num_layers", type=int, default=2)

    # CNN-LSTM specific
    parser.add_argument("--cnn_filters_1", type=int, default=32)
    parser.add_argument("--cnn_filters_2", type=int, default=64)
    parser.add_argument("--cnn_kernel_size", type=int, default=5)

    # SimpleCNN specific
    parser.add_argument("--cnn_depth", type=int, default=4)
    parser.add_argument(
        "--cnn_init_filters",
        type=int,
        default=8,
        help="Initial filter size for SimpleCNN",
    )

    # GNN specific
    parser.add_argument(
        "--rnn_hidden_size",
        type=int,
        default=64,
        help="Hidden size for the RNN encoder in GNN model.",
    )
    parser.add_argument(
        "--gnn_hidden_size",
        type=int,
        default=64,
        help="Hidden size for the GCN mixer in GNN model.",
    )
    parser.add_argument(
        "--temporal_model",
        type=str,
        default="gru",
        help="Type of temporal model for GNN (gru, lstm, or mamba)",
    )

    # Loss function weights
    parser.add_argument(
        "--r_loss_weight",
        type=float,
        default=1.0,
        help="Weight for R-wave segment in MSE loss",
    )
    parser.add_argument(
        "--other_loss_weight",
        type=float,
        default=1.0,
        help="Weight for non-R-wave segments in MSE loss",
    )

    parser.add_argument(
        "--model_type",
        type=str,
        required=False,
        help="Model type to train (e.g., cnn, cnn_lstm, unet, lstm, gnn_reconstruction)",
    )

    args = parser.parse_args()
    return args
