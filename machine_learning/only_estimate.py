import os
import sys
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import neurokit2 as nk
from torch.utils.data import DataLoader
from models import VAE
import Dataset
from config.settings import (
    RATE,
    RATE_15CH,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    OUTPUT_DIR,
)
from src.dataset_creation.Make_dataset_0120 import (
    CSVReader_16ch,
    linear_interpolation_resample_All,
    validate_integer_input,
    peak_sc_plot,
    peak_search_nk_15ch,
)


def ecg_clean_df_15ch(df_15ch, rate):
    df_15ch_cleaned = pd.DataFrame()
    for column in df_15ch.columns:
        ecg_signal = df_15ch[column].copy().values
        ecg_signal = nk.ecg_clean(ecg_signal, sampling_rate=rate, method="neurokit")
        df_15ch_cleaned[column] = ecg_signal
    fig = plt.figure(num=None, figsize=(12, 5), dpi=100, facecolor="w", edgecolor="k")
    axis_line_width = 2.0
    tick_label_size = 18
    plot_time = np.arange(len(df_15ch)) / RATE_15CH
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(plot_time, df_15ch["ch_1"], label="ch_1")
    ax1.plot(plot_time, df_15ch_cleaned["ch_1"], label="filtered ch1")
    ax1.legend(loc="upper right", fontsize=18, ncol=1, bbox_to_anchor=(1, 1))
    ax1.tick_params(labelsize=tick_label_size, direction="in")
    plt.xlim(3.8, 8)
    plt.ylim(-100, 200)
    for axis in ["top", "bottom", "left", "right"]:
        ax1.spines[axis].set_linewidth(axis_line_width)
    plt.tight_layout()
    plt.show()
    return df_15ch_cleaned


def tensor_to_ndarray(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    else:
        raise TypeError("Input must be a torch.Tensor.")


def plot_fig_15ch_only(recon_x, datalength, args, batch_size_num, label_name):
    sample_rate = 500
    sample_num = args.datalength
    dt = 1 / sample_rate
    xticks = np.linspace(0.0, 1.0 / sample_rate * sample_num, sample_num)
    recon_x2 = torch.reshape(recon_x, (-1, args.ecg_ch_num, datalength))
    for p in range(batch_size_num):
        for q in range(args.ecg_ch_num):
            plt.tight_layout()
            plt.plot(
                xticks,
                recon_x2[p][q].cpu().data.numpy(),
                color="red",
                linewidth=1.0,
                linestyle="-",
            )
            plt.xlabel("second")
            plt.axis("on")
            plt.minorticks_on()
            plt.grid(which="both", axis="x", alpha=0.8, linestyle="--", linewidth=1)
            plt.legend(
                ["predict", "ECG", "P_onset", "T_offset"],
                bbox_to_anchor=(1.05, 1),
                loc="upper left",
            )
            plt.title(label_name[p] + "_ch={}".format(ecg_ch_names[q]))
            plt.tight_layout()
            if not os.path.exists(os.path.join(args.fig_root)):
                os.mkdir(os.path.join(args.fig_root))
            plt.savefig(
                os.path.join(
                    args.fig_root,
                    args.TARGET_NAME,
                    "ch={}_15ch_only_test_x_xo_{}.png".format(
                        ecg_ch_names[q], label_name[p]
                    ),
                ),
                dpi=300,
            )
            plt.cla()
            plt.clf()
            plt.close()


def plot_fig_15ch(df_15ch, args, label_name):
    sample_rate = RATE
    sample_num = len(df_15ch)
    dt = 1 / sample_rate
    xticks = np.linspace(0.0, dt * sample_num, sample_num)
    fig, axs = plt.subplots(15, 1, figsize=(15, 30))
    for i, column in enumerate(df_15ch.columns):
        axs[i].plot(
            xticks, df_15ch[column].values, color="blue", linewidth=1.0, linestyle="-"
        )
        axs[i].set_xlabel("second")
        axs[i].set_title(column)
        axs[i].minorticks_on()
        axs[i].grid(which="both", axis="x", alpha=0.8, linestyle="--", linewidth=1)
    plt.tight_layout()
    if not os.path.exists(os.path.join(args.fig_root)):
        os.mkdir(os.path.join(args.fig_root))
    plt.savefig(
        os.path.join(
            args.fig_root,
            args.TARGET_NAME,
            "{}_15ch_data.png".format(label_name),
        ),
        dpi=300,
    )
    plt.cla()
    plt.clf()
    plt.close()


def output_csv(file_path, file_name, data):
    dt = 1.0 / RATE
    time_tmp = np.arange(len(data)) * dt
    time = pd.DataFrame()
    time["Time"] = time_tmp
    data = data.reset_index(drop=True)
    data_out = pd.concat([time, data], axis=1)
    data_out.to_csv(file_path + "/" + file_name, index=None)


def cut_heartbeats(ecg_data, center_idxs, file_path, time_length):
    create_directory_if_not_exists(file_path)
    range = int(time_length * RATE / 2)
    for i, center_idx in enumerate(center_idxs):
        data = ecg_data[center_idx - range : center_idx + range].copy()
        file_name = "dataset_{}.csv".format(str(i).zfill(3))
        output_csv(file_path=file_path, file_name=file_name, data=data.copy())


def save_15ch_data(df_15ch, file_path, file_name):
    df_15ch.to_csv(os.path.join(file_path, file_name), index=False)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    dir_path = args.raw_datas_dir
    csv_reader_16ch = CSVReader_16ch(dir_path)
    df_16ch = csv_reader_16ch.process_files()
    cols = df_16ch.columns
    df_15ch = pd.DataFrame()
    for col in cols:
        df_15ch[col] = df_16ch[col] - df_16ch["ch_16"]
    df_15ch = df_15ch.drop(columns=["ch_16"])
    df_15ch_pf = ecg_clean_df_15ch(df_15ch=df_15ch.copy(), rate=RATE_15CH)
    df_resample_15ch = linear_interpolation_resample_All(
        df=df_15ch_pf.copy(), sampling_rate=RATE_15CH, new_sampling_rate=RATE
    )
    df_15ch_pf = df_resample_15ch.copy()
    reverse = args.reverse
    TARGET_CHANNEL_15CH = validate_integer_input()
    ecg_ch_names = [
        "A1",
        "A2",
        "A3",
        "aVR",
        "aVL",
        "aVF",
        "V1",
        "V2",
        "V3",
        "V4",
        "V5",
        "V6",
    ]
    if reverse == "off":
        rpeaks, val = peak_search_nk_15ch(
            df_15ch_pf[TARGET_CHANNEL_15CH].copy(), RATE=RATE
        )
        peak_sc_plot(df_15ch_pf.copy(), RATE=RATE, TARGET=TARGET_CHANNEL_15CH)
        file_path = PROCESSED_DATA_DIR + f"/15ch_only/{args.TARGET_NAME}"
        cut_heartbeats(df_15ch_pf, rpeaks, file_path=file_path, time_length=0.8)
    else:
        df_15ch_reverse = df_15ch_pf.copy()
        df_15ch_reverse[TARGET_CHANNEL_15CH] = (-1) * df_15ch_pf.copy()[
            TARGET_CHANNEL_15CH
        ]
        df_resample_15ch = df_15ch_reverse.copy()
        sc_15ch = peak_sc_15ch(
            df_15ch_reverse.copy(), RATE=RATE, TARGET=TARGET_CHANNEL_15CH
        )
        peak_sc_plot(df_15ch_reverse.copy(), RATE=RATE, TARGET=TARGET_CHANNEL_15CH)

    datalength = args.datalength
    latent_size = 4
    vae = VAE(
        datalength=datalength,
        enc_convlayer_sizes=args.enc_convlayer_sizes,
        enc_fclayer_sizes=args.enc_fclayer_sizes,
        dec_fclayer_sizes=args.dec_fclayer_sizes,
        dec_convlayer_sizes=args.dec_convlayer_sizes,
        latent_size=latent_size,
        conditional=args.conditional,
        num_labels=20 if args.conditional else 0,
    ).to(device)
    test_dataset = Dataset.Dataset_setup_15ch_only(
        TARGET_NAME=args.TARGET_NAME,
        transform_type=args.transform_type,
        Dataset_name="15ch_only",
        dataset_num=10,
    )
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    vae.load_state_dict(torch.load(args.pth, map_location=lambda storage, loc: storage))
    vae.eval()

    z_temps = np.empty((0, latent_size))
    label_temps = []
    with torch.no_grad():
        for j, (x, label_name) in enumerate(test_loader):
            x = x.to(device)
            recon_x, mean, log_var, z = vae(x)
            z_temp = tensor_to_ndarray(z)
            z_temps = np.append(z_temps, z_temp, axis=0)
            for i in range(len(label_name)):
                label_temps.append(label_name[i])
            ecg_ch = args.ecg_ch_num
            if j != -1:
                numplotfig = len(x)
                recon_x = recon_x.view(-1, ecg_ch, datalength)
                sample_rate = 500
                sample_num = args.datalength
                xticks = np.linspace(0.0, 1.0 / sample_rate * sample_num, sample_num)
                if numplotfig > 6:
                    numplotfig = 6
                for p in range(numplotfig):
                    for q in range(ecg_ch):
                        plt.subplot(4, 3, q + 1)
                        plt.plot(
                            xticks,
                            recon_x[p][q].cpu().data.numpy(),
                            color="red",
                            linewidth=1.0,
                            linestyle="-",
                        )
                        plt.axis("on")
                        plt.minorticks_on()
                        plt.grid(
                            which="both",
                            axis="x",
                            alpha=0.8,
                            linestyle="--",
                            linewidth=1,
                        )
                        plt.title("ch={}".format(ecg_ch_names[q]), fontsize=5)
                    plt.legend(
                        ["predict", "ECG", "P_onset", "T_offset"],
                        bbox_to_anchor=(1.05, 1),
                        loc="upper left",
                    )
                    plt.suptitle(label_name[p])
                    plt.tight_layout()
                    plt.gcf().set_size_inches(10, 5)
                    plt.savefig(
                        os.path.join(
                            args.fig_root,
                            args.TARGET_NAME,
                            label_name[p] + "test" + str(j) + ".png",
                        ),
                        dpi=300,
                    )
                    plt.show()
                    plt.cla()
                    plt.clf()
                    plt.close()

    # 推定に用いた15チャネルのデータを画像として保存
    plot_fig_15ch(df_15ch_pf, args, args.TARGET_NAME)


def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--augumentation", type=str, default="")
    parser.add_argument("--Dataset_name", type=str, default="")
    parser.add_argument("--dataset_num", type=int, default=20)
    parser.add_argument("--TARGET_NAME", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--val_batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--datalength", type=int, default=400)
    parser.add_argument("--enc_convlayer_sizes", type=list, default=[[15, 1], [30, 2]])
    parser.add_argument("--enc_fclayer_sizes", type=list, default=[6000, 500, 64])
    parser.add_argument("--dec_fclayer_sizes", type=list, default=[64, 512, 3200])
    parser.add_argument("--dec_convlayer_sizes", type=list, default=[[16, 2], [8, 1]])
    parser.add_argument("--latent_size", type=int, default=2)
    parser.add_argument("--print_every", type=int, default=2000)
    parser.add_argument(
        "--fig_root", type=str, default=OUTPUT_DIR + "/" + f"figs_newref/15ch_only"
    )
    parser.add_argument("--conditional", action="store_true")
    parser.add_argument("--train_off", action="store_false")
    parser.add_argument("--pth", type=str, default=r"model_pth/vae_cross_z4.pth")
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--loss_fn_type", type=str, default="mse")
    parser.add_argument("--beta", type=int, default=1)
    parser.add_argument("--alpha", type=int, default=300000)
    parser.add_argument("--dim_red", type=str, default="")
    parser.add_argument("--transform_type", type=str, default="normal")
    parser.add_argument("--ecg_ch_num", type=int, default=8)
    parser.add_argument("--current_time", type=str)
    parser.add_argument("--ave_data_flg", type=int, default=0)
    args = parser.parse_args()
    # 標準入力からTARGET_NAMEを受け取る
    args.TARGET_NAME = input("TARGET_NAMEを入力してください: ")
    create_directory_if_not_exists(args.fig_root + "/" + args.TARGET_NAME)
    args.reverse = "off"
    args.raw_datas_dir = RAW_DATA_DIR + "/sheet_sensor_csvdatas/{}".format("0304")
    main(args)
