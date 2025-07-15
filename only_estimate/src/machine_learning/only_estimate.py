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
from only_estimate.src.dataset_creation.for_16ch_data.Make_dataset_0120 import (
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
    print("sample_num", sample_num)
    print("dt", dt)
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


def plot_fig_8ch(df_8ch, args, label_name):
    sample_rate = RATE
    sample_num = len(df_8ch)
    dt = 1 / sample_rate
    print("sample_num", sample_num)
    print("dt", dt)
    xticks = np.linspace(0.0, dt * sample_num, sample_num)
    fig, axs = plt.subplots(8, 1, figsize=(15, 30))
    for i, column in enumerate(df_8ch.columns):
        axs[i].plot(
            xticks, df_8ch[column].values, color="blue", linewidth=1.0, linestyle="-"
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
            "{}_8ch_data.png".format(label_name),
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


def concatenate_heartbeats(heartbeats, baseline_range=50):
    concatenated_signal = []
    for heartbeat in heartbeats:
        # 基線部分を除去
        heartbeat = heartbeat[baseline_range:-baseline_range]
        concatenated_signal.extend(heartbeat)
    return np.array(concatenated_signal)


def align_waveform(recon_wave, target_peak, reference_peak):
    """再構成波形のピーク(200番目)を指定した元波形のピーク位置に合わせる"""
    shift = reference_peak - target_peak  # シフト量
    return np.roll(recon_wave, shift)  # シフト後の波形を返す


def merge_dataframes_with_farthest_from_05(list1, list2):
    """
    2つのリストを比較し、各要素ごとに0.5からより遠い値を採用する。
    """
    # NumPy配列に変換
    arr1 = np.array(list1, dtype=np.float64)
    arr2 = np.array(list2, dtype=np.float64)
    # NaN が含まれる場合はもう片方の値を採用
    merged_array = np.where(
        np.isnan(arr1),
        arr2,
        np.where(
            np.isnan(arr2),
            arr1,
            np.where(abs(arr1 - 0.5) > abs(arr2 - 0.5), arr1, arr2),
        ),
    )

    return merged_array.tolist()


def plot_concatenated_signal(concatenated_signal, sample_rate, args, label_name):
    dt = 1 / sample_rate
    sample_num = len(concatenated_signal)
    xticks = np.linspace(0.0, dt * sample_num, sample_num)
    plt.figure(figsize=(15, 5))
    plt.plot(xticks, concatenated_signal, color="blue", linewidth=1.0, linestyle="-")
    plt.xlabel("second")
    plt.title("Concatenated Signal")
    plt.minorticks_on()
    plt.grid(which="both", axis="x", alpha=0.8, linestyle="--", linewidth=1)
    plt.tight_layout()
    if not os.path.exists(os.path.join(args.fig_root)):
        os.mkdir(os.path.join(args.fig_root))
    plt.savefig(
        os.path.join(
            args.fig_root,
            args.TARGET_NAME,
            "{}_concatenated_signal.png".format(label_name),
        ),
        dpi=300,
    )
    plt.cla()
    plt.clf()
    plt.close()


def merge_dataframes_with_overlap(df1, df2, overlap=200):
    """
    2つのDataFrameを結合する際、200データを重ね、0.5から遠い値を採用する。
    """
    # 入力データの長さを確認
    assert (
        df1.shape[0] == 400 and df2.shape[0] == 400
    ), "両方のデータフレームは400行である必要があります。"

    # ① 重なる200行の部分を取得
    overlap_df1 = df1.iloc[-overlap:].to_numpy()
    overlap_df2 = df2.iloc[:overlap].to_numpy()

    # ② 0.5 から遠い値を採用
    merged_overlap = np.where(
        abs(overlap_df1 - 0.5) > abs(overlap_df2 - 0.5), overlap_df1, overlap_df2
    )

    # ③ 最終的なDataFrameを作成
    merged_data = np.vstack(
        [
            df1.iloc[:-overlap].to_numpy(),  # df1 の最初の 200 行
            merged_overlap,  # 重なった部分（0.5から遠い値を選択）
            df2.iloc[overlap:].to_numpy(),  # df2 の後半（200～400）
        ]
    )

    return pd.DataFrame(merged_data)


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
    TARGET_CHANNEL_15CH = "ch_1"
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
    heartbeats = []
    all_reconx = pd.DataFrame()
    sample_rate = 500
    sample_num = args.datalength
    datanum = 10
    rpeaks = rpeaks[:datanum]
    with torch.no_grad():
        target_peak = 0
        # 最終的に全てのデータを格納するリスト
        all_length = rpeaks[datanum - 1] + 200
        all_data = np.full((all_length, 8), 0.5)
        for j, (x, label_name) in enumerate(test_loader):
            x = x.to(device)
            recon_x, mean, log_var, z = vae(x)

            ecg_ch = args.ecg_ch_num
            numplotfig = len(x)
            print(f"numplotfig{numplotfig}")
            recon_x = recon_x.view(-1, ecg_ch, datalength)

            # ❶ `recon_x` を NumPy に変換
            recon_x_np = recon_x.cpu().numpy()  # Tensor → NumPy
            recon_peak_position = 200  # 再構成波形のピーク位置
            xticks = np.linspace(0.0, 1.0 / sample_rate * sample_num, sample_num)
            for p in range(numplotfig):
                row_data = []  # 1つの波形セット（p単位の行データ）

                # R波ピークに合わせてシフトした波形を格納するリスト
                aligned_recon_waves = []

                for q in range(ecg_ch):
                    wave_data = recon_x_np[p, q]  # (400,) の1D配列

                    aligned_recon_waves.append(wave_data)  # シフト後の波形を格納
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

                # 8チャネル分を (400, 8) にする
                aligned_recon_waves = np.column_stack(aligned_recon_waves)

                # ❷ 4000データの適切な位置に配置
                start_idx = (
                    rpeaks[target_peak] - recon_peak_position
                )  # 開始インデックス
                end_idx = start_idx + 400  # 終了インデックス
                # print(f"start_idx: {start_idx}, end_idx: {end_idx}")
                # print(all_data[start_idx:end_idx])
                # print(all_data.shape)
                # 0.5から遠い値を採用しながら更新
                all_data[start_idx:end_idx] = merge_dataframes_with_farthest_from_05(
                    aligned_recon_waves, all_data[start_idx:end_idx]
                )
                target_peak = target_peak + 1
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
                # plt.show()
                plt.cla()
                plt.clf()
                plt.close()
            print(all_data)
        # 全データを NumPy 配列に変換し、行方向に結合
        all_data = np.vstack(all_data)  # (numplotfig * 400, ecg_ch)

        # DataFrame に変換
        all_df = pd.DataFrame(all_data)
        # all_dfのカラムを設定
        all_df.columns = ["A1", "A2", "V1", "V2", "V3", "V4", "V5", "V6"]

    # 推定に用いた15チャネルのデータを画像として保存
    plot_fig_15ch(df_15ch_pf.iloc[: len(all_df)], args, args.TARGET_NAME)
    # 推定した8チャネルのデータを画像として保存
    plot_fig_8ch(all_df, args, args.TARGET_NAME)

    plt.figure(figsize=(12, 6))
    plt.plot(all_df.iloc[:, 0])  # 最初の1行目をプロット
    plt.title("First Sample of Reconstructed ECG")
    plt.xlabel("Time Step")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()


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
    args.raw_datas_dir = RAW_DATA_DIR + "/sheet_sensor_csvdatas/{}".format(
        "for_estimate"
    )
    main(args)
