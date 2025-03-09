import os
import re
from re import A
import sys
import time
from tkinter import W
from matplotlib.transforms import Bbox
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import codecs
import time
import datetime
import neurokit2 as nk

# import pywt
import torchvision
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torch.autograd import Variable
from collections import defaultdict
from sklearn.manifold import TSNE
from scipy import signal
import matplotlib.cm as cm
import matplotlib

matplotlib.use("TkAgg")

import random
import pandas as pd

from models import VAE

# =================
import Dataset
from torch.utils.data import TensorDataset
from torchsummary import summary
from tqdm import tqdm
from time import sleep
import json
import matplotlib.ticker as ticker
import gc
import csv

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_dir)
from config.settings import (
    DATA_DIR,
    BASE_DIR,
    PROCESSED_DATA_DIR,
    OUTPUT_DIR,
    RAW_DATA_DIR,
    TEST_DIR,
    RATE,
    RATE_15CH,
    TIME,
    DATASET_MADE_DATE,
    OUTPUT_MAE_DIR,
)
from src.dataset_creation.Make_dataset_0120 import (
    CSVReader_16ch,
    linear_interpolation_resample_All,
    validate_integer_input,
    peak_sc_15ch,
    peak_sc_plot,
    peak_search_nk_15ch,
)

# =====================


def min_max(x, minth, maxth):
    min = minth  # x.min(axis=axis, keepdims=True)
    max = maxth  # x.max(axis=axis, keepdims=True)
    if (max - min) != 0:
        result = (x - min) / (max - min)
        return np.clip(result, 0, 1.0)
    else:
        return x * 0


def ecg_clean_df_15ch(df_15ch, rate):
    ecg_signal = df_15ch.copy()["ch_1"]
    # cleand_signal=nk.ecg_clean(ecg_signal,sampling_rate=500,method="neurokit")
    # print(cleaned_signal)
    # print(type(cleaned_signal))
    # plt.plot(cleaned_signal)
    # plt.plot(df_15ch["ch_1"])
    # plt.title("org")
    # plt.show()
    df_15ch_cleaned = pd.DataFrame()
    for i, column in enumerate(df_15ch.columns):
        # df0_mul.plot()
        # df1=df.iloc[:,i]
        ecg_signal = df_15ch[column].copy().values
        ecg_signal = nk.ecg_clean(ecg_signal, sampling_rate=rate, method="neurokit")
        df_15ch_cleaned[column] = ecg_signal
    fig = plt.figure(num=None, figsize=(12, 5), dpi=100, facecolor="w", edgecolor="k")
    axis_line_width = 2.0
    tick_label_size = 18
    # 最初のグラフ（8プロット）
    plot_time = np.arange(len(df_15ch)) / RATE_15CH
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(plot_time, df_15ch["ch_1"], label="ch_1")
    ax1.plot(plot_time, df_15ch_cleaned["ch_1"], label="filtered ch1")
    # ax1.legend(fontsize=12, ncol=1)
    ax1.legend(loc="upper right", fontsize=18, ncol=1, bbox_to_anchor=(1, 1))
    ax1.tick_params(labelsize=tick_label_size, direction="in")
    plt.xlim(3.8, 8)
    plt.ylim(-100, 200)
    for axis in ["top", "bottom", "left", "right"]:
        ax1.spines[axis].set_linewidth(axis_line_width)
    # ax2 = fig.add_subplot(2, 1, 2)
    # ax2.plot(df_15ch_cleaned["ch_1"],label='filtered ch1')
    # ax2.legend(loc='center left', fontsize=12, ncol=1, bbox_to_anchor=(1, 0.5))
    # ax2.tick_params(labelsize=tick_label_size,direction='in')
    # # print(type(df_15ch_cleaned))
    # plt.plot(df_15ch_cleaned["ch_1"])
    # plt.title("cleaned")
    # plt.savefig("taniguchi_filter.svg")
    plt.tight_layout()
    plt.show()
    return df_15ch_cleaned


def tensor_to_ndarray(tensor):
    if isinstance(tensor, torch.Tensor):
        z = tensor.detach().cpu().numpy()
        # print(z)
        return z
    else:
        raise TypeError("Input must be a torch.Tensor.")


def plot_fig_15ch_only(recon_x, datalength, args, batch_size_num, label_name):
    sample_rate = 500
    sample_num = args.datalength
    dt = 1 / sample_rate
    # print(sc)
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
    sample_num = args.datalength
    xticks = np.linspace(0.0, 1.0 / sample_rate * sample_num, sample_num)
    ecg_ch = args.ecg_ch_num
    recon_x2 = torch.reshape(
        recon_x, (-1, ecg_ch, datalength)
    )  # datalength*2じゃないはず
    for p in range(batch_size_num):
        print("p={}".format(p))
        # for p in range(2):
        for q in range(ecg_ch):
            plt.tight_layout()
            plt.plot(
                xticks,
                recon_x2[p][q].cpu().data.numpy(),
                color="red",
                linewidth=1.0,
                linestyle="-",
            )
            # plt.axvline(x=sc[p][0],color='black',linewidth=2,linestyle='--')
            # plt.axvline(x=sc[p][1],color='black',linewidth=2,linestyle='--')
            plt.xlabel("second")
            plt.axis("on")
            plt.minorticks_on()
            plt.grid(which="both", axis="x", alpha=0.8, linestyle="--", linewidth=1)
            # plt.legend(["predict","ECG"],bbox_to_anchor=(1.05,1),loc="upper left")
            plt.legend(
                ["predict", "ECG", "P_onset", "T_offset"],
                bbox_to_anchor=(1.05, 1),
                loc="upper left",
            )
            plt.title(label_name[p] + "_ch={}".format(ecg_ch_names[q]))
            plt.tight_layout()

            if not os.path.exists(
                os.path.join(
                    args.fig_root,
                )
            ):
                os.mkdir(os.path.join(args.fig_root, str(ts)))

            plt.savefig(
                # os.path.join(args.fig_root, str(ts),"epoch{:d}_iteration{:d}".format(epoch_num,iteration_num),"ch{:d}_train_x_xo_{:d}.png".format(q,p)),
                os.path.join(
                    args.fig_root,
                    args.TARGET_NAME,
                    "ch={}_15ch_only_test_x_xo_{}.png".format(
                        ecg_ch_names[q], label_name[p]
                    ),
                ),
                dpi=300,
            )
            # plt.show()
            plt.cla()
            plt.clf()
            plt.close()


def output_csv(file_path, file_name, data):
    dt = 1.0 / RATE
    time_tmp = np.arange(len(data)) * dt
    # time_data=pd.DataFrame(time,column="Time")
    time = pd.DataFrame()
    time["Time"] = time_tmp
    print(time)
    data = data.reset_index(drop=True)
    data_out = pd.concat([time, data], axis=1)
    print(data_out)
    data_out.to_csv(file_path + "/" + file_name, index=None)


def cut_heartbeats(ecg_data, center_idxs, file_path, time_length):
    create_directory_if_not_exists(file_path)
    range = int(time_length * RATE / 2)
    for i, center_idx in enumerate(center_idxs):
        data = ecg_data[center_idx - range : center_idx + range].copy()
        print("{}番目の心拍切り出し".format(i + 1))
        # print(data)
        file_name = "dataset_{}.csv".format(str(i).zfill(3))
        output_csv(file_path=file_path, file_name=file_name, data=data.copy())


def main(args):
    # ==============================================goto=======================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # ここから推定用データセット作成

    dir_path = args.raw_datas_dir
    csv_reader_16ch = CSVReader_16ch(dir_path)
    df_16ch = csv_reader_16ch.process_files()
    print(df_16ch)
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
    print("TARGET_CHNNEL_15chは")
    TARGET_CHANNEL_15CH = validate_integer_input()
    if reverse == "off":
        # sc_15ch = peak_sc_15ch(df_15ch_pf.copy(), RATE=RATE, TARGET=TARGET_CHANNEL_15CH)
        rpeaks, val = peak_search_nk_15ch(
            df_15ch_pf[TARGET_CHANNEL_15CH].copy(), RATE=RATE
        )
        print(rpeaks)
        print("iiiiiiiiiiiiii")
        peak_sc_plot(df_15ch_pf.copy(), RATE=RATE, TARGET=TARGET_CHANNEL_15CH)
        file_path = PROCESSED_DATA_DIR + f"/15ch_only/{args.TARGET_NAME}"
        cut_heartbeats(df_15ch_pf, rpeaks, file_path=file_path, time_length=0.8)

    else:
        df_15ch_reverse = df_15ch_pf.copy()
        df_15ch_reverse[TARGET_CHANNEL_15CH] = (-1) * df_15ch_pf.copy()[
            TARGET_CHANNEL_15CH
        ]
        # reverseを採用
        df_resample_15ch = df_15ch_reverse.copy()
        sc_15ch = peak_sc_15ch(
            df_15ch_reverse.copy(), RATE=RATE, TARGET=TARGET_CHANNEL_15CH
        )
        print(sc_15ch)
        print("aaaaaaaaaaaaaaaaa")
        peak_sc_plot(df_15ch_reverse.copy(), RATE=RATE, TARGET=TARGET_CHANNEL_15CH)

    # ここから推論

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
    print("15ch_only MODE::\n")
    test_dataset = Dataset.Dataset_setup_15ch_only(
        TARGET_NAME=args.TARGET_NAME,
        transform_type=args.transform_type,
        Dataset_name="15ch_only",
        dataset_num=10,
    )
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    print(len(test_dataset))
    pth = args.pth
    vae.load_state_dict(torch.load(pth, map_location=lambda storage, loc: storage))
    vae.eval()
    print(vae)
    # names = torchvision.models.feature_extraction.get_graph_node_names(vae)
    # extractor = torchvision.models.feature_extraction.create_feature_extractor(vae, ['encoder.MLP.AC0.weight', 'encoder.MLP.AC0.bias'])
    print(vae.state_dict().keys())
    print("===========================")
    print("encorder_Linear1")
    print(vae.encoder.linear_means.weight)
    print(vae.encoder.linear_means.bias)
    print("===========================")
    record_loss_eval = []
    min_test_loss = 2000

    z_temps = np.empty((0, latent_size))
    label_temps = []
    test_val_all = []
    print(test_loader)
    print(len(test_loader))
    with torch.no_grad():
        for j, (x, label_name) in enumerate(test_loader):
            print(j)
            print("------------------------")
            x = x.to(device)
            recon_x, mean, log_var, z = vae(x)
            print(x)
            print(len(x))
            print("suirondayo")
            print(z)
            print(recon_x)
            print(recon_x.shape)
            z_temp = tensor_to_ndarray(z)
            z_temps = np.append(z_temps, z_temp, axis=0)
            # print(label_name)
            for i in range(len(label_name)):
                label_temps.append(label_name[i])
            ecg_ch = args.ecg_ch_num
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

            # if j % args.print_every != -1:
            if j != -1:
                numplotfig = len(x)
                # print("numplotfig")
                # print(numplotfig)
                print("koko2")
                print(label_name)
                # plot_fig(numplotfig,recon_x,xo,datalength,ts,args,label_name)
                # plot_fig(numplotfig,x,xo,datalength,ts,args)
                # recon_x_cpu = recon_x.to("cpu").view(-1, datalength).detach().numpy()
                # df_recon_x = pd.DataFrame(recon_x_cpu)
                # df_recon_x.to_csv(os.path.join(args.fig_root, str(ts),"predict_"+str(args.target_set))+".csv",index=False,header=False)
                # df_recon_x.to_csv(os.path.join(args.fig_root, str(ts),"predict_"+str(args.TARGET_NAME))+".csv",index=False,header=False)

                print("koko3")
                print(recon_x.view(-1, datalength).shape)

                recon_x = recon_x.view(-1, ecg_ch, datalength)
                print(recon_x.shape)

                # test_val=cul_val(pt_index,acc_rmse)

                sample_rate = 500
                sample_num = args.datalength
                xticks = np.linspace(0.0, 1.0 / sample_rate * sample_num, sample_num)

                if numplotfig > 6:
                    numplotfig = 6
                for p in range(numplotfig):
                    for q in range(ecg_ch):
                        plt.subplot(4, 3, q + 1)
                        if args.conditional:
                            plt.text(
                                0,
                                0,
                                "c={:d}".format(c[p].item()),
                                color="black",
                                backgroundcolor="white",
                                fontsize=8,
                            )
                        # recon_x2 = torch.reshape(
                        #     recon_x, (-1, ecg_ch, datalength)
                        # )  # datalength*2じゃないはず
                        # print(recon_x2)
                        # print("uuuuuuuuuuuuu")
                        plt.plot(
                            xticks,
                            recon_x[p][q].cpu().data.numpy(),
                            color="red",
                            linewidth=1.0,
                            linestyle="-",
                        )
                        # plt.subplots_adjust(left=0.125,bottom=0.1,right=0.9,top=0.9,wspace=0.2,hspace=0.35)

                        # plt.axvline(x=sc_pt[p][0],color='black',linewidth=2,linestyle='--',label='P_onset')
                        # plt.axvline(x=sc_pt[p][1],color='black',linewidth=2,linestyle='--',label='T_offset')

                        plt.axis("on")
                        plt.minorticks_on()
                        plt.grid(
                            which="both",
                            axis="x",
                            alpha=0.8,
                            linestyle="--",
                            linewidth=1,
                        )
                        # plt.title('ch{}'.format(str(q)))
                        plt.title("ch={}".format(ecg_ch_names[q]), fontsize=5)

                    # if not os.path.exists(os.path.join(args.fig_root, str(ts))):
                    #     if not(os.path.exists(os.path.join(args.fig_root))):
                    #         os.mkdir(os.path.join(args.fig_root))
                    #     os.mkdir(os.path.join(args.fig_root, str(ts)))

                    plt.legend(
                        ["predict", "ECG", "P_onset", "T_offset"],
                        bbox_to_anchor=(1.05, 1),
                        loc="upper left",
                    )
                    # plt.suptitle(label_name[p]+'RMSE_all='+str(acc_rmse_per_batch[p]))
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

            # recon_xf=fourier(recon_x)
            print(j)
            # if j==0:
            print("z")
            print(z)
            batch_size_now = len(label_name)
            print(batch_size_now)
            # plot_fig_15ch_only(
            #     recon_x=recon_x,
            #     datalength=datalength,
            #     args=args,
            #     label_name=label_name,
            #     batch_size_num=batch_size_now,
            # )
            # plot_fig_test_name(recon_x=recon_x,xo=xo,datalength=datalength,ts=ts,args=args,batch_size_num=batch_size_now,label_name=label_name,acc=acc_rmse_per_batch_ch,pt_index=pt_index)
            # plot_fig_train(recon_x=recon_x,xo=xo,datalength=datalength,ts=ts,args=args,iteration_num=iteration,batch_size_num=batch_size_now,epoch_num=epoch)
        # plot_scatter_2d(z=z_temps,labels=label_temps,latent_size=latent_size,ts=ts,args=args)


def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--augumentation", type=str, default="")
    parser.add_argument("--Dataset_name", type=str, default="")
    parser.add_argument("--dataset_num", type=int, default=20)  # ICCEの際は15心拍分で

    parser.add_argument("--TARGET_NAME", type=str, default="yoshikura")
    # parser.add_argument("--dname", type = str ,default = "test")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--train_batch_size", type=int, default=4)  # default=256
    parser.add_argument("--val_batch_size", type=int, default=1)  # default=256
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--datalength", type=int, default=400)
    # parser.add_argument("--enc_convlayer_sizes", type=list, default=[[16, 1], [32, 1], [64, 2]]) #畳み込み層の設定　増やしすぎると過学習の可能性　前から２ペアずつ読み込む　２つ目の[]の第二引数はストライド
    # parser.add_argument("--enc_convlayer_sizes", type=list, default=[[15, 1], [30, 2],[60, 2],[120, 2],[240,2]])#[[入力,ストライド],]
    # parser.add_argument("--enc_convlayer_sizes", type=list, default=[[15, 1], [30, 2],[60, 2],[120,2],[240,2]])#[[入力,ストライド],]
    parser.add_argument(
        "--enc_convlayer_sizes",
        type=list,
        default=[[15, 1], [30, 2]],
    )  # [[入力,ストライド],]
    parser.add_argument(
        "--enc_fclayer_sizes", type=list, default=[6000, 500, 64]
    )  # 一番目の適切な値は畳み込み層次第　エラーメッセージをみて調整するのが早い
    # parser.add_argument("--dec_fclayer_sizes", type=list, default=[64, 512, 4800])#12chのとき
    parser.add_argument(
        "--dec_fclayer_sizes", type=list, default=[64, 512, 3200]
    )  # 8chのとき
    # parser.add_argument("--dec_convlayer_sizes", type=list, default=[[24,2],[12,1]]) #通常畳み込み層が続くが、出力が単純（三角波）なので畳み込み層があってもなくても結果がそこまで変わらなかったらしい
    parser.add_argument(
        "--dec_convlayer_sizes", type=list, default=[[16, 2], [8, 1]]
    )  # 通常畳み込み層が続くが、出力が単純（三角波）なので畳み込み層があってもなくても結果がそこまで変わらなかったらしい
    # parser.add_argument("--dec_convlayer_sizes", type=list, default=[]) #通常畳み込み層が続くが、出力が単純（三角波）なので畳み込み層があってもなくても結果がそこまで変わらなかったらしい
    parser.add_argument("--latent_size", type=int, default=2)

    parser.add_argument("--print_every", type=int, default=2000)
    parser.add_argument(
        "--fig_root", type=str, default=OUTPUT_DIR + "/" + f"figs_newref/15ch_only"
    )  # 学習過程及びテスト結果を出力するフォルダ
    parser.add_argument("--conditional", action="store_true")
    parser.add_argument("--train_off", action="store_false")

    parser.add_argument(
        "--pth", type=str, default=r"model_pth/vae_cross_z4.pth"
    )  # 学習モデルのファイル名指定
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument(
        "--loss_fn_type", type=str, default="mse"
    )  # ECGの正規化なしだと上手くいった。
    parser.add_argument("--beta", type=int, default=1)
    parser.add_argument("--alpha", type=int, default=300000)
    parser.add_argument("--dim_red", type=str, default="")
    parser.add_argument("--transform_type", type=str, default="normal")

    # parser.add_argument("--loss_fn_type", type=str, default='bce')
    parser.add_argument("--ecg_ch_num", type=int, default=8)
    parser.add_argument("--current_time", type=str)
    parser.add_argument(
        "--ave_data_flg", type=int, default=0
    )  # 平均心拍を利用するか否か
    args = parser.parse_args()
    create_directory_if_not_exists(args.fig_root + "/" + args.TARGET_NAME)
    args.reverse = "off"
    args.raw_datas_dir = RAW_DATA_DIR + "/sheet_sensor_csvdatas/{}".format("0304")
    main(args)
