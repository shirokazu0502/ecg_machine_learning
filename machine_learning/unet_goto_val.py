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
from torch.utils.tensorboard import SummaryWriter
from scipy.stats import pearsonr

matplotlib.use("TkAgg")

import random
import pandas as pd

from models_Unet import UNet1D

# =================
import Dataset
from torch.utils.data import DataLoader
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

# =====================
cmap = "tab10"


def extract_between_third_and_fourth_underscore(input_string):
    # 文字列をアンダースコア（_）で分割します
    parts = input_string.split("_")

    # 分割した結果が4つ以上の要素を持つことを確認します
    if len(parts) >= 4:
        # 3番目から4番目のアンダースコアの間の要素を取得します
        result = parts[3]
        return result
    else:
        # アンダースコアの数が足りない場合、Noneを返します
        return None


def hpf(d_in, sampling_rate, fp, fs):
    # """ high pass filter """
    # fp = 0.5   # 通過域端周波数[Hz] 入力引数に変更
    # fs = 0.1   # 阻止域端周波数[Hz] 入力引数に変更
    gpass = 1  # 通過域最大損失量[dB]
    gstop = 40  # 阻止域最小減衰量[dB]

    norm_pass = fp / (sampling_rate / 2)
    norm_stop = fs / (sampling_rate / 2)
    N, Wn = signal.cheb2ord(
        wp=norm_pass, ws=norm_stop, gpass=gpass, gstop=gstop, analog=0
    )
    b, a = signal.cheby2(N, gstop, Wn, "high")
    d_out = signal.filtfilt(b, a, d_in)
    return d_out


def lpf(d_in, sampling_rate, fp, fs):
    """low pass filter"""
    # fp = 30    # 通過域端周波数[Hz]→入力引数に変更
    # fs = 50    # 阻止域端周波数[Hz]→入力引数に変更
    gpass = 1  # 通過域最大損失量[dB]
    gstop = 40  # 阻止域最小減衰量[dB]

    norm_pass = fp / (sampling_rate / 2)
    norm_stop = fs / (sampling_rate / 2)
    N, Wn = signal.cheb2ord(
        wp=norm_pass, ws=norm_stop, gpass=gpass, gstop=gstop, analog=0
    )
    b, a = signal.cheby2(N, gstop, Wn, "low")
    d_out = signal.filtfilt(b, a, d_in)
    return d_out


def min_max_old(x):
    # print("x.shape")
    # print(type(x))
    min = x.min(axis=None, keepdims=True)
    max = x.max(axis=None, keepdims=True)
    if (max - min) != 0:
        result = (x - min) / (max - min)
        # print(result.shape)
        return result
    else:
        return x * 0


def min_max_2(x):
    # print("x.shape")
    # print(type(x))
    num = x.shape[0]
    print("num_of_")
    print(num)
    for i in range(num):
        min = x[i].min(axis=None, keepdims=True)
        max = x[i].max(axis=None, keepdims=True)
        if (max - min) != 0:
            # print(abs(max),abs(min))
            a = 0
            if abs(max) < abs(min):
                a = abs(min)
            else:
                a = abs(max)
            # print(a)
            x[i] = x[i] / (2.0 * a) + 0.5
    return x

    # print(result.shape)


def min_max(x, minth, maxth):
    min = minth  # x.min(axis=axis, keepdims=True)
    max = maxth  # x.max(axis=axis, keepdims=True)
    if (max - min) != 0:
        result = (x - min) / (max - min)
        return np.clip(result, 0, 1.0)
    else:
        return x * 0


def plot_fig(numplotfig, recon_x, xo, datalength, ts, args, label_name, ecg_ch_names):
    sample_rate = 500
    sample_num = datalength
    xticks = np.linspace(0.0, 1.0 / sample_rate * sample_num, sample_num)
    ecg_ch = args.ecg_ch_num
    # if(ecg_ch==12):
    #     ecg_ch_names=["A1","A2","A3","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
    # if(ecg_ch==8):
    #     ecg_ch_names=["A1","A2","V1","V2","V3","V4","V5","V6"]
    for p in range(numplotfig):

        for q in range(ecg_ch):
            # plt.subplot(4, 2, q+1)
            # plt.figure(figsize=(10,5))

            recon_x2 = torch.reshape(
                recon_x, (-1, ecg_ch, datalength)
            )  # datalength*2じゃないはず
            xo2 = torch.reshape(xo, (-1, ecg_ch, datalength))

            # print(xo.shape)
            # plt.plot(x[p].cpu().data.numpy()[0:datalength], color="green", linewidth=1.0, linestyle="-")
            # plt.plot(min_max_old(recon_x2[p][q].cpu().data.numpy()), color="red", linewidth=1.0, linestyle="-")
            plt.rcParams["font.size"] = 16
            plt.rcParams["xtick.direction"] = "in"
            plt.rcParams["ytick.direction"] = "in"
            # plt.plot(xticks,recon_x2[p][q].cpu().data.numpy(), color="0.1", linewidth=1.0, linestyle="-")
            plt.plot(
                xticks,
                recon_x2[p][q].cpu().data.numpy(),
                color="red",
                linewidth=1.0,
                linestyle="-",
            )
            # plt.plot(xo[p].cpu().data.numpy(), color="blue", linewidth=1.0, linestyle="-")#xoが（３２，４００）かたちになってる、
            plt.plot(
                xticks,
                xo2[p][q].cpu().data.numpy(),
                color="blue",
                linewidth=1.0,
                linestyle="-",
            )  # xoが（３２，４００）かたちになってる、
            # plt.subplots_adjust(left=0.125,bottom=0.1,right=0.9,top=0.9,wspace=0.2,hspace=0.35)
            # plt.ylim(0,1)
            plt.xlim(0.0, sample_num / sample_rate)
            plt.xlabel("second")
            plt.ylabel("amplitude")
            plt.axis("on")
            plt.minorticks_on()
            plt.grid(which="both", axis="x", alpha=0.8, linestyle="--", linewidth=1)
            plt.legend(
                ["predict", "ECG"],
                bbox_to_anchor=(0.60, 1),
                loc="upper left",
                fontsize=16,
                framealpha=1.0,
            )
            # plt.title(label_name[p])
            # plt.title(label_name[p]+'_ch{}'.format(str(q)))
            plt.title(label_name[p] + "{}".format(ecg_ch_names[q]))
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    args.fig_root, str(ts), "test{:d}_ch0{:d}.png".format(0, q)
                ),
                dpi=300,
            )
            # os.path.join("mask_test","takahashi",
            # label_name+"0.png")
            # ,dpi=300,
            # )
            # plt.show()
            plt.cla()
            plt.clf()
            plt.close()


def plot_fig_15ch_only(
    recon_x, datalength, ts, args, batch_size_num, label_name, pt_index
):
    sample_rate = 500
    # sample_num=750
    sample_num = args.datalength
    dt = 1 / sample_rate
    sc = pt_index * dt
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
    # input("")
    # sample_num=750
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

            if not os.path.exists(os.path.join(args.fig_root, str(ts))):
                os.mkdir(os.path.join(args.fig_root, str(ts)))

            plt.savefig(
                # os.path.join(args.fig_root, str(ts),"epoch{:d}_iteration{:d}".format(epoch_num,iteration_num),"ch{:d}_train_x_xo_{:d}.png".format(q,p)),
                os.path.join(
                    args.fig_root,
                    str(ts),
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

            # for p in range(numplotfig):

            #     for q in range(ecg_ch):
            #         if(ecg_ch==12):
            #             plt.subplot(4, 3, q+1)
            #         if(ecg_ch==8):
            #             plt.subplot(4, 2, q+1)
            #         #plt.figure(figsize=(10,5))

            #         # print("koko1")
            #         if args.conditional:
            #             plt.text(0, 0, "c={:d}".format(c[p].item()), color='black', backgroundcolor='white', fontsize=8)
            #         recon_x2 = torch.reshape(recon_x, (-1,ecg_ch,datalength))#datalength*2じゃないはず
            #         xo2=torch.reshape(xo,(-1,ecg_ch,datalength))
            #         plt.plot(xticks,recon_x2[p][q].cpu().data.numpy(), color="red", linewidth=1.0, linestyle="-")
            #         #plt.plot(xo[p].cpu().data.numpy(), color="blue", linewidth=1.0, linestyle="-")#xoが（３２，４００）かたちになってる、
            #         plt.plot(xticks,xo2[p][q].cpu().data.numpy(), color="blue", linewidth=1.0, linestyle="-")#xoが（３２，４００）かたちになってる、
            #         #plt.subplots_adjust(left=0.125,bottom=0.1,right=0.9,top=0.9,wspace=0.2,hspace=0.35)

            #         plt.axvline(x=sc_pt[p][0],color='black',linewidth=2,linestyle='--',label='P_onset')
            #         plt.axvline(x=sc_pt[p][1],color='black',linewidth=2,linestyle='--',label='T_offset')

            #         plt.axis('on')
            #         plt.minorticks_on()
            #         plt.grid(which="both", axis ="x", alpha=0.8,linestyle ="--",linewidth=1)
            #         # plt.title('ch{}'.format(str(q)))
            #         plt.title('{}_RMSE_per_batch_ch={}'.format(ecg_ch_names[q],str(acc_rmse_per_batch_ch[p][q])),fontsize=5)


def plot_fig_test_name_8ch_2row(
    recon_x,
    xo,
    datalength,
    ts,
    args,
    batch_size_num,
    label_name,
    acc,
    pt_index,
    ecg_ch_names,
):
    sample_rate = 500
    # sample_num=750
    sample_num = args.datalength
    dt = 1 / sample_rate
    sc = pt_index * dt
    # print(sc)
    # ecg_ch_names=["A1","A2","A3","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
    # input("")
    # sample_num=750
    sample_num = args.datalength
    xticks = np.linspace(0.0, 1.0 / sample_rate * sample_num, sample_num) - 0.4
    ecg_ch = args.ecg_ch_num
    recon_x2 = torch.reshape(
        recon_x, (-1, ecg_ch, datalength)
    )  # datalength*2じゃないはず
    xo2 = torch.reshape(xo, (-1, ecg_ch, datalength))
    print(label_name)
    print(recon_x)
    for p in range(batch_size_num):
        # print("p={}".format(p))
        # plt.figure(figsize=(5, 5))
        # plt.rcParams['xtick.labelsize'] = 20
        # plt.rcParams['ytick.labelsize'] = 20
        fig = plt.figure(figsize=(10, 5))
        # plt.tight_layout()
        # for p in range(2):
        for q in range(ecg_ch):
            # plt.subplot(3, 3, q+1)
            plt.rcParams["xtick.labelsize"] = 16
            plt.rcParams["ytick.labelsize"] = 16
            fig.add_subplot(2, 4, q + 1)
            # plt.subplot(4, 2, q+1)
            # plt.figure(figsize=(10,5))
            # recon_x2=recon_x.view(-1,datalength)
            # xo2=torch.reshape(xo,(-1,datalength))
            # print(xo.shape)
            # plt.plot(x[p].cpu().data.numpy()[0:datalength], color="green", linewidth=1.0, linestyle="-")
            # plt.plot(min_max_old(recon_x2[p][q].cpu().data.numpy()), color="red", linewidth=1.0, linestyle="-")
            # plt.tight_layout()
            plt.plot(
                xticks,
                recon_x2[p][q].cpu().data.numpy(),
                color="red",
                linewidth=1.0,
                linestyle="-",
            )
            # plt.plot(xo[p].cpu().data.numpy(), color="blue", linewidth=1.0, linestyle="-")#xoが（３２，４００）かたちになってる、
            plt.plot(
                xticks,
                xo2[p][q].cpu().data.numpy(),
                color="blue",
                linewidth=1.0,
                linestyle="-",
            )  # xoが（３２，４００）かたちになってる、
            # plt.subplots_adjust(left=0.125,bottom=0.1,right=0.9,top=0.9,wspace=0.2,hspace=0.35)
            # plt.ylim(0,1)

            # plt.axvline(x=sc[p][0],color='black',linewidth=2,linestyle='--')
            # plt.axvline(x=sc[p][1],color='black',linewidth=2,linestyle='--')

            plt.xlabel("(s)")
            plt.axis("on")
            plt.minorticks_on()
            plt.grid(which="both", axis="x", alpha=0.8, linestyle="--", linewidth=1)
            plt.xticks([-0.4, 0, 0.4], ["-0.4", "0.0", "0.4"])
            # plt.legend(["predict","ECG"],bbox_to_anchor=(1.05,1),loc="upper left")
            # plt.title('{} MAE={}'.format(ecg_ch_names[q],str(acc[p][q])),fontsize="8")
            plt.title("{}".format(ecg_ch_names[q]), fontsize="20")
            # plt.suptitle("Subjet")
            plt.tight_layout()

        # plt.suptitle(label_name[p])
        # plt.suptitle("II")
        # plt.legend(["predict","ECG","P_onset","T_offset"],bbox_to_anchor=(1.4,1),loc="upper left")
        # plt.legend(["Predicted waveform","Correct waveform"],bbox_to_anchor=(1.4,1),loc="upper left")
        # plt.tight_layout()
        # plt.subplots_adjust(left=0.125,
        #                     bottom=0.1,
        #                     right=0.9,
        #                     top=0.9,
        #                     wspace=0.2,
        #                     hspace=0.35)

        if not os.path.exists(os.path.join(args.fig_root, str(ts))):
            os.mkdir(os.path.join(args.fig_root, str(ts)))

        # plt.savefig(
        #     # os.path.join(args.fig_root, str(ts),"epoch{:d}_iteration{:d}".format(epoch_num,iteration_num),"ch{:d}_train_x_xo_{:d}.png".format(q,p)),
        #     os.path.join(args.fig_root, str(ts),"plot_all_channel_8ch_{}test_x_xo.png".format(label_name[p])),
        #     # os.path.join(args.fig_root, str(ts),"plot_all_channel_8ch_{}test_x_xo.SVG".format(label_name[p])),
        #     dpi=300,
        #     )
        plt.savefig(
            # os.path.join(args.fig_root, str(ts),"epoch{:d}_iteration{:d}".format(epoch_num,iteration_num),"ch{:d}_train_x_xo_{:d}.png".format(q,p)),
            # os.path.join("Outputs/rec_ecgs/plot_all_channel_8ch_{}test_x_xo.png".format(label_name[p])),
            os.path.join(
                args.fig_root,
                str(ts),
                "plot_all_channel_8ch_2row_{}test_x_xo.SVG".format(label_name[p]),
            ),
            dpi=300,
        )
        # plt.show()
        plt.cla()
        plt.clf()
        plt.close()


def plot_fig_test_name_8ch(
    recon_x,
    xo,
    datalength,
    ts,
    args,
    batch_size_num,
    label_name,
    acc,
    pt_index,
    ecg_ch_names,
):
    sample_rate = 500
    # sample_num=750
    sample_num = args.datalength
    dt = 1 / sample_rate
    sc = pt_index * dt
    # print(sc)
    # ecg_ch_names=["A1","A2","A3","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
    # input("")
    # sample_num=750
    sample_num = args.datalength
    xticks = np.linspace(0.0, 1.0 / sample_rate * sample_num, sample_num)
    ecg_ch = args.ecg_ch_num
    recon_x2 = torch.reshape(
        recon_x, (-1, ecg_ch, datalength)
    )  # datalength*2じゃないはず
    xo2 = torch.reshape(xo, (-1, ecg_ch, datalength))
    print(label_name)
    for p in range(batch_size_num):
        # print("p={}".format(p))
        # plt.figure(figsize=(5, 5))
        fig = plt.figure(figsize=(5, 5))
        # plt.tight_layout()
        # for p in range(2):
        for q in range(ecg_ch):
            # plt.subplot(3, 3, q+1)
            fig.add_subplot(3, 3, q + 1)
            # plt.subplot(4, 2, q+1)
            # plt.figure(figsize=(10,5))
            # recon_x2=recon_x.view(-1,datalength)
            # xo2=torch.reshape(xo,(-1,datalength))
            # print(xo.shape)
            # plt.plot(x[p].cpu().data.numpy()[0:datalength], color="green", linewidth=1.0, linestyle="-")
            # plt.plot(min_max_old(recon_x2[p][q].cpu().data.numpy()), color="red", linewidth=1.0, linestyle="-")
            # plt.tight_layout()
            plt.plot(
                xticks,
                recon_x2[p][q].cpu().data.numpy(),
                color="red",
                linewidth=1.0,
                linestyle="-",
            )
            # plt.plot(xo[p].cpu().data.numpy(), color="blue", linewidth=1.0, linestyle="-")#xoが（３２，４００）かたちになってる、
            plt.plot(
                xticks,
                xo2[p][q].cpu().data.numpy(),
                color="blue",
                linewidth=1.0,
                linestyle="-",
            )  # xoが（３２，４００）かたちになってる、
            # plt.subplots_adjust(left=0.125,bottom=0.1,right=0.9,top=0.9,wspace=0.2,hspace=0.35)
            # plt.ylim(0,1)

            # plt.axvline(x=sc[p][0],color='black',linewidth=2,linestyle='--')
            # plt.axvline(x=sc[p][1],color='black',linewidth=2,linestyle='--')

            plt.xlabel("(s)")
            plt.axis("on")
            plt.minorticks_on()
            plt.grid(which="both", axis="x", alpha=0.8, linestyle="--", linewidth=1)
            # plt.legend(["predict","ECG"],bbox_to_anchor=(1.05,1),loc="upper left")
            # plt.title('{} MAE={}'.format(ecg_ch_names[q],str(acc[p][q])),fontsize="8")
            plt.title("{}".format(ecg_ch_names[q]), fontsize="8")
            plt.suptitle("Subjet")
            plt.tight_layout()

        # plt.suptitle(label_name[p])
        # plt.suptitle("II")
        # plt.legend(["predict","ECG","P_onset","T_offset"],bbox_to_anchor=(1.4,1),loc="upper left")
        # plt.legend(["Predicted waveform","Correct waveform"],bbox_to_anchor=(1.4,1),loc="upper left")
        # plt.tight_layout()
        # plt.subplots_adjust(left=0.125,
        #                     bottom=0.1,
        #                     right=0.9,
        #                     top=0.9,
        #                     wspace=0.2,
        #                     hspace=0.35)

        if not os.path.exists(os.path.join(args.fig_root, str(ts))):
            os.mkdir(os.path.join(args.fig_root, str(ts)))

        # plt.savefig(
        #     # os.path.join(args.fig_root, str(ts),"epoch{:d}_iteration{:d}".format(epoch_num,iteration_num),"ch{:d}_train_x_xo_{:d}.png".format(q,p)),
        #     os.path.join(args.fig_root, str(ts),"plot_all_channel_8ch_{}test_x_xo.png".format(label_name[p])),
        #     # os.path.join(args.fig_root, str(ts),"plot_all_channel_8ch_{}test_x_xo.SVG".format(label_name[p])),
        #     dpi=300,
        #     )
        plt.savefig(
            # os.path.join(args.fig_root, str(ts),"epoch{:d}_iteration{:d}".format(epoch_num,iteration_num),"ch{:d}_train_x_xo_{:d}.png".format(q,p)),
            # os.path.join("Outputs/rec_ecgs/plot_all_channel_8ch_{}test_x_xo.png".format(label_name[p])),
            os.path.join(
                args.fig_root,
                str(ts),
                "plot_all_channel_8ch_{}test_x_xo.SVG".format(label_name[p]),
            ),
            dpi=300,
        )
        # plt.show()
        plt.cla()
        plt.clf()
        plt.close()


def plot_fig_test_name(
    recon_x,
    xo,
    datalength,
    ts,
    args,
    batch_size_num,
    label_name,
    acc,
    pt_index,
    ecg_ch_names,
):
    sample_rate = 500
    # sample_num=750
    sample_num = args.datalength
    dt = 1 / sample_rate
    sc = pt_index * dt
    # print(sc)
    # ecg_ch_names=["A1","A2","A3","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
    # input("")
    # sample_num=750
    sample_num = args.datalength
    xticks = np.linspace(0.0, 1.0 / sample_rate * sample_num, sample_num)
    ecg_ch = args.ecg_ch_num
    recon_x2 = torch.reshape(
        recon_x, (-1, ecg_ch, datalength)
    )  # datalength*2じゃないはず
    xo2 = torch.reshape(xo, (-1, ecg_ch, datalength))
    for p in range(batch_size_num):
        print("p={}".format(p))
        # for p in range(2):

        for q in range(ecg_ch):
            # plt.subplot(4, 2, q+1)
            # plt.figure(figsize=(10,5))

            # recon_x2=recon_x.view(-1,datalength)
            # xo2=torch.reshape(xo,(-1,datalength))

            # print(xo.shape)
            # plt.plot(x[p].cpu().data.numpy()[0:datalength], color="green", linewidth=1.0, linestyle="-")
            # plt.plot(min_max_old(recon_x2[p][q].cpu().data.numpy()), color="red", linewidth=1.0, linestyle="-")
            plt.tight_layout()
            plt.plot(
                xticks,
                recon_x2[p][q].cpu().data.numpy(),
                color="red",
                linewidth=1.0,
                linestyle="-",
            )
            # plt.plot(xo[p].cpu().data.numpy(), color="blue", linewidth=1.0, linestyle="-")#xoが（３２，４００）かたちになってる、
            plt.plot(
                xticks,
                xo2[p][q].cpu().data.numpy(),
                color="blue",
                linewidth=1.0,
                linestyle="-",
            )  # xoが（３２，４００）かたちになってる、
            # plt.subplots_adjust(left=0.125,bottom=0.1,right=0.9,top=0.9,wspace=0.2,hspace=0.35)
            # plt.ylim(0,1)

            plt.axvline(x=sc[p][0], color="black", linewidth=2, linestyle="--")
            plt.axvline(x=sc[p][1], color="black", linewidth=2, linestyle="--")

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
            plt.title(
                label_name[p] + "_acc={}_ch={}".format(str(acc[p][q]), ecg_ch_names[q])
            )
            plt.tight_layout()

            if not os.path.exists(os.path.join(args.fig_root, str(ts))):
                os.mkdir(os.path.join(args.fig_root, str(ts)))

            plt.savefig(
                # os.path.join(args.fig_root, str(ts),"epoch{:d}_iteration{:d}".format(epoch_num,iteration_num),"ch{:d}_train_x_xo_{:d}.png".format(q,p)),
                os.path.join(
                    args.fig_root,
                    str(ts),
                    "ch={}_test_x_xo_{}_MAE={}.png".format(
                        ecg_ch_names[q], label_name[p], str(acc[p][q])
                    ),
                ),
                # os.path.join(args.fig_root, str(ts),"ch={}_test_x_xo_{}_MAE={}.SVG".format(ecg_ch_names[q],label_name[p],str(acc[p][q]))),
                dpi=300,
            )
            # plt.show()
            plt.cla()
            plt.clf()
            plt.close()


def loss_fn_unet_mse(recon_x, x):
    """
    U-Net における回帰用 MSE loss（単純な再構成目的）
    """
    mse = torch.nn.MSELoss(reduction="mean")
    return mse(recon_x, x)


def loss_fn_mse_PRT(
    recon_x, x, mean, log_var, datalength, args, current_weight, pt_index
):
    weight_P = current_weight["P"]
    weight_R = current_weight["R"]
    weight_T = current_weight["T"]
    Q_peaks = pt_index[:, 5]
    S_peaks = pt_index[:, 6]

    # new_length=datalength-(index2-index1)
    batch_size = x.shape[0]
    MSE = torch.nn.MSELoss(reduction="sum")
    MSE_loss = 0.0
    for i in range(batch_size):
        index1 = Q_peaks[i]
        index2 = S_peaks[i]
        # print(index1)
        # print(index2)
        index1 = 170  # 　一旦はこれでやらせてみる 0401
        index2 = 230
        recon_x_before_R1 = recon_x.view(-1, args.ecg_ch_num, datalength)[i, :, :index1]
        recon_x_after_R2 = recon_x.view(-1, args.ecg_ch_num, datalength)[i, :, index2:]
        x_before_R1 = x.view(-1, args.ecg_ch_num, datalength)[i, :, :index1]
        x_after_R2 = x.view(-1, args.ecg_ch_num, datalength)[i, :, index2:]
        recon_x_R = recon_x.view(-1, args.ecg_ch_num, datalength)[i, :, index1:index2]
        x_R = x.view(-1, args.ecg_ch_num, datalength)[i, :, index1:index2]
        MSE_loss_1 = MSE(
            recon_x_before_R1, x_before_R1
        )  # 重みを小さくするR波より前のインデックスでのMSE（sum）
        MSE_loss_2 = MSE(
            recon_x_after_R2, x_after_R2
        )  # 重みを小さくするR波より後ろのインデックスでのMSE（sum）
        MSE_loss_R = MSE(
            recon_x_R, x_R
        )  # 重みを小さくするR波（０．０４秒）のMSE（sum）
        MSE_loss_keep = (
            MSE_loss_1 * weight_P + MSE_loss_2 * weight_T + MSE_loss_R * weight_R
        )  # （インデックス長さ×チャネル数）で割る。まだバッチの数では割れていない。
        MSE_loss = MSE_loss + MSE_loss_keep
    # print(MSE_loss/args.ecg_ch_num/datalength/batch_size)
    # input("mse_loss_qs")
    MSE_loss = MSE_loss / args.ecg_ch_num / datalength / batch_size
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    alpha = args.alpha
    beta = args.beta
    # print(MSE_loss_1)#重みを小さくするR波より前のインデックスでのMSE（sum）
    return (
        (alpha * MSE_loss + beta * KLD / batch_size),
        alpha * MSE_loss,
        beta * KLD / batch_size,
    )
    # return (beta*KLD) / batch_size, MSE_loss/batch_size, KLD/batch_size


def noise_make(mean, scale, datanum, ch_num):
    rnd = np.random.normal(loc=mean, scale=scale, size=datanum * ch_num)
    rnd = rnd.reshape(-1, ch_num, datanum)


def tensor_to_ndarray(tensor):
    if isinstance(tensor, torch.Tensor):
        z = tensor.detach().cpu().numpy()
        # print(z)
        return z
    else:
        raise TypeError("Input must be a torch.Tensor.")


def extract_pos_name(string):
    # 人物名のパターンを定義します（例: "姓_スポーツ_数字" の形式）
    # pattern = r'(\w+)_\w+_\w+_\w+'
    # pattern = r'(\w+)_\w+_\w+_\w+_\w+'
    pattern = r"\w+_\w+_\w+_(\w+)_\w+"

    # 文字列中の人物名を抽出します
    match = re.search(pattern, string)

    if match:
        # 姓を取得します
        last_name = match.group(1)

        # 人物名を返します
        print(last_name)
        return last_name
    else:
        # 人物名が見つからない場合はNoneを返します
        return None


def extract_person_name(string):
    # 人物名のパターンを定義します（例: "姓_スポーツ_数字" の形式）
    # pattern = r'(\w+)_\w+_\w+_\w+'
    # pattern = r'(\w+)_\w+_\w+_\w+_\w+'
    pattern = r"(\w+)_\w+"

    # 文字列中の人物名を抽出します
    match = re.search(pattern, string)
    # print(string)

    if match:
        # 姓を取得します
        last_name = match.group(1)

        # 人物名を返します
        return last_name
    else:
        # 人物名が見つからない場合はNoneを返します
        return None


class RMSELoss(torch.nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.mse = torch.nn.MSELoss(reduction=reduction)
        print(reduction)

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))


class MAE(
    torch.nn.Module
):  # MSEの個々の損失の平方根はMAEと同じなる。reduction=Noneの時だけ
    def __init__(self, reduction="none"):
        super().__init__()
        self.mse = torch.nn.MSELoss(reduction=reduction)  # MSELossの
        print(reduction)

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))


class MAE_2(
    torch.nn.Module
):  # 上のMAEと書き方は違うけど同じ値になった。こっちのが一般的な書き方。
    def __init__(self, reduction="none"):
        super().__init__()
        self.l1_loss = torch.nn.L1Loss(reduction=reduction)

    def forward(self, yhat, y):
        return self.l1_loss(yhat, y)


def cul_val_no_pt(acc):
    batch_size = acc.shape[0]
    acc_list = []
    for i in range(batch_size):
        acc_pt_12ch = torch.mean(acc[i, :, :], dim=(0, 1))
        # print("pwave_onset{},twave_offset{}".format(pt[i,0],pt[i,1]))
        # acc_pt=acc[i,:,pt[i,0]:pt[i,1]]
        acc_pt_12ch = acc_pt_12ch.to("cpu").detach().numpy()
        # print(acc_pt_12ch)
        acc_list.append(acc_pt_12ch)
    # print(acc_list)
    # print(acc_list[0])
    return acc_list


def cul_val(pt, acc):
    batch_size = acc.shape[0]
    acc_list = []
    for i in range(batch_size):
        acc_pt_12ch = torch.mean(acc[i, :, pt[i, 0] : pt[i, 1]], dim=(0, 1))
        # print("pwave_onset{},twave_offset{}".format(pt[i,0],pt[i,1]))
        # acc_pt=acc[i,:,pt[i,0]:pt[i,1]]
        acc_pt_12ch = acc_pt_12ch.to("cpu").detach().numpy()
        # print(acc_pt_12ch)
        acc_list.append(acc_pt_12ch)
    # print(acc_list)
    # print(acc_list[0])
    return acc_list
    # print(np.concatenate(acc_list))


def cul_val_per_12ch_no_pt(acc):
    batch_size = acc.shape[0]
    acc_list = []
    for i in range(batch_size):
        acc_pt_12ch = torch.mean(acc[i, :, :], dim=(1))
        # print("pwave_onset{},twave_offset{}".format(pt[i,0],pt[i,1]))
        # acc_pt=acc[i,:,pt[i,0]:pt[i,1]]
        acc_pt_12ch = acc_pt_12ch.to("cpu").detach().numpy()
        # print(acc_pt_12ch)
        acc_list.append(acc_pt_12ch)
    # print(acc_list)
    # print(acc_list[0])
    return acc_list
    # print(np.concatenate(acc_list))


def cul_val_per_12ch(pt, acc):
    batch_size = acc.shape[0]
    acc_list = []
    for i in range(batch_size):
        acc_pt_12ch = torch.mean(acc[i, :, pt[i, 0] : pt[i, 1]], dim=(1))
        # print("pwave_onset{},twave_offset{}".format(pt[i,0],pt[i,1]))
        # acc_pt=acc[i,:,pt[i,0]:pt[i,1]]
        acc_pt_12ch = acc_pt_12ch.to("cpu").detach().numpy()
        # print(acc_pt_12ch)
        acc_list.append(acc_pt_12ch)
    # print(acc_list)
    # print(acc_list[0])
    return acc_list
    # print(np.concatenate(acc_list))


def write_to_csv(file_path, data):
    # ファイルが存在するか確認
    file_exists = os.path.exists(file_path)

    # CSVファイルを開く
    with open(file_path, "a", newline="") as csvfile:
        fieldnames = [
            "TARGET_NAME",
            "MAE_all",
            "MAE_A1",
            "MAE_A2",
            "MAE_V1",
            "MAE_V2",
            "MAE_V3",
            "MAE_V4",
            "MAE_V5",
            "MAE_V6",
            "pearson_score",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # ファイルが存在しない場合、ヘッダを書き込む
        if not file_exists:
            writer.writeheader()

        # データを書き込む
        writer.writerow(data)


def save_csv2(data, ts, args, label_name, data_rec_or_xo):
    """
    Args:
    - data: 入力データ（batch_size×8×400の形状）
    - output_path: 出力先のディレクトリのパス
    - output_filename: 出力ファイル名
    """
    # 入力データの形状が正しいことを確認
    # recon_x2 = torch.reshape(recon_x, (-1,ecg_ch,datalength))#datalength*2じゃないはず
    data = torch.reshape(data, (-1, 8, args.datalength))
    data_np = data.cpu().numpy() if data.is_cuda else data.numpy()
    assert len(data.shape) == 3, "入力データの形状が不正です。"

    # ディレクトリが存在しない場合は作成
    # output_path = os.path.join(args.fig_root, str(ts),"waveforms")
    # output_path = os.path.join("leave_1_out_{}_R_weight_{}_augumentation={}".format(args.Dataset_name,str(args.loss_pt_on_off_R_weight),args.augumentation),"waveforms","{}".format(args.TARGET_NAME))
    output_path = os.path.join(
        OUTPUT_DIR
        + "/{}/PRTweight_{}_{}_{}/Waveforms".format(
            args.Dataset_name,
            str(args.loss_pt_on_off_P_weight),
            str(args.loss_pt_on_off_R_weight),
            str(args.loss_pt_on_off_T_weight),
        ),
        "{}".format(args.TARGET_NAME),
    )
    os.makedirs(output_path, exist_ok=True)

    # batch_sizeを取得
    batch_size = data_np.shape[0]
    # ファイルのパスを作成
    for p in range(batch_size):
        if data_rec_or_xo == "recon_x":
            # output_file = os.path.join(args.fig_root, str(ts),"waveforms","{}_reconx.csv".format(label_name[p]))
            output_file = os.path.join(
                output_path, "{}_reconx.csv".format(label_name[p])
            )
        else:
            # output_file = os.path.join(args.fig_root, str(ts),"waveforms","{}_xo.csv".format(label_name[p]))
            output_file = os.path.join(output_path, "{}_xo.csv".format(label_name[p]))
        df_data = pd.DataFrame(data_np[p])
        df_data = df_data.T
        new_columns = ["A1", "A2", "V1", "V2", "V3", "V4", "V5", "V6"]
        df_data.columns = new_columns
        df_data.to_csv(output_file, index=None)
    return output_path


def save_csv(data, ts, args, label_name, data_rec_or_xo):
    """
    Args:
    - data: 入力データ（batch_size×8×400の形状）
    - output_path: 出力先のディレクトリのパス
    - output_filename: 出力ファイル名
    """
    # 入力データの形状が正しいことを確認
    # recon_x2 = torch.reshape(recon_x, (-1,ecg_ch,datalength))#datalength*2じゃないはず
    data = torch.reshape(data, (-1, 8, args.datalength))
    data_np = data.cpu().numpy() if data.is_cuda else data.numpy()
    assert len(data.shape) == 3, "入力データの形状が不正です。"

    # ディレクトリが存在しない場合は作成
    # output_path = os.path.join(args.fig_root, str(ts),"waveforms")
    # output_path = os.path.join("leave_1_out_{}_R_weight_{}_augumentation={}".format(args.Dataset_name,str(args.loss_pt_on_off_R_weight),args.augumentation),"waveforms","{}".format(args.TARGET_NAME))
    if (
        args.loss_pt_on_off_R_weight != None
        and args.loss_pt_on_off_P_weight != None
        and args.loss_pt_on_off_T_weight != None
    ):
        output_path = os.path.join(
            "leave_1_out_{}_PRTweight_{}_{}_{}".format(
                args.Dataset_name,
                str(args.loss_pt_on_off_P_weight),
                str(args.loss_pt_on_off_R_weight),
                str(args.loss_pt_on_off_T_weight),
            ),
            "waveforms",
            "{}".format(args.TARGET_NAME),
        )
    else:
        output_path = os.path.join(
            "leave_1_out_{}_R_weight_{}".format(
                args.Dataset_name, str(args.loss_pt_on_off_R_weight)
            ),
            "waveforms",
            "{}".format(args.TARGET_NAME),
        )
    os.makedirs(output_path, exist_ok=True)

    # batch_sizeを取得
    batch_size = data_np.shape[0]
    # ファイルのパスを作成
    for p in range(batch_size):
        if data_rec_or_xo == "recon_x":
            # output_file = os.path.join(args.fig_root, str(ts),"waveforms","{}_reconx.csv".format(label_name[p]))
            output_file = os.path.join(
                output_path, "{}_reconx.csv".format(label_name[p])
            )
        else:
            # output_file = os.path.join(args.fig_root, str(ts),"waveforms","{}_xo.csv".format(label_name[p]))
            output_file = os.path.join(output_path, "{}_xo.csv".format(label_name[p]))
        df_data = pd.DataFrame(data_np[p])
        df_data = df_data.T
        new_columns = ["A1", "A2", "V1", "V2", "V3", "V4", "V5", "V6"]
        df_data.columns = new_columns
        df_data.to_csv(output_file, index=None)
    return output_path


def train_unet(
    model, train_loader, test_loader, optimizer, criterion, epochs, device, writer
):
    model.train()
    for epoch in range(epochs):
        total_train_loss = 0.0
        for x, xo, _, _ in train_loader:
            x, xo = x.to(device), xo.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, xo)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        print(
            f"Epoch [{epoch + 1}/{epochs}], Loss: {total_train_loss / len(train_loader):.4f}"
        )
        writer.add_scalar(
            "Loss/train_loss", total_train_loss / len(train_loader), epoch
        )
        model.eval()
        total_test_loss = 0.0
        with torch.no_grad():
            for x, xo, label_name, _ in test_loader:
                x, xo = x.to(device), xo.to(device)
                outputs = model(x)
                loss = criterion(outputs, xo)
                total_test_loss += loss.item()
        writer.add_scalar("Loss/test_loss", total_test_loss / len(test_loader), epoch)


def test_unet(model, test_loader, device, args, ts):
    model.eval()
    all_outputs = []
    all_targets = []
    label_temps = []
    test_val_all = []
    test_val_12ch_all = []
    pearson_scores = []
    datalength = args.datalength
    ecg_ch = args.ecg_ch_num
    if ecg_ch == 12:
        ecg_ch_names = [
            "Ⅰ",
            "Ⅱ",
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
    if ecg_ch == 8:
        ecg_ch_names = ["Ⅰ", "Ⅱ", "V1", "V2", "V3", "V4", "V5", "V6"]
    with torch.no_grad():
        for j, (x, xo, label_name, pt_index) in enumerate(test_loader):
            x, xo = x.to(device), xo.to(device)
            recon_x = model(x)
            # all_outputs.append(recon_x.cpu().numpy())
            # all_targets.append(xo.cpu().numpy())
            for i in range(len(label_name)):
                label_temps.append(label_name[i])
            numplotfig = len(x)
            print(label_name)
            recon_x_cpu = recon_x.to("cpu").view(-1, datalength).detach().numpy()
            df_recon_x = pd.DataFrame(recon_x_cpu)
            print(recon_x.view(-1, datalength).shape)
            # mse_loss=torch.nn.MSELoss(reduction="none")
            # acc_mse=mse_loss(recon_x.view(-1,ecg_ch, datalength), xo.view(-1,ecg_ch, datalength))
            # rmse_loss=RMSELoss(reduction="none")#RMSELoss関数を使っているけどreducutionがnoneやから
            # mae_loss=MAE(reduction="none")#RMSELoss関数を使っているけどreducutionがnoneやから
            mae_loss = MAE_2(
                reduction="none"
            )  # RMSELoss関数を使っているけどreducutionがnoneやから
            # mae_loss_2=MAE_2(reduction="none")#RMSELoss関数を使っているけどreducutionがnoneやから
            # acc_rmse=rmse_loss(recon_x.view(-1,ecg_ch, datalength), xo.view(-1,ecg_ch, datalength))
            recon_x = recon_x.view(-1, ecg_ch, datalength)
            xo = xo.view(-1, ecg_ch, datalength)
            # acc_rmse=rmse_loss(recon_x,xo)
            acc_mae = mae_loss(recon_x, xo)
            # ここに振幅の比の計算を入れたい

            if args.loss_pt_on_off == "off":
                # test_val=cul_val_no_pt(acc_rmse)
                test_val = cul_val_no_pt(acc_mae)
            else:
                # test_val=cul_val(pt_index,acc_mse)
                test_val = cul_val(pt_index, acc_mae)

            test_val = [array_item.item() for array_item in test_val]
            test_val = np.array(test_val, dtype=np.float32)
            test_val_all = test_val_all + test_val.tolist()

            if (
                args.loss_pt_on_off == "off"
            ):  # 精度評価でptの範囲だけを計算するかどうか。offで０．８秒間全体を評価計算する。
                test_val_12ch = cul_val_per_12ch_no_pt(acc_mae)
            else:
                test_val_12ch = cul_val_per_12ch(pt_index, acc_mae)

            test_val_12ch = np.array(
                [array_item.tolist() for array_item in test_val_12ch],
                dtype=np.float32,
            )
            test_val_12ch = test_val_12ch.reshape(-1, args.ecg_ch_num)
            test_val_12ch_all = test_val_12ch_all + test_val_12ch.tolist()

            recon_x_np = recon_x.view(-1, datalength).cpu().numpy().astype(np.float64)
            xo_np = xo.view(-1, datalength).cpu().numpy().astype(np.float64)
            r, _ = pearsonr(
                recon_x_np.ravel(),
                xo_np.ravel(),
            )
            pearson_scores.append(r)

            sample_rate = 500
            sc_pt = pt_index / sample_rate
            sample_num = args.datalength
            xticks = np.linspace(0.0, 1.0 / sample_rate * sample_num, sample_num)

            if numplotfig > 6:
                numplotfig = 6
            batch_size_now = xo.shape[0]

            plot_fig_test_name_8ch_2row(
                recon_x=recon_x,
                xo=xo,
                datalength=datalength,
                ts=ts,
                args=args,
                batch_size_num=batch_size_now,
                label_name=label_name,
                acc=test_val_12ch,
                pt_index=pt_index,
                ecg_ch_names=ecg_ch_names,
            )
            output_path = save_csv2(
                data=recon_x,
                args=args,
                ts=ts,
                label_name=label_name,
                data_rec_or_xo="recon_x",
            )
            output_path = save_csv2(
                data=xo,
                args=args,
                ts=ts,
                label_name=label_name,
                data_rec_or_xo="xo",
            )

    return test_val_all, test_val_12ch_all, pearson_scores


def main(args):
    # ==============================================goto=======================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    writer = SummaryWriter(log_dir=f"runs/{args.current_time}/{args.TARGET_NAME}")
    # PからTまでそれぞれにおけるデータセット作成
    # train_dataset_dict = {}
    # test_dataset_dict = {}
    # train_dataset_dict["P_train_dataset"], test_dataset_dict["P_test_dataset"] = (
    #     Dataset.Dataset_setup_8ch_pt_augmentation(
    #         TARGET_NAME=args.TARGET_NAME,
    #         transform_type=args.transform_type,
    #         Dataset_name=args.Dataset_name,
    #         dataset_num=args.dataset_num,
    #         DataAugumentation=args.p_augumentation,
    #         ave_data_flg=args.ave_data_flg,
    #     )
    # )
    # train_dataset_dict["R_train_dataset"], test_dataset_dict["R_test_dataset"] = (
    #     Dataset.Dataset_setup_8ch_pt_augmentation(
    #         TARGET_NAME=args.TARGET_NAME,
    #         transform_type=args.transform_type,
    #         Dataset_name=args.Dataset_name,
    #         dataset_num=args.dataset_num,
    #         DataAugumentation=args.r_augumentation,
    #         ave_data_flg=args.ave_data_flg,
    #     )
    # )
    # train_dataset_dict["T_train_dataset"], test_dataset_dict["T_test_dataset"] = (
    #     Dataset.Dataset_setup_8ch_pt_augmentation(
    #         TARGET_NAME=args.TARGET_NAME,
    #         transform_type=args.transform_type,
    #         Dataset_name=args.Dataset_name,
    #         dataset_num=args.dataset_num,
    #         DataAugumentation=args.t_augumentation,
    #         ave_data_flg=args.ave_data_flg,
    #     )
    # )
    # # 最終的にテストを行うためのデータセット(本質的にはどれも同じなので適当にR波を採用)
    # all_test_dataset = test_dataset_dict["R_test_dataset"]

    train_dataset, test_dataset = Dataset.Dataset_setup_8ch_pt_augmentation(
        TARGET_NAME=args.TARGET_NAME,
        transform_type=args.transform_type,
        Dataset_name=args.Dataset_name,
        dataset_num=args.dataset_num,
        DataAugumentation=args.p_augumentation,
        ave_data_flg=args.ave_data_flg,
    )
    print("gotooooooooooo_end")
    # ==================================================================
    # ts = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H%M%S')
    # ts = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H%M%S')+'_'+args.mode+'_targetset='+str(args.target_set)
    P_weight = args.loss_pt_on_off_P_weight
    R_weight = args.loss_pt_on_off_R_weight
    T_weight = args.loss_pt_on_off_T_weight
    PRT_weight = "{}{}{}".format(P_weight, R_weight, T_weight)
    # ts = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H%M%S')+'_Dataset_name='+args.Dataset_name+'_transform_type='+args.transform_type+'_'+args.mode+'_TARGETNAME='+str(args.TARGET_NAME)+'_beta='+str(args.beta)+'_alpha='+str(args.alpha)+'_loss_pt_on_off='+args.loss_pt_on_off
    ts = (
        datetime.datetime.fromtimestamp(time.time()).strftime("%Y%m%d_%H%M%S")
        + "_Dataset_name="
        + args.Dataset_name
        + "_transform_type="
        + args.transform_type
        + "_"
        + args.mode
        + "_TARGETNAME="
        + str(args.TARGET_NAME)
        + "_beta="
        + str(args.beta)
        + "_alpha="
        + str(args.alpha)
        + "_loss_pt_on_off="
        + args.loss_pt_on_off
        + "PRT="
        + PRT_weight
        + "_augument="
        + args.p_augumentation
        + "_"
        + args.r_augumentation
        + "_"
        + args.t_augumentation
    )

    if not os.path.exists(os.path.join(args.fig_root, str(ts))):
        if not (os.path.exists(os.path.join(args.fig_root))):
            os.mkdir(os.path.join(args.fig_root))
        os.mkdir(os.path.join(args.fig_root, str(ts)))

    with open(os.path.join(args.fig_root, str(ts), "args.json"), mode="w") as f:
        json.dump(args.__dict__, f, indent=4)

    datalength = args.datalength
    transform = transforms.Compose([transforms.ToTensor()])

    ecg_ch = args.ecg_ch_num
    if ecg_ch == 12:
        ecg_ch_names = [
            "Ⅰ",
            "Ⅱ",
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
    if ecg_ch == 8:
        ecg_ch_names = ["Ⅰ", "Ⅱ", "V1", "V2", "V3", "V4", "V5", "V6"]

    # overlap オーバーラップ幅はSTEPで指定
    STEP = 1
    DSRATE = 1

    # input()
    # summary(unet, input_size=(1,16 , 384))
    in_channels = 15
    out_channels = 8
    base_filters = 64
    # common_kwargs = {
    #     "datalength": datalength,
    #     "enc_convlayer_sizes": args.enc_convlayer_sizes,
    #     "dec_convlayer_sizes": args.dec_convlayer_sizes,
    #     "in_channels": in_channels,
    #     "out_channels": out_channels,
    # }
    common_kwargs = {
        "in_channels": in_channels,
        "out_channels": out_channels,
        "base_filters": base_filters,
    }
    # unet_dict = {
    #     "P": UNet1D(**common_kwargs).to(device),
    #     "R": UNet1D(**common_kwargs).to(device),
    #     "T": UNet1D(**common_kwargs).to(device),
    # }
    unet = UNet1D(**common_kwargs).to(device)
    if args.mode == "train":
        print("TRAINING MODE::\n")

        train_loader = DataLoader(
            dataset=train_dataset, batch_size=args.train_batch_size, shuffle=True
        )
        # dataset=train_dataset, batch_size=args.train_batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

        optimizer = torch.optim.Adam(unet.parameters(), lr=args.learning_rate)

        logs = defaultdict(list)
        label_temps = []
        train_acc_temps = []
        test_acc_temps = []

        # 訓練
        print("Training U-Net...")
        train_unet(
            unet,
            train_loader,
            test_loader,
            optimizer,
            loss_fn_unet_mse,
            args.epochs,
            device,
            writer,
        )

        torch.save(
            unet.state_dict(),
            os.path.join("model_pth", "unet.pth"),
        )
        torch.save(
            unet.state_dict(),
            os.path.join(args.fig_root, str(ts), "unet.pth"),
        )

        writer.close()

        if args.loss_fn_type == "bce":
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.plot(logs["loss"], color="blue")
            plt.plot(logs["bce"], color="red")
            plt.plot(logs["kdl"], color="green")
            plt.legend(["loss_all", "bce", "kdl"])
            plt.savefig(
                os.path.join(args.fig_root, str(ts), "loss_epoch_all.png"), dpi=300
            )
            # plt.show()
            plt.close()

            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.plot(logs["loss"], color="blue")
            plt.legend(["loss"])
            plt.savefig(
                os.path.join(args.fig_root, str(ts), "loss_epoch_epoch.png"), dpi=300
            )
            # torch.save(unet.state_dict(), args.pth)
            plt.close()

    elif args.mode == "test":
        print("TEST MODE::\n")
        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
        print("aaaaaaaaaaaaaaaaaa")
        print(test_loader)
        unet.load_state_dict(
            torch.load(
                os.path.join("model_pth", "unet.pth"),
                map_location=lambda storage, loc: storage,
            )
        )
        test_val_all, test_val_12ch_all, pearson_scores = test_unet(
            unet, test_loader, device, args, ts
        )

        test_val_all = np.array(test_val_all)
        test_val_mean = np.mean(test_val_all)
        test_val_12ch_all = np.array(test_val_12ch_all)
        print(test_val_12ch_all.shape)
        test_val_12ch_mean = np.mean(test_val_12ch_all, axis=0)
        print(test_val_12ch_mean.shape)
        pearson_score = np.mean(pearson_scores)
        data_to_write = {
            "TARGET_NAME": args.TARGET_NAME,
            "MAE_all": test_val_mean,
            "MAE_A1": test_val_12ch_mean[0],
            "MAE_A2": test_val_12ch_mean[1],
            "MAE_V1": test_val_12ch_mean[2],
            "MAE_V2": test_val_12ch_mean[3],
            "MAE_V3": test_val_12ch_mean[4],
            "MAE_V4": test_val_12ch_mean[5],
            "MAE_V5": test_val_12ch_mean[6],
            "MAE_V6": test_val_12ch_mean[7],
            "pearson_score": pearson_score,
        }
        # write_to_csv(file_path='cross_val_results/Datasets={},MAE={}.csv'.format(args.Dataset_name,args.loss_pt_on_off),data=data_to_write)
        output_file = os.path.join(
            args.mae_folder
            + "/MAE_leave_1_out_{}_PRTweight_{}_{}_{}_augumentation={}.csv".format(
                args.Dataset_name,
                str(args.loss_pt_on_off_P_weight),
                str(args.loss_pt_on_off_R_weight),
                str(args.loss_pt_on_off_T_weight),
                args.p_augumentation,
                args.r_augumentation,
                args.t_augumentation,
            )
        )
        write_to_csv(output_file, data=data_to_write)

    elif args.mode == "15ch_only":
        print("15ch_only MODE::\n")
        test_dataset = Dataset.Dataset_setup_12ch_pt_15ch_only(
            TARGET_NAME="osaka",
            transform_type=args.transform_type,
            Dataset_name="15ch_only",
            dataset_num=3,
        )
        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
        print(len(test_dataset))
        pth = args.pth
        unet.load_state_dict(
            torch.load(
                os.path.join("model_pth", pth),
                map_location=lambda storage, loc: storage,
            )
        )
        unet.eval()
        print(unet)
        # names = torchvision.models.feature_extraction.get_graph_node_names(unet)
        # extractor = torchvision.models.feature_extraction.create_feature_extractor(unet, ['encoder.MLP.AC0.weight', 'encoder.MLP.AC0.bias'])
        print(unet.state_dict().keys())
        print("===========================")
        print("encorder_Linear1")
        print(unet.encoder.linear_means.weight)
        print(unet.encoder.linear_means.bias)
        print("===========================")

        label_temps = []
        test_val_all = []
        with torch.no_grad():
            for j, (x, xo, label_name, pt_index) in enumerate(test_loader):
                x, xo = x.to(device), xo.to(device)
                recon_x, mean, log_var, z = unet(x)
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
                    print("koko2")
                    print(label_name)
                    recon_x_cpu = (
                        recon_x.to("cpu").view(-1, datalength).detach().numpy()
                    )
                    df_recon_x = pd.DataFrame(recon_x_cpu)

                    print("koko3")
                    print(recon_x.view(-1, datalength).shape)
                    # mse_loss=torch.nn.MSELoss(reduction="none")
                    # acc_mse=mse_loss(recon_x.view(-1,ecg_ch, datalength), xo.view(-1,ecg_ch, datalength))
                    rmse_loss = RMSELoss(reduction="none")
                    # acc_rmse=rmse_loss(recon_x.view(-1,ecg_ch, datalength), xo.view(-1,ecg_ch, datalength))

                    recon_x = recon_x.view(-1, ecg_ch, datalength)

                    # test_val=cul_val(pt_index,acc_rmse)

                    sample_rate = 500
                    sc_pt = pt_index / sample_rate
                    sample_num = args.datalength
                    xticks = np.linspace(
                        0.0, 1.0 / sample_rate * sample_num, sample_num
                    )

                    if numplotfig > 6:
                        numplotfig = 6
                    for p in range(numplotfig):

                        for q in range(ecg_ch):
                            plt.subplot(4, 3, q + 1)
                            # plt.figure(figsize=(10,5))

                            # print("koko1")
                            if args.conditional:
                                plt.text(
                                    0,
                                    0,
                                    "c={:d}".format(c[p].item()),
                                    color="black",
                                    backgroundcolor="white",
                                    fontsize=8,
                                )
                            recon_x2 = torch.reshape(
                                recon_x, (-1, ecg_ch, datalength)
                            )  # datalength*2じゃないはず
                            plt.plot(
                                xticks,
                                recon_x2[p][q].cpu().data.numpy(),
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
                                str(ts),
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
                plot_fig_15ch_only(
                    recon_x=recon_x,
                    datalength=datalength,
                    ts=ts,
                    args=args,
                    label_name=label_name,
                    batch_size_num=batch_size_now,
                    pt_index=pt_index,
                )
                # plot_fig_test_name(recon_x=recon_x,xo=xo,datalength=datalength,ts=ts,args=args,batch_size_num=batch_size_now,label_name=label_name,acc=acc_rmse_per_batch_ch,pt_index=pt_index)
                # plot_fig_train(recon_x=recon_x,xo=xo,datalength=datalength,ts=ts,args=args,iteration_num=iteration,batch_size_num=batch_size_now,epoch_num=epoch)
            # plot_scatter_2d(z=z_temps,labels=label_temps,latent_size=latent_size,ts=ts,args=args)

    else:
        print("DEBUG MODE::\n")
    # filelist_fp.close()


def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


if __name__ == "__main__":
    current_time = "0527_1540_ch15_t_extend_unet"

    parser = argparse.ArgumentParser()
    # parser.add_argument("--augumentation", type=str, default="")
    parser.add_argument("--p_augumentation", type=str, default="")
    parser.add_argument("--r_augumentation", type=str, default="")
    parser.add_argument("--t_augumentation", type=str, default="")
    parser.add_argument("--Dataset_name", type=str, default="")
    parser.add_argument("--loss_pt_on_off", type=str, default="off")
    parser.add_argument("--loss_pt_on_off_R_weight", type=str, default="")
    parser.add_argument("--loss_pt_on_off_P_weight", type=str, default="")
    parser.add_argument("--loss_pt_on_off_T_weight", type=str, default="")
    parser.add_argument("--dataset_num", type=int, default=100)  # ICCEの際は16心拍分で

    parser.add_argument("--TARGET_NAME", type=str, default="")
    # parser.add_argument("--dname", type = str ,default = "test")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--train_batch_size", type=int, default=4)  # default=256
    parser.add_argument("--val_batch_size", type=int, default=1)  # default=256
    parser.add_argument("--learning_rate", type=float, default=0.00005)
    parser.add_argument("--datalength", type=int, default=400)
    # parser.add_argument("--enc_convlayer_sizes", type=list, default=[[16, 1], [32, 1], [64, 2]]) #畳み込み層の設定　増やしすぎると過学習の可能性　前から２ペアずつ読み込む　２つ目の[]の第二引数はストライド
    # parser.add_argument("--enc_convlayer_sizes", type=list, default=[[16, 1], [30, 2],[60, 2],[120, 2],[240,2]])#[[入力,ストライド],]
    # parser.add_argument("--enc_convlayer_sizes", type=list, default=[[16, 1], [30, 2],[60, 2],[120,2],[240,2]])#[[入力,ストライド],]
    parser.add_argument(
        "--enc_convlayer_sizes",
        type=list,
        default=[[15, 1], [32, 2], [64, 2], [128, 2]],
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
        "--dec_convlayer_sizes", type=list, default=[[128, 2], [64, 2], [32, 2], [8, 1]]
    )  # 通常畳み込み層が続くが、出力が単純（三角波）なので畳み込み層があってもなくても結果がそこまで変わらなかったらしい
    # parser.add_argument("--dec_convlayer_sizes", type=list, default=[]) #通常畳み込み層が続くが、出力が単純（三角波）なので畳み込み層があってもなくても結果がそこまで変わらなかったらしい
    parser.add_argument("--latent_size", type=int, default=2)

    parser.add_argument("--print_every", type=int, default=2000)
    parser.add_argument(
        "--fig_root", type=str, default=OUTPUT_DIR + "/" + f"figs_newref/{current_time}"
    )  # 学習過程及びテスト結果を出力するフォルダ
    parser.add_argument(
        "--mae_folder", type=str, default=OUTPUT_MAE_DIR + "/" + f"/{current_time}"
    )  # MAE結果を出力するフォルダ
    parser.add_argument("--conditional", action="store_true")
    parser.add_argument("--train_off", action="store_false")

    parser.add_argument(
        "--pth", type=str, default=r"unet_prt_sep.pth"
    )  # 学習モデルのファイル名指定
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument(
        "--loss_fn_type", type=str, default="mse"
    )  # ECGの正規化なしだと上手くいった。
    parser.add_argument("--beta", type=int, default=1)
    # parser.add_argument("--alpha", type=int, default=300000)
    parser.add_argument("--alpha", type=int, default=10)
    parser.add_argument("--dim_red", type=str, default="")
    parser.add_argument("--transform_type", type=str, default="normal")

    # parser.add_argument("--loss_fn_type", type=str, default='bce')
    parser.add_argument("--ecg_ch_num", type=int, default=8)
    parser.add_argument("--current_time", type=str, default=current_time)
    parser.add_argument(
        "--ave_data_flg", type=int, default=0
    )  # 平均心拍を利用するか否か
    args = parser.parse_args()
    create_directory_if_not_exists(args.fig_root)
    create_directory_if_not_exists(args.mae_folder)
    main(args)
