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

matplotlib.use("TkAgg")

import random
import pandas as pd

from models import VAE

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
    RATE_16CH,
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


def draw3d(
    data, x_axis, y_axis, cb_min, cb_max
):  # cb_min,cb_max:カラーバーの下端と上端の値
    """PSD描画用の関数"""
    Y, X = np.meshgrid(y_axis, x_axis)
    # 図を描くのに何色用いるか（大きくすると重くなる。小さくすると荒くなる。）
    div = 30.0
    delta = (cb_max - cb_min) / div
    interval = np.arange(cb_min, abs(cb_max) * 2 + delta, delta)[0 : int(div) + 1]
    # plt.rcParams["font.size"] = 3
    plt.contourf(X, Y, data, interval)


# class MyDataset(torch.utils.data.Dataset):

#     def __init__(self, in_data, out_data, label, transform=None):
#         self.transform = transform
#         self.in_data = in_data
#         self.out_data = out_data
#         self.data_num = len(in_data)
#         self.label = label

#     def __len__(self):
#         return self.data_num

#     def __getitem__(self, idx):
#         if self.transform:
#             in_data = self.transform(self.in_data)[0][idx]
#             out_data = self.transform(self.out_data)[0][idx]
#             out_label = self.label[idx]
#         else:
#             in_data = self.in_data[idx]
#             out_data = self.out_data[idx]
#             out_label =  self.label[idx]

#         return in_data, out_data, out_label

# def trans_tri(x, P_LEN):
#     for idx, val in enumerate(x):
#         if val == 1.0:
#             for i in range(P_LEN):
#                 if idx - i > 0:
#                     x[idx - i] = 1.0 - i / P_LEN
#                 if idx + i < len(x):
#                     x[idx + i] = 1.0 - i / P_LEN
#     return x

# def parseDataFile(filename, X_LIM_MIN, X_LIM_MAX, datalength, STEP, DSRATE, datasetlist, datasetoutlist, label):
#     numofdataset = 0
#     times = []
#     ch = []
#     mecg_flag = []
#     fecg_flag = []
#     for i in range(16):
#         ch.append([])
#     wave_fp = codecs.open(filename, 'r')
#     for lineidx, line in enumerate(wave_fp):
#         if lineidx >= 1:#一行目は説明
#             split_data = line.rstrip('\r\n').split(',')
#             times.append(lineidx)#ToDo
#             for i in range(16):
#                 ch[i].append(float(split_data[3 + i]))
#                 #ch[i].append(float(split_data[3 + i])*0.1) #case68 (不整脈)のとき
#             mecg_flag.append(float(split_data[19]))
#             fecg_flag.append(float(split_data[20]))
#         else:
#             print(line)
#     wave_fp.close()

#     chfilt = []
#     for i in range(16):
#         chfilt.append(hpf(lpf(ch[i], 125, fp=45, fs=50), 125, fp=3, fs=2)) #original
#     mecg_flag = trans_tri(mecg_flag, 16) #パルス波を三角波に置き換える
#     fecg_flag = trans_tri(fecg_flag, 16)

#     timesnp = np.array(times)[X_LIM_MIN:X_LIM_MAX:DSRATE]

#     chnp = []
#     for i in range(16):
#         #chnp.append(np.array(ch[i])[X_LIM_MIN:X_LIM_MAX:DSRATE])
#         #chnp.append(chfilt[i][X_LIM_MIN:X_LIM_MAX:DSRATE])
#         chnp.append((chfilt[i] - chfilt[3])[X_LIM_MIN:X_LIM_MAX:DSRATE]) #差分を取る
#     mecg_flag = np.array(mecg_flag)[X_LIM_MIN:X_LIM_MAX:DSRATE]
#     fecg_flag = np.array(fecg_flag)[X_LIM_MIN:X_LIM_MAX:DSRATE]

#     # num_subplot = 2
#     # plt.figure(num=None, figsize=(8, 6), dpi=150, facecolor='w', edgecolor='k')
#     # ax = plt.subplot(num_subplot, 1, 1)
#     # plt.plot(mecg_flag, color="blue", linewidth=1.0, linestyle="-")
#     # plt.plot(fecg_flag, color="red", linewidth=1.0, linestyle="-")

#     # ax = plt.subplot(num_subplot, 1, 2)
#     # plt.plot(chnp[2], color="red", linewidth=1.0, linestyle="-")
#     # plt.plot(chnp[7], color="blue", linewidth=1.0, linestyle="-")

#     # plt.tight_layout()
#     # #plt.savefig("test.png",dpi=300)
#     # plt.show()

#     for i in range(int((len(timesnp) - datalength) / STEP)):
#         numofdataset += 1
#         # datastack = min_max(chnp[0][i * STEP:i * STEP + datalength], -1500, 1500)
#         # for ch in range(16):
#         #     datastack = np.hstack((datastack, min_max(chnp[ch + 1][i * STEP:i * STEP + datalength], -1500, 1500)))
#         #datastack = min_max_old(chnp[0][i * STEP:i * STEP + datalength])
#         datastack = min_max(chnp[0][i * STEP:i * STEP + datalength], -400, 400) #正規化
#         for ch in range(16):
#             datastack = np.hstack((datastack, min_max(chnp[ch + 1][i * STEP:i * STEP + datalength], -400, 400))) #chを入れ替えない時#これは正規化の幅を決めてる？
#         if datastack.max(axis=None, keepdims=True) != 0:
#             datasetlist.append(datastack)

#             #outdatastack = mecg_flag[i * STEP:i * STEP + datalength]
#             outdatastack = np.hstack((mecg_flag[i * STEP:i * STEP + datalength], fecg_flag[i * STEP:i * STEP + datalength]))
#             datasetoutlist.append(outdatastack)
#             label.append(0)
#             #print(f"{ctgnp[i * STEP]},{ctgmin_avg[0]},{ctgmax[0]}")
#     return datasetlist, datasetoutlist, label, chfilt, mecg_flag, fecg_flag

# def createTraindataset(filelist_fp, datalength, transform, STEP, filelist_log_fp, DSRATE):
#     datasetlist = []
#     datasetoutlist = []
#     label = []

#     for idx, filename in enumerate(filelist_fp):
#         #ファイル読み込み
#         filelist_log_fp.write(filename)
#         split_fpdata = filename.rstrip('\r\n').split(',')
#         if len(split_fpdata) == 5:
#             print(f"Input file: {split_fpdata[0]}, {split_fpdata[1]}, {split_fpdata[2]}, {split_fpdata[3]}, {split_fpdata[4]}")
#             if split_fpdata[2] == '0':
#                 print("0:= not used")
#             else:
#                 X_LIM_MIN = int(split_fpdata[1])
#                 X_LIM_MAX = int(split_fpdata[2])
#                 filename = split_fpdata[0]
#                 datasetlist, datasetoutlist, label, chfilt, mecg_flag, fecg_flag = parseDataFile(filename, X_LIM_MIN, X_LIM_MAX, datalength, STEP, DSRATE, datasetlist, datasetoutlist, label)
#                 #datasetlist:フィルタ処理して差分を取った16chの信号、datasetoutlist:mecg_flagとfecg_flagを横につないだ配列、label:0信号

#     datasetlist = np.array(datasetlist)
#     print("datasetlist")
#     print(datasetlist.shape)
#     datasetoutlist = np.array(datasetoutlist)
#     print(datasetoutlist.shape)
#     label = np.array(label)
#     print(label.shape)
#     #return MyDataset(datasetlist.astype(np.float32), datasetoutlist.astype(np.float32), label, transform)
#     return MyDataset(datasetlist.astype(np.float32), datasetoutlist.astype(np.float32), label, transform)

# def createTestdataset(filename, X_LIM_MIN, X_LIM_MAX, datalength, transform, STEP, DSRATE):
#     datasetlist = []
#     datasetoutlist = []
#     label = []
#     datasetlist, datasetoutlist, label, chfilt, mecg_flag, fecg_flag = parseDataFile(filename, X_LIM_MIN, X_LIM_MAX, datalength, STEP, DSRATE, datasetlist, datasetoutlist, label)

#     datasetlist = np.array(datasetlist)
#     print(datasetlist.shape)
#     datasetoutlist = np.array(datasetoutlist)
#     print(datasetoutlist.shape)
#     label = np.array(label)
#     print(label.shape)
#     return MyDataset(datasetlist.astype(np.float32), datasetoutlist.astype(np.float32), label, transform), chfilt, mecg_flag, fecg_flag


def fourier(x, ch=8, dt=0.002, datalength=400, batch_size=4):
    xo = x.view(-1, ch, datalength)
    y = torch.ones(batch_size, ch, datalength)
    for batch in range(batch_size):
        for num in range(ch):
            y[batch, num] = torch.fft.fft(xo[batch, num])

    # print (y)
    return y


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


def plot_fig_train(
    recon_x, xo, datalength, ts, args, iteration_num, batch_size_num, epoch_num
):
    sample_rate = 500
    sample_num = 750
    xticks = np.linspace(0.0, 1.0 / sample_rate * sample_num, sample_num)
    ecg_ch = args.ecg_ch_num
    for p in range(batch_size_num):

        for q in range(ecg_ch):
            # plt.subplot(4, 2, q+1)
            # plt.figure(figsize=(10,5))

            # recon_x2 = torch.reshape(recon_x, (-1,ecg_ch,datalength))#datalength*2じゃないはず
            # xo2=torch.reshape(xo,(-1,ecg_ch,datalength))
            recon_x2 = recon_x.view(-1, datalength)
            xo2 = torch.reshape(xo, (-1, datalength))

            # print(xo.shape)
            # plt.plot(x[p].cpu().data.numpy()[0:datalength], color="green", linewidth=1.0, linestyle="-")
            # plt.plot(min_max_old(recon_x2[p][q].cpu().data.numpy()), color="red", linewidth=1.0, linestyle="-")
            plt.plot(
                xticks,
                recon_x2[p].cpu().data.numpy(),
                color="red",
                linewidth=1.0,
                linestyle="-",
            )
            # plt.plot(xo[p].cpu().data.numpy(), color="blue", linewidth=1.0, linestyle="-")#xoが（３２，４００）かたちになってる、
            plt.plot(
                xticks,
                xo2[p].cpu().data.numpy(),
                color="blue",
                linewidth=1.0,
                linestyle="-",
            )  # xoが（３２，４００）かたちになってる、
            # plt.subplots_adjust(left=0.125,bottom=0.1,right=0.9,top=0.9,wspace=0.2,hspace=0.35)
            # plt.ylim(0,1)
            plt.xlabel("second")
            plt.axis("on")
            plt.minorticks_on()
            plt.grid(which="both", axis="x", alpha=0.8, linestyle="--", linewidth=1)
            plt.legend(["predict", "ECG"], bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.tight_layout()

            if not os.path.exists(
                os.path.join(
                    args.fig_root,
                    str(ts),
                    "epoch{:d}_iteration{:d}".format(epoch_num, iteration_num),
                )
            ):
                os.mkdir(
                    os.path.join(
                        args.fig_root,
                        str(ts),
                        "epoch{:d}_iteration{:d}".format(epoch_num, iteration_num),
                    )
                )

            plt.savefig(
                os.path.join(
                    args.fig_root,
                    str(ts),
                    "epoch{:d}_iteration{:d}".format(epoch_num, iteration_num),
                    "train_x_xo_{:d}_ch{:d}.png".format(p, q),
                ),
                dpi=300,
            )
            # plt.show()
            plt.cla()
            plt.clf()
            plt.close()


def plot_fig_train_name(
    recon_x,
    xo,
    datalength,
    ts,
    args,
    iteration_num,
    batch_size_num,
    epoch_num,
    label_name,
):
    sample_rate = 500
    sample_num = args.datalength
    xticks = np.linspace(0.0, 1.0 / sample_rate * sample_num, sample_num)
    ecg_ch = args.ecg_ch_num
    recon_x2 = torch.reshape(
        recon_x, (-1, ecg_ch, datalength)
    )  # datalength*2じゃないはず
    xo2 = torch.reshape(xo, (-1, ecg_ch, datalength))
    for p in range(batch_size_num):
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
            plt.xlabel("second")
            plt.axis("on")
            plt.minorticks_on()
            plt.grid(which="both", axis="x", alpha=0.8, linestyle="--", linewidth=1)
            plt.legend(["predict", "ECG"], bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.title(label_name[p] + "_ch{}".format(str(q)))
            plt.tight_layout()

            if not os.path.exists(
                os.path.join(
                    args.fig_root,
                    str(ts),
                    "epoch{:d}_iteration{:d}".format(epoch_num, iteration_num),
                )
            ):
                os.mkdir(
                    os.path.join(
                        args.fig_root,
                        str(ts),
                        "epoch{:d}_iteration{:d}".format(epoch_num, iteration_num),
                    )
                )

            plt.savefig(
                # os.path.join(args.fig_root, str(ts),"epoch{:d}_iteration{:d}".format(epoch_num,iteration_num),"ch{:d}_train_x_xo_{:d}.png".format(q,p)),
                os.path.join(
                    args.fig_root,
                    str(ts),
                    "epoch{:d}_iteration{:d}".format(epoch_num, iteration_num),
                    "ch{:d}_train_x_xo_{}.png".format(q, label_name[p]),
                ),
                dpi=300,
            )
            # plt.show()
            plt.cla()
            plt.clf()
            plt.close()


def plot_fig_16ch_only(
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
                    "ch={}_16ch_only_test_x_xo_{}.png".format(
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


def loss_fn_bce(recon_x, x, mean, log_var, datalength):
    # print("recon_x,x")
    # print(recon_x.shape,x.shape)
    # print("recon_x")
    # print(recon_x)
    # recon_xf=fourier(recon_x)
    # xf=fourier(x)
    # fft=torch.nn.MSELoss(reduction="sum")

    # fft_loss=fft(recon_xf,xf)

    BCE = torch.nn.functional.binary_cross_entropy(
        # recon_x.view(-1, datalength * 2), x.view(-1, datalength * 2), reduction='sum') #二値交差エントロピー損失、この関数内のxはxoのことつまり予測値
        recon_x.view(-1, datalength),
        x.view(-1, datalength),
        reduction="sum",
    )  # 二値交差エントロピー損失、この関数内のxはxoのことつまり予測値
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    # alpha=100000
    beta = 1
    KLD = beta * KLD
    # BCE=beta*BCE

    return (BCE + KLD) / x.size(0), BCE / x.size(0), KLD / x.size(0)


def loss_fn_mse_pt(recon_x, x, mean, log_var, datalength, pt_index, args):
    ecg_ch_num = x.shape[1]
    batch_size = x.shape[0]
    MSE = torch.nn.MSELoss(
        reduction="none"
    )  # x,recon_xと全く同じ形状で(y_n-y'_n)^2の計算して要素に並べる。
    MSE_loss = MSE(
        recon_x.view(-1, ecg_ch_num, datalength), x.view(-1, ecg_ch_num, datalength)
    )
    num = 0
    MSE_loss_temp = 0
    for i in range(batch_size):
        MSE_loss_temp += torch.mean(
            MSE_loss[i, :, pt_index[i][0] : pt_index[i][1]]
        )  # ミニバッチ内のデータそれぞれについてP波の始まりとT波の終わりまでの範囲でMSE計算
    # MSE_loss_pt=MSE_loss_temp/batch_size
    MSE_loss_pt = MSE_loss_temp

    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    alpha = args.alpha
    beta = args.beta
    # KLD=beta*KLD
    return (
        (alpha * MSE_loss_pt + beta * KLD) / batch_size,
        MSE_loss_pt,
        KLD / batch_size,
    )


def loss_fn_mse(recon_x, x, mean, log_var, datalength, args):
    batch_size = x.shape[0]
    # print(x.shape)
    # print(recon_x.shape)
    # MSE=torch.nn.MSELoss(reduction="sum")#reducitonをsumにしていた。これはミニバッチの全ての要素の二乗誤差の和。meanのときの要素数倍になる。
    MSE = torch.nn.MSELoss(reduction="mean")
    MSE_loss = MSE(
        recon_x.view(-1, datalength), x.view(-1, datalength)
    )  # ミニバッチ内全てで平均。ミニバッチ内のそれぞれのデータの形状が同じ
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    alpha = args.alpha
    beta = args.beta
    # print(MSE_loss)
    # # print("pure_MSE")
    # KLD=beta*KLD
    # return (alpha*MSE_loss + beta*KLD) / batch_size, MSE_loss/batch_size, KLD/batch_size
    return (
        alpha * MSE_loss + beta * KLD / batch_size,
        MSE_loss / batch_size,
        KLD / batch_size,
    )  # MSEはreduction=meanで既にバッチで割られているからここではバッチ数で割らない
    # return (alpha*MSE_loss + KLD) / x.size(0), MSE_loss/x.size(0), KLD/x.size(0)


def loss_fn_mse_without_R(recon_x, x, mean, log_var, datalength, args):
    # index1=190
    # index2=210
    # input(x.shape)
    # input(recon_x.shape)
    index1 = 190
    index2 = 210
    new_length = datalength - (index2 - index1)
    batch_size = x.shape[0]
    MSE = torch.nn.MSELoss(reduction="sum")
    recon_x_without_R1 = recon_x.view(-1, args.ecg_ch_num, datalength)[:, :, :index1]
    recon_x_without_R2 = recon_x.view(-1, args.ecg_ch_num, datalength)[:, :, index2:]
    x_without_R1 = x.view(-1, args.ecg_ch_num, datalength)[:, :, :index1]
    x_without_R2 = x.view(-1, args.ecg_ch_num, datalength)[:, :, index2:]
    MSE_loss_1 = MSE(
        recon_x_without_R1, x_without_R1
    )  # ミニバッチ内全てで平均。ミニバッチ内のそれぞれのデータの形状が同じ
    MSE_loss_2 = MSE(
        recon_x_without_R2, x_without_R2
    )  # ミニバッチ内全てで平均。ミニバッチ内のそれぞれのデータの形状が同じ
    MSE_loss = (MSE_loss_1 + MSE_loss_2) / args.ecg_ch_num / new_length
    # MSE_loss=(MSE_loss_1+MSE_loss_2)/batch_size/args.ecg_ch_num/new_length
    # MSE_loss=(MSE_loss_1)/batch_size/args.ecg_ch_num/new_length
    # MSE_loss=0

    # MSE_loss=MSE(recon_x.view(-1,args.ecg_ch_num, datalength), x.view(-1,args.ecg_ch_num, datalength))/batch_size/args.ecg_ch_num/datalength#ミニバッチ内全てで平均。ミニバッチ内のそれぞれのデータの形状が同じ
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    alpha = args.alpha
    beta = args.beta
    # print(MSE_loss)
    return (
        (alpha * MSE_loss + beta * KLD) / batch_size,
        MSE_loss / batch_size,
        KLD / batch_size,
    )
    # return (beta*KLD) / batch_size, MSE_loss/batch_size, KLD/batch_size


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
        index1 = 190  # 　一旦はこれでやらせてみる 0401
        index2 = 210
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


# def loss_fn_mse_PRT_weight_QS(recon_x, x, mean, log_var,datalength,args,pt_index):
#     weight_P=args.loss_pt_on_off_P_weight
#     weight_R=args.loss_pt_on_off_R_weight
#     weight_T=args.loss_pt_on_off_T_weight
#     Q_peaks=pt_index[:,5]
#     S_peaks=pt_index[:,6]

#     # new_length=datalength-(index2-index1)
#     batch_size=x.shape[0]
#     MSE=torch.nn.MSELoss(reduction="sum")
#     MSE_loss=0.0
#     sum_all=0.0
#     MSE_loss_all_1=0.0
#     MSE_loss_all_2=0.0
#     MSE_loss_all_R=0.0
#     for i in range(batch_size):
#         index1=Q_peaks[i]
#         index2=S_peaks[i]
#         # print(index1)
#         # print(index2)
#         # index1=190
#         # index2=210
#         length_P=index1
#         length_T=datalength-index2
#         length_R=index2-index1
#         # print(length_P+length_R+length_T)
#         recon_x_before_R1=recon_x.view(-1,args.ecg_ch_num, datalength)[i,:,:index1]
#         recon_x_after_R2=recon_x.view(-1,args.ecg_ch_num, datalength)[i,:,index2:]
#         x_before_R1=x.view(-1,args.ecg_ch_num, datalength)[i,:,:index1]
#         x_after_R2=x.view(-1,args.ecg_ch_num, datalength)[i,:,index2:]
#         recon_x_R=recon_x.view(-1,args.ecg_ch_num, datalength)[i,:,index1:index2]
#         x_R=x.view(-1,args.ecg_ch_num, datalength)[i,:,index1:index2]
#         # print(x_R.shape)
#         # print(x_R)
#         MSE_loss_1=MSE(recon_x_before_R1,x_before_R1)#重みを小さくするR波より前のインデックスでのMSE（sum）
#         MSE_loss_2=MSE(recon_x_after_R2,x_after_R2)#重みを小さくするR波より後ろのインデックスでのMSE（sum）
#         MSE_loss_R=MSE(recon_x_R,x_R)#重みを小さくするR波（０．０４秒）のMSE（sum）
#         # MSE_loss=(MSE_loss_1+MSE_loss_2)/args.ecg_ch_num/new_length+MSE_loss_R/args.ecg_ch_num/(index2-index1)*weight#（インデックス長さ×チャネル数）で割る。まだバッチの数では割れていない。
#         # MSE_loss_keep=(MSE_loss_1/args.ecg_ch_num/length_P)+(MSE_loss_2/args.ecg_ch_num/length_T)+(MSE_loss_R/args.ecg_ch_num/length_R)#（インデックス長さ×チャネル数）で割る。まだバッチの数では割れていない。
#         MSE_loss_keep=(MSE_loss_1/args.ecg_ch_num/length_P)*weight_P+(MSE_loss_2/args.ecg_ch_num/length_T)*weight_T+(MSE_loss_R/args.ecg_ch_num/length_R)*weight_R#（インデックス長さ×チャネル数）で割る。まだバッチの数では割れていない。
#         sum=MSE_loss_1+MSE_loss_2+MSE_loss_R
#         sum_all=sum_all+sum
#         MSE_loss_all_1=MSE_loss_all_1+MSE_loss_1/length_P
#         # print("MSE_loss_all_1")
#         # print(MSE_loss_all_1)
#         # print(weight_P)
#         # print(weight_T)
#         MSE_loss_all_2=MSE_loss_all_2+MSE_loss_2/length_T
#         MSE_loss_all_R=MSE_loss_all_R+MSE_loss_R/length_R
#         # print(sum)
#         # print(sum_all)
#         # print(MSE_loss_keep)
#         MSE_loss=MSE_loss+MSE_loss_keep
#     print(MSE_loss)
#     MSE_loss_all=(MSE_loss_all_1*weight_P+MSE_loss_all_2*weight_T+MSE_loss_all_R*weight_R)/args.ecg_ch_num
#     print(MSE_loss_all)
#     print(sum_all)
#     # input()
#     KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
#     alpha=args.alpha
#     beta=args.beta
#     # print(MSE_loss_1)#重みを小さくするR波より前のインデックスでのMSE（sum）
#     return (alpha*MSE_loss + beta*KLD) / batch_size, MSE_loss/batch_size, KLD/batch_size
#     # return (beta*KLD) / batch_size, MSE_loss/batch_size, KLD/batch_size

# def loss_fn_mse_PRT_weight(recon_x, x, mean, log_var,datalength,args):#これ間違ってる。重みの計算が
#     weight_P=args.loss_pt_on_off_P_weight
#     weight_R=args.loss_pt_on_off_R_weight
#     weight_T=args.loss_pt_on_off_T_weight

#     index1=190
#     index2=210
#     length_P=index1
#     length_T=datalength-index2
#     length_R=index2-index1
#     # new_length=datalength-(index2-index1)
#     batch_size=x.shape[0]
#     MSE=torch.nn.MSELoss(reduction="sum")
#     recon_x_before_R1=recon_x.view(-1,args.ecg_ch_num, datalength)[:,:,:index1]
#     recon_x_after_R2=recon_x.view(-1,args.ecg_ch_num, datalength)[:,:,index2:]
#     x_before_R1=x.view(-1,args.ecg_ch_num, datalength)[:,:,:index1]
#     x_after_R2=x.view(-1,args.ecg_ch_num, datalength)[:,:,index2:]
#     recon_x_R=recon_x.view(-1,args.ecg_ch_num, datalength)[:,:,index1:index2]
#     x_R=x.view(-1,args.ecg_ch_num, datalength)[:,:,index1:index2]
#     # print(x_R.shape)
#     MSE_loss_1=MSE(recon_x_before_R1,x_before_R1)#重みを小さくするR波より前のインデックスでのMSE（sum）
#     MSE_loss_2=MSE(recon_x_after_R2,x_after_R2)#重みを小さくするR波より後ろのインデックスでのMSE（sum）
#     MSE_loss_R=MSE(recon_x_R,x_R)#重みを小さくするR波（０．０４秒）のMSE（sum）
#     # MSE_loss=(MSE_loss_1+MSE_loss_2)/args.ecg_ch_num/new_length+MSE_loss_R/args.ecg_ch_num/(index2-index1)*weight#（インデックス長さ×チャネル数）で割る。まだバッチの数では割れていない。
#     MSE_loss=(MSE_loss_1/args.ecg_ch_num/length_P)*weight_P+(MSE_loss_2/args.ecg_ch_num/length_T)*weight_T+(MSE_loss_R/args.ecg_ch_num/length_R)*weight_R#（インデックス長さ×チャネル数）で割る。まだバッチの数では割れていない。
#     sum_all=MSE_loss_1+MSE_loss_2+MSE_loss_R
#     print(sum_all)
#     KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
#     alpha=args.alpha
#     beta=args.beta
#     print(MSE_loss)
#     return (alpha*MSE_loss + beta*KLD) / batch_size, MSE_loss/batch_size, KLD/batch_size
#     # return (beta*KLD) / batch_size, MSE_loss/batch_size, KLD/batch_size

# def loss_fn_mse_R_weight(recon_x, x, mean, log_var,datalength,args):
#     weight=args.loss_pt_on_off_R_weight
#     index1=190
#     index2=210
#     new_length=datalength-(index2-index1)
#     batch_size=x.shape[0]
#     MSE=torch.nn.MSELoss(reduction="sum")
#     recon_x_before_R1=recon_x.view(-1,args.ecg_ch_num, datalength)[:,:,:index1]
#     # print(recon_x_before_R1)
#     recon_x_after_R2=recon_x.view(-1,args.ecg_ch_num, datalength)[:,:,index2:]
#     x_before_R1=x.view(-1,args.ecg_ch_num, datalength)[:,:,:index1]
#     x_after_R2=x.view(-1,args.ecg_ch_num, datalength)[:,:,index2:]
#     recon_x_R=recon_x.view(-1,args.ecg_ch_num, datalength)[:,:,index1:index2]
#     x_R=x.view(-1,args.ecg_ch_num, datalength)[:,:,index1:index2]
#     # print(x_R)
#     MSE_loss_1=MSE(recon_x_before_R1,x_before_R1)#重みを小さくするR波より前のインデックスでのMSE（sum）
#     MSE_loss_2=MSE(recon_x_after_R2,x_after_R2)#重みを小さくするR波より後ろのインデックスでのMSE（sum）
#     MSE_loss_R=MSE(recon_x_R,x_R)#重みを小さくするR波（０．０４秒）のMSE（sum）
#     MSE_loss=(MSE_loss_1+MSE_loss_2)/args.ecg_ch_num/new_length+MSE_loss_R/args.ecg_ch_num/(index2-index1)*weight#（インデックス長さ×チャネル数）で割る。まだバッチの数では割れていない。
#     KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
#     alpha=args.alpha
#     beta=args.beta
#     return (alpha*MSE_loss + beta*KLD) / batch_size, MSE_loss/batch_size, KLD/batch_size
#     # return (beta*KLD) / batch_size, MSE_loss/batch_size, KLD/batch_size


# def loss_fn_mse_without_R_1_10(recon_x, x, mean, log_var,datalength,args):
#     # index1=190
#     # index2=210
#     # input(x.shape)
#     # input(recon_x.shape)
#     index1=190
#     index2=210
#     new_length=datalength-(index2-index1)
#     batch_size=x.shape[0]
#     MSE=torch.nn.MSELoss(reduction="sum")
#     recon_x_without_R1=recon_x.view(-1,args.ecg_ch_num, datalength)[:,:,:index1]
#     recon_x_without_R2=recon_x.view(-1,args.ecg_ch_num, datalength)[:,:,index2:]
#     x_without_R1=x.view(-1,args.ecg_ch_num, datalength)[:,:,:index1]
#     x_without_R2=x.view(-1,args.ecg_ch_num, datalength)[:,:,index2:]

#     recon_x_R=recon_x.view(-1,args.ecg_ch_num, datalength)[:,:,index1:index2]
#     x_R=x.view(-1,args.ecg_ch_num, datalength)[:,:,index1:index2]
#     # print(x_R)

#     MSE_loss_1=MSE(recon_x_without_R1,x_without_R1)#ミニバッチ内全てで平均。ミニバッチ内のそれぞれのデータの形状が同じ
#     MSE_loss_2=MSE(recon_x_without_R2,x_without_R2)#ミニバッチ内全てで平均。ミニバッチ内のそれぞれのデータの形状が同じ
#     MSE_loss_R=MSE(recon_x_R,x_R)#ミニバッチ内全てで平均。ミニバッチ内のそれぞれのデータの形状が同じ
#     # print("MSE_loss_1+2")
#     # print((MSE_loss_1+MSE_loss_2)/args.ecg_ch_num/new_length)
#     # print("MSE_loss_R")
#     # print(MSE_loss_R/args.ecg_ch_num/(index2-index1)/2)
#     # input("MSE_loss_R")
#     MSE_loss=(MSE_loss_1+MSE_loss_2)/args.ecg_ch_num/new_length+MSE_loss_R/args.ecg_ch_num/(index2-index1)/10
#     # MSE_loss=(MSE_loss_1+MSE_loss_2)/batch_size/args.ecg_ch_num/new_length
#     # MSE_loss=(MSE_loss_1)/batch_size/args.ecg_ch_num/new_length
#     # MSE_loss=0


#     # MSE_loss=MSE(recon_x.view(-1,args.ecg_ch_num, datalength), x.view(-1,args.ecg_ch_num, datalength))/batch_size/args.ecg_ch_num/datalength#ミニバッチ内全てで平均。ミニバッチ内のそれぞれのデータの形状が同じ
#     KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
#     alpha=args.alpha
#     beta=args.beta
#     # print(MSE_loss)
#     # input("")
#     return (alpha*MSE_loss + beta*KLD) / batch_size, MSE_loss/batch_size, KLD/batch_size
#     # return (beta*KLD) / batch_size, MSE_loss/batch_size, KLD/batch_size
def noise_make(mean, scale, datanum, ch_num):
    rnd = np.random.normal(loc=mean, scale=scale, size=datanum * ch_num)
    rnd = rnd.reshape(-1, ch_num, datanum)
    # print(rnd.shape)
    # rnd=np.random.randn(datanum)

    # print(rnd.shape)
    # plt.hist(rnd,bins=datanum)
    # plt.plot(rnd)
    # plt.show()


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


def plot_scatter_2d_pos(
    z,
    labels,
    latent_size,
    ts,
    fig_num=1,
    figsize=(4, 3),
    dpi=300,
    facecolor="w",
    edgecolor="k",
):
    # fig = plt.figure(num=fig_num, figsize=figsize, dpi=dpi, facecolor=facecolor, edgecolor=edgecolor)
    # fig = plt.figure(dpi=dpi, facecolor=facecolor, edgecolor=edgecolor)
    subxy = 2 if latent_size > 2 else 1
    ax = plt.subplot(subxy, subxy, 1)
    legend_dict = {}
    # colors = {'A': 'red', 'B': 'blue', 'C': 'green'}
    # ラベルごとに異なる色を設定
    pos_names = []
    # print(labels)
    # input()
    for item in labels:
        pos_name = extract_pos_name(item)
        if pos_name:
            pos_names.append(pos_name)
            # print(person_name)
    labels = pos_names

    unique_labels = np.unique(labels)
    num_labels = len(unique_labels)
    colors = plt.cm.get_cmap("tab10", num_labels)  # カラーマップから色を取得
    for i in range(z.shape[0]):
        label = labels[i]
        color = colors(unique_labels.tolist().index(label) % num_labels)
        if label not in legend_dict:
            legend_dict[label] = ax.scatter(
                z[i, 0], z[i, 1], s=40, label=label, color=color
            )

        else:
            ax.scatter(z[i, 0], z[i, 1], s=40, label=labels[i], color=color)
    legend_elements = [legend_dict[label] for label in sorted(legend_dict.keys())]
    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-4.5, 4.5)
    ax.set_aspect("equal", adjustable="box")
    # plt.legend(handles=legend_elements, title='Labels',fontsize=8)
    ax.legend(
        handles=legend_elements,
        title="Labels",
        fontsize=8,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )
    # cbar = fig.colorbar(sc, aspect=30)
    # cbar.ax.tick_params(direction='out')
    # cbar.ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.savefig(os.path.join(args.fig_root, str(ts), "plot_z.png"), dpi=300)
    # plt.show()
    plt.close()
    # if Z_DIM > 2:
    #     ax = plt.subplot(subxy, subxy, 2)
    #     for i in range(Y_DIM):
    #         zplot = ztest_pred[ltest_pred == i]
    #         ax.scatter(zplot[:, 0], zplot[:, 2], s=5, label=i, color=colors((2*i+1)/(Y_DIM*2)))
    #         zplot = np.vstack((zplot[:, 0], zplot[:, 2])).T
    #         zavg = zplot.sum(axis=0) / zplot.shape[0]
    #         ax.annotate(str(i), zavg)
    #         plt.legend()

    #     ax = plt.subplot(subxy, subxy, 3)
    #     for i in range(Y_DIM):
    #         zplot = ztest_pred[ltest_pred == i]
    #         ax.scatter(zplot[:, 1], zplot[:, 2], s=5, label=i, color=colors((2*i+1)/(Y_DIM*2)))
    #         zplot = np.vstack((zplot[:, 1], zplot[:, 2])).T
    #         zavg = zplot.sum(axis=0) / zplot.shape[0]
    #         ax.annotate(str(i), zavg)
    #         plt.legend()

    # fig.tight_layout()
    # fig.savefig(DIR_OUT + "VAEclustering_2d_test_sma" + str(SMA_NUM) + ".png")
    # plt.show()


def plot_scatter_2d(
    z,
    labels,
    latent_size,
    ts,
    args,
    fig_num=1,
    figsize=(4, 3),
    dpi=300,
    facecolor="w",
    edgecolor="k",
):
    if latent_size == 2:
        z = np.array(z)
    else:
        if args.dim_red == "TSNE":
            from sklearn.manifold import TSNE

            z = TSNE(n_components=2).fit_transform(z)
            # print(z)
            # input("TSNE_OK?")
        elif args.dim_red == "PCA":
            from sklearn.decomposition import PCA

            # print(z.shape)
            z = PCA(n_components=2).fit(z.T).components_.T
            print(z)
            # input("PCA_OK?")
        else:
            raise ValueError()
    fig = plt.figure(
        num=fig_num, figsize=figsize, dpi=dpi, facecolor=facecolor, edgecolor=edgecolor
    )
    # fig = plt.figure(dpi=dpi, facecolor=facecolor, edgecolor=edgecolor)
    # subxy = 2 if  latent_size> 2 else 1
    subxy = 1
    ax = fig.add_subplot(subxy, subxy, 1)
    legend_dict = {}
    # colors = {'A': 'red', 'B': 'blue', 'C': 'green'}
    # ラベルごとに異なる色を設定
    person_names = []
    # print(labels)
    # input()
    for item in labels:
        person_name = extract_person_name(item)
        # print(person_name)
        # input()
        if person_name:
            person_names.append(person_name)
            # print(person_name)
    labels = person_names

    unique_labels = np.unique(labels)
    num_labels = len(unique_labels)
    colors = plt.cm.get_cmap("tab10", num_labels)  # カラーマップから色を取得
    for i in range(z.shape[0]):
        label = labels[i]
        color = colors(unique_labels.tolist().index(label) % num_labels)
        if label not in legend_dict:
            legend_dict[label] = ax.scatter(
                z[i, 0], z[i, 1], s=10, label=label, color=color
            )

        else:
            # ax.scatter(z[i,0], z[i,1], s=40,label=labels[i],color=color)
            ax.scatter(z[i, 0], z[i, 1], s=10, label=labels[i], color=color)
    legend_elements = [legend_dict[label] for label in sorted(legend_dict.keys())]
    # ax.set_xlim(-4.5, 4.5)
    # ax.set_ylim(-4.5, 4.5)
    ax.set_aspect("equal", adjustable="box")
    # plt.legend(handles=legend_elements, title='Labels',fontsize=8)
    ax.legend(
        handles=legend_elements,
        title="Labels",
        fontsize=8,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )
    plt.title(
        "epoch={},beta={},latent_size={},dim_red={},".format(
            args.epochs, args.beta, args.latent_size, args.dim_red
        )
    )
    # cbar = fig.colorbar(sc, aspect=30)
    # cbar.ax.tick_params(direction='out')
    # cbar.ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    # plt.show()
    plt.savefig(os.path.join(args.fig_root, str(ts), "plot_z.png"), dpi=300)
    # plt.show()
    plt.close()
    # if Z_DIM > 2:
    #     ax = plt.subplot(subxy, subxy, 2)
    #     for i in range(Y_DIM):
    #         zplot = ztest_pred[ltest_pred == i]
    #         ax.scatter(zplot[:, 0], zplot[:, 2], s=5, label=i, color=colors((2*i+1)/(Y_DIM*2)))
    #         zplot = np.vstack((zplot[:, 0], zplot[:, 2])).T
    #         zavg = zplot.sum(axis=0) / zplot.shape[0]
    #         ax.annotate(str(i), zavg)
    #         plt.legend()

    #     ax = plt.subplot(subxy, subxy, 3)
    #     for i in range(Y_DIM):
    #         zplot = ztest_pred[ltest_pred == i]
    #         ax.scatter(zplot[:, 1], zplot[:, 2], s=5, label=i, color=colors((2*i+1)/(Y_DIM*2)))
    #         zplot = np.vstack((zplot[:, 1], zplot[:, 2])).T
    #         zavg = zplot.sum(axis=0) / zplot.shape[0]
    #         ax.annotate(str(i), zavg)
    #         plt.legend()

    # fig.tight_layout()
    # fig.savefig(DIR_OUT + "VAEclustering_2d_test_sma" + str(SMA_NUM) + ".png")
    # plt.show()


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


def main(args):
    # ==============================================goto=======================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    train_dataset, test_dataset = Dataset.Dataset_setup_8ch_pt_augmentation(
        TARGET_NAME=args.TARGET_NAME,
        transform_type=args.transform_type,
        Dataset_name=args.Dataset_name,
        dataset_num=args.dataset_num,
        DataAugumentation=args.augumentation,
        ave_data_flg=args.ave_data_flg,
    )
    print("len(train_dataset)")
    print(len(train_dataset))
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
        + args.augumentation
    )

    if not os.path.exists(os.path.join(args.fig_root, str(ts))):
        if not (os.path.exists(os.path.join(args.fig_root))):
            os.mkdir(os.path.join(args.fig_root))
        os.mkdir(os.path.join(args.fig_root, str(ts)))

    with open(os.path.join(args.fig_root, str(ts), "args.json"), mode="w") as f:
        json.dump(args.__dict__, f, indent=4)

    datalength = args.datalength
    latent_size = args.latent_size
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

    print(len(train_dataset))
    print(len(train_dataset[5]))
    print("fdfdfdfed")
    print(train_dataset[0][3])
    print(train_dataset[0][2])

    # input()
    # summary(vae, input_size=(1,16 , 384))
    common_kwargs = {
        "datalength": datalength,
        "enc_convlayer_sizes": args.enc_convlayer_sizes,
        "enc_fclayer_sizes": args.enc_fclayer_sizes,
        "dec_fclayer_sizes": args.dec_fclayer_sizes,
        "dec_convlayer_sizes": args.dec_convlayer_sizes,
        "latent_size": latent_size,
        "conditional": args.conditional,
        "num_labels": 20 if args.conditional else 0,
    }
    vae_dict = {
        "P": VAE(**common_kwargs).to(device),
        "R": VAE(**common_kwargs).to(device),
        "T": VAE(**common_kwargs).to(device),
    }
    if args.mode == "train":
        print("TRAINING MODE::\n")

        # train_size =27
        # val_size = len(dataset) - train_size
        # val_size = 1
        # train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        train_data_loader = DataLoader(
            dataset=train_dataset, batch_size=args.train_batch_size, shuffle=True
        )
        # dataset=train_dataset, batch_size=args.train_batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
        weights = {
            "P": args.loss_pt_on_off_P_weight,
            "R": args.loss_pt_on_off_R_weight,
            "T": args.loss_pt_on_off_T_weight,
        }
        weight_keys = list(weights.keys())
        for target_weight in weight_keys:
            current_weight = {}
            for weight_key in weight_keys:
                left, right = weights[weight_key].split()
                current_weight[weight_key] = (
                    float(right) if target_weight == weight_key else float(left)
                )
            vae = vae_dict[target_weight]
            print(vae)
            optimizer = torch.optim.Adam(vae.parameters(), lr=args.learning_rate)

            logs = defaultdict(list)
            z_temps = np.empty((0, latent_size))
            label_temps = []
            train_acc_temps = []
            test_acc_temps = []
            for epoch in range(args.epochs):
                vae.train()
                tracker_epoch = defaultdict(lambda: defaultdict(dict))

                loss_keep = 0.0
                mse_keep = 0.0
                bce_keep = 0.0
                kdl_keep = 0.0
                acc_keep = 0.0

                print(len(train_data_loader))
                print("loader")
                for iteration, (x, xo, label_name, pt_index) in enumerate(
                    train_data_loader
                ):
                    # print("pt_index", pt_index)
                    x, xo = x.to(device), xo.to(device)
                    x_check = torch.reshape(x, (-1, 16, args.datalength))
                    recon_x, mean, log_var, z = vae(x)
                    # print(recon_x.shape)
                    # print(xo.shape)
                    if args.loss_fn_type == "bce":
                        loss, bce, kdl = loss_fn_bce(
                            recon_x, xo, mean, log_var, datalength
                        )
                    if args.loss_fn_type == "mse":
                        if args.loss_pt_on_off == "off":
                            # if args.loss_pt_on_off_R_weight!=None:
                            #     loss,mse,kdl = loss_fn_mse_R_weight(recon_x, xo, mean, log_var,datalength,args)#0.8秒間ｓのデータ全体で損失計算
                            if len(current_weight) == 3:
                                # loss,mse,kdl = loss_fn_mse_PRT_weight(recon_x, xo, mean, log_var,datalength,args)#0.8秒間のデータ全体で損失計算
                                # loss_org,mseaa,kdlaaaa = loss_fn_mse_PRT_weight(recon_x, xo, mean, log_var,datalength,args)#0.8秒間のデータ全体で損失計算
                                # loss,mse,kdl = loss_fn_mse_PRT_weight_QS(recon_x, xo, mean, log_var,datalength,args,pt_index)#0.8秒間のデータ全体で損失計算,ただしQRSに重みをつける
                                loss, mse, kdl = loss_fn_mse_PRT(
                                    recon_x,
                                    xo,
                                    mean,
                                    log_var,
                                    datalength,
                                    args,
                                    current_weight,
                                    pt_index,
                                )  # 0.8秒間ｓのデータ全体で損失計算,ただしP,R,Sそれぞれに重みをつける
                                # loss_pure,mse_bbb,kdlbbb = loss_fn_mse(recon_x, xo, mean, log_var,datalength,args)#0.8秒間ｓのデータ全体で損失計算
                                # print(loss)
                                # print(loss_pure)
                                # input("loss_check")
                                # input()

                            else:
                                loss, mse, kdl = loss_fn_mse(
                                    recon_x, xo, mean, log_var, datalength, args
                                )  # 0.8秒間ｓのデータ全体で損失計算

                        if args.loss_pt_on_off == "on":
                            loss, mse, kdl = loss_fn_mse_pt(
                                recon_x,
                                xo,
                                mean,
                                log_var,
                                datalength,
                                pt_index=pt_index,
                                args=args,
                            )
                    # loss,bce,kdl = loss_fn(recon_x, xo, mean, log_var,datalength)

                    # print(loss)

                    loss_keep += loss
                    if args.loss_fn_type == "bce":
                        bce_keep += bce
                    if args.loss_fn_type == "mse":
                        mse_keep += mse
                    kdl_keep += kdl
                    # -----------評価--------------------------------
                    mse_loss = torch.nn.MSELoss(reduction="mean")
                    # print(recon_x.shape[0])
                    acc = mse_loss(
                        recon_x.view(-1, datalength), xo.view(-1, datalength)
                    )
                    acc_keep += acc
                    # -----------評価--------------------------------

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # iteration 100ごとにLoss表示
                    # epochの最初のitertationごとにLoss表示

                    # if iteration % args.print_every == 0 or iteration == len(train_data_loader)-1:
                    if iteration % args.print_every == 0:
                        print(
                            "Epoch {:02d}/{:02d} Batch {:04d}/{:d}, Loss {:9.4f},Acc(mse){:9.4f}".format(
                                epoch,
                                args.epochs,
                                iteration,
                                len(train_data_loader) - 1,
                                loss_keep.item() / (iteration + 1),
                                acc_keep.item() / (iteration + 1),
                            )
                        )

                        # ifxは使ってない→inferenceも使ってない
                        if args.conditional:
                            c = torch.arange(0, 10).long().unsqueeze(1).to(device)
                            ifx = vae.inference(n=c.size(0), c=c)
                        else:
                            ifx = vae.inference(n=16)

                        ###学習途中の生成の様子を描画###

                        # plt.figure(figsize=(5, 10))
                        numplotfig = len(x)
                        if numplotfig > 6:
                            numplotfig = 6

                    # if epoch==0 or epoch==100:
                    if epoch == args.epochs - 1:
                        # 潜在変数の保存
                        z_temp = tensor_to_ndarray(z)
                        z_temps = np.append(z_temps, z_temp, axis=0)
                        # print(z_temps)
                        # print(label_name)
                        for i in range(len(label_name)):
                            label_temps.append(label_name[i])
                        # print(label_temps)
                    if epoch == args.epochs - 1 and iteration == 0:
                        print("z")
                        print(z)
                        print(xo.shape)
                        batch_size_now = xo.shape[0]
                        # plot_fig_train(recon_x=recon_x,xo=xo,datalength=datalength,ts=ts,args=args,iteration_num=iteration,batch_size_num=batch_size_now,epoch_num=epoch)

                        # plot_fig_train_name(recon_x=recon_x,xo=xo,datalength=datalength,ts=ts,args=args,iteration_num=iteration,batch_size_num=batch_size_now,epoch_num=epoch,label_name=label_name)
                    # # 訓練データで推論してMAE計算
                    # acc_temp = []
                    # vae.eval()
                    # train_val = []
                    # test_val = []
                    # with torch.no_grad():
                    #     for j, (x, xo, label_name, pt_index) in enumerate(
                    #         train_data_loader
                    #     ):
                    #         x, xo = x.to(device), xo.to(device)
                    #         recon_x, mean, log_var, z = vae(x)
                    #         for i in range(len(label_name)):
                    #             label_temps.append(label_name[i])
                    #         mae_loss = MAE_2(reduction="none")
                    #         recon_x = recon_x.view(-1, ecg_ch, datalength)
                    #         xo = xo.view(-1, ecg_ch, datalength)
                    #         acc_mae = mae_loss(recon_x, xo)

                    #         if args.loss_pt_on_off == "off":
                    #             train_val = train_val + cul_val_no_pt(
                    #                 acc_mae
                    #             )  # cul_val_no_ptは波形全体でmae計算
                    #         else:
                    #             train_val = train_val + cul_val(
                    #                 pt_index, acc_mae
                    #             )  # cul_valはptの範囲でmae計算i
                    # train_values = [array_item.item() for array_item in train_val]
                    # train_val_ndarray = np.array(train_values, dtype=np.float32)
                    # # print(test_val_ndarray.mean())
                    # train_acc_temps.append(train_val_ndarray.mean())

                    # logs["train_loss"].append(loss_keep.item() / (iteration + 1))
                    # if args.loss_fn_type == "bce":
                    #     logs["train_bce"].append(bce_keep.item() / (iteration + 1))
                    # if args.loss_fn_type == "mse":
                    #     logs["train_mse"].append(mse_keep.item() / (iteration + 1))
                    # logs["train_kdl"].append(kdl_keep.item() / (iteration + 1))
                    # logs["train_acc"].append(acc_keep.item() / (iteration + 1))

                    # # testデータで推論してMAE計算
                    # acc_temp = []
                    # vae.eval()
                    # test_val = []
                    # with torch.no_grad():
                    #     # for j,(x,xo,label_name) in enumerate(test_loader):
                    #     for j, (x, xo, label_name, pt_index) in enumerate(test_loader):
                    #         x, xo = x.to(device), xo.to(device)
                    #         recon_x, mean, log_var, z = vae(x)
                    #         z_temp = tensor_to_ndarray(z)
                    #         z_temps = np.append(z_temps, z_temp, axis=0)
                    #         for i in range(len(label_name)):
                    #             label_temps.append(label_name[i])
                    #         mae_loss = MAE_2(reduction="none")
                    #         recon_x = recon_x.view(-1, ecg_ch, datalength)
                    #         xo = xo.view(-1, ecg_ch, datalength)
                    #         acc_mae = mae_loss(recon_x, xo)

                    #         if args.loss_pt_on_off == "off":
                    #             # test_val=test_val+cul_val_no_pt(acc_rmse)#cul_val_no_ptは波形全体でrmse計算
                    #             test_val = test_val + cul_val_no_pt(
                    #                 acc_mae
                    #             )  # cul_val_no_ptは波形全体でmae計算
                    #         else:
                    #             # test_val=test_val+cul_val(pt_index,acc_rmse)#cul_valはptの範囲でRMSE計算
                    #             test_val = test_val + cul_val(
                    #                 pt_index, acc_mae
                    #             )  # cul_valはptの範囲でmae計算i
                    # test_values = [array_item.item() for array_item in test_val]
                    # test_val_ndarray = np.array(test_values, dtype=np.float32)
                    # # print(test_val_ndarray.mean())
                    # test_acc_temps.append(test_val_ndarray.mean())

                    # logs["test_loss"].append(loss_keep.item() / (iteration + 1))
                    # if args.loss_fn_type == "bce":
                    #     logs["test_bce"].append(bce_keep.item() / (iteration + 1))
                    # if args.loss_fn_type == "mse":
                    #     logs["test_mse"].append(mse_keep.item() / (iteration + 1))
                    # logs["test_kdl"].append(kdl_keep.item() / (iteration + 1))
                    # logs["tesx_acc"].append(acc_keep.item() / (iteration + 1)
            if target_weight == "P":
                torch.save(
                    vae.state_dict(), os.path.join("model_pth", "vae_pwave_weight.pth")
                )
                torch.save(
                    vae.state_dict(),
                    os.path.join(args.fig_root, str(ts), "vae_pwave_weight.pth"),
                )
            elif target_weight == "R":
                torch.save(
                    vae.state_dict(), os.path.join("model_pth", "vae_rwave_weight.pth")
                )
                torch.save(
                    vae.state_dict(),
                    os.path.join(args.fig_root, str(ts), "model_rwave_weight.pth"),
                )
            elif target_weight == "T":
                torch.save(
                    vae.state_dict(), os.path.join("model_pth", "vae_twave_weight.pth")
                )
                torch.save(
                    vae.state_dict(),
                    os.path.join(args.fig_root, str(ts), "vae_twave_weight.pth"),
                )

        "テストデータのMAE推移プロット"
        plt.xlabel("epoch")
        plt.ylabel("MAE")
        plt.plot(train_acc_temps, color="blue")
        plt.plot(test_acc_temps, color="red")
        plt.ylim(0, 0.1)
        # plt.plot(test_val,color="red")
        plt.legend(["train_acc", "test_acc"])
        # plt.title("beta={}".format(str(args.beta)))
        plt.title("MAE values of train and test data epoch 0-{}".format(epoch))
        plt.savefig(os.path.join(args.fig_root, str(ts), "test_acc"), dpi=300)
        # plt.show()
        plt.close()

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
            # torch.save(vae.state_dict(), args.pth)
            plt.close()

        # #モデル保存
        if args.loss_fn_type == "mse":
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.plot(logs["loss"], color="blue")
            plt.plot(logs["mse"], color="red")
            plt.plot(logs["kdl"], color="green")
            plt.legend(["Loss_all", "MSE", "KLD"])
            plt.title("beta={}".format(str(args.beta)))
            #    plt.show()
            plt.savefig(
                os.path.join(args.fig_root, str(ts), "loss_epoch_all.png"), dpi=300
            )
            #    plt.show()
            plt.close()
            plt.ylim(0, 100)

            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.plot(logs["loss"], color="blue")
            plt.legend(["loss"])
            plt.title("beta={}".format(str(args.beta)))
            plt.savefig(
                os.path.join(args.fig_root, str(ts), "loss_epoch_epoch.png"), dpi=300
            )
            #    plt.show()
            plt.close()

        # plot_scatter_2d(z=z_temps,labels=label_temps,latent_size=latent_size,ts=ts,args=args)

    elif args.mode == "zplot":
        print("ZPLOT MODE::\n")
        # train_data_loader = DataLoader(
        #     dataset=train_dataset, batch_size=args.train_batch_size, shuffle=False
        # )
        # # dataset=train_dataset, batch_size=len(train_dataset), shuffle=False)

        # pth = args.pth
        # vae.load_state_dict(
        #     torch.load(
        #         os.path.join("model_pth", pth),
        #         map_location=lambda storage, loc: storage,
        #     )
        # )
        # vae.eval()
        # z_temps = np.empty((0, latent_size))
        # label_temps = []
        # with torch.no_grad():
        #     for iteration, (x, xo, label_name, pt_index) in enumerate(
        #         train_data_loader
        #     ):
        #         x, xo = x.to(device), xo.to(device)
        #         recon_x, mean, log_var, z = vae(x)
        #         # 潜在変数の保存
        #         z_temp = tensor_to_ndarray(z.cpu().detach())

        #         z_temps = np.append(z_temps, z_temp, axis=0)
        #         print("z_temps")
        #         print(z_temps)
        #         # print(label_name)
        #         for i in range(len(label_name)):
        #             label_temps.append(label_name[i])
        #         # print(label_temps)
        #         xo = xo.cpu().detach()[::50]
        #         recon_x = recon_x.cpu().detach()[::50]
        #         batch_size_now = xo.shape[0]
        #         # plot_fig_train_name(recon_x=recon_x,xo=xo,datalength=datalength,ts=ts,args=args,iteration_num=0,batch_size_num=batch_size_now,epoch_num=0,label_name=label_name)

        #     plot_scatter_2d(
        #         z=z_temps, labels=label_temps, latent_size=latent_size, ts=ts, args=args
        #     )
        #     # plot_scatter_2d_pos(z=z_temps,labels=label_temps,latent_size=latent_size,ts=ts)

    elif args.mode == "test":
        print("TEST MODE::\n")
        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
        print(len(test_dataset))
        print("aaaaaaaaaaaaaaaaaa")
        print(test_loader)
        # input()
        # P波、R波、T波のVAEをそれぞれ読み込む
        vae_dict["P"].load_state_dict(
            torch.load(
                os.path.join("model_pth", "vae_pwave_weight.pth"),
                map_location=lambda storage, loc: storage,
            )
        )
        vae_dict["R"].load_state_dict(
            torch.load(
                os.path.join("model_pth", "vae_rwave_weight.pth"),
                map_location=lambda storage, loc: storage,
            )
        )
        vae_dict["T"].load_state_dict(
            torch.load(
                os.path.join("model_pth", "vae_twave_weight.pth"),
                map_location=lambda storage, loc: storage,
            )
        )

        vae_dict["P"].eval()
        vae_dict["R"].eval()
        vae_dict["T"].eval()

        # print(vae.state_dict().keys())
        print("===========================")
        print("encorder_Linear1")
        # print(vae.encoder.linear_means.weight)
        # print(vae.encoder.linear_means.bias)
        print("===========================")
        record_loss_eval = []
        min_test_loss = 2000
        z_temps = np.empty((0, latent_size))
        label_temps = []
        test_val_all = []
        test_val_12ch_all = []
        with torch.no_grad():
            for j, (x, xo, label_name, pt_index) in enumerate(test_loader):
                x, xo = x.to(device), xo.to(device)
                recon_x_p, mean_p, log_var_p, z_p = vae_dict["P"](x)
                recon_x_r, mean_r, log_var_r, z_r = vae_dict["R"](x)
                recon_x_t, mean_t, log_var_t, z_t = vae_dict["T"](x)
                before_r_index = 170
                after_r_index = 230
                recon_x = recon_x_p.view(-1, ecg_ch, datalength)
                # before_r_indexより前のデータはrecon_x_pのデータを活用
                x_before_R = recon_x_p.view(-1, ecg_ch, datalength)[
                    :, :, :before_r_index
                ]
                recon_x[:, :, :before_r_index] = x_before_R
                # R波の部分はbefore_r_indexからafter_r_indexまでのデータを活用
                recon_x_r = recon_x_r.view(-1, ecg_ch, datalength)
                x_R = recon_x_r.view(-1, ecg_ch, datalength)[
                    :, :, before_r_index:after_r_index
                ]
                recon_x[:, :, before_r_index:after_r_index] = x_R
                # after_r_indexより後のデータはrecon_x_tのデータを活用
                x_after_R = recon_x_t.view(-1, args.ecg_ch_num, datalength)[
                    :, :, after_r_index:
                ]
                recon_x[:, :, after_r_index:] = x_after_R

                # print(label_name)
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

                acc_rmae_per_batch = test_val

                sample_rate = 500
                sc_pt = pt_index / sample_rate
                sample_num = args.datalength
                xticks = np.linspace(0.0, 1.0 / sample_rate * sample_num, sample_num)

                if numplotfig > 6:
                    numplotfig = 6
                batch_size_now = xo.shape[0]
                # print(batch_size_now)

                pos_begin = extract_between_third_and_fourth_underscore(label_name[0])
                pos_end = extract_between_third_and_fourth_underscore(label_name[-1])

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
                # plot_fig_test_name(recon_x=recon_x,xo=xo,datalength=datalength,ts=ts,args=args,batch_size_num=batch_size_now,label_name=label_name,acc=test_val_12ch,pt_index=pt_index,ecg_ch_names=ecg_ch_names)
                # output_path=save_csv(data=recon_x,args=args,ts=ts,label_name=label_name,data_rec_or_xo='recon_x')
                # output_path=save_csv(data=xo,args=args,ts=ts,label_name=label_name,data_rec_or_xo='xo')
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
                # plot_fig_train(recon_x=recon_x,xo=xo,datalength=datalength,ts=ts,args=args,iteration_num=iteration,batch_size_num=batch_size_now,epoch_num=epoch)
            # plot_scatter_2d(z=z_temps,labels=label_temps,latent_size=latent_size,ts=ts,args=args)
        test_val_all = np.array(test_val_all)
        test_val_mean = np.mean(test_val_all)
        test_val_12ch_all = np.array(test_val_12ch_all)
        print(test_val_12ch_all.shape)
        test_val_12ch_mean = np.mean(test_val_12ch_all, axis=0)
        print(test_val_12ch_mean.shape)
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
        }
        # write_to_csv(file_path='cross_val_results/Datasets={},MAE={}.csv'.format(args.Dataset_name,args.loss_pt_on_off),data=data_to_write)
        output_file = os.path.join(
            args.mae_folder
            + "/MAE_leave_1_out_{}_PRTweight_{}_{}_{}_augumentation={}.csv".format(
                args.Dataset_name,
                str(args.loss_pt_on_off_P_weight),
                str(args.loss_pt_on_off_R_weight),
                str(args.loss_pt_on_off_T_weight),
                args.augumentation,
            )
        )
        write_to_csv(output_file, data=data_to_write)

    elif args.mode == "16ch_only":
        print("16ch_only MODE::\n")
        test_dataset = Dataset.Dataset_setup_12ch_pt_16ch_only(
            TARGET_NAME="osaka",
            transform_type=args.transform_type,
            Dataset_name="16ch_only",
            dataset_num=3,
        )
        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
        print(len(test_dataset))
        pth = args.pth
        vae.load_state_dict(
            torch.load(
                os.path.join("model_pth", pth),
                map_location=lambda storage, loc: storage,
            )
        )
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
        with torch.no_grad():
            for j, (x, label_name, pt_index) in enumerate(test_loader):
                x = x.to(device)
                recon_x, mean, log_var, z = vae(x)
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
                    recon_x_cpu = (
                        recon_x.to("cpu").view(-1, datalength).detach().numpy()
                    )
                    df_recon_x = pd.DataFrame(recon_x_cpu)
                    # df_recon_x.to_csv(os.path.join(args.fig_root, str(ts),"predict_"+str(args.target_set))+".csv",index=False,header=False)
                    # df_recon_x.to_csv(os.path.join(args.fig_root, str(ts),"predict_"+str(args.TARGET_NAME))+".csv",index=False,header=False)

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
                plot_fig_16ch_only(
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
    current_time = "0403_1610_prt_sep"

    parser = argparse.ArgumentParser()
    parser.add_argument("--augumentation", type=str, default="")
    parser.add_argument("--Dataset_name", type=str, default="")
    parser.add_argument("--loss_pt_on_off", type=str, default="off")
    parser.add_argument("--loss_pt_on_off_R_weight", type=str, default="")
    parser.add_argument("--loss_pt_on_off_P_weight", type=str, default="")
    parser.add_argument("--loss_pt_on_off_T_weight", type=str, default="")
    parser.add_argument("--dataset_num", type=int, default=20)  # ICCEの際は16心拍分で

    parser.add_argument("--TARGET_NAME", type=str, default="")
    # parser.add_argument("--dname", type = str ,default = "test")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--train_batch_size", type=int, default=4)  # default=256
    parser.add_argument("--val_batch_size", type=int, default=1)  # default=256
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--datalength", type=int, default=400)
    # parser.add_argument("--enc_convlayer_sizes", type=list, default=[[16, 1], [32, 1], [64, 2]]) #畳み込み層の設定　増やしすぎると過学習の可能性　前から２ペアずつ読み込む　２つ目の[]の第二引数はストライド
    # parser.add_argument("--enc_convlayer_sizes", type=list, default=[[16, 1], [30, 2],[60, 2],[120, 2],[240,2]])#[[入力,ストライド],]
    # parser.add_argument("--enc_convlayer_sizes", type=list, default=[[16, 1], [30, 2],[60, 2],[120,2],[240,2]])#[[入力,ストライド],]
    parser.add_argument(
        "--enc_convlayer_sizes",
        type=list,
        default=[[16, 1], [30, 2]],
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
        "--fig_root", type=str, default=OUTPUT_DIR + "/" + f"figs_newref/{current_time}"
    )  # 学習過程及びテスト結果を出力するフォルダ
    parser.add_argument(
        "--mae_folder", type=str, default=OUTPUT_MAE_DIR + "/" + f"/{current_time}"
    )  # MAE結果を出力するフォルダ
    parser.add_argument("--conditional", action="store_true")
    parser.add_argument("--train_off", action="store_false")

    parser.add_argument(
        "--pth", type=str, default=r"vae_prt_sep.pth"
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
    create_directory_if_not_exists(args.fig_root)
    create_directory_if_not_exists(args.mae_folder)
    main(args)
