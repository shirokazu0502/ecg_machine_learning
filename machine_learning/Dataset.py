import re
from re import A
from tkinter import W
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset
import random
import time
import gc
from scipy.interpolate import interp1d
import neurokit2 as nk
import sys

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
)


def replace_slash_with_underscore(input_string):
    print(input_string.replace("/", "_"))
    return input_string.replace("/", "_")


def Train_Test_person_datas(dirnames, target_name):
    # 人物名のパターンを定義します（例: "姓_スポーツ_数字" の形式）
    # pattern = r'(\w+)_\w+_\w+_\w+'
    train_list = []
    test_list = []
    for string in dirnames:
        pattern = r"(\w+)_\w+_\w+_\w+"
        # 文字列中の人物名を抽出します
        match = re.search(pattern, string)

        if match:
            # 姓を取得します
            last_name = match.group(1)
            if last_name != target_name:
                train_list.append(string)
            else:
                test_list.append(string)
            # 人物名を返します
    return train_list, test_list


def Train_Test_person_datas2(dirnames, target_name):
    # 人物名のパターンを定義します（例: "姓_スポーツ_数字" の形式）
    # pattern = r'(\w+)_\w+_\w+_\w+'
    train_list = []
    test_list = []
    for string in dirnames:
        pattern = r"(\w+)_\w+_\w+"
        # 文字列中の人物名を抽出します
        match = re.search(pattern, string)

        if match:
            # 姓を取得します
            last_name = match.group(1)
            if last_name != target_name:
                train_list.append(string)
            else:
                test_list.append(string)
            # 人物名を返します
    return train_list, test_list


def get_directory_names(directory_path):
    directory_names = []

    for entry in os.scandir(directory_path):
        if entry.is_dir():
            directory_names.append(entry.name)

    return directory_names


def get_directory_names_all(directory_path):
    directory_names = []

    for entry in os.scandir(directory_path):
        if entry.is_dir():
            for entry_in in os.scandir(directory_path + "/" + entry.name):
                if entry_in.is_dir():
                    # print(entry.name+'/'+entry_in.name)
                    dir_name = entry.name + "/" + entry_in.name
                    directory_names.append(dir_name)

    return directory_names


def data_plot_after_splitting2(
    ecg_list: list,
    doppler_list: list,
    npeaks: int,
    target_name: str,
    label_list: list,
    sampling_rate: float = 500,
    figtitle: str = "title",
    savefig: bool = True,
    figpath: str = "./plot_target",
    fontsize: int = 15,
) -> None:
    """心拍を分割したものをまとめてプロットする関数。

    1行目：ECGの時系列波形
    2行目：Dopplerの時系列波形
    3行目：DopplerのCWTスペクトログラム
    4行目：ECG 時系列
    5行目：Doppler 時系列
    6行目：Doppler CWT
          ・
          ・
          ・
    これを繰り返す。

    Parameters
    ----------
    ecg_list : list
        ECGの時系列波形のリスト(要素数は心拍数となる)
    doppler_list : list
        Dopplerの時系列波形のリスト(要素数は心拍数となる)
    doppler_cwt_list : list
        DopplerのCWTスペクトログラムのリスト(要素数は心拍数となる)
    freqs_list : list
        CWTをプロットするときの縦軸に必要な、周波数情報のリスト
    sequence_num_list : list
        データ取得時、BLE通信でデータが欠落していないか確かめるために出力していた連番のリスト
        現在は使っていないので無視してよい
        (適当に空のリスト[[], [], ...(心拍数の数だけ続ける)]をいれるか、そもそもこの関数から消すか)
    npeaks : int
        心拍数
    sampling_rate : float
        サンプリングレート
    figtitle : str
        出力する画像の上部に記載するタイトル（fig.suptitle()）
    savefig: bool=False
        Trueにしたら画像を後述するfigpathに保存する
    figpath: str='./title.png'
        画像出力先パス
    fontsize: int=20
        画像のフォントサイズ


    Returns
    -------
    None

    """
    if npeaks <= 10:  # 心拍数が10を超えたら次の行に移るようにする
        nrow = 3
        ncol = npeaks
    else:
        ncol = 10
        nrow = -(-npeaks // ncol) * 3  # 切り上げ

    fig = plt.figure(figsize=(18, 5 * nrow / 3))

    # 時系列波形
    for peak_idx in range(npeaks):
        ecg = ecg_list[peak_idx]
        doppler = doppler_list[peak_idx]

        N = len(ecg[0])  # サンプル点数
        time_array = np.arange(0, N) / sampling_rate  # グラフ横軸（時間）

        # ECG
        ax1 = fig.add_subplot(nrow, ncol, peak_idx + (peak_idx // ncol) * 2 * ncol + 1)
        for i in range(12):
            ax1.plot(time_array, ecg[i][:])
        ax1.set_title(label_list[peak_idx], fontsize=5)
        # data drop idx：連番(sequence_number)は0-255が連なっていることが期待されるが、そうなっていないものを抽出する。

        # doppler(time)
        ax2 = fig.add_subplot(
            nrow, ncol, peak_idx + (peak_idx // ncol) * 2 * ncol + ncol + 1, sharex=ax1
        )
        # ax2.set_ylim([-100,100])
        # print(len(doppler))
        for i in range(15):
            # ax2.plot(time_array, doppler[:][i])
            ax2.plot(time_array, doppler[i][:])
        # ax2.set_xticks(np.linspace(0, 2, 3),fontsize=5)
        # ax2.set_xticks(np.linspace(0, 2, 5), minor=True)
        # x軸共有のため、上二つのラベルは非表示にする
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.setp(ax2.get_xticklabels(), visible=False)

        # 右と上の枠線を消すcf)https://qiita.com/irs/items/fe909442be057f0efb48
        for ax in [ax1, ax2]:
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.spines["left"].set_linewidth(5)
            ax.spines["bottom"].set_linewidth(5)
            ax.xaxis.set_tick_params(
                direction="in",
                bottom=True,
                top=False,
                left=False,
                right=False,
                length=10,
                width=5,
            )
            ax.yaxis.set_tick_params(
                direction="in",
                bottom=True,
                top=False,
                left=False,
                right=False,
                length=5,
                width=5,
            )

    fig.suptitle(figtitle, fontsize=fontsize)
    fig.tight_layout(
        rect=[0, 0, 1, 0.96]
    )  # rect指定の順番は左下原点で(left,bottom,right,top). suptitle+tight_layout組み合わせる場合は注意
    fig.patch.set_facecolor("white")  # 背景色を白にする

    plt.show()
    # if savefig:
    # fig.savefig(figpath+'/'+target_name+'.png') # 背景を透明にしたければtransparent=Trueオプションを付ければ良いっぽい

    plt.close()


def data_plot_after_splitting(
    ecg_list: list,
    doppler_list: list,
    npeaks: int,
    target_name: str,
    sampling_rate: float = 500,
    figtitle: str = "title",
    savefig: bool = True,
    figpath: str = "./plot_target",
    fontsize: int = 20,
) -> None:
    """心拍を分割したものをまとめてプロットする関数。

    1行目：ECGの時系列波形
    2行目：Dopplerの時系列波形
    3行目：DopplerのCWTスペクトログラム
    4行目：ECG 時系列
    5行目：Doppler 時系列
    6行目：Doppler CWT
          ・
          ・
          ・
    これを繰り返す。

    Parameters
    ----------
    ecg_list : list
        ECGの時系列波形のリスト(要素数は心拍数となる)
    doppler_list : list
        Dopplerの時系列波形のリスト(要素数は心拍数となる)
    doppler_cwt_list : list
        DopplerのCWTスペクトログラムのリスト(要素数は心拍数となる)
    freqs_list : list
        CWTをプロットするときの縦軸に必要な、周波数情報のリスト
    sequence_num_list : list
        データ取得時、BLE通信でデータが欠落していないか確かめるために出力していた連番のリスト
        現在は使っていないので無視してよい
        (適当に空のリスト[[], [], ...(心拍数の数だけ続ける)]をいれるか、そもそもこの関数から消すか)
    npeaks : int
        心拍数
    sampling_rate : float
        サンプリングレート
    figtitle : str
        出力する画像の上部に記載するタイトル（fig.suptitle()）
    savefig: bool=False
        Trueにしたら画像を後述するfigpathに保存する
    figpath: str='./title.png'
        画像出力先パス
    fontsize: int=20
        画像のフォントサイズ


    Returns
    -------
    None

    """
    if npeaks <= 10:  # 心拍数が10を超えたら次の行に移るようにする
        nrow = 3
        ncol = npeaks
    else:
        ncol = 10
        nrow = -(-npeaks // ncol) * 3  # 切り上げ

    fig = plt.figure(figsize=(18, 5 * nrow / 3))

    # 時系列波形
    for peak_idx in range(npeaks):
        ecg = ecg_list[peak_idx]
        doppler = doppler_list[peak_idx]

        N = len(ecg[0])  # サンプル点数
        time_array = np.arange(0, N) / sampling_rate  # グラフ横軸（時間）

        # ECG
        ax1 = fig.add_subplot(nrow, ncol, peak_idx + (peak_idx // ncol) * 2 * ncol + 1)
        ax1.plot(time_array, ecg[1][:], color="tab:blue")
        # data drop idx：連番(sequence_number)は0-255が連なっていることが期待されるが、そうなっていないものを抽出する。

        # doppler(time)
        ax2 = fig.add_subplot(
            nrow, ncol, peak_idx + (peak_idx // ncol) * 2 * ncol + ncol + 1, sharex=ax1
        )
        # ax2.set_ylim([-100,100])
        # print(len(doppler))
        for i in range(15):
            # ax2.plot(time_array, doppler[:][i])
            ax2.plot(time_array, doppler[i][:])

        # x軸共有のため、上二つのラベルは非表示にする
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.setp(ax2.get_xticklabels(), visible=False)

        # 右と上の枠線を消すcf)https://qiita.com/irs/items/fe909442be057f0efb48
        for ax in [ax1, ax2]:
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.spines["left"].set_linewidth(5)
            ax.spines["bottom"].set_linewidth(5)
            ax.xaxis.set_tick_params(
                direction="in",
                bottom=True,
                top=False,
                left=False,
                right=False,
                length=10,
                width=5,
            )
            ax.yaxis.set_tick_params(
                direction="in",
                bottom=True,
                top=False,
                left=False,
                right=False,
                length=5,
                width=5,
            )

    fig.suptitle(figtitle, fontsize=fontsize)
    fig.tight_layout(
        rect=[0, 0, 1, 0.96]
    )  # rect指定の順番は左下原点で(left,bottom,right,top). suptitle+tight_layout組み合わせる場合は注意
    fig.patch.set_facecolor("white")  # 背景色を白にする

    plt.show()
    if savefig:
        fig.savefig(
            figpath + "/" + target_name + ".png"
        )  # 背景を透明にしたければtransparent=Trueオプションを付ければ良いっぽい

    plt.close()


class NormalizeMinorMax_bat(object):

    def __call__(self, in_data):
        size = in_data.shape[0]
        normalized_data = torch.zeros_like(in_data)
        # print(in_data.shape)
        for i in range(size):
            time_data = in_data[i]
            max_val = torch.max(time_data)
            min_val = torch.min(time_data)
            val = max(abs(max_val), abs(min_val))

            normalized_data_tmp = 0.5 * (time_data / val) + 0.5
            # normalized_data_tmp=(time_data-min_val)/(max_val - min_val)
            normalized_data[i, :, :] = normalized_data_tmp
        # plt.plot(in_data[0][0])
        # plt.show()
        # print(in_data.shape)
        return normalized_data


class NormalizeMinMax_bat(object):

    def __call__(self, in_data):
        size = in_data.shape[0]
        normalized_data = torch.zeros_like(in_data)
        # print(in_data.shape)
        for i in range(size):
            time_data = in_data[i]
            max_val = torch.max(time_data)
            min_val = torch.min(time_data)
            normalized_data_tmp = (time_data - min_val) / (max_val - min_val)
            normalized_data[i, :, :] = normalized_data_tmp
        # plt.plot(in_data[0][0])
        # plt.show()
        # print(in_data.shape)
        return normalized_data


class NormalizeMinMax(object):

    def __call__(self, in_data):
        size = in_data.shape[0]
        normalized_data = torch.zeros_like(in_data)
        # print(in_data.shape)
        for i in range(size):
            time_data = in_data[i]
            for ch in range(in_data.shape[1]):
                ch_data = time_data[ch, :]
                max_val = torch.max(ch_data)
                min_val = torch.min(ch_data)
                normalized_ch = (ch_data - min_val) / (max_val - min_val)
                normalized_data[i, ch, :] = normalized_ch
        # plt.plot(in_data[0][0])
        # plt.show()
        # print(in_data.shape)
        return normalized_data


class NormalizeTimeSeries(object):

    def __call__(self, in_data):
        size = in_data.shape[0]
        print(in_data.shape)
        for i in range(size):
            time_data = in_data[i]
            mean = torch.mean(time_data)
            std = torch.std(time_data)
            normalized_time_data = (time_data - mean) / std
            in_data[i] = normalized_time_data
            # print(in_data[i].shape)
        # plt.plot(in_data[0][0])
        # plt.show()
        # print(in_data.shape)
        return in_data


def Normalize(in_data):
    size = in_data.shape[0]
    # print(in_data.shape)
    for i in range(size):
        time_data = in_data[i]
        mean = torch.mean(time_data)
        std = torch.std(time_data)
        normalized_time_data = (time_data - mean) / std
        in_data[i] = normalized_time_data
    # plt.plot(in_data[0][0])
    # plt.show()
    # print(in_data.shape)
    return in_data


class random_slide2(object):
    def __call__(self, data, random_number):
        # random_number=random.randint(0,249)
        # print(data.shape)
        slide_data = torch.zeros_like(data[:, :750])
        slide_data = data[:, random_number : random_number + 750]
        # print(data.shape)
        # data[:,2,:]=data[:,2,:]
        return slide_data


class random_slide(object):
    def __call__(self, data, random_numbers):
        # random_number=random.randint(0,249)
        print(data.shape)
        slide_data = torch.zeros_like(data[:, :, :750])
        for i in range(len(random_numbers)):
            slide_data[i] = data[i, :, random_numbers[i] : random_numbers[i] + 750]
        # print(data.shape)
        # data[:,2,:]=data[:,2,:]
        return slide_data


class Original_Compose(object):
    def __call__(self, data, random_number):
        Normalize = NormalizeTimeSeries()
        data = Normalize(data)
        random_slider = random_slide()
        data = random_slider(data, random_number)
        return data


class MyDataset(TensorDataset):

    def __init__(self, in_data, out_data, name, transform=None, transform2=None):
        self.in_data = in_data
        self.out_data = out_data
        self.data_num = len(in_data)
        self.name = name
        self.transform = transform
        self.transform2 = transform2

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        if self.transform:
            # input("transform___________")
            # print(self.in_data.shape)
            # random_number=random.randint(0,249)
            # mul_data = self.transform(self.in_data,random_number)[idx]
            # ecg_data = self.transform(self.out_data,random_number)[idx]
            mul_data = self.transform(self.in_data)[idx]
            # mul_data = self.transform2(mul_data,random_number)[idx]
            ecg_data = self.transform(self.out_data)[idx]
            # ecg_data = self.transform2(ecg_data,random_number)[idx]
            # mul_data = self.transform(self.in_data,random_number)[idx]
            # ecg_data = self.transform(self.out_data,random_number)[idx]
            name = self.name[idx]
            # print(idx)
            # print(mul_data.shape)
            # plt.plot(in_data[idx][0])

        else:
            mul_data = self.in_data[idx]
            ecg_data = self.out_data[idx]
            name = self.name[idx]

        return mul_data, ecg_data, name


class MyDataset_15ch_only(TensorDataset):
    def __init__(self, in_data, name, pt_index, transform=None):
        self.in_data = in_data
        self.data_num = len(in_data)
        self.name = name
        self.transform = transform
        self.pt_index = pt_index

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        # print(idx)
        # print(self.in_data)
        # print(self.in_data.shape)

        mul_data = self.in_data[idx]
        name = self.name[idx]
        pt_index = self.pt_index[idx]

        return mul_data, name, pt_index


class MyDataset5(TensorDataset):
    def __init__(self, in_data, out_data, name, pt_index, transform=None):
        self.in_data = in_data
        self.out_data = out_data
        self.data_num = len(in_data)
        self.name = name
        self.transform = transform
        self.pt_index = pt_index

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        # print(self.in_data)
        # print(self.in_data.shape)

        mul_data = self.in_data[idx]
        ecg_data = self.out_data[idx]
        name = self.name[idx]
        pt_index = self.pt_index[idx]

        return mul_data, ecg_data, name, pt_index


class MyDataset4(TensorDataset):
    def __init__(self, in_data, out_data, name, pt_index, transform=None):
        self.in_data = in_data
        self.out_data = out_data
        self.data_num = len(in_data)
        self.name = name
        self.transform = transform
        self.pt_index = pt_index

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        batch_size = self.in_data.shape[0]  # バッチサイズじゃない！全体のデータ数
        # print("idx")
        # print(idx)
        # print("batch_size")
        # print(batch_size)
        # random_numbers = torch.randint(low=0, high=251, size=(batch_size,))
        random_number = torch.randint(low=0, high=251, size=(1,))
        # print("random_number")
        # print(random_number)
        # print("random_numbers")
        # print(random_numbers)
        mul_data = self.in_data[idx]
        ecg_data = self.out_data[idx]
        mul_data = self.transform(mul_data, random_number)
        ecg_data = self.transform(ecg_data, random_number)
        # mul_data = self.transform(self.in_data,random_number)[idx]
        # ecg_data = self.transform(self.out_data,random_number)[idx]
        # print("mul_data")
        # print(mul_data.shape[0])
        name = self.name[idx]
        # print("name")
        # print(name)
        # print(random_number)
        # print("pt_index")
        # print(self.pt_index[idx])
        pt_index = self.pt_index[idx] - random_number.detach().numpy().copy()
        # print("pt_index")
        # print(pt_index)

        return mul_data, ecg_data, name, pt_index


class MyDataset3(TensorDataset):
    def __init__(self, in_data, out_data, name, transform=None):
        self.in_data = in_data
        self.out_data = out_data
        self.data_num = len(in_data)
        self.name = name
        self.transform = transform

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        batch_size = self.in_data.shape[0]  # バッチサイズ
        random_numbers = torch.randint(low=0, high=251, size=(batch_size,))
        mul_data = self.transform(self.in_data, random_numbers)[idx]
        ecg_data = self.transform(self.out_data, random_numbers)[idx]
        name = self.name[idx]

        return mul_data, ecg_data, name


class MyDataset2(TensorDataset):
    def __init__(self, in_data, out_data, name, transform=None, transform2=None):
        self.in_data = in_data
        self.out_data = out_data
        self.data_num = len(in_data)
        self.name = name
        self.transform = transform
        self.transform2 = transform2

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        if self.transform:
            if self.transform2:
                # input("transform___________")
                # print(self.in_data.shape)
                # random_number=random.randint(0,249)
                batch_size = self.in_data.shape[0]  # バッチサイズ
                random_numbers = torch.randint(low=0, high=251, size=(batch_size,))
                # random_numbers = torch.from_numpy(random_numbers)
                # mul_data = self.transform(self.in_data,random_number)[idx]
                # ecg_data = self.transform(self.out_data,random_number)[idx]
                mul_data = self.transform(self.in_data)
                ecg_data = self.transform(self.out_data)
                mul_data = self.transform2(mul_data, random_numbers)[idx]
                ecg_data = self.transform2(ecg_data, random_numbers)[idx]
                # mul_data = self.transform(self.in_data,random_number)[idx]
                # ecg_data = self.transform(self.out_data,random_number)[idx]
                # name = self.name[idx]
                # print(idx)
                # print(mul_data.shape)
                # plt.plot(in_data[idx][0])
            else:
                mul_data = self.transform(self.in_data[:, :, 125:875])[idx]
                ecg_data = self.transform(self.out_data[:, :, 125:875])[idx]
                # print(mul_data.shape)

        else:
            if self.transform2:
                batch_size = self.in_data.shape[0]  # バッチサイズ
                random_numbers = torch.randint(low=0, high=251, size=(batch_size,))
                mul_data = self.transform2(self.in_data, random_numbers)[idx]
                ecg_data = self.transform2(self.out_data, random_numbers)[idx]
                # mul_data = self.in_data[idx]
                # ecg_data = self.out_data[idx]
            # name =  self.name[idx]
            else:
                mul_data = self.in_data[:, :, 125:875][idx]
                ecg_data = self.out_data[:, :, 125:875][idx]
                # print(mul_data.shape)

        name = self.name[idx]

        return mul_data, ecg_data, name


def noise_make(mean, scale, datanum, ch_num):
    rnd = np.random.normal(loc=mean, scale=scale, size=datanum * ch_num)
    rnd = rnd.reshape(-1, datanum, ch_num)
    # plt.hist(rnd,bins=ch_num*datanum)
    # plt.show()
    return rnd


def create_noise_data(PGV_torch, mean, scale, datanum, ch_num):
    noise = noise_make(mean, scale, datanum, ch_num)
    PGV_noise = PGV_torch + noise

    return PGV_noise


def min_max_2(x):
    # print("x.shape")
    print(type(x))
    x = x.to("cpu").detach().numpy().copy()
    num = x.shape[0]
    # print("num_of_chanel")
    # print(num)
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
    x = torch.FloatTensor(x)
    return x


# Dataset_make()
def normalize_tensor_data(tensor):
    # データとチャネルの次元を取得
    num_data = tensor.size(0)

    normalized_data = torch.zeros_like(tensor)
    # print(in_data.shape)
    for i in range(num_data):
        time_data = tensor[i]
        max_val = torch.max(time_data)
        min_val = torch.min(time_data)
        val = max(abs(max_val), abs(min_val))

        normalized_data_tmp = 0.5 * (time_data / val) + 0.5
        # normalized_data_tmp=(time_data-min_val)/(max_val - min_val)
        normalized_data[i, :, :] = normalized_data_tmp
    # plt.plot(in_data[0][0])
    # plt.show()
    # print(in_data.shape)
    return normalized_data


def pt_extend(tensor, pt_array):
    # データとチャネルの次元を取得
    num_data = tensor.size(0)
    data_length = tensor.size(2)
    new_data = tensor
    # print(in_data.shape)
    for i in range(num_data):
        time_data = tensor[i]
        pwave = time_data[:, pt_array[0]]
        twave = time_data[:, pt_array[1]]
        # print(pwave)
        # print(twave)
        for j in range(pt_array[0]):
            new_data[i, :, j] = pwave
        for j in range(data_length - pt_array[1]):
            new_data[i, :, pt_array[1] + j] = twave

    return new_data


def linear_interpolation_All(extation_range_ECG, extation_range_PGV, extation_rate):
    # 時系列データの時間情報を正規化
    # print(extantion_range_ECG.shape[1])
    length = extation_range_ECG.shape[1]
    # print(extation_range_ECG)

    x = np.arange(length)
    new_x = np.linspace(0, length - 1, int((length) * extation_rate))
    ECG_shape = (extation_range_ECG.shape[0], len(new_x))
    # print(extation_range_ECG.shape)
    # print(ECG_shape)
    # input("")
    new_tensor_ECG = torch.zeros(ECG_shape, dtype=torch.float32)
    # print(new_tensor_ECG)
    PGV_shape = (extation_range_PGV.shape[0], len(new_x))
    new_tensor_PGV = torch.zeros(PGV_shape, dtype=torch.float32)
    for i in range(extation_range_ECG.shape[0]):
        data = extation_range_ECG[i, :].numpy().copy()
        interpolator = interp1d(x, data)
        new_data = interpolator(new_x)
        new_data_tensor_ECG = torch.tensor(new_data)
        new_tensor_ECG[i] = new_data_tensor_ECG
        # print(new_tensor_ECG[i])
    for i in range(extation_range_PGV.shape[0]):
        data = extation_range_PGV[i, :].numpy().copy()
        interpolator = interp1d(x, data)
        new_data = interpolator(new_x)
        new_data_tensor = torch.tensor(new_data)
        new_tensor_PGV[i] = new_data_tensor
    return new_tensor_ECG, new_tensor_PGV


def make_p_onset_extension_datas(
    PGV_datas, ECG_datas, pt_array, label_name, extation_rate
):
    p_offset_org = pt_array[2]
    r_onset = 190
    # print(ECG_datas)
    extation_range_ECG = ECG_datas[:, p_offset_org:r_onset]
    extation_range_PGV = PGV_datas[:, p_offset_org:r_onset]
    new_extation_range_ECG, new_extation_range_PGV = linear_interpolation_All(
        extation_range_ECG=extation_range_ECG,
        extation_range_PGV=extation_range_PGV,
        extation_rate=extation_rate,
    )
    new_ECG_data = torch.concat(
        [ECG_datas[:, :p_offset_org], new_extation_range_ECG, ECG_datas[:, r_onset:]],
        dim=1,
    )
    # print(new_extation_range_ECG.shape)
    new_PGV_data = torch.concat(
        [PGV_datas[:, :p_offset_org], new_extation_range_PGV, PGV_datas[:, r_onset:]],
        dim=1,
    )
    slide_index = new_ECG_data.shape[1] - 400
    if new_ECG_data.shape[1] > 400:
        new_ECG_data_400 = new_ECG_data[:, (new_ECG_data.shape[1] - 400) :]
        new_PGV_data_400 = new_PGV_data[:, (new_ECG_data.shape[1] - 400) :]
        # print(new_ECG_data_400.shape)
        # print(new_ECG_data.shape)
        # print(new_PGV_data_400.shape)
        # print(new_PGV_data.shape)
    else:
        First_ECG_value_tensor = new_ECG_data[:, 0]
        # 形状を[8, 1]に変更
        First_ECG_value_tensor_view = First_ECG_value_tensor.view(
            First_ECG_value_tensor.shape[0], 1
        )  # または tensor_a.reshape(8, 1)
        First_ECG_value_view_tensors = torch.cat(
            [First_ECG_value_tensor_view] * (400 - new_ECG_data.shape[1]), dim=1
        )
        new_ECG_data_400 = torch.concat(
            [First_ECG_value_view_tensors, new_ECG_data], dim=1
        )

        First_PGV_value_tensor = new_PGV_data[:, -1]
        # 形状を[15, 1]に変更
        First_PGV_value_tensor_view = First_PGV_value_tensor.view(
            First_PGV_value_tensor.shape[0], 1
        )  # または tensor_a.reshape(8, 1)
        First_PGV_value_view_tensors = torch.cat(
            [First_PGV_value_tensor_view] * (400 - new_PGV_data.shape[1]), dim=1
        )
        new_PGV_data_400 = torch.concat(
            [First_PGV_value_view_tensors, new_PGV_data], dim=1
        )
    # print(new_ECG_data_400)
    # print(new_ECG_data_400.shape)
    # print(new_PGV_data_400)
    # print(new_PGV_data_400.shape)
    # input("")
    new_label_name = label_name + "extraction_P=" + str(extation_rate)
    pt_array_augumentation = pt_array.copy()  # numpy　参照渡ししないように気を付ける。
    pt_array_augumentation[0] = (
        pt_array_augumentation[0] - slide_index
    )  # P_onset,P_offsetの新しいやつ
    pt_array_augumentation[2] = pt_array_augumentation[2] - slide_index
    # plot_augumentation(ECG_datas,PGV_datas,pt_array,new_ECG_data_400,new_PGV_data_400,pt_array_augumentation,extation_rate,augumentation="pr")
    # plt.title("ECG_A2+augmentation_rate={}".format(extation_rate))
    # plt.plot(np.arange(0,0.8,0.002),ECG_datas[1],label="org")
    # plt.scatter(pt_array[2]*0.002,ECG_datas[1,pt_array[2]])
    # plt.axvline(x=190*0.002,color='black',linewidth=2,linestyle='--')
    # plt.plot(np.arange(0,0.8,0.002),new_ECG_data_400[1],label="augumentation")
    # plt.scatter(pt_array_augumentation[2]*0.002,new_ECG_data_400[1,pt_array_augumentation[2]])
    # # plt.scatter(pt_array_augumentation[0]*0.002,new_ECG_data_400[1,pt_array_augumentation[0]])
    # plt.legend()
    # plt.show()
    # plt.cla()
    # plt.close()
    # plt.title("15ch_ch2")
    # plt.plot(np.arange(0,0.8,0.002),PGV_datas[1],label="org")
    # plt.scatter(pt_array[0]*0.002,PGV_datas[1,pt_array[0]])
    # plt.plot(np.arange(0,0.8,0.002),new_PGV_data_400[1],label="augumentation")
    # plt.scatter(pt_array_augumentation[0]*0.002,new_PGV_data_400[1,pt_array_augumentation[0]])
    # plt.legend()
    # plt.show()

    # print(type(pt_array))
    # print(pt_array)
    print(pt_array_augumentation)
    # input()
    return new_ECG_data_400, new_PGV_data_400, new_label_name, pt_array_augumentation


def make_t_onset_extension_datas(
    PGV_datas, ECG_datas, pt_array, label_name, extation_rate
):
    t_onset_org = pt_array[3]
    r_offset = 210
    extation_range_ECG = ECG_datas[:, r_offset:t_onset_org]
    extation_range_PGV = PGV_datas[:, r_offset:t_onset_org]
    new_extation_range_ECG, new_extation_range_PGV = linear_interpolation_All(
        extation_range_ECG=extation_range_ECG,
        extation_range_PGV=extation_range_PGV,
        extation_rate=extation_rate,
    )
    # print(new_extation_range_ECG.shape)
    new_ECG_data = torch.concat(
        [ECG_datas[:, :r_offset], new_extation_range_ECG, ECG_datas[:, t_onset_org:]],
        dim=1,
    )
    new_PGV_data = torch.concat(
        [PGV_datas[:, :r_offset], new_extation_range_PGV, PGV_datas[:, t_onset_org:]],
        dim=1,
    )
    slide_index = new_ECG_data.shape[1] - 400
    if new_ECG_data.shape[1] > 400:
        new_ECG_data_400 = new_ECG_data[:, :400]
        new_PGV_data_400 = new_PGV_data[:, :400]

    else:
        last_ECG_value_tensor = new_ECG_data[:, -1]
        # 形状を[8, 1]に変更
        last_ECG_value_tensor_view = last_ECG_value_tensor.view(
            last_ECG_value_tensor.shape[0], 1
        )  # または tensor_a.reshape(8, 1)
        last_ECG_value_view_tensors = torch.cat(
            [last_ECG_value_tensor_view] * (400 - new_ECG_data.shape[1]), dim=1
        )
        new_ECG_data_400 = torch.concat(
            [new_ECG_data, last_ECG_value_view_tensors], dim=1
        )

        last_PGV_value_tensor = new_PGV_data[:, -1]
        # 形状を[8, 1]に変更
        last_PGV_value_tensor_view = last_PGV_value_tensor.view(
            last_PGV_value_tensor.shape[0], 1
        )  # または tensor_a.reshape(8, 1)
        last_PGV_value_view_tensors = torch.cat(
            [last_PGV_value_tensor_view] * (400 - new_PGV_data.shape[1]), dim=1
        )
        new_PGV_data_400 = torch.concat(
            [new_PGV_data, last_PGV_value_view_tensors], dim=1
        )
    # print(new_ECG_data_400)
    # print(new_ECG_data_400.shape)
    # print(new_PGV_data_400)
    # print(new_PGV_data_400.shape)
    # input("")
    new_label_name = label_name + str(extation_rate)
    pt_array_augumentation = pt_array.copy()  # numpy　参照渡ししないように気を付ける。
    pt_array_augumentation[1] = pt_array_augumentation[1] + slide_index
    pt_array_augumentation[3] = pt_array_augumentation[3] + slide_index
    # print(type(pt_array))
    # input()
    # plot_augumentation(ECG_datas,PGV_datas,pt_array,new_ECG_data_400,new_PGV_data_400,pt_array_augumentation,extation_rate,augumentation="rt")
    return new_ECG_data_400, new_PGV_data_400, new_label_name, pt_array_augumentation


def make_pq_extension_datas(PGV_datas, ECG_datas, pt_array, label_name, extation_rate):
    p_offset_org = pt_array[2]
    q_peak = pt_array[5]
    # r_onset=190
    # print(ECG_datas)
    extation_range_ECG = ECG_datas[:, p_offset_org:q_peak]
    extation_range_PGV = PGV_datas[:, p_offset_org:q_peak]
    new_extation_range_ECG, new_extation_range_PGV = linear_interpolation_All(
        extation_range_ECG=extation_range_ECG,
        extation_range_PGV=extation_range_PGV,
        extation_rate=extation_rate,
    )
    new_ECG_data = torch.concat(
        [ECG_datas[:, :p_offset_org], new_extation_range_ECG, ECG_datas[:, q_peak:]],
        dim=1,
    )
    # print(new_extation_range_ECG.shape)
    new_PGV_data = torch.concat(
        [PGV_datas[:, :p_offset_org], new_extation_range_PGV, PGV_datas[:, q_peak:]],
        dim=1,
    )
    slide_index = new_ECG_data.shape[1] - 400
    if new_ECG_data.shape[1] >= 400:
        new_ECG_data_400 = new_ECG_data[:, (new_ECG_data.shape[1] - 400) :]
        new_PGV_data_400 = new_PGV_data[:, (new_ECG_data.shape[1] - 400) :]
        # print(new_ECG_data_400.shape)
        # print(new_ECG_data.shape)
        # print(new_PGV_data_400.shape)
        # print(new_PGV_data.shape)
    else:
        First_ECG_value_tensor = new_ECG_data[:, 0]
        # 形状を[8, 1]に変更
        First_ECG_value_tensor_view = First_ECG_value_tensor.view(
            First_ECG_value_tensor.shape[0], 1
        )  # または tensor_a.reshape(8, 1)
        First_ECG_value_view_tensors = torch.cat(
            [First_ECG_value_tensor_view] * (400 - new_ECG_data.shape[1]), dim=1
        )
        new_ECG_data_400 = torch.concat(
            [First_ECG_value_view_tensors, new_ECG_data], dim=1
        )

        First_PGV_value_tensor = new_PGV_data[:, -1]
        # 形状を[15, 1]に変更
        First_PGV_value_tensor_view = First_PGV_value_tensor.view(
            First_PGV_value_tensor.shape[0], 1
        )  # または tensor_a.reshape(8, 1)
        First_PGV_value_view_tensors = torch.cat(
            [First_PGV_value_tensor_view] * (400 - new_PGV_data.shape[1]), dim=1
        )
        new_PGV_data_400 = torch.concat(
            [First_PGV_value_view_tensors, new_PGV_data], dim=1
        )
    # print(new_ECG_data_400)
    # print(new_ECG_data_400.shape)
    # print(new_PGV_data_400)
    # print(new_PGV_data_400.shape)
    # input("")
    new_label_name = label_name + "extraction_P=" + str(extation_rate)
    pt_array_augumentation = pt_array.copy()  # numpy　参照渡ししないように気を付ける。
    pt_array_augumentation[0] = (
        pt_array_augumentation[0] - slide_index
    )  # P_onset,P_offsetの新しいやつ
    pt_array_augumentation[2] = pt_array_augumentation[2] - slide_index
    pt_array_augumentation[4] = pt_array_augumentation[4] - slide_index
    print(pt_array_augumentation)
    # input("")
    # plot_augumentation_v2(ECG_datas,PGV_datas,pt_array,new_ECG_data_400,new_PGV_data_400,pt_array_augumentation,extation_rate,augumentation="pq")
    # plt.title("ECG_A2+augmentation_rate={}".format(extation_rate))
    # plt.plot(np.arange(0,0.8,0.002),ECG_datas[1],label="org")
    # plt.scatter(pt_array[2]*0.002,ECG_datas[1,pt_array[2]])
    # plt.axvline(x=190*0.002,color='black',linewidth=2,linestyle='--')
    # plt.plot(np.arange(0,0.8,0.002),new_ECG_data_400[1],label="augumentation")
    # plt.scatter(pt_array_augumentation[2]*0.002,new_ECG_data_400[1,pt_array_augumentation[2]])
    # # plt.scatter(pt_array_augumentation[0]*0.002,new_ECG_data_400[1,pt_array_augumentation[0]])
    # plt.legend()
    # plt.show()
    # plt.cla()
    # plt.close()
    # plt.title("15ch_ch2")
    # plt.plot(np.arange(0,0.8,0.002),PGV_datas[1],label="org")
    # plt.scatter(pt_array[0]*0.002,PGV_datas[1,pt_array[0]])
    # plt.plot(np.arange(0,0.8,0.002),new_PGV_data_400[1],label="augumentation")
    # plt.scatter(pt_array_augumentation[0]*0.002,new_PGV_data_400[1,pt_array_augumentation[0]])
    # plt.legend()
    # plt.show()

    # print(type(pt_array))
    # print(pt_array)
    # print(pt_array_augumentation)
    # input()
    return new_ECG_data_400, new_PGV_data_400, new_label_name, pt_array_augumentation


def make_st_extension_datas(PGV_datas, ECG_datas, pt_array, label_name, extation_rate):
    t_onset_org = pt_array[3]
    s_peak = pt_array[6]
    extation_range_ECG = ECG_datas[:, s_peak:t_onset_org]
    extation_range_PGV = PGV_datas[:, s_peak:t_onset_org]
    new_extation_range_ECG, new_extation_range_PGV = linear_interpolation_All(
        extation_range_ECG=extation_range_ECG,
        extation_range_PGV=extation_range_PGV,
        extation_rate=extation_rate,
    )
    # print(new_extation_range_ECG.shape)
    new_ECG_data = torch.concat(
        [ECG_datas[:, :s_peak], new_extation_range_ECG, ECG_datas[:, t_onset_org:]],
        dim=1,
    )
    new_PGV_data = torch.concat(
        [PGV_datas[:, :s_peak], new_extation_range_PGV, PGV_datas[:, t_onset_org:]],
        dim=1,
    )
    slide_index = new_ECG_data.shape[1] - 400
    if new_ECG_data.shape[1] >= 400:
        new_ECG_data_400 = new_ECG_data[:, :400]
        new_PGV_data_400 = new_PGV_data[:, :400]

    else:
        last_ECG_value_tensor = new_ECG_data[:, -1]
        # 形状を[8, 1]に変更
        last_ECG_value_tensor_view = last_ECG_value_tensor.view(
            last_ECG_value_tensor.shape[0], 1
        )  # または tensor_a.reshape(8, 1)
        last_ECG_value_view_tensors = torch.cat(
            [last_ECG_value_tensor_view] * (400 - new_ECG_data.shape[1]), dim=1
        )
        new_ECG_data_400 = torch.concat(
            [new_ECG_data, last_ECG_value_view_tensors], dim=1
        )

        last_PGV_value_tensor = new_PGV_data[:, -1]
        # 形状を[8, 1]に変更
        last_PGV_value_tensor_view = last_PGV_value_tensor.view(
            last_PGV_value_tensor.shape[0], 1
        )  # または tensor_a.reshape(8, 1)
        last_PGV_value_view_tensors = torch.cat(
            [last_PGV_value_tensor_view] * (400 - new_PGV_data.shape[1]), dim=1
        )
        new_PGV_data_400 = torch.concat(
            [new_PGV_data, last_PGV_value_view_tensors], dim=1
        )
    # print(new_ECG_data_400)
    # print(new_ECG_data_400.shape)
    # print(new_PGV_data_400)
    # print(new_PGV_data_400.shape)
    # input("")
    new_label_name = label_name + str(extation_rate)
    pt_array_augumentation = pt_array.copy()  # numpy　参照渡ししないように気を付ける。
    pt_array_augumentation[1] = pt_array_augumentation[1] + slide_index
    pt_array_augumentation[3] = pt_array_augumentation[3] + slide_index
    pt_array_augumentation[7] = pt_array_augumentation[7] + slide_index
    # print(type(pt_array))
    # input()
    # plot_augumentation_v2(ECG_datas,PGV_datas,pt_array,new_ECG_data_400,new_PGV_data_400,pt_array_augumentation,extation_rate)
    # input("")
    print(pt_array_augumentation)
    print(ECG_datas)
    # plot_augumentation_v2(ECG_datas,PGV_datas,pt_array,new_ECG_data_400,new_PGV_data_400,pt_array_augumentation,extation_rate,augumentation="st")
    return new_ECG_data_400, new_PGV_data_400, new_label_name, pt_array_augumentation


# def make_t_height_extation(PGV_datas,ECG_datas,pt_array,label_name,extation_rate):
#     t_onset_org=pt_array[3]
#     t_offset_org=pt_array[1]
#     r_offset=210
#     base_lines_tensor_ECG=ECG_datas[:,pt_array[0]]
#     base_lines_tensor_PGV=PGV_datas[:,pt_array[0]]
#     extation_range_ECG=ECG_datas[:,t_onset_org:t_offset_org]
#     extation_range_PGV=PGV_datas[:,t_onset_org:t_offset_org]
#     # new_tensor_ECG = torch.zeros(extation_range_ECG.shape, dtype=torch.float32)


#     # print(new_tensor_ECG-base_lines_tensor_ECG[:,None])
#     new_extation_range_ECG=(extation_range_ECG-base_lines_tensor_ECG.view(ECG_datas.shape[0],1))*extation_rate+base_lines_tensor_ECG.view(ECG_datas.shape[0],1)
#     new_extation_range_PGV=(extation_range_PGV-base_lines_tensor_PGV.view(PGV_datas.shape[0],1))*extation_rate+base_lines_tensor_PGV.view(PGV_datas.shape[0],1)
#     # *extation_rate+base_lines_tensor_PGV
#     # print(new_extation_range_ECG)
#     # print(new_extation_range_ECG.shape)
#     new_ECG_data=torch.concat([ECG_datas[:,:t_onset_org],new_extation_range_ECG,ECG_datas[:,t_offset_org:]],dim=1)
#     new_PGV_data=torch.concat([PGV_datas[:,:t_onset_org],new_extation_range_PGV,PGV_datas[:,t_offset_org:]],dim=1)
#     new_label_name=label_name+str(extation_rate)
#     plot_augumentation_height(ECG_datas=ECG_datas,PGV_datas=PGV_datas,new_ECG_data_400=new_ECG_data,new_PGV_data_400=new_PGV_data,pt_array=pt_array,pt_array_augumentation=pt_array,extation_rate=extation_rate)
#     return new_ECG_data,new_PGV_data ,new_label_name,pt_array
def sin_wave(point_num, extation_rate):
    A = extation_rate - 1
    frequency = 0.5  # 正弦波の周波数
    # point_num=200
    duration = 1  # 生成する波形の時間（秒）
    # サンプル点を生成
    t = np.linspace(0, duration, int(point_num * 1), endpoint=True)
    y = A * np.sin(2 * np.pi * frequency * t) + 1
    # plt.plot(t, y)
    # plt.xlabel('Time [s]')
    # plt.ylabel('Amplitude')
    # plt.title('Sine Wave')
    # plt.grid(True)
    # plt.show()
    return y


def make_t_height_extation(PGV_datas, ECG_datas, pt_array, label_name, extation_rate):
    t_onset_org = pt_array[3]
    t_offset_org = pt_array[1]
    r_offset = 210
    base_lines_tensor_ECG = ECG_datas[:, pt_array[0]]
    base_lines_tensor_PGV = PGV_datas[:, pt_array[0]]
    extation_range_ECG = ECG_datas[:, t_onset_org:t_offset_org]
    extation_range_PGV = PGV_datas[:, t_onset_org:t_offset_org]
    point_num = t_offset_org - t_onset_org
    extation_rate_sin = torch.tensor(sin_wave(point_num, extation_rate=extation_rate))
    # print(extation_rate_sin.dtype)
    extation_rate_sin = extation_rate_sin.float()
    # input()
    # extation_rate_sin=torch.tensor(sin_y*extation_rate)
    # print(extation_rate_sin.shape)
    # print((extation_range_ECG-base_lines_tensor_ECG.view(ECG_datas.shape[0],1)).shape)
    # input()
    # new_tensor_ECG = torch.zeros(extation_range_ECG.shape, dtype=torch.float32)

    # print(new_tensor_ECG-base_lines_tensor_ECG[:,None])
    new_extation_range_ECG = (
        extation_range_ECG - base_lines_tensor_ECG.view(ECG_datas.shape[0], 1)
    ) * extation_rate_sin + base_lines_tensor_ECG.view(ECG_datas.shape[0], 1)
    new_extation_range_PGV = (
        extation_range_PGV - base_lines_tensor_PGV.view(PGV_datas.shape[0], 1)
    ) * extation_rate_sin + base_lines_tensor_PGV.view(PGV_datas.shape[0], 1)
    # *extation_rate+base_lines_tensor_PGV
    # print(new_extation_range_ECG)
    # print(new_extation_range_ECG.shape)
    new_ECG_data = torch.concat(
        [
            ECG_datas[:, :t_onset_org],
            new_extation_range_ECG,
            ECG_datas[:, t_offset_org:],
        ],
        dim=1,
    )
    new_PGV_data = torch.concat(
        [
            PGV_datas[:, :t_onset_org],
            new_extation_range_PGV,
            PGV_datas[:, t_offset_org:],
        ],
        dim=1,
    )
    new_label_name = label_name + str(extation_rate)
    # plot_augumentation_height(ECG_datas=ECG_datas,PGV_datas=PGV_datas,new_ECG_data_400=new_ECG_data,new_PGV_data_400=new_PGV_data,pt_array=pt_array,pt_array_augumentation=pt_array,extation_rate=extation_rate)
    return new_ECG_data, new_PGV_data, new_label_name, pt_array


def peak_histgram(pt_test_set, pt_train_set, TARGET_NAME, augumentation):
    np_test_pt_set = np.concatenate([arr.reshape(1, -1) for arr in pt_test_set], axis=0)
    np_train_pt_set = np.concatenate(
        [arr.reshape(1, -1) for arr in pt_train_set], axis=0
    )
    # np_test_pt_set=np.concatenate(pt_test_set,axis=0)
    print(np_test_pt_set)
    # print(pt_test_set[0][:])
    print(pt_test_set[:])
    if augumentation == "pr":
        combined_data = np.concatenate([np_test_pt_set[:, 0], np_train_pt_set[:, 0]])
    if augumentation == "rt":
        combined_data = np.concatenate([np_test_pt_set[:, 1], np_train_pt_set[:, 1]])
    if augumentation == "":
        combined_data = np.concatenate([np_test_pt_set[:, 1], np_train_pt_set[:, 1]])
    hist, bin_edges = np.histogram(combined_data * 0.002, bins=20)
    # ヒストグラムの各ビンに対する情報を表示
    for i in range(len(hist)):
        if i == len(hist) - 1:
            print(f"{bin_edges[i]:.2f}以上\t{hist[i]}")
        else:
            print(f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}\t{hist[i]}")
    bin_size = 0.01
    if augumentation == "pr":
        plt.title(TARGET_NAME + "_P_onsets")
        plt.hist(
            np_test_pt_set[:, 0] * 0.002,
            bins=np.arange(
                min(combined_data * 0.002),
                max(combined_data * 0.002) + bin_size,
                bin_size,
            ),
            alpha=0.5,
            color="blue",
            label="test",
        )
        plt.hist(
            np_train_pt_set[:, 0] * 0.002,
            bins=np.arange(
                min(combined_data * 0.002),
                max(combined_data * 0.002) + bin_size,
                bin_size,
            ),
            alpha=0.5,
            color="orange",
            label="train",
        )

    if augumentation == "rt":
        plt.title(TARGET_NAME + "_T_offsets")
        plt.hist(
            np_test_pt_set[:, 1] * 0.002,
            bins=np.arange(
                min(combined_data * 0.002),
                max(combined_data * 0.002) + bin_size,
                bin_size,
            ),
            alpha=0.5,
            color="blue",
            label="test",
        )
        plt.hist(
            np_train_pt_set[:, 1] * 0.002,
            bins=np.arange(
                min(combined_data * 0.002),
                max(combined_data * 0.002) + bin_size,
                bin_size,
            ),
            alpha=0.5,
            color="orange",
            label="train",
        )
    if augumentation == "":
        plt.title(TARGET_NAME + "_T_offsets")
        plt.hist(
            np_test_pt_set[:, 1] * 0.002,
            bins=np.arange(
                min(combined_data * 0.002),
                max(combined_data * 0.002) + bin_size,
                bin_size,
            ),
            alpha=0.5,
            color="blue",
            label="test",
        )
        plt.hist(
            np_train_pt_set[:, 1] * 0.002,
            bins=np.arange(
                min(combined_data * 0.002),
                max(combined_data * 0.002) + bin_size,
                bin_size,
            ),
            alpha=0.5,
            color="orange",
            label="train",
        )
    # plt.hist(np_train_pt_set[:,0]*0.002, bins=bin_edges, alpha=0.5, color='orange', label='train')
    plt.xlabel("second")
    plt.ylabel("number")
    plt.legend()
    plt.show()


def plot_augumentation_height(
    ECG_datas,
    PGV_datas,
    pt_array,
    new_ECG_data_400,
    new_PGV_data_400,
    pt_array_augumentation,
    extation_rate,
):

    plt.title("ECG_A2+augmentation_rate={}".format(extation_rate))
    plt.plot(np.arange(0, 0.8, 0.002), ECG_datas[1], label="org")
    # plt.scatter(pt_array[2]*0.002,ECG_datas[1,pt_array[2]])
    plt.axvline(x=pt_array[1] * 0.002, color="black", linewidth=2, linestyle="--")
    plt.axvline(x=pt_array[3] * 0.002, color="black", linewidth=2, linestyle="--")
    plt.plot(np.arange(0, 0.8, 0.002), new_ECG_data_400[1], label="augumentation")
    # plt.scatter(pt_array_augumentation[2]*0.002,new_ECG_data_400[1,pt_array_augumentation[2]])
    # plt.scatter(pt_array_augumentation[0]*0.002,new_ECG_data_400[1,pt_array_augumentation[0]])
    plt.legend()
    plt.show()
    plt.cla()
    plt.close()

    for i in range(15):
        plt.title("15ch_ch{}".format(i))
        plt.plot(np.arange(0, 0.8, 0.002), PGV_datas[i], label="org")
        plt.plot(np.arange(0, 0.8, 0.002), new_PGV_data_400[i], label="augumentation")
        plt.legend()
        plt.show()
        plt.cla()
        plt.close()
    # plt.title("15ch_ch2")
    # plt.plot(np.arange(0,0.8,0.002),PGV_datas[1],label="org")
    # plt.scatter(pt_array[0]*0.002,PGV_datas[1,pt_array[0]])
    # plt.plot(np.arange(0,0.8,0.002),new_PGV_data_400[1],label="augumentation")
    # plt.scatter(pt_array_augumentation[0]*0.002,new_PGV_data_400[1,pt_array_augumentation[0]])
    # plt.legend()
    # plt.show()


def plot_augumentation(
    ECG_datas,
    PGV_datas,
    pt_array,
    new_ECG_data_400,
    new_PGV_data_400,
    pt_array_augumentation,
    extation_rate,
    augumentation,
):

    plt.title("ECG_A2+augmentation_rate={}".format(extation_rate))
    plt.plot(np.arange(0, 0.8, 0.002), ECG_datas[1], label="org")
    if augumentation == "pr":
        plt.scatter(pt_array[2] * 0.002, ECG_datas[1, pt_array[2]])
        plt.axvline(x=190 * 0.002, color="black", linewidth=2, linestyle="--")
        plt.scatter(
            pt_array_augumentation[2] * 0.002,
            new_ECG_data_400[1, pt_array_augumentation[2]],
        )
    if augumentation == "rt":
        plt.scatter(pt_array[3] * 0.002, ECG_datas[1, pt_array[3]])
        plt.axvline(x=210 * 0.002, color="black", linewidth=2, linestyle="--")
        plt.scatter(
            pt_array_augumentation[3] * 0.002,
            new_ECG_data_400[1, pt_array_augumentation[3]],
        )
    plt.plot(np.arange(0, 0.8, 0.002), new_ECG_data_400[1], label="augumentation")
    # plt.scatter(pt_array_augumentation[0]*0.002,new_ECG_data_400[1,pt_array_augumentation[0]])
    plt.legend()
    plt.show()
    plt.cla()
    plt.close()
    for i in range(15):
        plt.title("15ch_ch{}".format(i))
        plt.plot(np.arange(0, 0.8, 0.002), PGV_datas[i], label="org")
        if augumentation == "pr":
            plt.scatter(pt_array[2] * 0.002, PGV_datas[i, pt_array[2]])
            plt.scatter(
                pt_array_augumentation[2] * 0.002,
                new_PGV_data_400[i, pt_array_augumentation[2]],
            )
        if augumentation == "rt":
            plt.scatter(pt_array[3] * 0.002, PGV_datas[i, pt_array[3]])
            plt.scatter(
                pt_array_augumentation[3] * 0.002,
                new_PGV_data_400[i, pt_array_augumentation[3]],
            )
        plt.plot(np.arange(0, 0.8, 0.002), new_PGV_data_400[i], label="augumentation")
        plt.legend()
        plt.show()
        plt.cla()
        plt.close()


def plot_augumentation_v2(
    ECG_datas,
    PGV_datas,
    pt_array,
    new_ECG_data_400,
    new_PGV_data_400,
    pt_array_augumentation,
    extation_rate,
    augumentation,
):

    # plt.plot(np.arange(0,0.8,0.002),ECG_datas[1],label="Original waveform")
    plt.plot(np.arange(0, 0.8, 0.002), ECG_datas[1], label="Waveform before extension")
    plt.xlim(0, 0.8)
    # plt.axvline(x=190*0.002,color='red',linewidth=2,linestyle='--')
    # plt.axvline(x=210*0.002,color='red',linewidth=2,linestyle='--')
    # plt.axvline(x=pt_array[5]*0.002,color='red',linewidth=2,linestyle='--')
    # plt.axvline(x=pt_array[6]*0.002,color='red',linewidth=2,linestyle='--')
    plt.legend(loc="upper left", fontsize=8)
    plt.savefig("before_waveform.svg")
    plt.show()
    plt.cla()
    plt.close()

    # plt.plot(np.arange(0,0.8,0.002),ECG_datas[1],label="Original waveform")
    plt.plot(np.arange(0, 0.8, 0.002), ECG_datas[1], label="Waveform before extension")
    plt.xlim(0, 0.8)
    if augumentation == "st":
        s_peak = pt_array[6]
        plt.scatter(
            pt_array[3] * 0.002,
            ECG_datas[1, pt_array[3]],
            c="g",
            label="T onset",
            marker="^",
        )
        plt.scatter(
            pt_array[1] * 0.002,
            ECG_datas[1, pt_array[1]],
            c="forestgreen",
            label="T offset",
            marker="v",
        )
        # plt.scatter(s_peak*0.002,ECG_datas[1,s_peak],c='brown',label="T_offset",marker='o')
        plt.scatter(
            s_peak * 0.002, ECG_datas[1, s_peak], c="brown", label="S peak", marker="o"
        )
        plt.axvline(x=s_peak * 0.002, color="black", linewidth=2, linestyle="--")
        plt.axvline(x=pt_array[3] * 0.002, color="black", linewidth=2, linestyle="--")
        plt.axvline(x=pt_array[1] * 0.002, color="black", linewidth=2, linestyle="--")
        # plt.scatter(pt_array_augumentation[3]*0.002,new_ECG_data_400[1,pt_array_augumentation[3]])
    if augumentation == "pq":
        q_peak = pt_array[5]
        plt.scatter(
            pt_array[0] * 0.002,
            ECG_datas[1, pt_array[0]],
            c="b",
            label="P_onset",
            marker="^",
        )
        plt.scatter(
            pt_array[2] * 0.002,
            ECG_datas[1, pt_array[2]],
            c="royalblue",
            label="P_offset",
            marker="v",
        )
        plt.scatter(
            q_peak * 0.002, ECG_datas[1, q_peak], c="y", label="q_peak", marker="o"
        )
        plt.axvline(x=q_peak * 0.002, color="black", linewidth=2, linestyle="--")
        plt.axvline(x=pt_array[0] * 0.002, color="black", linewidth=2, linestyle="--")
        plt.axvline(x=pt_array[2] * 0.002, color="black", linewidth=2, linestyle="--")
    # plt.plot(np.arange(0,0.8,0.002),new_ECG_data_400[1],label="augumentation")
    # plt.scatter(pt_array_augumentation[0]*0.002,new_ECG_data_400[1,pt_array_augumentation[0]])
    # plt.legend()
    plt.legend(loc="upper left", fontsize=8)
    plt.savefig("before_waveform_ST.svg")
    plt.show()
    plt.cla()
    plt.close()

    # plt.title("ECG_A2_extention_rate={}".format(extation_rate))
    plt.title("Extension Ratio={:.3f}".format(extation_rate))
    plt.plot(np.arange(0, 0.8, 0.002), ECG_datas[1], label="Waveform before extension")

    if augumentation == "st":
        # plt.scatter(pt_array[3]*0.002,ECG_datas[1,pt_array[3]])
        plt.scatter(
            pt_array[3] * 0.002,
            ECG_datas[1, pt_array[3]],
            c="g",
            label="T onset",
            marker="^",
        )
        plt.scatter(
            s_peak * 0.002, ECG_datas[1, s_peak], c="brown", label="S peak", marker="o"
        )
        # plt.axvline(x=s_peak*0.002,color='black',linewidth=2,linestyle='--')
        # plt.scatter(pt_array_augumentation[3]*0.002,new_ECG_data_400[1,pt_array_augumentation[3]],c="r",label="T_offset_augumentation",marker='^')
        plt.scatter(
            pt_array_augumentation[3] * 0.002,
            new_ECG_data_400[1, pt_array_augumentation[3]],
            c="r",
            label="T onset after extension",
            marker="^",
        )
    if augumentation == "pq":
        # plt.scatter(pt_array[3]*0.002,ECG_datas[1,pt_array[3]])
        plt.scatter(
            pt_array[2] * 0.002,
            ECG_datas[1, pt_array[2]],
            c="royalblue",
            label="P_offset",
            marker="v",
        )
        # plt.axvline(x=s_peak*0.002,color='black',linewidth=2,linestyle='--')
        plt.scatter(
            pt_array_augumentation[2] * 0.002,
            new_ECG_data_400[1, pt_array_augumentation[2]],
            c="r",
            label="P_offset_augumentation",
            marker="^",
        )
    plt.plot(
        np.arange(0, 0.8, 0.002), new_ECG_data_400[1], label="Waveform after extension"
    )
    # plt.scatter(pt_array_augumentation[0]*0.002,new_ECG_data_400[1,pt_array_augumentation[0]])
    # plt.legend()
    # plt.legend(loc='upper right')
    plt.legend(loc="upper left", fontsize=8)
    base_filename = "extension"
    extension = ".svg"
    st_svg_name = get_unique_filename(base_filename=base_filename, extension=extension)
    plt.savefig(st_svg_name)
    plt.show()
    plt.cla()
    plt.close()


def get_unique_filename(base_filename, extension):
    counter = 1
    unique_filename = base_filename + extension
    while os.path.exists(unique_filename):
        unique_filename = f"{base_filename}_{counter}{extension}"
        counter += 1
    return unique_filename


# 例：基本ファイル名と拡張子を設定
# for i in range(15):
#     plt.title("15ch_ch{}".format(i))
#     plt.plot(np.arange(0,0.8,0.002),PGV_datas[i],label="org")
#     if(augumentation=="pr"):
#         plt.scatter(pt_array[2]*0.002,PGV_datas[i,pt_array[2]])
#         plt.scatter(pt_array_augumentation[2]*0.002,new_PGV_data_400[i,pt_array_augumentation[2]])
#     if(augumentation=="rt"):
#         plt.scatter(pt_array[3]*0.002,PGV_datas[i,pt_array[3]])
#         plt.scatter(pt_array_augumentation[3]*0.002,new_PGV_data_400[i,pt_array_augumentation[3]])
#     plt.plot(np.arange(0,0.8,0.002),new_PGV_data_400[i],label="augumentation")
#     plt.legend()
#     plt.show()
#     plt.cla()
#     plt.close()
def Dataset_setup_8ch_pt_augmentation(
    TARGET_NAME, transform_type, Dataset_name, dataset_num, DataAugumentation, ave_data_flg
):
    datalength = 400
    ecg_ch_num = 8
    Data = []
    PGV_train_set = []
    ECG_train_set = []
    label_train_set = []
    pt_train_set = []
    PGV_test_set = []
    ECG_test_set = []
    label_test_set = []
    pt_test_set = []
    # directory_path = './Dataset/'+Dataset_name
    directory_path = PROCESSED_DATA_DIR + "/" + Dataset_name
    dir_names = get_directory_names_all(directory_path)
    print(dir_names)
    print(len(dir_names))
    Train_list, Test_list = Train_Test_person_datas2(dir_names, target_name=TARGET_NAME)
    print(Train_list)
    print(Test_list)
    ECG_center = []
    PGV_center = []
    drop_col_ecg = [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        18,
        19,
        20,
        21,
    ]
    drop_col_mul = [0, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]
    outputpeak_path = "peak_compare_{}".format(Dataset_name)
    os.makedirs(outputpeak_path, exist_ok=True)
    if ave_data_flg==1: # 平均心拍を利用する場合
        ave_path="moving_ave_datasets/"
    else:
        ave_path = ""
    print(ave_path)
    print("fafafafafafa")
    for j in range(len(Train_list)):
        path_to_dataset = directory_path + "/" + Train_list[j] + "/"
        # input(path_to_dataset)
        for i in range(dataset_num):
            path = path_to_dataset + ave_path + "dataset_{}.csv".format(str(i).zfill(3))
            pt_path = path_to_dataset + "ponset_toffset_{}.csv".format(str(i).zfill(3))

            if not os.path.isfile(path) or not os.path.isfile(pt_path):
                print("no file in " + Train_list[j])
                print(path)
            else:
                label_name = replace_slash_with_underscore(
                    Train_list[j]
                ) + "_dataset{}".format(str(i).zfill(3))
                data = pd.read_csv(path, header=None, skiprows=1)
                df_pt = pd.read_csv(pt_path, header=None, skiprows=1)
                pt_array = np.array(df_pt.iloc[0], dtype=int)
                data_ecg = data.drop(drop_col_ecg, axis=1)
                sh_ecg = data_ecg.shape
                data_ecg.columns = range(sh_ecg[1])

                data_mul = data.drop(drop_col_mul, axis=1)
                sh_mul = data_mul.shape
                data_mul.columns = range(sh_mul[1])

                PGV_train = torch.FloatTensor(data_mul.T.values)
                PGV_train = PGV_train.reshape(-1, 15, datalength)
                PGV_train = normalize_tensor_data(PGV_train)

                # if(transform_type=="normal"):
                # PGV_train = pt_extend(PGV_train.clone(), pt_array)
                PGV_train_set.append(PGV_train)
                ECG_train = torch.FloatTensor(data_ecg.T.values)
                ECG_train = ECG_train.reshape(-1, ecg_ch_num, datalength)
                ECG_train = normalize_tensor_data(ECG_train)
                print("aaaaaaaaaaaaaaaa")
                # xx=ECG_train[0,0]
                # xx2=ECG_train[0,1]
                # yy=PGV_train[0,0]

                # peaks_ecg=nk.ecg_peaks(xx2, 500)  # R波の位置を取得
                # rpeaks_ecg=peaks_ecg[1]['ECG_R_Peaks']
                # peaks_15ch=nk.ecg_peaks(yy, 500)  # R波の位置を取得
                # rpeaks_15ch=peaks_15ch[1]['ECG_R_Peaks']
                # print(rpeaks_ecg)
                # print(rpeaks_15ch)
                # plt.plot(yy,label="15ch_ch1")
                # # plt.plot(xx,label="ECG_A1",color="r")
                # plt.plot(xx2,label="ECG_A2",color="r")
                # plt.scatter(rpeaks_15ch,yy[rpeaks_15ch],label="15ch_ch1_peak")
                # plt.scatter(rpeaks_ecg,xx2[rpeaks_ecg],label="ecg_A2_peak")
                # plt.legend()
                # plt.title(label_name+'ecgpeak={}_15chpeak={}'.format(rpeaks_ecg,rpeaks_15ch))
                # # plt.show()
                # plt.savefig(outputpeak_path+"/{}.png".format(label_name))
                # plt.close()
                # plt.cla()

                # input("")
                # if transform_type == "normal":
                # ECG_train = pt_extend(ECG_train.clone(), pt_array)
                ECG_train_set.append(ECG_train)
                label_train_set.append(label_name)
                pt_train_set.append(pt_array)

                if (
                    DataAugumentation == "height"
                    or DataAugumentation == "rt_and_height"
                ):  # P_offsetからR_onsetを延長するデータ拡張
                    r_offset = 210  #
                    t_offset = pt_array[1]
                    t_onset = pt_array[3]
                    extend_t_height_rates = [
                        0.5,
                        0.6,
                        0.7,
                        0.8,
                        0.9,
                        1.1,
                        1.2,
                        1.3,
                        1.4,
                        1.5,
                    ]
                    for l in range(len(extend_t_height_rates)):
                        # check_value=(datalength-t_offset)/(t_onset-r_offset)#ST部を引き延ばす水増しをしても大丈夫か確かめる指標。1以上でOK
                        (
                            ECG_train_augment_data,
                            PGV_train_augment_data,
                            label_name_augment,
                            pt_array_augument,
                        ) = make_t_height_extation(
                            PGV_train[0],
                            ECG_train[0],
                            pt_array,
                            label_name,
                            extation_rate=extend_t_height_rates[l],
                        )
                        check_bool_ECG = (ECG_train_augment_data > 1).any().item()
                        check_bool_PGV = (PGV_train_augment_data > 1).any().item()
                        if check_bool_ECG == False and check_bool_PGV == False:
                            print(
                                "extend_t_offset_rate:{} is ablable".format(
                                    str(extend_t_height_rates[l])
                                )
                            )
                            PGV_train_set.append(
                                PGV_train_augment_data.view(1, 15, 400)
                            )
                            ECG_train_set.append(ECG_train_augment_data.view(1, 8, 400))
                            label_train_set.append(label_name_augment)
                            pt_train_set.append(pt_array_augument)
                            # input()
                        else:
                            print(
                                "extend_t_height_rate:{} is not ablable".format(
                                    str(extend_t_offset_rates[l])
                                )
                            )
                            print(
                                "name:{},heartbeat_num={}extation_rate={}".format(
                                    TARGET_NAME, str(i), str(l)
                                )
                            )
                            # input("")
                            break

                if DataAugumentation == "pr":  # P_offsetからR_onsetを延長するデータ拡張
                    # r_offset=210#
                    r_onset = 190
                    p_onset = pt_array[0]
                    p_offset = pt_array[2]
                    extend_p_offset_rates = [
                        0.5,
                        0.6,
                        0.7,
                        0.8,
                        0.9,
                        1.1,
                        1.2,
                        1.3,
                        1.4,
                        1.5,
                    ]
                    for l in range(len(extend_p_offset_rates)):
                        check_value_P = (p_onset) / (
                            r_onset - p_offset
                        )  # PR部を引き延ばす水増しをしても大丈夫か確かめる指標。1以上でOK
                        if check_value_P > (extend_p_offset_rates[l] - 1):
                            print(
                                "extend_p_offset_rate:{} is ablable".format(
                                    str(extend_p_offset_rates[l])
                                )
                            )
                            (
                                ECG_train_augment_data,
                                PGV_train_augment_data,
                                label_name_augment,
                                pt_array_augument,
                            ) = make_p_onset_extension_datas(
                                PGV_train[0],
                                ECG_train[0],
                                pt_array,
                                label_name,
                                extation_rate=extend_p_offset_rates[l],
                            )
                            PGV_train_set.append(
                                PGV_train_augment_data.view(1, 15, 400)
                            )
                            ECG_train_set.append(ECG_train_augment_data.view(1, 8, 400))
                            label_train_set.append(label_name_augment)
                            pt_train_set.append(pt_array_augument)
                            # print(pt_array_augument)
                            # input()
                        else:
                            print(
                                "extend_p_offset_rate:{} is not ablable".format(
                                    str(extend_t_offset_rates[l])
                                )
                            )
                            break
                if (
                    DataAugumentation == "rt" or DataAugumentation == "rt_and_height"
                ):  # P_onsetからR_onsetを延長するデータ拡張
                    r_offset = 210  #
                    t_offset = pt_array[1]
                    t_onset = pt_array[3]
                    extend_t_offset_rates = [
                        0.5,
                        0.6,
                        0.7,
                        0.8,
                        0.9,
                        1.1,
                        1.2,
                        1.3,
                        1.4,
                        1.5,
                    ]
                    for l in range(len(extend_t_offset_rates)):
                        check_value = (datalength - t_offset) / (
                            t_onset - r_offset
                        )  # ST部を引き延ばす水増しをしても大丈夫か確かめる指標。1以上でOK
                        if check_value > (extend_t_offset_rates[l] - 1):
                            print(
                                "extend_t_offset_rate:{} is ablable".format(
                                    str(extend_t_offset_rates[l])
                                )
                            )
                            (
                                ECG_train_augment_data,
                                PGV_train_augment_data,
                                label_name_augment,
                                pt_array_augument,
                            ) = make_t_onset_extension_datas(
                                PGV_train[0],
                                ECG_train[0],
                                pt_array,
                                label_name,
                                extation_rate=extend_t_offset_rates[l],
                            )
                            PGV_train_set.append(
                                PGV_train_augment_data.view(1, 15, 400)
                            )
                            ECG_train_set.append(ECG_train_augment_data.view(1, 8, 400))
                            label_train_set.append(label_name_augment)
                            pt_train_set.append(pt_array_augument)
                            # input()
                        else:
                            print(
                                "extend_t_offset_rate:{} is not ablable".format(
                                    str(extend_t_offset_rates[l])
                                )
                            )
                            break
                if (
                    DataAugumentation == "pq" or DataAugumentation == "pq_and_height"
                ):  # P_onsetからR_onsetを延長するデータ拡張
                    q_peak = pt_array[5]  #
                    p_offset = pt_array[2]
                    p_onset = pt_array[0]
                    # extend_t_offset_rates=[0.5,0.6,0.7,0.8,0.9,1.1,1.2,1.3,1.4,1.5]
                    pq_sample = q_peak - p_offset
                    # stte_sample=st_sample+te_sample
                    # rand_int=torch.randint(0,stte_sample,(10,))
                    # extend_t_offset_rates=(rand_int/st_sample).to(torch.float32)
                    extend_p_offset_rates = torch.rand(10) * (
                        1.0 + float(p_onset / pq_sample)
                    )
                    # extend_t_offset_rates=(rand_float/st_sample).to(torch.float32)
                    # print(rand_float)
                    # print()
                    # print(extend_t_offset_rates)
                    # input()
                    # print(rand_int)
                    # print(extend_t_offset_rates)
                    # input("rand_int----------")
                    for l in range(len(extend_p_offset_rates)):
                        check_value = (p_onset) / (
                            q_peak - p_offset
                        )  # PR部を引き延ばす水増しをしても大丈夫か確かめる指標。1以上でOK
                        if check_value > (extend_p_offset_rates[l] - 1):
                            print(
                                "extend_t_offset_rate:{} is ablable".format(
                                    str(extend_p_offset_rates[l])
                                )
                            )
                            (
                                ECG_train_augment_data,
                                PGV_train_augment_data,
                                label_name_augment,
                                pt_array_augument,
                            ) = make_pq_extension_datas(
                                PGV_train[0],
                                ECG_train[0],
                                pt_array,
                                label_name,
                                extation_rate=extend_p_offset_rates[l],
                            )
                            PGV_train_set.append(
                                PGV_train_augment_data.view(1, 15, 400)
                            )
                            ECG_train_set.append(ECG_train_augment_data.view(1, 8, 400))
                            label_train_set.append(label_name_augment)
                            pt_train_set.append(pt_array_augument)
                            # input()
                        else:
                            print(
                                "extend_p_offset_rate:{} is not ablable".format(
                                    str(extend_p_offset_rates[l])
                                )
                            )
                            break
                if (
                    DataAugumentation == "st" or DataAugumentation == "st_and_height"
                ):  # P_onsetからR_onsetを延長するデータ拡張
                    s_peak = pt_array[6]  #
                    t_offset = pt_array[1]
                    t_onset = pt_array[3]
                    # extend_t_offset_rates=[0.5,0.6,0.7,0.8,0.9,1.1,1.2,1.3,1.4,1.5]
                    st_sample = t_onset - s_peak
                    te_sample = datalength - t_offset
                    # stte_sample=st_sample+te_sample
                    # rand_int=torch.randint(0,stte_sample,(10,))
                    # extend_t_offset_rates=(rand_int/st_sample).to(torch.float32)
                    extend_t_offset_rates = torch.rand(10) * (
                        1.0 + float(te_sample / st_sample)
                    )
                    # extend_t_offset_rates=(rand_float/st_sample).to(torch.float32)
                    # print(rand_float)
                    # print()
                    # print(extend_t_offset_rates)
                    # input()
                    # print(rand_int)
                    # print(extend_t_offset_rates)
                    # input("rand_int----------")
                    # input()
                    for l in range(len(extend_t_offset_rates)):
                        check_value = (datalength - t_offset) / (
                            t_onset - s_peak
                        )  # ST部を引き延ばす水増しをしても大丈夫か確かめる指標。1以上でOK
                        if check_value > (extend_t_offset_rates[l] - 1):
                            print(
                                "extend_t_offset_rate:{} is ablable".format(
                                    str(extend_t_offset_rates[l])
                                )
                            )
                            (
                                ECG_train_augment_data,
                                PGV_train_augment_data,
                                label_name_augment,
                                pt_array_augument,
                            ) = make_st_extension_datas(
                                PGV_train[0],
                                ECG_train[0],
                                pt_array,
                                label_name,
                                extation_rate=extend_t_offset_rates[l],
                            )
                            PGV_train_set.append(
                                PGV_train_augment_data.view(1, 15, 400)
                            )
                            ECG_train_set.append(ECG_train_augment_data.view(1, 8, 400))
                            label_train_set.append(label_name_augment)
                            pt_train_set.append(pt_array_augument)
                            # input()
                        else:
                            print(
                                "extend_t_offset_rate:{} is not ablable".format(
                                    str(extend_t_offset_rates[l])
                                )
                            )
                            break

    for i in range(len(label_train_set)):
        print(label_train_set[i], pt_train_set[i])
    
    if ave_data_flg==1: # 平均心拍を利用する場合
        ave_path="moving_ave_datasets/"
    else:
        ave_path = ""
    for j in range(len(Test_list)):
        path_to_dataset = directory_path + "/" + Test_list[j] + "/"
        for i in range(dataset_num):
            path = path_to_dataset + ave_path + "dataset_{}.csv".format(str(i).zfill(3))
            pt_path = path_to_dataset + "ponset_toffset_{}.csv".format(str(i).zfill(3))
            if not os.path.isfile(path):
                print("no file in " + Test_list[j])
                pass
            else:
                label_name = replace_slash_with_underscore(
                    Test_list[j]
                ) + "_dataset{}".format(str(i).zfill(3))
                data = pd.read_csv(path, header=None, skiprows=1)
                df_pt = pd.read_csv(pt_path, header=None, skiprows=1)
                pt_array = np.array(df_pt.iloc[0], dtype=int)

                data_ecg = data.drop(drop_col_ecg, axis=1)
                sh_ecg = data_ecg.shape
                data_ecg.columns = range(sh_ecg[1])

                data_mul = data.drop(drop_col_mul, axis=1)
                sh_mul = data_mul.shape
                data_mul.columns = range(sh_mul[1])
                print("TARGET_NAME={}".format(Test_list[j]))
                PGV_test = torch.FloatTensor(data_mul.T.values)
                print(i)
                PGV_test = PGV_test.reshape(-1, 15, datalength)
                PGV_test = normalize_tensor_data(PGV_test)
                # if(transform_type=="normal"):#これは12ｎ誘導のP波T波の情報をつかってｋ埋めているから良くない⇒P波T波の位置を推論するｗモデルをつくるべき
                # PGV_test = pt_extend(PGV_test.clone(), pt_array)
                print("gggggggggggggggggggg")
                print(PGV_test)
                PGV_test_set.append(PGV_test)

                ECG_test = torch.FloatTensor(data_ecg.T.values)
                ECG_test = ECG_test.reshape(-1, ecg_ch_num, datalength)
                ECG_test = normalize_tensor_data(ECG_test)
                # if transform_type == "normal":
                #     ECG_test = pt_extend(ECG_test.clone(), pt_array)
                ECG_test_set.append(ECG_test)
                label_test_set.append(label_name)
                # print(label_name)
                pt_test_set.append(pt_array)
    # print(PGV_train_set)
    print(PGV_test_set)
    print("kakakakakakakakakkakaka")
    PGV_train_set = torch.cat(PGV_train_set, dim=0)
    ECG_train_set = torch.cat(ECG_train_set, dim=0)
    PGV_test_set = torch.cat(PGV_test_set, dim=0)
    ECG_test_set = torch.cat(ECG_test_set, dim=0)
    # peak_histgram(pt_test_set=pt_test_set,pt_train_set=pt_train_set,TARGET_NAME=TARGET_NAME,augumentation=DataAugumentation)
    # height_histgram(ECG_test_set=ECG_test_set,ECG_train_set=ECG_train_set,pt_test_set=pt_test_set,pt_train_set=pt_train_set,TARGET_NAME=TARGET_NAME)

    if transform_type == "random":
        print("len(pt_train_set)")
        print(len(pt_train_set))
        print(len(PGV_train_set))
        train_dataset = MyDataset4(
            PGV_train_set,
            ECG_train_set,
            label_train_set,
            transform=random_slide2(),
            pt_index=pt_train_set,
        )
        test_dataset = MyDataset4(
            PGV_test_set,
            ECG_test_set,
            label_test_set,
            transform=random_slide2(),
            pt_index=pt_test_set,
        )
        # train_dataset = MyDataset3(PGV_train_set, ECG_train_set,label_train_set,transform=random_slide())
        # test_dataset = MyDataset3(PGV_test_set, ECG_test_set,label_test_set,transform=random_slide())
    elif transform_type == "normal":
        train_dataset = MyDataset5(
            PGV_train_set,
            ECG_train_set,
            label_train_set,
            transform="",
            pt_index=pt_train_set,
        )
        # input(train_dataset)
        test_dataset = MyDataset5(
            PGV_test_set,
            ECG_test_set,
            label_test_set,
            transform="",
            pt_index=pt_test_set,
        )

    elif transform_type == "abnormal":
        train_dataset = MyDataset5(
            PGV_train_set,
            ECG_train_set,
            label_train_set,
            transform="",
            pt_index=pt_train_set,
        )
        # input(train_dataset)
        test_dataset = MyDataset5(
            PGV_test_set,
            ECG_test_set,
            label_test_set,
            transform="",
            pt_index=pt_test_set,
        )
    else:
        train_dataset = MyDataset2(PGV_train_set, ECG_train_set, label_train_set)
        test_dataset = MyDataset2(PGV_test_set, ECG_test_set, label_test_set)

    return train_dataset, test_dataset


# Dataset_setup_8ch_pt(TARGET_NAME="gosha",transform_type="normal",Dataset_name="pqrst_nkmodule",dataset_num=10)
# def height_histgram(ECG_test_set,ECG_train_set,pt_test_set,pt_train_set,TARGET_NAME):
#     np_test_pt_set = np.concatenate([arr.reshape(1, -1) for arr in pt_test_set], axis=0)
#     np_train_pt_set = np.concatenate([arr.reshape(1, -1) for arr in pt_train_set], axis=0)
#     # np_test_pt_set=np.concatenate(pt_test_set,axis=0)
#     # print(np_test_pt_set)
#     # print(pt_test_set[0][:])
#     print(ECG_test_set[:,1,:].shape)
#     selected_indices=torch.tensor(np_test_pt_set[:,0].reshape(10,1).tolist())
#     print(selected_indices)
#     baseline_tensors=torch.gather(ECG_test_set[:,1,:], 1, selected_indices)
#     print(baseline_tensors)
#     input()
#     selected_indices_t=torch.tensor(np_test_pt_set[:,].reshape(10,1).tolist())
#     ECG_test_height=(ECG_test_set[:,1,:]-baseline_tensors)
#     ECG_test_height_T=ECG_test_height[:,sel]
#     print(ECG_test_set[:,1,:]-baseline_tensors)
#     input()
#     combined_data = np.concatenate([np_test_pt_set[:,0],np_train_pt_set[:,0]])
#     hist, bin_edges = np.histogram(combined_data*0.002, bins=20)
#     # ヒストグラムの各ビンに対する情報を表示
#     for i in range(len(hist)):
#         if i == len(hist) - 1:
#             print(f"{bin_edges[i]:.2f}以上\t{hist[i]}")
#         else:
#             print(f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}\t{hist[i]}")
#     bin_size=0.01
#     plt.hist(np_test_pt_set[:,0]*0.002, bins=np.arange(min(combined_data*0.002), max(combined_data*0.002) + bin_size, bin_size), alpha=0.5, color='blue', label='test')
#     plt.hist(np_train_pt_set[:,0]*0.002, bins=np.arange(min(combined_data*0.002), max(combined_data*0.002) + bin_size, bin_size), alpha=0.5, color='orange', label='train')
#     # plt.hist(np_train_pt_set[:,0]*0.002, bins=bin_edges, alpha=0.5, color='orange', label='train')
#     plt.title(TARGET_NAME)
#     plt.xlabel("second")
#     plt.ylabel("number")
#     plt.legend()
#     plt.show()
# Dataset_setup_12ch_pt_15ch_only(TARGET_NAME="osaka",transform_type='normal',Dataset_name='new_sensor_N=6_pt_8s',dataset_num=5)
# train,test=Dataset_setup_12ch(TARGET_NAME="matumoto")
# train,test=Dataset_setup_12ch(0,0)
