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
import neurokit2 as nk
from matplotlib.ticker import MultipleLocator

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


def read_files_with_pattern(directory, pattern):
    """
    指定したディレクトリ内の正規表現パターンに一致するファイル名を読み込む関数。

    Parameters:
    - directory: ファイルを検索するディレクトリのパス
    - pattern: 正規表現パターン

    Returns:
    - 一致するファイル名のリスト
    """
    file_list = []

    # 指定ディレクトリ内のファイルを走査
    for filename in os.listdir(directory):
        # 正規表現でパターンに一致するか確認
        match = re.match(pattern, filename)
        if match:
            file_list.append(os.path.join(directory, filename))

    return file_list


def is_all_elements_integer(array_2d):
    # ndarrayのすべての要素が整数か確認
    if not np.issubdtype(array_2d.dtype, np.integer):
        warnings.warn("ndarray contains non-integer elements.", UserWarning)
        return False
    return True


def check_file_existence(file_path, file_name):
    # 指定されたファイルパスを作成
    full_path = os.path.join(file_path, file_name)

    # ファイルの存在を確認
    if os.path.exists(full_path):
        print(f"{file_name}は存在します。")
        return full_path
    else:
        # ファイルが存在しない場合、ディレクトリが存在するか確認
        if os.path.exists(file_path):
            print(
                f"{file_name}は存在しませんが、指定されたディレクトリ({file_path})は存在します。"
            )
        else:
            print(
                f"{file_name}は存在せず、指定されたディレクトリ({file_path})も存在しません。"
            )
        # return None
        return 0
        # sys.exit()


def df_read_csv(file_path):
    data = pd.read_csv(file_path)
    print(data)
    return data


def find_p_element(arr, target):
    # 配列が空の場合はNoneを返す
    if arr.size == 0:
        return None

    # 初期値として最初の要素を仮の最も近い要素として設定
    closest_element = arr[0]

    # 配列をループして最も近い要素を見つける
    for num in arr:
        # 指定した数以下かつ、現在の要素が仮の最も近い要素よりも近い場合
        if num <= target and abs(num - target) < abs(closest_element - target):
            closest_element = num

    return closest_element


def find_t_element(arr, target):
    # 配列が空の場合はNoneを返す
    if arr.size == 0:
        return None

    # 初期値として最初の要素を仮の最も近い要素として設定
    closest_element = arr[0]

    # 配列をループして最も近い要素を見つける
    for num in arr:
        # 指定した数以上かつ、現在の要素が仮の最も近い要素よりも近い場合
        if num >= target and abs(num - target) < abs(closest_element - target):
            closest_element = num

    return closest_element


class DataFrameBuilder:
    def __init__(self, columns):
        self.columns = columns
        self.df = pd.DataFrame(columns=self.columns)
        print(self.df)

    def add_data(self, data):
        # dataをデータフレームに追加
        new_row = pd.DataFrame([data], columns=self.columns)
        self.df = pd.concat([self.df, new_row], ignore_index=True)
        print(self.df)

    def get_dataframe(self):
        return self.df


def PTwave_search2(args, ecg_A2, header, sampling_rate, signal_type, time_range=0.8):
    # plt.plot(ecg_A2)
    # plt.show()
    ecg_signal_cleaned = nk.ecg_clean(
        ecg_A2, sampling_rate=sampling_rate, method="neurokit"
    )
    ecg_signal = ecg_A2
    # rpeaks = nk.ecg_peaks(ecg_signal, sampling_rate)[1]['ECG_R_Peaks']  # R波の位置を取得
    peaks = nk.ecg_peaks(ecg_signal_cleaned, sampling_rate)  # R波の位置を取得
    rpeaks = peaks[1]["ECG_R_Peaks"]
    print(rpeaks)
    print("uuuuuuuuuuuuuuuuuuuuuuu")
    print(nk.ecg_peaks(ecg_signal, sampling_rate))
    # type='peak'
    type = "cwt"
    # waves_peak_df, waves_peak_dict = nk.ecg_delineate(ecg_signal,
    waves_peak_df, waves_peak_dict = nk.ecg_delineate(
        ecg_signal_cleaned,
        rpeaks,
        sampling_rate=sampling_rate,
        method=type,
        #  method="cwt",
        show=False,
    )
    #  show=True)
    # R波のピーク値を追加
    waves_peak_dict["ECG_R_peaks"] = rpeaks.tolist()
    print(ecg_signal[rpeaks])
    print("eeeeeeeeeeeeeeeeee")
    print(waves_peak_dict)
    for key in waves_peak_dict.keys():
        print(key)
    # input()
    # nk.events_plot([waves_peak_df["ECG_P_Peaks"], waves_peak_df["ECG_T_Peaks"]], ecg_signal)
    # NaNを含まない部分を取り出してからプロット（整数型に変換）
    # input(waves_peak_dict)
    # fig=plt.figure(figsize=(12,12))
    plt.close()
    fig = plt.figure(figsize=(24, 12))
    ax = plt.axes()
    ax.plot(ecg_signal)
    ax.set_ylim(0.3, 0.8)
    color_dict = {
        "P_Onsets": "b",  # 青
        "P_Peaks": "deepskyblue",  # 緑
        "P_Offsets": "royalblue",  # 赤
        "Q_Peaks": "y",  # シアン
        "R_Peaks": "r",  # 青
        "R_Onsets": "darkred",  # マゼンタ
        "R_Offsets": "tomato",  # 黄
        "S_Peaks": "brown",  # 黒
        "T_Onsets": "g",  # 紫
        "T_Peaks": "limegreen",  # オレンジ
        "T_Offsets": "forestgreen",  # ブラウン
    }

    ecg_p_onsets = waves_peak_dict.get("ECG_P_Onsets")
    if ecg_p_onsets is not None:
        valid_ecg_p_onsets = np.array(ecg_p_onsets)[~np.isnan(ecg_p_onsets)].astype(int)
        ax.plot(
            valid_ecg_p_onsets,
            ecg_signal[valid_ecg_p_onsets],
            color_dict["P_Onsets"],
            label="P onset",
            marker="v",
            linestyle="None",
            alpha=0.7,
        )

    ecg_p_peaks = waves_peak_dict.get("ECG_P_Peaks")
    if ecg_p_peaks is not None:
        valid_ecg_p_peaks = np.array(ecg_p_peaks)[~np.isnan(ecg_p_peaks)].astype(int)
        ax.plot(
            valid_ecg_p_peaks,
            ecg_signal[valid_ecg_p_peaks],
            color_dict["P_Peaks"],
            label="P peaks",
            marker="o",
            linestyle="None",
            alpha=0.7,
        )
    # "ECG_S_Peaks"の処理

    if type == "cwt":
        ecg_p_offsets = waves_peak_dict.get("ECG_P_Offsets")
        if ecg_p_offsets is not None:
            valid_ecg_p_offsets = np.array(ecg_p_offsets)[
                ~np.isnan(ecg_p_offsets)
            ].astype(int)
            ax.plot(
                valid_ecg_p_offsets,
                ecg_signal[valid_ecg_p_offsets],
                color_dict["P_Offsets"],
                label="P Offsets",
                marker="^",
                linestyle="None",
                alpha=0.7,
            )

        ecg_q_peaks = waves_peak_dict.get("ECG_Q_Peaks")
        if ecg_q_peaks is not None:
            valid_ecg_q_peaks = np.array(ecg_q_peaks)[~np.isnan(ecg_q_peaks)].astype(
                int
            )
            ax.plot(
                valid_ecg_q_peaks,
                ecg_signal[valid_ecg_q_peaks],
                color_dict["Q_Peaks"],
                label="Q peaks",
                marker="o",
                linestyle="None",
                alpha=0.7,
            )
    # R波
    ax.plot(rpeaks, ecg_signal[rpeaks], "ro", label="R peaks", alpha=0.7)

    ax.set_ylim(0.3, 0.8)
    ecg_s_peaks = waves_peak_dict.get("ECG_S_Peaks")
    if ecg_s_peaks is not None:
        valid_ecg_s_peaks = np.array(ecg_s_peaks)[~np.isnan(ecg_s_peaks)].astype(int)
        ax.plot(
            valid_ecg_s_peaks,
            ecg_signal[valid_ecg_s_peaks],
            color_dict["S_Peaks"],
            label="S peaks",
            marker="o",
            linestyle="None",
            alpha=0.7,
        )

    if type == "cwt":
        ecg_t_onsets = waves_peak_dict.get("ECG_T_Onsets")
        if ecg_t_onsets is not None:
            valid_ecg_t_onsets = np.array(ecg_t_onsets)[~np.isnan(ecg_t_onsets)].astype(
                int
            )
            ax.plot(
                valid_ecg_t_onsets,
                ecg_signal[valid_ecg_t_onsets],
                color_dict["T_Onsets"],
                label="T onset",
                marker="v",
                linestyle="None",
                alpha=0.7,
            )

    ecg_t_peaks = waves_peak_dict.get("ECG_T_Peaks")
    if ecg_t_peaks is not None:
        valid_ecg_t_peaks = np.array(ecg_t_peaks)[~np.isnan(ecg_t_peaks)].astype(int)
        ax.plot(
            valid_ecg_t_peaks,
            ecg_signal[valid_ecg_t_peaks],
            color_dict["T_Peaks"],
            label="T peaks",
            marker="o",
            linestyle="None",
            alpha=0.7,
        )

    ecg_t_offsets = waves_peak_dict.get("ECG_T_Offsets")
    if ecg_t_offsets is not None:
        valid_ecg_t_offsets = np.array(ecg_t_offsets)[~np.isnan(ecg_t_offsets)].astype(
            int
        )
        ax.plot(
            valid_ecg_t_offsets,
            ecg_signal[valid_ecg_t_offsets],
            color_dict["T_Offsets"],
            label="T Offsets",
            marker="^",
            linestyle="None",
            alpha=0.7,
        )

    plt.axvline(x=800, color="red", linewidth=2, linestyle="--")
    plt.axvline(x=8400, color="red", linewidth=2, linestyle="--")
    ax.set_title("{}_{}_{}".format(args.TARGET_NAME, args.TARGET_CHANNEL, signal_type))
    ax.legend()
    ax.grid(True)
    ax.set_ylim(0.3, 0.9)
    # ax.ylim(0.3,0.8)
    plt.tight_layout()
    plt.savefig(
        "{}/{}_{}.png".format(args.Peaks_path, args.TARGET_NAME, args.signal_type)
    )
    # plt.show()
    plt.close()
    return waves_peak_dict, ecg_A2


def test():
    ecg = nk.ecg_simulate(duration=2, sampling_rate=500)
    plt.plot(ecg)
    plt.show()
    _, rpeaks = nk.ecg_peaks(ecg, sampling_rate=500)
    signals, waves = nk.ecg_delineate(ecg, rpeaks, sampling_rate=500)
    nk.events_plot([waves["ECG_P_Peaks"], waves["ECG_T_Peaks"]], ecg)
    # Delineate cardiac cycle
    signals, waves = nk.ecg_delineate(ecg, rpeaks, sampling_rate=500)
    plt.show()
    # input("")


def get_value_in_range_before_R(data, init, datalength):
    # 初期値から終了値までの範囲内に値があるか確認
    print(data)
    target = 0
    for i in range(len(data)):
        if init < data[i] and data[i] < init + (datalength / 2):
            target = data[i]
            # break
    if target == 0:
        return None
    return target


def get_value_in_range_after_R(data, init, datalength):
    # 初期値から終了値までの範囲内に値があるか確認
    print(data)
    target = 0
    for i in range(len(data)):
        if init + (datalength / 2) < data[i] and data[i] < init + datalength:
            target = data[i]
            # break
    if target == 0:
        return None
    return target


def get_value_in_range(data, init, end):
    # 初期値から終了値までの範囲内に値があるか確認
    print(data)
    target = 0
    for i in range(len(data)):
        if init < data[i] and data[i] < end:
            target = data[i]
            # break
    if target == 0:
        return None
    return target


class DataFrameNoneChecker:
    def __init__(self, df, target_column_names):
        """
        データフレームの各行に None が含まれているかどうかを確認し、新しいカラムに結果を格納するクラス。

        Parameters:
        - df: データフレーム
        - new_column_name: 結果を格納する新しいカラムの名前
        """
        self.df = df.copy()
        self.target_column = target_column_names
        new_column_name = "{}_{}_HasNone".format(
            target_column_names[0], target_column_names[1]
        )
        self.new_column_name = new_column_name
        self.df[new_column_name] = False  # 新しいカラムを初期化

    def check_none_in_rows(self):
        """
        各行に対して None の有無を確認し、新しいカラムに結果を格納するメソッド。
        """
        for index, row in self.df.iterrows():
            if row.isnull().any():
                self.df.at[index, self.new_column_name] = True

    def check_none_in_specific_columns(self):
        column1 = self.target_column[0]
        column2 = self.target_column[1]
        """
        指定した二つのカラムに None の有無を確認し、新しいカラムに結果を格納するメソッド。

        Parameters:
        - column1: 検査する最初のカラム名
        - column2: 検査する二番目のカラム名
        """
        for index, row in self.df.iterrows():
            # 指定した二つのカラムに None が含まれている場合に True を格納
            if pd.isnull(row[column1]) or pd.isnull(row[column2]):
                self.df.at[index, self.new_column_name] = True
            else:
                if row[column1] > row[column2]:
                    self.df.at[index, self.new_column_name] = True

    def get_result(self):
        """
        結果を含むデータフレームを取得するメソッド。

        Returns:
        - データフレーム（新しいカラムが追加されたもの）
        """
        return self.df


def find_nearest(array, value):
    """
    配列の中で指定した値に最も近い値のインデックスを返す関数。

    Parameters:
    - array: 対象の配列
    - value: 指定した値

    Returns:
    - 最も近い値のインデックス
    """
    print(array)
    print(value)
    array = np.array(array)
    idx = (np.abs(array - value)).argmin()
    return idx


# def subtract_row_index_times_400(df):
#     """
#     データフレーム内の各要素からその行の行番号×400を引く関数。

#     Parameters:
#     - df: 対象のデータフレーム

#     Returns:
#     - 処理後のデータフレーム
#     """
#     def subtract_row_index_times_400_for_column(series):

#         return series - series.index * 400

#     def subtract_400(value):
#         if pd.api.types.is_integer(value):
#             return value - 400
#         else:
#             return value


#     df_result = df.apply(subtract_row_index_times_400_for_column, axis=0)
#     return df_result
def subtract_row_index_times_400_except(
    df, args, exclude_columns=["baseline", "ECG_P_Heights", "ECG_T_Heights"]
):
    """
    データフレーム内の指定したカラム以外の各要素からその行の行番号×400を引く関数。

    Parameters:
    - df: 対象のデータフレーム
    - exclude_columns: 処理を適用しないカラム名のリスト

    Returns:
    - 処理後のデータフレーム
    """

    def subtract_row_index_times_400_for_column(series):
        if exclude_columns is not None and series.name in exclude_columns:
            return series  # 処理を適用しないカラムはそのまま返す
        return series - series.index * (args.space_length + args.datalength)

    df_result = df.apply(subtract_row_index_times_400_for_column, axis=0)
    return df_result


# def subtract_row_index_times_400(df):
#     """
#     データフレーム内の整数型要素に対してその行の行番号×400を引く関数。

#     Parameters:
#     - df: 対象のデータフレーム

#     Returns:
#     - 処理後のデータフレーム
#     """
#     def subtract_row_index_times_400_for_column(series):
#         # シリーズの各要素が整数型の場合のみ-400する
#         return series.apply(lambda x: x - series.index * 400 if pd.api.types.is_integer_dtype(x) else x)


#     df_result = df.apply(subtract_row_index_times_400_for_column, axis=0)
#     return df_result
def interpolate_values(start, end, space_length):
    # 引数の値が同じ場合はその値を含むリストを返す
    if start == end:
        return [start]

    # 引数の大小を確認して、小さい方から大きい方までの400の等間隔な値を作成する
    if start < end:
        values = [
            start + i * (end - start) / (space_length + 1)
            for i in range(space_length + 2)
        ]
    else:
        values = [
            start - i * (start - end) / (space_length + 1)
            for i in range(space_length + 2)
        ]

    return values[1:-1]


class Append_Peak_Value:
    def __init__(self, df, ecg_signal):
        self.df = df
        self.ecg_signal = ecg_signal
        pass

    def convert_to_numpy(self):
        P_peak_np = self.df.copy()["ECG_P_Peaks"].to_numpy().T
        T_peak_np = self.df.copy()["ECG_T_Peaks"].to_numpy().T
        P_onset_np = self.df.copy()["ECG_P_Onsets"].to_numpy().T
        return P_peak_np, T_peak_np, P_onset_np

    def create_array_with_selected_indices(
        self, original_array, selected_indices, baseline_array
    ):
        result_array = np.empty_like(selected_indices, dtype=object)
        # # print(original_array)
        # print(result_array.shape)
        # array=np.ndarray(shape=(len(selected_indices),))
        # print(original_array)
        # print(selected_indices)

        for i, index in enumerate(selected_indices):
            # print(i,index)
            if index != None and baseline_array[i] != None:
                result_array[i] = (
                    original_array[index] - original_array[baseline_array[i]]
                )

                # print(original_array[index])
            else:
                result_array[i] = None
        # print(result_array)
        # return array
        # print(original_array[index])
        return result_array

    def get_peak_values(self):
        P_peak_np, T_peak_np, P_onset_np = self.convert_to_numpy()
        # print(P_)
        P_Values = self.create_array_with_selected_indices(
            self.ecg_signal, P_peak_np, P_onset_np
        )
        T_Values = self.create_array_with_selected_indices(
            self.ecg_signal, T_peak_np, P_onset_np
        )
        # print(P_peak_np)
        # print(P_Values)
        # print(P_Values.shape)
        # input("")
        return P_Values, T_Values

    def concat_df(self, P_Values, T_Values):
        df_P = pd.Series(P_Values, name="ECG_P_Heights")
        df_T = pd.Series(T_Values, name="ECG_T_Heights")
        new_df = pd.concat([self.df.copy(), df_P, df_T], axis=1)
        return new_df

    def plot_signals(self):
        pass

    def process(self):
        P_Values, T_Values = self.get_peak_values()
        new_df = self.concat_df(P_Values=P_Values, T_Values=T_Values)
        print(self.df)
        print(new_df)
        # input()
        return new_df


def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def main(args):
    # signal_type='xo'
    # signal_type='reconx'
    signal_type = args.signal_type
    file_path = args.file_path
    heart_num = 10
    path = os.path.join(file_path, args.TARGET_NAME)
    dataset_num = 0
    next_data = 0
    for i in range(heart_num):
        # file_name="matumoto_0718_0.8s_0_dataset{:03}_{}.csv".format(i,signal_type)
        file_name = "{}_{}_0.8s_{}_dataset{:03}_{}.csv".format(
            args.TARGET_NAME, args.date, args.pos, i, signal_type
        )
        print(file_name)
        # input()
        full_path = check_file_existence(file_name=file_name, file_path=path)
        if full_path == 0:
            file_name = "{}_{}_0.8s_{}_dataset{:03}_{}.csv".format(
                args.TARGET_NAME, args.date, "left", i, signal_type
            )
            full_path = check_file_existence(file_name=file_name, file_path=path)
        if full_path != None:
            dataset_num += 1
            # print("hello")
            data = df_read_csv(full_path)
            # ecg_A2=data["A2"]
            ecg_A2 = data[args.TARGET_CHANNEL]

            # print(ecg_A2)
            if i == 0:
                ecg_keep = ecg_A2
                # ecg_keep=pd.concat([ecg_keep,ecg_A2],ignore_index=True)
                # print(ecg_keep)
                space_start_data = ecg_A2.iloc[-1]
                print(space_start_data)
                # input("")
            else:
                space_end_data = ecg_A2[0]
                space_values = interpolate_values(
                    start=space_start_data,
                    end=space_end_data,
                    space_length=args.space_length,
                )
                # space_values=pd.Series(space_values)
                space_values = pd.Series(space_values)
                ecg_keep = pd.concat([ecg_keep, space_values], ignore_index=True)

                # input("")
                ecg_keep = pd.concat([ecg_keep, ecg_A2], ignore_index=True)
                space_start_data = ecg_A2.iloc[-1]
                if i == heart_num - 1:
                    last_heart = ecg_A2
            print(ecg_keep)
            # input("")

        else:
            print("{}番目は存在しない".format(dataset_num + 1))
            # input()
            break
    print(ecg_keep)
    space_values = interpolate_values(
        start=space_start_data, end=ecg_keep[0], space_length=args.space_length
    )
    space_values = pd.Series(space_values)
    print(space_values)
    # input("")
    # 一番後ろに８００インデックス分先頭心拍くっつける。
    ecg_keep = pd.concat([ecg_keep, space_values], ignore_index=True)
    ecg_keep = pd.concat([ecg_keep, ecg_keep.copy()[:400]], ignore_index=True)
    # 一番前に８００インデックス分先頭心拍くっつける。
    ecg_periodic = pd.concat([last_heart, space_values], ignore_index=True)
    ecg_keep = pd.concat([ecg_periodic, ecg_keep.copy()], ignore_index=True)

    # print(ecg_keep)
    # input("")
    # print("aaaaaaa")
    ecg_all = ecg_keep.to_numpy().T
    # plt.plot(ecg_all)
    # plt.show()
    # plt.show()
    waves_peak_dict, ecg_signal = PTwave_search2(
        args=args,
        ecg_A2=ecg_all,
        header="A2",
        sampling_rate=500,
        signal_type=signal_type,
    )  # ecg_cleanされてない元の波形
    columns = waves_peak_dict.keys()
    print(type(columns))
    df_builder = DataFrameBuilder(columns=columns)
    # df_builder.add_data([0,0,0,0,0,0,0,0,0,0])
    baseline_values = []
    before_list = [
        "ECG_P_Onsets",
        "ECG_P_Peaks",
        "ECG_P_Offsets",
        "ECG_Q_Peaks",
        "ECG_R_Onsets",
    ]
    for i in range(
        dataset_num + 2
    ):  # periodicにしているから２心拍分要らないのが入ってる。最初と最後の800は使わない。
        datalength = 400
        init_index = (datalength + args.space_length) * i
        # init_index=400*i
        # datalength=2*datalength
        # init_index=2*datalength*i
        baseline_value = ecg_signal[
            init_index
        ]  # ０．８秒間のデータの始めの値を基準線としてS_offsetの計算に用いる。
        new_data = []
        for key in waves_peak_dict.keys():
            wave_name = key
            print(wave_name)
            if wave_name in before_list:
                wave_index = get_value_in_range_before_R(
                    waves_peak_dict.get(key), init=init_index, datalength=datalength
                )
                # input("{}=before".format(wave_name))
            elif wave_name == "ECG_R_peaks":
                for R_peak in waves_peak_dict.get(key):
                    if R_peak >= init_index and R_peak <= init_index + datalength:
                        wave_index = R_peak
                        break
            else:
                wave_index = get_value_in_range_after_R(
                    waves_peak_dict.get(key), init=init_index, datalength=datalength
                )
                # input("{}=after".format(wave_name))
            # p_onsets=get_value_in_range(waves_peak_dict.get("ECG_P_Onsets"),init=init_index,end=init_index+datalength)
            print(wave_index)
            print("oooooooooooo")
            new_data.append(wave_index)
        print(new_data)
        baseline_values.append(baseline_value)
        df_builder.add_data(new_data)

    # ecg_A2_np=ecg_A2.to_numpy().T

    baseline_Series = pd.Series(baseline_values, name="baseline")
    df_waves = df_builder.get_dataframe()
    df_waves_base = pd.concat([df_waves.copy(), baseline_Series], axis=1)
    print(df_waves_base)
    # print(df_waves.copy()["ECG_P_Peaks"].to_numpy().T)
    append_peak_value = Append_Peak_Value(df_waves_base.copy(), ecg_signal)
    df_waves_base_peak_values = append_peak_value.process()
    # input("")
    # df_st=df_waves_base[["ECG_S_Peaks","ECG_T_Onsets","baseline"]].copy()
    # print(df_st)

    none_checker = DataFrameNoneChecker(
        df_waves_base_peak_values, target_column_names=["ECG_S_Peaks", "ECG_T_Onsets"]
    )
    # none_checker.check_none_in_rows()
    none_checker.check_none_in_specific_columns()
    df_ST_checked = none_checker.get_result()
    # print(none_checker.get_result())
    df_new_ST = df_ST_checked.copy()
    print("df_ST_checked")
    print(df_ST_checked)
    # input("")
    S_offsets = []
    for i in range(dataset_num + 2):
        datalength = 400
        init_index = (datalength + args.space_length) * i
        print("{}番目".format(i))
        # print(df_ST_checked.copy()["ECG_S_Peaks"][i])
        print(
            ecg_signal[
                df_ST_checked.copy()["ECG_S_Peaks"][i] : df_ST_checked.copy()[
                    "ECG_T_Onsets"
                ][i]
            ]
        )  # S波とT波の間のecg_波形を表示している。確認済み。
        # print(find_nearest(ecg_signal[df_ST_checked.copy()["ECG_S_Peaks"][i]:df_ST_checked.copy()["ECG_T_Onsets"][i]],value=df_ST_checked.copy()["baseline"][i]))
        if df_ST_checked["ECG_S_Peaks_ECG_T_Onsets_HasNone"][i] != True:
            S_offset_from_S_peak = find_nearest(
                ecg_signal[
                    df_ST_checked.copy()["ECG_S_Peaks"][i] : df_ST_checked.copy()[
                        "ECG_T_Onsets"
                    ][i]
                ],
                value=df_ST_checked.copy()["baseline"][i],
            )  # ベースラインと一番近い値のインデックスをS_onsetとする。
            S_offsets.append(
                int(S_offset_from_S_peak + df_ST_checked.copy()["ECG_S_Peaks"][i])
            )
            # S_offsets.append(S_offset_from_S_peak)
        else:
            S_offsets.append(None)
        # input("")
        # df_new_ST.iloc[i,:-2]=df_ST_checked.copy().iloc[i,:-2]-init_index
        # ecg_signal[]#０．８秒間のデータの始めの値を基準線としてS_offsetの計算に用いる。
    print(S_offsets)
    # print(df_new_ST)
    S_offsets_Series = pd.Series(S_offsets, name="S_Offsets", dtype=pd.Int64Dtype())
    print(S_offsets_Series)
    df_waves_added_S_Offsets = pd.concat(
        [df_waves_base_peak_values.copy(), S_offsets_Series], axis=1
    )
    # print(df_waves_added_S_Offsets)
    df_waves_complite = subtract_row_index_times_400_except(
        df_waves_added_S_Offsets, args=args
    )
    # df_waves_complite["ST_interval"]=df_waves_complite["ECG_S_Offsets"]-df_waves_complite["ECG_T_Onsets"]
    df_STI = calculate_difference(
        df_waves_complite.copy(), "ECG_T_Onsets", "S_Offsets", "ST_interval"
    )
    # print(df_waves_complite)
    # print(df_STI)
    # df_STI.to_csv()
    # input()
    # df_waves_complite.to_csv(os.path.join(path,signal_type+'.csv'),na_rep='<NA>')
    # df_waves_complite.to_csv(os.path.join(args.file_path_os,"{}_{}.csv".format(args.TARGET_NAME,signal_type)),na_rep='')
    # df_waves_complite.to_csv(os.path.join(path,signal_type+'.csv'),na_rep='<NA>')
    # df_st=df_waves.copy()["ECG_S_Peaks","ECG_P_Onsets"]
    df_STI["NAME"] = args.TARGET_NAME
    df_STI = df_STI.copy().iloc[1:-1]
    # input("")
    existing_csv_path = "{}/{}.csv".format(args.file_path, args.signal_type)
    append_to_csv(existing_csv_path, df_STI, na_rep="")
    # ecg_A2_np=ecg_A2.to_numpy().T
    # ecg_A2_np=ecg_A2.to_numpy().T
    # prt_eles=PTwave_search2(ecg_A2=ecg_A2_np,header="A2",sampling_rate=500)
    print(df_STI.columns)
    if args.signal_type == "reconx":
        df_STI = df_STI.add_prefix("recon_")

    return df_STI


# class Compare_df:
#     def __init__(self, df,columns,path):
#         """
#         Compare_df クラスのコンストラクタ

#         Parameters:
#         - df1 (pd.DataFrame): 1つ目のデータフレーム
#         - df2 (pd.DataFrame): 2つ目のデータフレーム
#         """
#         self.columns=columns
#         self.df=df
#         self.path=path
#     def calculate_correlation(self,col1,col2):
#         df=self.df
#         """
#         データフレーム内の二つのカラムの相関係数を計算する関数

#         Parameters:
#         - df (pd.DataFrame): データフレーム
#         - col1 (str): 1つ目のカラム名
#         - col2 (str): 2つ目のカラム名

#         Returns:
#         - float: 二つのカラムの相関係数
#         """
#         # 指定されたカラムの相関係数を計算
#         correlation_coefficient = df[col1].corr(df[col2])

#         return correlation_coefficient

#     def scatter_seconds(self, x_col, y_col):
#         """
#         二つのデータフレームから散布図を描画するメソッド


#         Parameters:
#         - x_col1 (str): 1つ目のデータフレームの x 軸のカラム名
#         - y_col1 (str): 1つ目のデータフレームの y 軸のカラム名
#         - x_col2 (str): 2つ目のデータフレームの x 軸のカラム名
#         - y_col2 (str): 2つ目のデータフレームの y 軸のカラム名
#         """
#         # corr=self.calculate_correlation(col1=x_col,col2=y_col)
#         plt.figure(figsize=(6, 6))

#         plt.plot([0,400],[0,400],label="y=x",color="red")
#         # データフレーム1の散布図
#         plt.scatter(self.df[x_col]*0.002, self.df[y_col]*0.002, label='data', color='blue', alpha=0.7)
#         max_x=self.df[x_col].max()
#         max_y=self.df[y_col].max()
#         min_x=self.df[x_col].min()
#         min_y=self.df[y_col].min()
#         max_range=max(max_x,max_y)
#         min_range=min(min_x,min_y)
#         center=(max_range+min_range)*0.5
#         range_width=(max_range-min_range)*3
#         # print(center,range_width)
#         # input()
#         plt.xlim(center-range_width,center+range_width)
#         plt.ylim(center-range_width,center+range_width)
#         # plt.ylim(min_range,max_range)

#         # グラフのタイトルと軸ラベル
#         # plt.title('Scatter Plot_{},corr={}'.format(y_col,str(corr)))
#         # plt.title('{}_{}'.format(self.df["NAME"][0],y_col))
#         plt.xlabel('{}'.format("12-lead(second)"))
#         plt.ylabel('{}'.format("reconstruction(second)"))

#         # 凡例の表示
#         plt.legend()

#         # グリッドの表示
#         plt.grid(True)

#         # 描画
#         png_path=os.path.join(self.path,"Scatters_png")
#         create_directory_if_not_exists(png_path)
#         # plt.savefig(os.path.join(self.path,"Scatters_png","{}_{}.png".format(self.df["NAME"][0],y_col)))
#         plt.savefig(os.path.join(self.path,"Scatters_png","{}_{}.png".format(self.df["NAME"][0],y_col)))
#         plt.savefig(os.path.join(self.path,"Scatters_png","{}_{}.png".format(self.df["NAME"][0],y_col)))
#         plt.close()
#         plt.cla()
#         # plt.show()
#     def scatter_plot(self, x_col, y_col):
#         """
#         二つのデータフレームから散布図を描画するメソッド


#         Parameters:
#         - x_col1 (str): 1つ目のデータフレームの x 軸のカラム名
#         - y_col1 (str): 1つ目のデータフレームの y 軸のカラム名
#         - x_col2 (str): 2つ目のデータフレームの x 軸のカラム名
#         - y_col2 (str): 2つ目のデータフレームの y 軸のカラム名
#         """
#         # corr=self.calculate_correlation(col1=x_col,col2=y_col)
#         plt.figure(figsize=(6, 6))

#         plt.plot([0,400],[0,400],label="y=x",color="red")
#         # データフレーム1の散布図
#         plt.scatter(self.df[x_col], self.df[y_col], label='data', color='blue', alpha=0.7)
#         max_x=self.df[x_col].max()
#         max_y=self.df[y_col].max()
#         min_x=self.df[x_col].min()
#         min_y=self.df[y_col].min()
#         max_range=max(max_x,max_y)
#         min_range=min(min_x,min_y)
#         center=(max_range+min_range)*0.5
#         range_width=(max_range-min_range)*3
#         # print(center,range_width)
#         # input()
#         plt.xlim(center-range_width,center+range_width)
#         plt.ylim(center-range_width,center+range_width)
#         # plt.ylim(min_range,max_range)

#         # グラフのタイトルと軸ラベル
#         # plt.title('Scatter Plot_{},corr={}'.format(y_col,str(corr)))
#         plt.title('{}_{}'.format(self.df["NAME"][0],y_col))
#         plt.xlabel('{}'.format("12-lead_A2"))
#         plt.ylabel('{}'.format("reconstruction"))

#         # 凡例の表示
#         plt.legend()

#         # グリッドの表示
#         plt.grid(True)

#         # 描画
#         png_path=os.path.join(self.path,"Scatters_png")
#         create_directory_if_not_exists(png_path)
#         plt.savefig(os.path.join(self.path,"Scatters_png","{}_{}.png".format(self.df["NAME"][0],y_col)))
#         plt.close()
#         plt.cla()
#         # plt.show()

#     def process_plot(self):
#         for target_column in self.columns:
#             target_column_reconx="recon_{}".format(target_column)
#             target_column_xo=target_column
#             if(target_column not in ["ECG_P_Heights","ECG_T_Heights","baseline"]):
#                 self.scatter_seconds(y_col=target_column_reconx,x_col=target_column_xo)
#             # self.scatter_plot(x_col=target_column_reconx,y_col=target_column_xo)
#             else:
#                 self.scatter_plot(y_col=target_column_reconx,x_col=target_column_xo)


class Compare_df_all:
    def __init__(self, df, columns, png_path, args):
        """
        Compare_df クラスのコンストラクタ

        Parameters:
        - df1 (pd.DataFrame): 1つ目のデータフレーム
        - df2 (pd.DataFrame): 2つ目のデータフレーム
        """
        self.columns = columns
        self.df = df
        self.png_path = png_path
        self.weights = "{}_{}_{}".format(
            str(args.P_weight), str(args.R_weight), str(args.T_weight)
        )
        self.condition_name = args.condition_name

    def calculate_correlation(self, col1, col2):
        df = self.df
        """
        データフレーム内の二つのカラムの相関係数を計算する関数

        Parameters:
        - df (pd.DataFrame): データフレーム
        - col1 (str): 1つ目のカラム名
        - col2 (str): 2つ目のカラム名

        Returns:
        - float: 二つのカラムの相関係数
        """
        # print(df)
        # input()
        # 指定されたカラムの相関係数を計算
        print(df[col1])
        print(df[col2])
        df_c = df.copy()
        # input()
        df_c[col1]
        df_c[col1] = pd.to_numeric(df_c[col1], errors="coerce").fillna(0).astype(int)
        df_c[col2] = pd.to_numeric(df_c[col2], errors="coerce").fillna(0).astype(int)
        correlation_coefficient = df_c[col1].corr(df_c[col2])

        return correlation_coefficient

    def scatter_seconds(self, x_col, y_col, title):
        df = self.df
        dt = 1 / 500
        """
        二つのデータフレームから散布図を描画するメソッド


        Parameters:
        - x_col1 (str): 1つ目のデータフレームの x 軸のカラム名
        - y_col1 (str): 1つ目のデータフレームの y 軸のカラム名
        - x_col2 (str): 2つ目のデータフレームの x 軸のカラム名
        - y_col2 (str): 2つ目のデータフレームの y 軸のカラム名
        """
        plt.figure(figsize=(6, 6))

        plt.plot([0, 400], [0, 400], label="y=x", color="lime")
        # plt.plot([0,400],[0,380],label="+-5%",color="magenta")
        # plt.plot([0,400],[0,420],label='',color="magenta")
        count_rows_with_none_by_name = df.groupby("NAME").apply(
            lambda x: x[[x_col, y_col]].isna().any(axis=1).sum()
        )
        # print(count_rows_with_none_by_name)
        count_heart = (10 - count_rows_with_none_by_name) / 10.0 * 100
        # print(count_rows_with_none_by_name.sum())
        # input("")
        if count_rows_with_none_by_name.sum() == 0:
            # print(x_col,y_col)
            # input()
            corr = self.calculate_correlation(col1=x_col, col2=y_col)
        else:
            corr = "None"
        # print(df)
        # print(count_heart)
        # input("")

        unique_labels = df["NAME"].unique()
        # print(unique_labels)
        # print(df.index)
        # input("")
        subject_dic = {
            unique_labels[0]: "subject 1",
            unique_labels[1]: "subject 2",
            unique_labels[2]: "subject 3",
            unique_labels[3]: "subject 4",
            unique_labels[4]: "subject 5",
            unique_labels[5]: "subject 6",
        }

        # unique_labels = np.unique(labels)  # ユニークな名前ラベルの取得
        if len(unique_labels) > 10:
            colors = plt.cm.viridis(
                np.linspace(0, 1, len(unique_labels))
            )  # カラーマップから色を生成
        else:
            colors = [
                "red",
                "green",
                "blue",
                "yellow",
                "purple",
                "orange",
                "cyan",
                "magenta",
                "lime",
                "pink",
            ]
        for i, label in enumerate(unique_labels):
            # print(df.loc[df["NAME"] == label])
            indices = df[df["NAME"] == label].index  # 各名前ラベルのインデックスを取得
            # print(label)
            # print(indices)
            # plt.scatter(df.loc[indices, x_col]*dt, df.loc[indices, y_col]*dt, label=label, color=colors[i], alpha=0.7)
            plt.scatter(
                df.loc[indices, x_col] * dt,
                df.loc[indices, y_col] * dt,
                label=subject_dic[label],
                color=colors[i],
                alpha=0.7,
            )
            # plt.scatter(df.loc[indices, x_col]*dt, df.loc[indices, y_col]*dt, label=label+"_"+str(count_heart[label])+"%", color=colors[i], alpha=0.7)
            # print("{}番目".format(i+1))
            # print(count_heart[label])
            # input("")
            # plt.show()
            # print(colors[i])

        # データフレーム1の散布図
        # plt.scatter(df[x_col], df[y_col], label='data', color='blue', alpha=0.7)
        max_x = df[x_col].max()
        max_y = df[y_col].max()
        min_x = df[x_col].min()
        min_y = df[y_col].min()
        max_range = max(max_x, max_y)
        min_range = min(min_x, min_y)
        center = (max_range + min_range) * 0.5
        range_width = (max_range - min_range) * 1.5
        # print(center,range_width)
        # input()
        plt.xlim(dt * (center - range_width), dt * (center + range_width))
        plt.ylim(dt * (center - range_width), dt * (center + range_width))
        if x_col == "ECG_P_Peaks":
            plt.xlim(0.1, 0.3)
            plt.ylim(0.1, 0.3)
        if x_col == "ECG_T_Peaks":
            plt.xlim(0.55, 0.75)
            plt.ylim(0.55, 0.75)
        # plt.ylim(min_range,max_range)

        # グラフのタイトルと軸ラベル
        # plt.title('Scatter Plot_{},corr={}'.format(y_col,str(corr)))
        # plt.title('{}'.format(title))
        # plt.title('{}_corr={}'.format(title,corr))
        # plt.xlabel('{}'.format(x_col))
        # plt.ylabel('{}'.format(y_col))

        # if(x_col=="ECG_T_Peaks"):
        #     plt.xlabel('{}'.format("T-wave peak position of induction II of 12-lead ECG (s)"))
        #     plt.ylabel('{}'.format("T-wave peak position of induction II of reconstruction ECG (s)"))
        # else:
        #     plt.xlabel('{}'.format("12-lead(second)"))
        #     plt.ylabel('{}'.format("reconstruction(second)"))

        # plt.xlabel('{}'.format("12-lead_A2"))
        # plt.ylabel('{}'.format("reconstruction"))
        # plt.xticks(fontsize=16)
        plt.yticks(fontsize=20)
        plt.gca().xaxis.set_minor_locator(MultipleLocator(0.01))
        plt.gca().xaxis.set_major_locator(MultipleLocator(0.05))
        plt.gca().yaxis.set_minor_locator(MultipleLocator(0.01))
        plt.gca().yaxis.set_major_locator(MultipleLocator(0.05))
        plt.xticks(
            [0.55, 0.60, 0.65, 0.70, 0.75],
            ["0.15", "0.20", "0.25", "0.30", "0.35"],
            fontsize=20,
        )
        plt.yticks(
            [0.55, 0.60, 0.65, 0.70, 0.75],
            ["0.15", "0.20", "0.25", "0.30", "0.35"],
            fontsize=20,
        )
        # 主目盛りのグリッドをより目立たせる
        plt.grid(which="major", linestyle="-", linewidth=1)
        plt.grid(which="minor", linestyle="-", linewidth=0.5)
        # plt.minorticks_on()
        # 凡例の表示
        # plt.legend(fontsize=16,ncol=2,bbox_to_anchor=(1.05,1),loc="upper left")

        # グリッドの表示
        plt.grid(True)

        # 描画
        png_path = self.png_path
        create_directory_if_not_exists(png_path)
        if x_col == "ECG_T_Peaks":
            plt.savefig(
                os.path.join(
                    self.png_path, "{}_{}.png".format(self.condition_name, x_col)
                )
            )
            plt.savefig(
                os.path.join(
                    self.png_path, "{}_{}.svg".format(self.condition_name, x_col)
                )
            )
        # plt.show()
        plt.close()
        plt.cla()
        # plt.show()

    def scatter_plot(self, x_col, y_col, title):
        df = self.df
        """
        二つのデータフレームから散布図を描画するメソッド


        Parameters:
        - x_col1 (str): 1つ目のデータフレームの x 軸のカラム名
        - y_col1 (str): 1つ目のデータフレームの y 軸のカラム名
        - x_col2 (str): 2つ目のデータフレームの x 軸のカラム名
        - y_col2 (str): 2つ目のデータフレームの y 軸のカラム名
        """
        # corr=self.calculate_correlation(col1=x_col,col2=y_col)
        plt.figure(figsize=(6, 6))

        plt.plot([0, 400], [0, 400], label="y=x", color="lime")
        plt.plot([0, 400], [0, 360], label="+-10%", color="magenta")
        plt.plot([0, 400], [0, 440], label="", color="magenta")

        unique_labels = df["NAME"].unique()

        count_rows_with_none_by_name = df.groupby("NAME").apply(
            lambda x: x[[x_col, y_col]].isna().any(axis=1).sum()
        )
        print(count_rows_with_none_by_name)
        count_heart = (10 - count_rows_with_none_by_name) / 10.0 * 100
        if count_rows_with_none_by_name.sum() == 0:
            corr = self.calculate_correlation(col1=x_col, col2=y_col)
        else:
            corr = "None"
        if len(unique_labels) > 10:
            colors = plt.cm.viridis(
                np.linspace(0, 1, len(unique_labels))
            )  # カラーマップから色を生成
        else:
            colors = [
                "red",
                "green",
                "blue",
                "yellow",
                "purple",
                "orange",
                "cyan",
                "magenta",
                "lime",
                "pink",
            ]
        for i, label in enumerate(unique_labels):
            # print(df.loc[df["NAME"] == label])
            indices = df[df["NAME"] == label].index  # 各名前ラベルのインデックスを取得
            # print(label)
            # print(indices)
            plt.scatter(
                df.loc[indices, x_col],
                df.loc[indices, y_col],
                label=label + "_" + str(count_heart[label]) + "%",
                color=colors[i],
                alpha=0.7,
            )
            # plt.show()
            # print(colors[i])

        # データフレーム1の散布図
        # plt.scatter(df[x_col], df[y_col], label='data', color='blue', alpha=0.7)
        max_x = df[x_col].max()
        max_y = df[y_col].max()
        min_x = df[x_col].min()
        min_y = df[y_col].min()
        max_range = max(max_x, max_y)
        min_range = min(min_x, min_y)
        center = (max_range + min_range) * 0.5
        range_width = (max_range - min_range) * 1
        # print(center,range_width)
        # input()
        plt.xlim(center - range_width, center + range_width)
        plt.ylim(center - range_width, center + range_width)
        if x_col == "ECG_P_Heights":
            plt.xlim(0.0, 0.1)
            plt.ylim(0.0, 0.1)
            if max_range > 0.2 or min_range < 0.0:
                plt.xlim(center - range_width, center + range_width)
                plt.ylim(center - range_width, center + range_width)

        if x_col == "ECG_T_Heights":
            plt.xlim(0.0, 0.25)
            plt.ylim(0.0, 0.25)
            if max_range > 0.25 or min_range < 0.0:
                plt.xlim(center - range_width, center + range_width)
                plt.ylim(center - range_width, center + range_width)
        # plt.ylim(center-range_width,center+range_width)
        plt.xlabel("{}".format("12-lead_A2"))
        plt.ylabel("{}".format("reconstruction"))

        plt.title("{}_corr={}".format(title, corr))

        # 凡例の表示
        plt.legend()

        # グリッドの表示
        plt.grid(True)

        # 描画
        png_path = self.png_path
        create_directory_if_not_exists(png_path)

        plt.savefig(
            os.path.join(self.png_path, "{}_{}.png".format(self.condition_name, x_col))
        )
        # plt.savefig(os.path.join(self.png_path,"{}_PRTweights_{}.png".format(x_col,self.weights)))
        # plt.show()
        plt.close()
        plt.cla()
        # plt.show()

    def process_plot(self):
        for target_column in self.columns:
            target_column_reconx = "recon_{}".format(target_column)
            target_column_xo = target_column
            if target_column not in ["ECG_P_Heights", "ECG_T_Heights", "baseline"]:
                self.scatter_seconds(
                    y_col=target_column_reconx,
                    x_col=target_column_xo,
                    title=target_column,
                )
            else:
                self.scatter_plot(
                    y_col=target_column_reconx,
                    x_col=target_column_xo,
                    title=target_column,
                )


def append_to_csv(existing_csv, new_data, na_rep):
    """
    既存のCSVファイルにデータフレームを追記する関数。ファイルが存在しない場合は新規作成。

    Parameters:
    - existing_csv: 既存のCSVファイルのパス
    - new_data: 追加するデータフレーム

    Returns:
    - なし
    """
    try:
        # 既存のCSVファイルが存在するか確認
        if os.path.exists(existing_csv):
            # 既存のCSVファイルを読み込み
            existing_df = pd.read_csv(existing_csv)

            # 新しいデータを既存のデータフレームに追加
            updated_df = pd.concat([existing_df, new_data], ignore_index=True)
        else:
            # 既存のCSVファイルが存在しない場合は新しく作成
            updated_df = new_data

        # 更新後のデータフレームをCSVファイルに書き込み
        updated_df.to_csv(existing_csv, index=False, na_rep=na_rep)

        print("Data appended successfully.")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


def calculate_difference(df, column1, column2, new_column):
    """
    DataFrameの指定した2つのカラムの差を計算し、新しいカラムに格納する関数。

    Parameters:
    - df: 対象のDataFrame
    - column1: 差を計算する元となるカラム1
    - column2: 差を計算する元となるカラム2
    - new_column: 結果を格納する新しいカラムの名前

    Returns:
    - 処理後のDataFrame
    """
    df[new_column] = None  # 新しいカラムをNoneで初期化

    # カラム1とカラム2が整数である場合、差を計算し新しいカラムに格納
    df[new_column] = df.apply(
        lambda row: (
            row[column1] - row[column2]
            if pd.notna(row[column1])
            and pd.notna(row[column2])
            and isinstance(row[column1], int)
            and isinstance(row[column2], int)
            else None
        ),
        axis=1,
    )

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--TARGET_NAME", type=str, default="")
    parser.add_argument("--signal_type", type=str, default="")
    parser.add_argument("--TARGET_CHANNEL", type=str, default="A2")
    parser.add_argument("--date", type=str, default="")
    parser.add_argument("--file_path", type=str, default="")
    parser.add_argument("--file_path_os", type=str, default="")
    parser.add_argument("--dataset_date", type=str, default="")
    parser.add_argument("--names", type=str, default="")
    parser.add_argument("--measure_dates", type=str, default="")
    parser.add_argument("--dataset_ver", type=str, default="")
    parser.add_argument("--subjects_group", type=str, default="")
    parser.add_argument("--space_length", type=int, default=400)
    parser.add_argument("--datalength", type=int, default=400)
    parser.add_argument("--peak_method", type=str, default="cwt")
    parser.add_argument("--ping_path", type=str, default="")
    parser.add_argument("--pos", type=str, default="")
    parser.add_argument("--subject_group", type=str, default="")
    parser.add_argument("--augumentation", type=str, default="")
    parser.add_argument("--P_weight", type=str, default="")
    parser.add_argument("--R_weight", type=str, default="")
    parser.add_argument("--T_weight", type=str, default="")
    parser.add_argument("--condition_name", type=str, default="")
    args = parser.parse_args()
    # args.file_path_os="leave_1_out_1216_normal"
    # args.file_path_os="leave_1_out_pqrst_nkmodule_since1215_cwt_R_weight_1"
    # args.file_path_os="leave_1_out_pqrst_nkmodule_since1215_cwt_R_weight_0.1"
    # args.file_path_os="leave_1_out_pqrst_nkmodule_since1215_cwt_without_R_1_10"
    # args.file_path_os="leave_1_out_pqrst_nkmodule_since1215_cwt_R_weight_0.0001"
    # args.file_path_os="leave_1_out_pqrst_nkmodule_since1215_cwt_R_weight_0.1"
    # args.file_path_os="pqrst_nkmodule_since{}_cwt_{}/PRTweight_{}_{}_{}_augumentation={}".format(args.dataset_date,args.dataset_ver,args.P_weight,args.R_weight,args.T_weight,args.augumentation)

    # デバッグ用
    # args.dataset_date="0427"
    # args.subject_group="1"
    # args.P_weight="1.0"
    # args.R_weight="0.001"
    # args.T_weight="1.0"
    args.file_path_os = (
        OUTPUT_DIR
        + "/pqrst_nkmodule_since{}_cwt/PRTweight_{}_{}_{}_augumentation={}".format(
            args.dataset_date,
            args.P_weight,
            args.R_weight,
            args.T_weight,
            args.augumentation,
        )
    )
    #         # python3 output_data_pqrst_v2.py --file_path_os "leave_1_out_pqrst_nkmodule_since""$dataset_date""_cwt_""$dataset""_PRTweight_""$P_weight""_""$R_weight""_""$T_weight""_augumentation=""$augumentation"
    args.file_path = "/{}/Waveforms".format(args.file_path_os)
    args.png_path = OUTPUT_DIR + "/Scatters".format(args.file_path_os)
    args.Peaks_path = "{}/Peaks".format(args.file_path_os)
    args.Errors_path = OUTPUT_DIR + "/Errors".format(args.file_path_os)

    dataset_name = "pqrst_nkmodule_since{}_cwt".format(args.dataset_date)
    PRT_weight = "{}_{}_{}".format(args.P_weight, args.R_weight, args.T_weight)
    augumentation = args.augumentation
    args.condition_name = "{}_{}_{}".format(dataset_name, PRT_weight, augumentation)
    create_directory_if_not_exists(args.png_path)
    create_directory_if_not_exists(args.Errors_path)

    # 文字列をリストに変換
    names_list = args.names.split()
    dates_list = args.measure_dates.split()

    # 名前と日付の辞書を作成
    name_date_dict = dict(zip(names_list, dates_list))
    # if(args.subject_group=="1"):
    #     name_date_dict={"matumoto":"1128" ,"yoshikura":"1130","takahashi":"1102","taniguchi":"1107","kawai":"1115","goto":"1219", "asano":"0710"}
    # if(args.subject_group=="2"):
    #     name_date_dict={"asano":"0710","gosha":"0712","matumoto":"0718","mori":"0712","taniguchi":"0712","sato":"0719"}

    signal_types = ["reconx", "xo"]
    # for i in range(len(name_data_dict)):
    df_all_keep = pd.DataFrame()
    for name, date in name_date_dict.items():
        args.TARGET_NAME = name
        args.date = date
        args.pos = 0
        df_output_reconx = pd.DataFrame()
        df_output_xo = pd.DataFrame()
        for j in range(2):
            args.signal_type = signal_types[j]
            if signal_types[j] == "xo":
                df_output_xo = main(args)
            elif signal_types[j] == "reconx":
                df_output_reconx = main(args)

        # df_both=pd.concat([df_output_xo,df_output_reconx],axis=1)
        df_both = pd.concat([df_output_reconx, df_output_xo], axis=1)
        print("aaaaaaaaaaaaaaaaa")
        existing_csv_path = "{}/xo_reconx.csv".format(args.file_path, args.signal_type)
        print(df_both)
        # comaparer=Compare_df(df_both,columns=["ECG_P_Peaks","ECG_P_Values","ECG_T_Peaks","ECG_T_Values"],path=args.file_path)
        # comaparer.process_plot()
        # append_to_csv(existing_csv_path, df_both,na_rep='')
        df_all_keep = pd.concat([df_all_keep, df_both], axis=0)
    # print(df_all_keep)
    # print(df_all_keep)
    df_all_keep = df_all_keep.reset_index(drop=True)
    df_all_keep["recon_ECG_ST_Peaks_interval"] = (
        df_all_keep["recon_ECG_T_Peaks"] - df_all_keep["recon_ECG_S_Peaks"]
    )
    df_all_keep["ECG_ST_Peaks_interval"] = (
        df_all_keep["ECG_T_Peaks"] - df_all_keep["ECG_S_Peaks"]
    )
    # print(df_all_keep['recon_ECG_ST_Peaks_interval'])
    # input("")
    All_Compareler = Compare_df_all(
        df_all_keep, columns=["ECG_T_Peaks"], png_path=args.png_path, args=args
    )
    # All_Compareler=Compare_df_all(df_all_keep,columns=["ECG_P_Peaks","ECG_P_Heights","ECG_T_Peaks","ECG_T_Heights","ECG_ST_Peaks_interval"],png_path=args.png_path,args=args)
    All_Compareler.process_plot()
    df_all_keep.to_csv(existing_csv_path, index=False, na_rep="")
    print("iiiiiiiiiiiiiiiiiiiiiiiiiiii")
    df_error = pd.DataFrame()
    df_error["T_Peaks_error"] = (
        df_all_keep["ECG_T_Peaks"] - df_all_keep["recon_ECG_T_Peaks"]
    ).abs() * 0.002
    df_error["P_Peaks_error"] = (
        df_all_keep["ECG_P_Peaks"] - df_all_keep["recon_ECG_P_Peaks"]
    ).abs() * 0.002
    df_error["ST_Peaks_interval_error"] = (
        df_all_keep["ECG_ST_Peaks_interval"]
        - df_all_keep["recon_ECG_ST_Peaks_interval"]
    ).abs() * 0.002
    df_error["NAME"] = df_all_keep["NAME"]

    # input("")
    df_error_per_subject = (
        df_error.groupby("NAME")
        .agg({"T_Peaks_error": "mean", "ST_Peaks_interval_error": "mean"})
        .reset_index()
    )
    df_error_per_subject["DATSET_NAME"] = dataset_name
    df_error_per_subject["PRT_weight"] = PRT_weight
    df_error_per_subject["augumentation"] = augumentation
    df_error_output = df_error_per_subject[
        [
            "DATSET_NAME",
            "PRT_weight",
            "augumentation",
            "NAME",
            "T_Peaks_error",
            "ST_Peaks_interval_error",
        ]
    ]
    print(df_error_per_subject)
    # 被験者ごとのエラーを計算する
    errors_file_name = args.Errors_path + "/Errors_per_conditions.csv"
    if not os.path.exists(errors_file_name):
        df_error_output.to_csv(errors_file_name, header=True, index=False)
    else:
        df_error_output.to_csv(errors_file_name, mode="a", header=False, index=False)
    # input()

    df_error_all = df_error.agg(
        {"T_Peaks_error": "mean", "ST_Peaks_interval_error": "mean"}
    ).reset_index()
    # print(df_error_all)
    df_transposed = df_error_all.set_index("index").T
    df_transposed.reset_index(drop=True, inplace=True)
    df_error_all = df_transposed.copy()
    # インデックス名をリセット（必要に応じて）
    # print(df_error_all)
    # input("")
    df_error_all["DATSET_NAME"] = dataset_name
    df_error_all["PRT_weight"] = PRT_weight
    df_error_all["augumentation"] = augumentation
    df_error_all_output = df_error_all[
        [
            "DATSET_NAME",
            "PRT_weight",
            "augumentation",
            "T_Peaks_error",
            "ST_Peaks_interval_error",
        ]
    ]
    # 全員のエラーの平均を計算する
    errors_all_file_name = args.Errors_path + "/Errors_all.csv"
    if not os.path.exists(errors_all_file_name):
        df_error_all_output.to_csv(errors_all_file_name, header=True, index=False)
    else:
        df_error_all_output.to_csv(
            errors_all_file_name, mode="a", header=False, index=False
        )
    print(df_all_keep)
    df_cc = df_all_keep.copy()
    col1 = "ECG_T_Peaks"
    col2 = "recon_ECG_T_Peaks"
    df_corr = pd.DataFrame()
    # print(df_cc)
    # input()
    df_corr[col1] = pd.to_numeric(df_cc[col1], errors="coerce").fillna(0).astype(int)
    df_corr[col2] = pd.to_numeric(df_cc[col2], errors="coerce").fillna(0).astype(int)
    # print(df_corr)
    correlation_coefficient = df_corr[col1].corr(df_corr[col2])
    print(correlation_coefficient)
    # input("corr")

    # corr_T_Peaks = df_all_keep["ECG_T_Peaks"].corr(df_all_keep["recon_ECG_T_Peaks"])
    df_output_corr = pd.DataFrame()
    df_output_corr = pd.DataFrame(
        {
            "DATASET_NAME": [dataset_name],
            "PRT_weight": [PRT_weight],
            "augumentation": [augumentation],
            "corr": [correlation_coefficient],
        }
    )
    print(df_output_corr)
    # input()

    corr_file_name = args.Errors_path + "/corrs.csv"
    if not os.path.exists(errors_all_file_name):
        df_output_corr.to_csv(corr_file_name, header=True, index=False)
    else:
        df_output_corr.to_csv(corr_file_name, mode="a", header=False, index=False)
