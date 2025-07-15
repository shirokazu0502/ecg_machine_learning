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
            print(f"{file_name}は存在しませんが、指定されたディレクトリ({file_path})は存在します。")
        else:
            print(f"{file_name}は存在せず、指定されたディレクトリ({file_path})も存在しません。")
        # return None
        return 0
        # sys.exit()




def df_read_csv(file_path):
    data=pd.read_csv(file_path)
    print(data)
    return data

def find_p_element(arr, target):
    # 配列が空の場合はNoneを返す
    if arr.size==0:
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


def PTwave_search2(args,ecg_A2, header,sampling_rate, signal_type,time_range=0.8):
    # plt.plot(ecg_A2)
    # plt.show()
    ecg_signal_cleaned=nk.ecg_clean(ecg_A2,sampling_rate=sampling_rate,method='neurokit')
    ecg_signal=ecg_A2
    # rpeaks = nk.ecg_peaks(ecg_signal, sampling_rate)[1]['ECG_R_Peaks']  # R波の位置を取得
    peaks=nk.ecg_peaks(ecg_signal_cleaned, sampling_rate)  # R波の位置を取得
    rpeaks=peaks[1]['ECG_R_Peaks']
    print(rpeaks)
    print("uuuuuuuuuuuuuuuuuuuuuuu")
    print(nk.ecg_peaks(ecg_signal, sampling_rate))
    # type='peak'
    type='cwt'
    # waves_peak_df, waves_peak_dict = nk.ecg_delineate(ecg_signal,
    waves_peak_df, waves_peak_dict = nk.ecg_delineate(ecg_signal_cleaned,
                                                     rpeaks,
                                                     sampling_rate=sampling_rate,
                                                     method=type,
                                                    #  method="cwt",
                                                     show=False)
                                                    #  show=True)
    # R波のピーク値を追加
    waves_peak_dict['ECG_R_peaks']=rpeaks.tolist()
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
    fig=plt.figure(figsize=(24,12))
    ax=plt.axes()
    ax.plot(ecg_signal)
    ax.set_ylim(0.3, 0.8)
    color_dict = {
        "P_Onsets": 'b',    # 青
        "P_Peaks": 'deepskyblue',     # 緑
        "P_Offsets": 'royalblue',   # 赤
        "Q_Peaks": 'y',     # シアン
        "R_Peaks": 'r',    # 青
        "R_Onsets": 'darkred',    # マゼンタ
        "R_Offsets": 'tomato',   # 黄
        "S_Peaks": 'brown',     # 黒
        "T_Onsets": 'g',  # 紫
        "T_Peaks": 'limegreen',  # オレンジ
        "T_Offsets": 'forestgreen'   # ブラウン
    }


    ecg_p_onsets = waves_peak_dict.get("ECG_P_Onsets")
    if ecg_p_onsets is not None:
        valid_ecg_p_onsets = np.array(ecg_p_onsets)[~np.isnan(ecg_p_onsets)].astype(int)
        ax.plot(valid_ecg_p_onsets, ecg_signal[valid_ecg_p_onsets],color_dict["P_Onsets"], label="P onset",marker='v',linestyle='None', alpha=0.7)

    ecg_p_peaks = waves_peak_dict.get("ECG_P_Peaks")
    if ecg_p_peaks is not None:
        valid_ecg_p_peaks = np.array(ecg_p_peaks)[~np.isnan(ecg_p_peaks)].astype(int)
        ax.plot(valid_ecg_p_peaks, ecg_signal[valid_ecg_p_peaks], color_dict["P_Peaks"], label="P peaks",marker='o',linestyle='None', alpha=0.7)
    # "ECG_S_Peaks"の処理

    if(type=='cwt'):
        ecg_p_offsets = waves_peak_dict.get("ECG_P_Offsets")
        if ecg_p_offsets is not None:
            valid_ecg_p_offsets = np.array(ecg_p_offsets)[~np.isnan(ecg_p_offsets)].astype(int)
            ax.plot(valid_ecg_p_offsets, ecg_signal[valid_ecg_p_offsets], color_dict["P_Offsets"], label="P Offsets",marker='^',linestyle='None', alpha=0.7)

        ecg_q_peaks = waves_peak_dict.get("ECG_Q_Peaks")
        if ecg_q_peaks is not None:
            valid_ecg_q_peaks = np.array(ecg_q_peaks)[~np.isnan(ecg_q_peaks)].astype(int)
            ax.plot(valid_ecg_q_peaks, ecg_signal[valid_ecg_q_peaks], color_dict["Q_Peaks"], label="Q peaks",marker='o',linestyle='None', alpha=0.7)
#R波
    ax.plot(rpeaks, ecg_signal[rpeaks], "ro", label="R peaks", alpha=0.7)

    ax.set_ylim(0.3, 0.8)
    ecg_s_peaks = waves_peak_dict.get("ECG_S_Peaks")
    if ecg_s_peaks is not None:
        valid_ecg_s_peaks = np.array(ecg_s_peaks)[~np.isnan(ecg_s_peaks)].astype(int)
        ax.plot(valid_ecg_s_peaks, ecg_signal[valid_ecg_s_peaks], color_dict["S_Peaks"], label="S peaks",marker='o',linestyle='None', alpha=0.7)

    if(type=='cwt'):
        ecg_t_onsets = waves_peak_dict.get("ECG_T_Onsets")
        if ecg_t_onsets is not None:
            valid_ecg_t_onsets = np.array(ecg_t_onsets)[~np.isnan(ecg_t_onsets)].astype(int)
            ax.plot(valid_ecg_t_onsets, ecg_signal[valid_ecg_t_onsets], color_dict["T_Onsets"], label="T onset",marker='v',linestyle='None', alpha=0.7)

    ecg_t_peaks = waves_peak_dict.get("ECG_T_Peaks")
    if ecg_t_peaks is not None:
        valid_ecg_t_peaks = np.array(ecg_t_peaks)[~np.isnan(ecg_t_peaks)].astype(int)
        ax.plot(valid_ecg_t_peaks, ecg_signal[valid_ecg_t_peaks], color_dict["T_Peaks"], label="T peaks",marker='o',linestyle='None', alpha=0.7)


    ecg_t_offsets = waves_peak_dict.get("ECG_T_Offsets")
    if ecg_t_offsets is not None:
        valid_ecg_t_offsets = np.array(ecg_t_offsets)[~np.isnan(ecg_t_offsets)].astype(int)
        ax.plot(valid_ecg_t_offsets, ecg_signal[valid_ecg_t_offsets], color_dict["T_Offsets"], label="T Offsets",marker='^',linestyle='None', alpha=0.7)

    plt.axvline(x=800,color='red',linewidth=2,linestyle='--')
    plt.axvline(x=8400,color='red',linewidth=2,linestyle='--')
    ax.set_title("{}_{}_{}".format(args.TARGET_NAME,args.TARGET_CHANNEL,signal_type))
    ax.legend()
    ax.grid(True)
    ax.set_ylim(0.3, 0.9)
    # ax.ylim(0.3,0.8)
    plt.tight_layout()
    plt.savefig("{}/{}_{}.png".format(args.Peaks_path,args.TARGET_NAME,args.signal_type))
    # plt.show()
    plt.close()
    return waves_peak_dict,ecg_A2
    
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

def get_value_in_range_before_R(data, init,datalength):
    # 初期値から終了値までの範囲内に値があるか確認
    print(data)
    target=0
    for i in range(len(data)):
        if(init<data[i] and data[i]<init+(datalength/2)):
            target=data[i]
            # break
    if(target==0):
        return None
    return target

def get_value_in_range_after_R(data,init,datalength):
    # 初期値から終了値までの範囲内に値があるか確認
    print(data)
    target=0
    for i in range(len(data)):
        if(init+(datalength/2)<data[i] and data[i]<init+datalength):
            target=data[i]
            # break
    if(target==0):
        return None
    return target

def get_value_in_range(data, init, end):
    # 初期値から終了値までの範囲内に値があるか確認
    print(data)
    target=0
    for i in range(len(data)):
        if(init<data[i] and data[i]<end):
            target=data[i]
            # break
    if(target==0):
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
        self.target_column=target_column_names
        new_column_name='{}_{}_HasNone'.format(target_column_names[0],target_column_names[1])
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
        column1=self.target_column[0]
        column2=self.target_column[1]
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
                if(row[column1]>row[column2]):
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
def subtract_row_index_times_400_except(df, args,exclude_columns=["baseline","ECG_P_Heights","ECG_T_Heights"]):
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
        return series - series.index *(args.space_length+args.datalength)

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
def interpolate_values(start, end,space_length):
    # 引数の値が同じ場合はその値を含むリストを返す
    if start == end:
        return [start]

    # 引数の大小を確認して、小さい方から大きい方までの400の等間隔な値を作成する
    if start < end:
        values = [start + i * (end - start) / (space_length+1) for i in range(space_length+2)]
    else:
        values = [start - i * (start - end) / (space_length+1) for i in range(space_length+2)]

    return values[1:-1]

class Append_Peak_Value:
    def __init__(self,df,ecg_signal):
        self.df=df
        self.ecg_signal=ecg_signal
        pass

    def convert_to_numpy(self):
        P_peak_np=(self.df.copy()["ECG_P_Peaks"].to_numpy().T)
        T_peak_np=(self.df.copy()["ECG_T_Peaks"].to_numpy().T)
        P_onset_np=(self.df.copy()["ECG_P_Onsets"].to_numpy().T)
        return P_peak_np,T_peak_np,P_onset_np

    def create_array_with_selected_indices(self,original_array, selected_indices,baseline_array):
        result_array = np.empty_like(selected_indices, dtype=object)
        # # print(original_array)
        # print(result_array.shape)
        # array=np.ndarray(shape=(len(selected_indices),))
        # print(original_array)
        # print(selected_indices)

        for i, index in enumerate(selected_indices):
            # print(i,index)
            if index !=None and baseline_array[i]!=None:
                result_array[i] = original_array[index] - original_array[baseline_array[i]]

                # print(original_array[index])
            else:
                result_array[i] = None
        # print(result_array)
        # return array
                # print(original_array[index])
        return result_array

    def get_peak_values(self):
        P_peak_np,T_peak_np,P_onset_np=self.convert_to_numpy()
        # print(P_)
        P_Values=self.create_array_with_selected_indices(self.ecg_signal,P_peak_np,P_onset_np)
        T_Values=self.create_array_with_selected_indices(self.ecg_signal,T_peak_np,P_onset_np)
        # print(P_peak_np)
        # print(P_Values)
        # print(P_Values.shape)
        # input("")
        return P_Values,T_Values
    def concat_df(self,P_Values,T_Values):
        df_P=pd.Series(P_Values,name="ECG_P_Heights")
        df_T=pd.Series(T_Values,name="ECG_T_Heights")
        new_df=pd.concat([self.df.copy(),df_P,df_T],axis=1)
        return new_df

    def plot_signals(self):
        pass

    def process(self):
        P_Values,T_Values=self.get_peak_values()
        new_df=self.concat_df(P_Values=P_Values,T_Values=T_Values)
        print(self.df)
        print(new_df)
        # input()
        return new_df

def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
x=torch.tensor([1,2,3])
print("TEST MODE::\n")
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
print(len(test_dataset))
# input()

pth = args.pth
vae.load_state_dict(torch.load(pth, map_location=lambda storage, loc: storage))
vae.eval()
print(vae)
# names = torchvision.models.feature_extraction.get_graph_node_names(vae)
# print(names)
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
test_val_12ch_all = []
with torch.no_grad():
    for j, (x, xo, label_name, pt_index) in enumerate(test_loader):
        x, xo = x.to(device), xo.to(device)
        recon_x, mean, log_var, z = vae(x)
        z_temp = tensor_to_ndarray(z)
        z_temps = np.append(z_temps, z_temp, axis=0)

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
create_directory_if_not_exists(OUTPUT_MAE_DIR)
output_file = os.path.join(
    OUTPUT_MAE_DIR
    + "/MAE_leave_1_out_{}_PRTweight_{}_{}_{}_augumentation={}.csv".format(
        args.Dataset_name,
        str(args.loss_pt_on_off_P_weight),
        str(args.loss_pt_on_off_R_weight),
        str(args.loss_pt_on_off_T_weight),
        args.augumentation,
    )
)
write_to_csv(output_file, data=data_to_write)