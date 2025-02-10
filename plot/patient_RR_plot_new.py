import argparse
import sys
import os
import codecs
import struct
import binascii
import numpy as np
from pylab import *
import matplotlib.cm as cm
import pandas as pd
from scipy import signal
import time

### グラフ表示が要らない場合はFalse
DEBUG_PLOT = True
# DEBUG_PLOT = False
###
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import neurokit2 as nk
import warnings
import csv
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
    RATE_15CH,
    TIME,
    DATASET_MADE_DATE,
)
from config.name_dic import select_name_and_date

HPF_FP = 2.0
HPF_FS = 1.0
# DATASET_MADE_DATE="0120" #packet_loss_data_{}の部分
# DATASET_MADE_DATE="icce0116" #packet_loss_data_{}の部分
# RATE_12ch=500
RATE_12ch = RATE
# RATE = 300
# RATE_lizmil = 128  # リズミル取扱説明書に期待
RATE_lizmil = 128.2
# RATE_15CH=122.06
# RATE=500
# TIME=24  #記録時間は24秒または10秒


class CSVReader_12ch:
    def __init__(self, directory):
        self.directory = directory

    def search_files(self):
        files_found = []
        for filename in os.listdir(self.directory):
            if filename.startswith("12ch") and filename.endswith(".csv"):
                files_found.append(filename)
        return files_found

    def read_csv_file(self, filename):
        file_path = os.path.join(self.directory, filename)
        df = pd.read_csv(file_path)
        print(f"ファイル {filename} を読み込みました。")
        # 読み込んだデータフレームの操作などを行う
        # ...
        print(df)
        return df

    def header_make(self, df):
        df = df.drop(columns=[15, 16])
        df = df.rename(columns=lambda x: "ch_" + str(x + 1))
        return df

    def process_files(self):
        files_found = self.search_files()
        if len(files_found) > 0:
            df = self.read_csv_file(files_found[0])
            # df=self.header_make(df)
            print(df)
        else:
            print("指定した条件のCSVファイルは存在しません。")
            print("12ch")
            exit()
        return df


class CSVReader_16ch:
    def __init__(self, directory):
        self.directory = directory

    def search_files(self):
        files_found = []
        for filename in os.listdir(self.directory):
            if filename.startswith("db") and filename.endswith(".csv"):
                files_found.append(filename)
        return files_found

    def read_csv_file(self, filename):
        file_path = os.path.join(self.directory, filename)
        df = pd.read_csv(file_path, header=None)
        print(f"ファイル {filename} を読み込みました。")
        # 読み込んだデータフレームの操作などを行う
        # ...
        print(df)
        return df

    def header_make(self, df):
        try:
            df = df.drop(columns=[16, 17])
        except:
            df = df.drop(columns=[16])
        df = df.rename(columns=lambda x: "ch_" + str(x + 1))
        return df

    def process_files(self):
        files_found = self.search_files()
        if len(files_found) > 0:
            df = self.read_csv_file(files_found[0])
            df = self.header_make(df)
            print(df)
        else:
            print("指定した条件のCSVファイルは存在しません。")
            print("15ch")
            exit()
        return df


def linear_interpolation_resample_All(df, sampling_rate, new_sampling_rate):
    # 時系列データの時間情報を正規化
    df_new = pd.DataFrame(columns=df.columns)
    dt = 1.0 / sampling_rate
    time = np.arange(len(df)) * dt
    time_normalized = (time - time[0]) / (time[-1] - time[0])

    data = df[df.columns[0]].copy()
    data = data.to_numpy()

    # 線形補間関数を作成ch1
    interpolator = interp1d(time_normalized, data)

    # 新しい時間情報を生成
    new_time_normalized = np.linspace(
        0, 1, int((time[-1] - time[0]) * new_sampling_rate)
    )

    # 線形補間によるリサンプリング
    new_data = interpolator(new_time_normalized)

    # 新しい時間情報を元のスケールに戻す
    new_time = new_time_normalized * (time[-1] - time[0]) + time[0]
    df_new[df.columns[0]] = new_data
    print("{}_線形補間リサンプリング完了".format(df.columns[0]))
    print("old_datalength={}".format(len(df)))
    print("new_datalength={}".format(len(df_new)))
    print(df_new)

    return df_new


def ecg_clean_df_lizmil(lizmil_df, rate=RATE):
    ecg_signal = lizmil_df.copy()[0]
    # cleand_signal=nk.ecg_clean(ecg_signal,sampling_rate=500,method="neurokit")
    # print(cleaned_signal)
    # print(type(cleaned_signal))
    # plt.plot(cleaned_signal)
    print(lizmil_df[0])
    print(lizmil_df)
    plt.plot(lizmil_df[0])
    plt.title("org")
    plt.close()
    plt.cla()
    # plt.show()
    lizmil_df_cleaned = pd.DataFrame()
    for i, column in enumerate(lizmil_df.columns):
        # df0_mul.plot()
        # df1=df.iloc[:,i]
        ecg_signal = lizmil_df[column].copy().values
        ecg_signal = nk.ecg_clean(ecg_signal, sampling_rate=rate, method="neurokit")
        lizmil_df_cleaned[column] = ecg_signal
        # print(df[column])
        # print("")

    # print(type(lizmil_df_cleaned))
    # plt.plot(lizmil_df_cleaned["A2"])
    # plt.title("cleaned")
    # plt.show()
    return lizmil_df_cleaned


def ecg_clean_df_15ch(df_15ch, rate):
    ecg_signal = df_15ch.copy()["ch_1"]
    # cleand_signal=nk.ecg_clean(ecg_signal,sampling_rate=500,method="neurokit")
    # print(cleaned_signal)
    # print(type(cleaned_signal))
    # plt.plot(cleaned_signal)
    print(type(df_15ch))
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

    # fig = plt.figure(num=None, figsize=(12, 5), dpi=100, facecolor="w", edgecolor="k")
    # axis_line_width = 2.0
    # tick_label_size = 18
    # # 最初のグラフ（8プロット）
    # plot_time = np.arange(len(df_15ch)) / RATE_15CH
    # ax1 = fig.add_subplot(1, 1, 1)
    # ax1.plot(plot_time, df_15ch["ch_1"], label="ch_1")
    # ax1.plot(plot_time, df_15ch_cleaned["ch_1"], label="filtered ch1")
    # # ax1.legend(fontsize=12, ncol=1)
    # ax1.legend(loc="upper right", fontsize=18, ncol=1, bbox_to_anchor=(1, 1))
    # ax1.tick_params(labelsize=tick_label_size, direction="in")
    # plt.xlim(3.8, 8)
    # plt.ylim(-100, 200)
    # for axis in ["top", "bottom", "left", "right"]:
    #     ax1.spines[axis].set_linewidth(axis_line_width)
    # # ax2 = fig.add_subplot(2, 1, 2)
    # # ax2.plot(df_15ch_cleaned["ch_1"],label='filtered ch1')
    # # ax2.legend(loc='center left', fontsize=12, ncol=1, bbox_to_anchor=(1, 0.5))
    # # ax2.tick_params(labelsize=tick_label_size,direction='in')
    # # # print(type(df_15ch_cleaned))
    # # plt.plot(df_15ch_cleaned["ch_1"])
    # # plt.title("cleaned")
    # print()
    # plt.savefig("taniguchi_filter.svg")
    # plt.tight_layout()
    # plt.show()
    return df_15ch_cleaned


# df2のtimeに最も近いdf1のtimeを探して結合
def find_nearest(row, df1_time, df1_diff):
    nearest_idx = (df1_time - row["time"]).abs().idxmin()
    min_abs_diff = abs(df1_diff.iloc[nearest_idx] - row["diff"])

    # 前後5つの範囲で探索
    for offset in range(1, 6):  # 1～5の範囲で前後のインデックスを確認
        if nearest_idx - offset >= 0:  # 前側のインデックスが範囲内の場合
            tmp_abs_diff = abs(df1_diff.iloc[nearest_idx - offset] - row["diff"])
            if tmp_abs_diff < min_abs_diff and tmp_abs_diff < 0.008:
                min_abs_diff = tmp_abs_diff
                nearest_idx = nearest_idx - offset

        if nearest_idx + offset < len(df1_diff):  # 後側のインデックスが範囲内の場合
            tmp_abs_diff = abs(df1_diff.iloc[nearest_idx + offset] - row["diff"])
            if tmp_abs_diff < min_abs_diff and tmp_abs_diff < 0.008:
                min_abs_diff = tmp_abs_diff
                nearest_idx = nearest_idx + offset

    return nearest_idx


# def find_nearest(row, df1_time, df1_diff):
#     nearest_idx = (df1_time - row["time"]).abs().idxmin()
#     return nearest_idx


class AutoIntegerFileHandler:
    def __init__(self, filename):
        self.filename = filename

    def check_file(self):
        path = self.filename
        if os.path.exists(path):
            if os.path.isfile(path):
                print(f"The path '{path}' exists and it is a file.")
                if input("ok? y or n") == "y":
                    return True
                # return True
        else:
            print(f"The path '{path}' does not exist.")
        return False

    def input_integer(self, RATE, cut_time):
        time = cut_time
        print("同期する時間は{}(s)".format(time))
        integer = int(RATE * time)
        print("integer={}".format(integer))
        return integer

    def write_integer(
        self, RATE, cut_time, target_15ch, reverse, target_12ch, cut_min_max_range
    ):
        integer = self.input_integer(RATE, cut_time)
        # with open(self.filename, 'w') as file:
        #    file.write(str(integer)+'\n')
        # #    file.write("TARGET_CHANNEL_15ch="+str(self.ch))
        #    file.write(str(target_15ch)+'\n')
        #    file.write(str(target_12ch)+'\n')
        #    file.write(str(cut_min_max_range[0])+'\n')
        #    file.write(str(cut_min_max_range[1])+'\n')

        data = {
            "INDEX": str(integer),
            "TARGET_CH_15ch": str(target_15ch),
            "REVERSE": reverse,  # ピーク検出するときにTARGET_15chの波形を反転させるかどうかを決める。
            "TARGET_CH_12ch": str(target_12ch),
            "START_TIME": str(
                cut_min_max_range[0]
            ),  # 書いてるだけで別に同期ファイルがある場合は使わないデータ
            "END_TIME": str(
                cut_min_max_range[1]
            ),  # 書いてるだけで別に同期ファイルがある場合は使わないデータ
        }

        column_order = [
            "INDEX",
            "TARGET_CH_15ch",
            "REVERSE",
            "TARGET_CH_12ch",
            "START_TIME",
            "END_TIME",
        ]
        with open(self.filename, mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=column_order)

            # カラム名を書き込む
            writer.writeheader()

            # データを書き込む
            writer.writerow(data)

    def read_integer(self):
        # CSVファイルを読み込む
        with open(self.filename, "r") as file:
            reader = csv.DictReader(file)

            # カラム名を取得
            columns = reader.fieldnames
            print(columns)
            data_list = []
            data_dict = {}

            # データを読み込み、表示
            for i, row in enumerate(reader):
                print(row)
                data_dict = row

                # データを辞書型に格納
            # data_dict = {row["INDEX"]: row for row in data_list}
            print(data_dict)

            # print("CSVファイルの内容を辞書に格納しました。")
            # input("posseeeeeeee")
            return (
                int(data_dict["INDEX"]),
                data_dict["TARGET_CH_15ch"],
                data_dict["REVERSE"],
                data_dict["TARGET_CH_12ch"],
            )


class ArrayComparator:
    def __init__(self, sc_15ch, sc_lizmil, cut_min_max_range):
        self.sc_lizmil = sc_lizmil
        self.sc_15ch = sc_15ch
        self.cut_min_max_range = cut_min_max_range

    def cul_diff(self):
        time_lizmil = self.sc_lizmil[0][1:].to_numpy()
        time_15ch = self.sc_15ch[0][1:].to_numpy()
        diff_lizmil = np.diff(self.sc_lizmil[0])
        diff_15ch = np.diff(self.sc_15ch[0])
        return time_lizmil, time_15ch, diff_lizmil, diff_15ch

    def peak_diff_plot(self):
        time1, time2, diff1, diff2 = self.cul_diff()
        print(diff1)
        print(diff2)
        # データ1のプロット
        plt.plot(time1, diff1, label="lizmil", color="r")
        plt.scatter(time1, diff1, label="lizmil", color="r")
        # データ2のプロット
        plt.plot(time2, diff2, label="15ch", color="b")
        plt.scatter(time2, diff2, label="15ch", color="b")

        # グラフのタイトルと凡例
        plt.title("compare of peak time diff")
        plt.legend()

        # 軸ラベルの設定
        plt.xlabel("time(s)")
        plt.ylabel("diff(s)")

        # グラフの表示
        # plt.show()
        plt.close()

    def find_best_cut_time(self):
        cut_min_max_range = self.cut_min_max_range
        # min_mse = float("inf")  # 初期値として最大値を設定
        best_index = 0
        # target=-15#後ろから3つを基準に平均二乗誤差でマッチするインデックスを探す。
        time1, time2, diff_12ch, diff_15ch = self.cul_diff()
        RESAMPLE_RESOLUTION = 0.1  # sec
        print(time1[0], time1[-1])
        res_time1 = np.arange(time1[0], time1[-1], RESAMPLE_RESOLUTION)
        f1 = interp1d(time1, diff_12ch, kind="linear")
        res_diff_12ch = f1(res_time1)
        res_time2 = np.arange(time2[0], time2[-1], RESAMPLE_RESOLUTION)
        f2 = interp1d(time2, diff_15ch, kind="linear")
        res_diff_15ch = f2(res_time2)
        corr_list = []
        search_win_len = len(res_time1)
        print(f"{search_win_len=}")
        print(f"{len(res_time2)=}")
        print(f"{len(res_diff_12ch)=}")
        print(f"{len(res_diff_15ch)=}")
        for idx in range(len(res_time2) - search_win_len):
            corr_list.append(
                # np.dot(res_diff_12ch, res_diff_15ch[idx : idx + search_win_len])
                np.sum((res_diff_12ch - res_diff_15ch[idx : idx + search_win_len]) ** 2)
            )

        corr_list = np.array(corr_list)
        corr_max_idx = np.argmin(corr_list)
        corr_max_time = corr_max_idx * RESAMPLE_RESOLUTION
        # best_index = 0
        # # target=-15#後ろから3つを基準に平均二乗誤差でマッチするインデックスを探す。
        # time1, time2, diff_12ch, diff_15ch = self.cul_diff()
        # # time1 = pd.Series(time1)
        # # time2 = pd.Series(time2)
        # # target=-len(diff_12ch)
        target = 0
        # # target=5#後ろから3つを基準に平均二乗誤差でマッチするインデックスを探す。
        # # diff_12ch=diff_12ch[target:]
        # large_size = len(diff_15ch)
        # small_size = len(diff_12ch)
        # # print(time2, time1)
        # nearest_idxs = []
        # for i in range(large_size - small_size + 1):
        #     if (
        #         time2[i] - time1[target] < cut_min_max_range[0]
        #         or time2[i] - time1[target] > cut_min_max_range[1]
        #     ):  # 始めの4.0秒は使わない
        #         continue
        #     current_subset = diff_15ch[i : i + small_size]
        #     mse = np.mean((current_subset - diff_12ch) ** 2)
        #     # print(mse, time2[i])
        #     # if time2[i] > 40 and time2[i] < 50:
        #     #     print(time2[i], mse)
        #     # if time2[i] > 772 and time2[i] < 773.5:
        #     #     min_mse = mse
        #     #     best_index = i

        #     if mse < min_mse:
        #         min_mse = mse
        #         best_index = i
        # print(current_subset, diff_12ch)
        # print("12chの最初のピークのtime={}".format(time1[target]))
        # print(i)
        # print("15chの対応するピークのtime={}".format(time2[best_index]))
        # cut_time = time2[corr_max_idx] - time1[target]
        best_index = np.argmin(np.abs(time2 - corr_max_time))
        print(corr_max_time, best_index)
        return corr_max_time, best_index

    def peak_diff_plot_move(self, cut_time):
        cut_time, best_index = self.find_best_cut_time()
        time1, time2, diff1, diff2 = self.cul_diff()
        # リズミルセンサの15chセンサに対しての同期時間
        time1_v2 = time1 + cut_time
        # print(diff1)
        # print(diff2)
        # データ1のプロット
        plt.plot(time1, diff1, label="12ch", color="r")
        plt.scatter(time1, diff1, label="12ch", color="r")
        # データ2のプロット
        plt.plot(time2, diff2, label="15ch", color="b")
        plt.scatter(time2, diff2, label="15ch", color="b")

        # データ1のプロットのcut_time分平行移動
        plt.plot(time1_v2, diff1, label="12ch_move", color="g")
        plt.scatter(time1_v2, diff1, label="12ch_move", color="g")
        # グラフのタイトルと凡例
        plt.title("compare of peak time diff")
        plt.legend()

        # 軸ラベルの設定
        plt.xlabel("time(s)")
        plt.ylabel("diff(s)")

        # グラフの表示
        # plt.show()
        plt.close()

    def create_corr_and_blanalt_plot(self, cut_time, loop_num, data_label):
        cut_time, best_index = self.find_best_cut_time()
        time1, time2, diff1, diff2 = self.cul_diff()
        time1_v2 = time1 + cut_time
        # time2, diff2をリズミルセンサの範囲に合わせる
        print(time2, time1_v2)
        cut_time2 = time2[best_index : best_index + len(diff1)]
        cut_diff2 = diff2[best_index : best_index + len(diff1)]
        print(len(time1_v2), len(diff1))
        con_lizmil_data = pd.DataFrame({"time": time1_v2, "diff": diff1})
        print(con_lizmil_data)
        con_15ch_data = pd.DataFrame({"time": cut_time2, "diff": cut_diff2})
        print(con_15ch_data)
        lizmil_time = con_lizmil_data["time"]
        lizmil_diff = con_lizmil_data["diff"]
        # 最も近いdf1のインデックスを探す
        con_15ch_data["nearest_idx"] = con_15ch_data.apply(
            find_nearest, df1_time=lizmil_time, df1_diff=lizmil_diff, axis=1
        )
        con_all_data = con_15ch_data.merge(
            con_lizmil_data,
            left_on="nearest_idx",
            right_index=True,
            suffixes=("_15ch", "_lizmil"),
        )
        con_all_data.to_csv("./con_data.csv")
        print(con_all_data)
        corr = con_all_data["diff_15ch"].corr(con_all_data["diff_lizmil"])
        print(corr)
        # サブプロット作成
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # 1行2列のプロット
        # 相関が1の直線 (y = x)
        x = np.linspace(
            min(con_all_data["diff_15ch"].min(), con_all_data["diff_lizmil"].min()),
            max(con_all_data["diff_15ch"].max(), con_all_data["diff_lizmil"].max()),
            100,
        )
        axes[0].plot(x, x, label="y = x (correlation=1)", color="blue", linestyle="--")
        axes[0].scatter(
            con_all_data["diff_15ch"],
            con_all_data["diff_lizmil"],
            label="Data Points",
            color="r",
        )
        axes[0].set_title(f"RRI Scatter Plot\nCorrelation: {corr:.3f}")
        axes[0].set_xlabel("15ch_RRI")
        axes[0].set_ylabel("lizmil_RRI")
        axes[0].legend()

        # サブプロット2: Bland-Altman プロット
        RRI_diff = con_all_data["diff_15ch"] - con_all_data["diff_lizmil"]
        BPM_15ch = 60 / con_all_data["diff_15ch"]
        BPM_lizmil = 60 / con_all_data["diff_lizmil"]
        mean_BPM = (BPM_15ch + BPM_lizmil) / 2
        axes[1].scatter(mean_BPM, RRI_diff, color="blue", label="BPM vs RRI Difference")
        axes[1].set_title("Bland-Altman Plot")
        axes[1].set_xlabel("mean_BPM")
        axes[1].set_ylabel("diff_RRI")
        axes[1].legend()
        # グラフ全体のタイトルとレイアウト調整
        fig.suptitle(f"{data_label} corr and blanaltman plot", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # 保存と表示
        combined_path = (
            DATA_DIR
            + f"/RRI_plot/{data_label}/corr_blanalt_cuttime{cut_time:.0f}_{loop_num}.png"
        )
        plt.savefig(combined_path)
        # plt.show()
        plt.close()
        con_all_data.to_csv(
            DATA_DIR + f"/RRI_plot/{data_label}/{loop_num}_corr{corr:.2f}.csv"
        )


def validate_integer_input():
    try:
        value = int(input("整数を入力してください（1から15までの範囲）: "))
        if value < 0 or value > 16:
            raise ValueError("入力された整数は範囲外です。")
        else:
            ch = "ch_" + str(value)
            return ch
    except ValueError as e:
        print("エラー:", e)
        return None


def peak_sc(dataframe, RATE, TARGET):
    times, val = peak_search_nk(dataframe[TARGET], RATE)
    dt = 1.0 / RATE
    N = len(dataframe)
    time_np = np.array(times)
    time1 = time_np * dt
    sc = pd.DataFrame(index=[])
    sc[0] = time1
    sc[1] = val
    # print(sc)
    return sc


def peak_sc_plot(dataframe, RATE, TARGET):
    # times,val=peak_search(dataframe[TARGET],RATE)
    times, val = peak_search_nk(dataframe[TARGET], RATE_lizmil)
    dt = 1.0 / RATE_lizmil
    N = len(dataframe)
    time_np = np.array(times)
    time1 = time_np * dt
    sc = pd.DataFrame(index=[])
    sc[0] = time1
    sc[1] = val
    plt.scatter(x=time1, y=val, color="red")
    time = np.arange(len(dataframe)) * dt
    plt.plot(time, dataframe[TARGET])
    plt.title(TARGET)
    # print(sc)
    # plt.show()
    plt.close()
    # print(sc)
    # input()
    return sc


def peak_search_nk_15ch(df_target, RATE):
    print("safe")
    ecg_signal = df_target.copy().to_numpy().T
    # ecg_signal=nk.ecg_clean(ecg_signal,sampling_rate=RATE,method='neurokit')
    print(ecg_signal)
    _, rpeaks = nk.ecg_peaks(ecg_signal, RATE)
    print(rpeaks["ECG_R_Peaks"])
    # ax = plt.axes()
    # ax.plot(ecg_signal)
    # ax.plot(rpeaks, ecg_signal[rpeaks], "ro", label="R peaks")
    # ax.set_title(args.name + "_" + args.pos + "_12ch_A2")
    # ax.legend()
    # ax.grid(True)
    # plt.show()
    vals = ecg_signal[rpeaks["ECG_R_Peaks"]]
    return rpeaks["ECG_R_Peaks"], vals


def peak_search_nk(df_target, RATE):
    print("safe")
    ecg_signal = df_target.copy().to_numpy().T
    # ecg_signal=nk.ecg_clean(ecg_signal,sampling_rate=RATE,method='neurokit')
    print(ecg_signal)
    _, rpeaks = nk.ecg_peaks(ecg_signal, RATE)
    print(rpeaks["ECG_R_Peaks"])
    vals = ecg_signal[rpeaks["ECG_R_Peaks"]]
    return rpeaks["ECG_R_Peaks"], vals


def peak_sc_15ch(dataframe, RATE, TARGET):
    print(dataframe)
    times, val = peak_search_nk_15ch(dataframe[TARGET], RATE)
    dt = 1.0 / RATE
    N = len(dataframe)
    time_np = np.array(times)
    time1 = time_np * dt
    sc = pd.DataFrame(index=[])
    sc[0] = time1
    sc[1] = val
    # print(sc)
    return sc


def set_header_from_first_row(df):
    # 1行目を取得
    first_row = df.iloc[0][0]
    print(first_row, type(first_row))
    # 1行目が文字列で構成されているか判定
    if type(first_row) is str:
        # ヘッダー行をデータから削除
        df = df[1:].reset_index(drop=True)
        df = df.astype(float)
    return df


def peak_sc_lizmil(dataframe, RATE):
    times, val = peak_search_nk_15ch(dataframe[0], RATE)
    dt = 1.0 / RATE
    N = len(dataframe)
    time_np = np.array(times)
    time1 = time_np * dt
    sc = pd.DataFrame(index=[])
    sc[0] = time1
    sc[1] = val
    # print(sc)
    return sc


def main(args):
    # TARGET_CHANNEL_15CH=args.TARGET_CHANNEL_15CH
    TARGET_CHANNEL_12CH = args.TARGET_CHANNEL_12CH
    cut_min_max_range = args.cut_min_max_range
    # ファイル読み込み
    # dir_path = "./0_packetloss_data/"+args.dir_name
    # dir_path = "./0_packetloss_data_{}/".format(DATASET_MADE_DATE)+args.dir_name
    # dir_path = args.dataset_dir
    dir_path = args.raw_datas_dir
    csv_reader_16ch = CSVReader_16ch(dir_path)
    df_16ch = csv_reader_16ch.process_files()
    print(df_16ch)
    cols = df_16ch.columns
    df_15ch = pd.DataFrame()
    for col in cols:
        df_15ch[col] = df_16ch[col] - df_16ch["ch_16"]
    df_15ch = df_15ch.drop(columns=["ch_16"])
    # df_15ch = df_15ch[:1000000]

    for tmp_name in os.listdir(dir_path):
        if "lizmil" in tmp_name:
            filename = tmp_name
            break
    lizmil_file_path = os.path.join(dir_path, filename)
    lizmil_df = pd.read_csv(lizmil_file_path, header=None)
    print(len(lizmil_df))
    print("iiiiii")
    lizmil_df = set_header_from_first_row(lizmil_df)
    # 同期時刻検出用
    # lizmil_df = lizmil_df[400000:450000]
    # lizmil_df = lizmil_df[100000:]
    # 全体プロット用
    df_15ch_size = len(df_15ch)
    # lizmil_df = lizmil_df[100000 : 100000 + df_15ch_size - 20000]
    # print(lizmil_df[0])
    reverse = args.reverse
    print("TARGET_CHANNEL_15chは1です。")
    TARGET_CHANNEL_15CH = "ch_1"

    df_15ch_pf = ecg_clean_df_15ch(df_15ch=df_15ch.copy(), rate=RATE_15CH)
    df_resample_15ch = linear_interpolation_resample_All(
        df=df_15ch_pf.copy(), sampling_rate=RATE_15CH, new_sampling_rate=RATE
    )
    df_15ch_pf = df_resample_15ch.copy()

    if reverse == "off":
        sc_15ch = peak_sc_15ch(df_15ch_pf.copy(), RATE=RATE, TARGET=TARGET_CHANNEL_15CH)
        sc_15ch["Time Interval"] = sc_15ch[0].diff()
        print("kokodayo")
        print(sc_15ch)

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

    print("reverse=={}".format(reverse))
    print(sc_15ch)

    step_size = 50000
    # ループ処理
    loop_num = 0
    for start_idx in range(0, df_15ch_size, step_size):
        loop_num = loop_num + 1
        end_idx = start_idx + step_size

        # 残りのデータが50000行未満の場合はスキップ
        if end_idx > df_15ch_size:
            print(f"Skipping rows {start_idx} to {end_idx} (not enough data)")
            continue

        print(f"Processing rows {start_idx} to {end_idx}")
        print(df_15ch_size)

        # lizmil_dfのスライス
        lizmil_df_slice = lizmil_df[start_idx:end_idx]

        # 必要な処理をここに記述
        df_lizmil_pf = ecg_clean_df_lizmil(
            lizmil_df=lizmil_df_slice.copy(), rate=RATE_15CH
        )
        df_resample_lizmil = linear_interpolation_resample_All(
            df=df_lizmil_pf.copy(), sampling_rate=RATE_lizmil, new_sampling_rate=RATE
        )
        times, val = peak_search_nk(lizmil_df_slice[0], RATE_lizmil)
        dt = 1.0 / RATE_lizmil
        time_np = np.array(times)
        time1 = time_np * dt
        sc_lizmil = pd.DataFrame(index=[])
        sc_lizmil[0] = time1
        sc_lizmil[1] = val
        print(sc_lizmil)

        comparator = ArrayComparator(
            sc_15ch=sc_15ch,
            sc_lizmil=sc_lizmil,
            cut_min_max_range=cut_min_max_range,
        )
        comparator.peak_diff_plot()
        cut_time = comparator.find_best_cut_time()
        print(cut_time)
        print("aaaaaaaaaaaaaaa")
        comparator.peak_diff_plot_move(cut_time)
        comparator.create_corr_and_blanalt_plot(cut_time, loop_num, args.patient)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--dir_name", type=str, default='goto_0604/goto_0604_normal2')

    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--date", type=str, default="")
    parser.add_argument("--peak_method", type=str, default="")
    parser.add_argument("--pos", type=str, default="")
    parser.add_argument("--type", type=str, default="")
    parser.add_argument("--dir_name", type=str, default="")
    parser.add_argument("--png_path", type=str, default="")
    parser.add_argument("--output_filepath", type=str, default="")
    # parser.add_argument("--TARGET_CHANNEL_15CH", type=str, default='ch_1')
    parser.add_argument("--TARGET_CHANNEL_12CH", type=str, default="")
    # parser.add_argument("--cut_min_max_range", type=list, default=[0,10])
    parser.add_argument("--cut_min_max_range", type=list, default="")
    parser.add_argument("--time_range", type=float)
    parser.add_argument(
        "--reverse", type=str, default=""
    )  # onだと波形逆さまにしてピーク検出。これはシートセンサを逆向きに貼ったとき
    parser.add_argument("--project_path", type=str, default="")
    parser.add_argument("--raw_datas_dir", type=str, default="")
    parser.add_argument("--raw_datas_os", type=str, default="")
    parser.add_argument("--dataset_made_date", type=str, default="")
    parser.add_argument("--dataset_output_path", type=str, default="")
    parser.add_argument("--test_images_path", type=str, default="")
    args = parser.parse_args()
    # args.name='goto'#yoshikura takahashi matumoto
    # args.date='1219'
    args.peak_method = (
        "cwt"  # neurokitのピーク検出アルゴリズムについてcwtかpeakがある。
    )
    args.pos = "0"
    args.type = ""
    args.dir_name = "{}/{}".format(args.name, args.type)
    args.png_path = ""
    args.time_range = 0.8
    args.output_filepath = "{}_{}_{}s/{}".format(
        args.name, args.date, str(args.time_range), args.pos
    )
    args.TARGET_CHANNEL_12CH = "A2"
    args.cut_min_max_range = [0, 50000.0]
    args.reverse = "off"
    args.type = "{}_{}_{}".format(args.name, args.date, args.pos)
    args.dir_name = "{}/{}".format(args.name, args.type)
    # args.project_path='/home/cs28/share/goto/goto/ecg_project'
    # args.raw_datas_os=RAW_DATA_DIR
    # args.processed_datas_os=args.project_path+'/data/processed'
    # args.processed_datas_os=PROCESSED_DATA_DIR
    args.dataset_made_date = DATASET_MADE_DATE
    args.raw_datas_dir = RAW_DATA_DIR + "/patient_data/patient1/RR_plot"
    args.dataset_output_path = (
        PROCESSED_DATA_DIR
        + "/synchro_data/patient1_{}_{}".format(
            args.dataset_made_date, args.peak_method
        )
    )
    args.patient = "patient4"
    args.test_images_path = TEST_DIR + "/raw_datas_test"
    main(args)
