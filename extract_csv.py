from heapq import merge
import pandas as pd
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt
import sys
import matplotlib.pyplot as plt
from sympy import plot

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
from config.settings import RAW_DATA_DIR, OUTPUT_DIR


def bandpass_filter(signal, lowcut=0.5, highcut=50.0, fs=250, order=4):
    """バンドパスフィルタでノイズを除去"""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)


def detect_r_peaks(ecg_signal, fs=250):
    """R波ピークを検出する"""
    print(ecg_signal)
    peaks, _ = find_peaks(
        ecg_signal, distance=fs * 0.6, height=np.mean(ecg_signal)
    )  # 0.6秒以上の間隔
    return peaks


def compute_rri(peaks, fs=250):
    """R-R間隔を計算する（ミリ秒単位）"""
    rr_intervals = np.diff(peaks) / fs * 1000  # ミリ秒単位に変換
    rr_times = peaks[1:] / fs  # Rピークの時間（秒）
    return rr_times, rr_intervals


def plot_rri(rr_times, rr_intervals):
    """RRIプロットを作成する"""
    plt.figure(figsize=(10, 4))
    plt.plot(rr_times, rr_intervals, marker="o", linestyle="-", color="b")
    plt.xlabel("Time (seconds)")
    plt.ylabel("R-R Interval (ms)")
    plt.title("RRI Plot")
    plt.grid()
    plt.show()
    plt.close()


process_flg = 2
if process_flg == 0:
    patient_datas_dir = RAW_DATA_DIR + "/takahashi_test/patient1"
    # df = pd.read_csv(patient_datas_dir+"/db20240513_180746.csv",header=None)
    df = pd.read_csv(patient_datas_dir + "/dbraw_20240513_180746.csv", header=None)
    # あるデータフレームの範囲を抽出する
    # # 5000刻みで3つfor文で作成する
    # for i in range(3):
    #     extract_df = df[5000*i:5000*(i+1)]
    #     extract_df.to_csv(RAW_DATA_DIR+'/extract_patient_data/db_patient_16ch_data{}_{}_{}.csv'.format(i+1, 5000*i, 5000*(i+1)), header=False, index=False)
    #     print("データを抽出しました。")
    extract_df = df[0:100000]
    extract_df.to_csv(
        RAW_DATA_DIR + "/takahashi_test/patient1/db_patient1_16ch_data_0s_820s.csv",
        header=False,
        index=False,
    )
elif process_flg == 1:
    merge_df = pd.DataFrame(columns=["A1", "A2", "V1", "V2", "V3", "V4", "V5", "V6"])
    patient_output_dir = (
        OUTPUT_DIR
        + "/patient_data0427/PRTweight_1.0_0.001_1.0_augumentation=/Waveforms/patient"
    )
    # 10個のCSVをつなげる
    for i in range(10):
        df = pd.read_csv(
            patient_output_dir
            + "/patient_0427_0.8s_0_dataset0{:02}_reconx.csv".format(i),
            header=0,
        )
        merge_df = pd.concat([merge_df, df])
    merge_df.to_csv(
        patient_output_dir + "/patient_merge.csv", header=False, index=False
    )
elif process_flg == 2:
    patient_num = "patient9"
    dir_path = RAW_DATA_DIR + f"/patient_data/{patient_num}"
    for tmp_name in os.listdir(dir_path):
        if "lizmil" in tmp_name:
            filename = tmp_name
            break
    lizmil_file_path = os.path.join(dir_path, filename)
    lizmil_df = pd.read_csv(lizmil_file_path, header=None)
    extract_range = 1000000
    for i in range(0, len(lizmil_df) // extract_range):
        extract_df = lizmil_df[i * extract_range : (i + 1) * extract_range]
        extract_df.to_csv(
            RAW_DATA_DIR + f"/patient_data/{patient_num}/lizmil_{i}.csv",
            header=False,
            index=False,
        )
        extract_df = extract_df[0]
        fs = 128.2
        # filtered_signal = bandpass_filter(extract_df, fs)
        print(extract_df)
        r_peaks = detect_r_peaks(extract_df, fs)
        rr_times, rri = compute_rri(r_peaks, fs)
        plot_rri(rr_times, rri)

else:
    print("i")
