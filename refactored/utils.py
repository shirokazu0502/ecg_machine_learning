import os
import csv
import numpy as np
import pandas as pd
from scipy import signal
from scipy.interpolate import interp1d
import neurokit2 as nk
import matplotlib.pyplot as plt

def create_directory_if_not_exists(directory_path):
    """指定されたパスにディレクトリが存在しない場合、作成する"""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"ディレクトリ {directory_path} を作成しました。")
    else:
        print(f"ディレクトリ {directory_path} は既に存在しています。")

def lpf(x, sampling_rate, fp, fs):
    """Low-pass filter"""
    gpass = 1
    gstop = 20
    norm_pass = fp / (sampling_rate / 2)
    norm_stop = fs / (sampling_rate / 2)
    N, Wn = signal.cheb2ord(wp=norm_pass, ws=norm_stop, gpass=gpass, gstop=gstop, analog=0)
    b, a = signal.cheby2(N, gstop, Wn, "low")
    z = signal.lfilter(b, a, x)
    return z

def hpf(x, sampling_rate, fp, fs):
    """High-pass filter"""
    gpass = 1
    gstop = 20
    norm_pass = fp / (sampling_rate / 2)
    norm_stop = fs / (sampling_rate / 2)
    N, Wn = signal.cheb2ord(wp=norm_pass, ws=norm_stop, gpass=gpass, gstop=gstop, analog=0)
    b, a = signal.cheby2(N, gstop, Wn, "high")
    z = signal.lfilter(b, a, x)
    return z

def apply_filter(df, sampling_rate, hpf_fp=2.0, hpf_fs=1.0, lpf_fp=0, lpf_fs=0):
    """データフレームの各列にフィルタを適用する"""
    df_filtered = df.copy()
    for column in df.columns:
        data_col = df[column].values
        if lpf_fp != 0 and lpf_fs != 0:
            data_col = lpf(data_col, sampling_rate, lpf_fp, lpf_fs)
        if hpf_fp != 0 and hpf_fs != 0:
            data_col = hpf(data_col, sampling_rate, hpf_fp, hpf_fs)
        df_filtered[column] = data_col
    return df_filtered

def resample_dataframe(df, old_rate, new_rate):
    """データフレームを線形補間でリサンプリングする"""
    if old_rate == new_rate:
        return df
    
    original_time = np.arange(len(df)) / old_rate
    new_time = np.arange(int(len(df) * new_rate / old_rate)) / new_rate
    
    resampled_df = pd.DataFrame(index=new_time, columns=df.columns)
    
    for column in df.columns:
        interpolator = interp1d(original_time, df[column], kind='linear', fill_value="extrapolate")
        resampled_df[column] = interpolator(new_time)
        
    return resampled_df

def find_r_peaks(ecg_signal, sampling_rate):
    """NeuroKit2を使用してR波ピークを検出する"""
    _, rpeaks = nk.ecg_peaks(ecg_signal, sampling_rate=sampling_rate)
    return rpeaks['ECG_R_Peaks']

def find_optimal_shift(peaks1, peaks2, sampling_rate):
    """2つのピーク時系列のピーク間隔の差分を比較し、最適な時間的シフト（サンプル数）とMSEを返す"""
    diff1 = np.diff(peaks1)
    diff2 = np.diff(peaks2)

    min_mse = float('inf')
    best_shift_idx = 0

    # サイズが小さい方を基準にする
    if len(diff1) < len(diff2):
        shorter_diff = diff1
        longer_diff = diff2
    else:
        shorter_diff = diff2
        longer_diff = diff1

    len_short = len(shorter_diff)
    len_long = len(longer_diff)

    if len_short == 0:
        return 0, 0

    for i in range(len_long - len_short + 1):
        current_subset = longer_diff[i : i + len_short]
        mse = np.mean((current_subset - shorter_diff) ** 2)

        if mse < min_mse:
            min_mse = mse
            # 元のpeaks配列におけるインデックス差をシフト量とする
            if len(diff1) < len(diff2):
                best_shift_idx = peaks2[i] - peaks1[0]
            else:
                best_shift_idx = peaks1[i] - peaks2[0]

    return best_shift_idx, min_mse


def get_pqrst_waves(ecg_signal, rpeaks, sampling_rate):
    """NeuroKit2を使用してPQRST波の各点を検出する"""
    _, waves = nk.ecg_delineate(
        ecg_signal,
        rpeaks,
        sampling_rate=sampling_rate,
        method="dwt", # DWT法はロバスト性が高い
    )
    return waves

def plot_ecg_with_events(ecg_signal, rpeaks, waves):
    """ECG信号と検出されたイベント（PQRST）をプロットする"""
    plt.figure(figsize=(15, 6))
    plt.plot(ecg_signal, label="ECG Signal", color="grey")
    
    # R-peaks
    plt.scatter(rpeaks, ecg_signal[rpeaks], color='red', s=50, label='R-peaks')
    
    # P, Q, S, T waves
    for wave_type in ['ECG_P_Onsets', 'ECG_P_Peaks', 'ECG_P_Offsets', 
                      'ECG_Q_Peaks', 'ECG_S_Peaks', 
                      'ECG_T_Onsets', 'ECG_T_Peaks', 'ECG_T_Offsets']:
        points = waves.get(wave_type, [])
        valid_points = [p for p in points if not np.isnan(p)]
        if valid_points:
            plt.scatter(valid_points, ecg_signal[valid_points], label=wave_type, s=40)
            
    plt.legend()
    plt.title("ECG Signal with PQRST Delineation")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()

