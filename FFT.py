import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt

from scipy import signal
import sys
import os
from matplotlib.font_manager import FontProperties
from spectrum import *
from scipy.fft import fft, fftfreq
from scipy.signal import get_window

# CSVファイルの読み込み

# ch17まで


# データが激しいところのみ抽出（ブレが大きくなる）
def hpf(d_in, sampling_rate, fp, fs):
    """high pass filter"""
    # fp = 0.5   # 通過域端周波数[Hz]→入力引数に変更済
    # fs = 0.1   # 阻止域端周波数[Hz]→入力引数に変更済
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


def process_csv_file(csv_file_path):
    column_names = [
        "ch1",
        "ch2",
        "ch3",
        "ch4",
        "ch5",
        "ch6",
        "ch7",
        "ch8",
        "ch9",
        "ch10",
        "ch11",
        "ch12",
        "ch13",
        "ch14",
        "ch15",
        "ch16",
        "ch17",
    ]

    # 周波数解析
    df = pd.read_csv(csv_file_path, names=column_names)
    print(df)
    df = df.fillna(0)

    for i in range(1, 17):
        # 検索する文字列
        # 引数随時変更
        target_string = f"ch{i}"

        # 指定した文字列を含む列を見つける
        target_column_index = None
        for col in df.columns:
            if target_string in col:
                target_column_index = df.columns.get_loc(col)
                break

        # 指定した文字列のデータを抽出
        if target_column_index is not None:
            print(target_column_index)
            time_series_data = df.iloc[:, target_column_index].values
            print(time_series_data)
        else:
            print(f'"{target_string}"を含む列が見つかりませんでした。')
        # #データ切り抜き 開始から２秒後からを解析している（変更可能）
        # time_series_data =time_series_data[800:]

        # ハイパス処理
        sampling_rate = 122.06
        hp_fp = 0.5
        hp_fs = 0.2
        time_series_data_after_hpf = hpf(time_series_data, sampling_rate, hp_fp, hp_fs)
        # print(time_series_data_after_hpf)
        # plt.plot(time_series_data_after_hpf)
        # plt.title("after hpf")

        # print(np.size(time_series_data_after_hpf))
        # print(type(time_series_data_after_hpf))

        # 扱うデータ
        data = time_series_data_after_hpf

        # 周波数解析
        time_size = np.arange(0, np.size(data) / sampling_rate, 1.0 / sampling_rate)

        ### CWT
        ### https://pywavelets.readthedocs.io/en/v0.5.1/ref/cwt.html
        wavename = "cgau8"  # 8次の複素ガウスウェーブレットを表示
        totalscal = 256  # 解析に使うスケール（周波数の逆数）の総数
        fc = pywt.central_frequency(wavename)
        cparam = 2 * fc * totalscal
        # scales = np.arange(1, 255)#cparam / np.arange(totalscal, 1, -1)
        scales = cparam / np.arange(totalscal, 1, -1)
        print(f"scales:{scales}")
        print(f"scales shape:{np.shape(scales)}")
        [cwtmatr, frequencies] = pywt.cwt(data, scales, wavename, 1.0 / sampling_rate)
        print(np.size(cwtmatr))
        print(np.size(frequencies))

        wavelet = "mexh"  # ウェーブレットの種類（連続Morletウェーブレット）
        scales = np.arange(1, 128)  # 尺度の範囲
        [coefficients2, frequencies2] = pywt.cwt(
            data, scales, wavelet, 1.0 / sampling_rate
        )
        print(np.shape(coefficients2))

        # 窓関数（ハニング窓）を適用
        N = len(data)
        window = get_window("hann", N)
        windowed_signal = data  # * window

        ### STFT
        f, ti, Zxx = signal.stft(windowed_signal, fs=sampling_rate, nperseg=totalscal)

        ### fft
        N = len(windowed_signal)
        fft_result = fft(windowed_signal)
        fft_fre_result = fftfreq(N, 1.0 / sampling_rate)
        print(fft_result)
        print(fft_fre_result)
        print(np.shape(fft_result))
        print(np.shape(fft_fre_result))

        # 比較
        xlim_min = 25
        x_lin_max = 26
        plt.figure(figsize=(12, 10))

        ax = plt.subplot(611)
        plt.plot(time_size, time_series_data)
        ax.set_title(f"Input Data ch{i}")
        plt.xlim(time_size[0], time_size[-1])
        # plt.xlim(xlim_min, x_lin_max)

        ax = plt.subplot(612)
        plt.plot(time_size, data)
        ax.set_title(f"Input Data (HPF fs: {hp_fs}Hz - fp: {hp_fp}Hz)")
        plt.xlim(time_size[0], time_size[-1])
        # plt.xlim(xlim_min, x_lin_max)

        ax = plt.subplot(613)
        plt.contourf(time_size, frequencies, np.log(abs(cwtmatr)), aspect="auto")
        plt.ylim(0, 60)
        # plt.xlim(xlim_min, x_lin_max)
        plt.ylabel("freq(Hz)")
        ax.set_title(f"CWT wavelet:{wavename}")

        ax = plt.subplot(614)
        plt.contourf(time_size, frequencies2, np.log(abs(coefficients2)))
        # plt.xlim(xlim_min, x_lin_max)
        plt.ylim(0, 60)
        plt.ylabel("freq(Hz)")
        ax.set_title(f"CWT wavelet:{wavelet}")

        ax = plt.subplot(615)
        # plt.contourf(ti, f, np.log(np.abs(Zxx)), vmin=0, vmax=2 * np.sqrt(2))
        plt.contourf(ti, f, np.log(np.abs(Zxx)))
        # plt.xlim(xlim_min, x_lin_max)
        plt.ylim(0, 100)
        ax.set_title("STFT")
        plt.xlabel("time(s)")

        ax = plt.subplot(616)
        # plt.contourf(ti, f, np.log(np.abs(Zxx)), vmin=0, vmax=2 * np.sqrt(2))
        plt.plot(fft_fre_result[: N // 2], fft_result[: N // 2])
        plt.xlim(0, 130)
        ax.set_title("FFT")
        plt.xlabel("frequency(Hz)")

        plt.tight_layout()

        # グラフを保存するディレクトリを取得し、保存
        file_name = f"wavelet_ch{i}_sampling122hz.jpg"
        save_path = f"../data/FFT_result/{file_name}.jpg"
        plt.savefig(save_path, format="jpg")
        print(f"グラフを {save_path} に保存しました。")

        plt.show()
        print("exit")


if __name__ == "__main__":
    # csv読み込み
    csv_file_path = "../data/raw/db20241009_144725.csv"

    process_csv_file(csv_file_path)
