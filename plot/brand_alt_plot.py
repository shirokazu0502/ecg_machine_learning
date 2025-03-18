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
import glob

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
    RATE_16CH,
    TIME,
    DATASET_MADE_DATE,
)


def plot_corr_and_bland_altman(csv_path, output_path):
    # CSVファイルを読み込む
    df = pd.read_csv(csv_path)

    # 15chとlizmilのRRIの差分を計算
    RRI_diff = df["diff_15ch"] - df["diff_lizmil"]
    BPM_15ch = 60 / df["diff_15ch"]
    BPM_lizmil = 60 / df["diff_lizmil"]
    mean_BPM = (BPM_15ch + BPM_lizmil) / 2

    # 相関係数を計算
    corr = df["diff_15ch"].corr(df["diff_lizmil"])

    # サブプロット作成
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # 相関プロット
    x = np.linspace(
        min(df["diff_15ch"].min(), df["diff_lizmil"].min()),
        max(df["diff_15ch"].max(), df["diff_lizmil"].max()),
        100,
    )
    axes[0].plot(x, x, label="y = x (correlation=1)", color="blue", linestyle="--")
    axes[0].scatter(
        df["diff_15ch"], df["diff_lizmil"], label="Data Points", color="r", alpha=0.2
    )
    axes[0].set_title(f"RRI Scatter Plot\nCorrelation: {corr:.3f}")
    axes[0].set_xlabel("15ch_RRI")
    axes[0].set_ylabel("lizmil_RRI")
    axes[0].legend()

    # Bland-Altman プロット
    axes[1].scatter(
        mean_BPM, RRI_diff, color="blue", label="BPM vs RRI Difference", alpha=0.2
    )
    axes[1].set_title("Bland-Altman Plot")
    axes[1].set_xlabel("Mean BPM")
    axes[1].set_ylabel("RRI Difference")
    axes[1].set_ylim([-1, 1])  # y軸の範囲を-1から1に設定
    axes[1].legend()

    # グラフ全体のタイトルとレイアウト調整
    fig.suptitle("Correlation and Bland-Altman Plot", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # 画像を保存
    plt.savefig(output_path)
    plt.show()


def process_all_corr_csv(directory, patient_num):
    merge_df_path = f"{directory}/{patient_num}_merge_df.csv"
    output_path = merge_df_path.replace(".csv", ".png")
    plot_corr_and_bland_altman(merge_df_path, output_path)
    # csv_files = glob.glob(f"{directory}/*_corr*.csv")
    # for csv_file in csv_files:
    #     output_path = csv_file.replace(".csv", ".png")
    #     print(f"Processing {csv_file}...")
    #     plot_corr_and_bland_altman(csv_file, output_path)


patient_num = "patient1"
directory = DATA_DIR + f"/RRI_plot/{patient_num}"  # CSVファイルがあるディレクトリを指定
print(directory)
process_all_corr_csv(directory, patient_num)
