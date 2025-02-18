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
import re

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


def create_corr_blanalt_plot(dataframe):
    corr = dataframe["diff_15ch"].corr(dataframe["diff_lizmil"])
    # サブプロット作成
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # 1行2列のプロット
    # 相関が1の直線 (y = x)
    x = np.linspace(
        min(dataframe["diff_15ch"].min(), dataframe["diff_lizmil"].min()),
        max(dataframe["diff_15ch"].max(), dataframe["diff_lizmil"].max()),
        100,
    )
    axes[0].plot(x, x, label="y = x (correlation=1)", color="blue", linestyle="--")
    axes[0].scatter(
        dataframe["diff_15ch"],
        dataframe["diff_lizmil"],
        label="Data Points",
        color="r",
    )
    axes[0].set_title(f"RRI Scatter Plot\nCorrelation: {corr:.3f}")
    axes[0].set_xlabel("15ch_RRI")
    axes[0].set_ylabel("lizmil_RRI")
    axes[0].legend()

    # サブプロット2: Bland-Altman プロット
    RRI_diff = dataframe["diff_15ch"] - dataframe["diff_lizmil"]
    BPM_15ch = 60 / dataframe["diff_15ch"]
    BPM_lizmil = 60 / dataframe["diff_lizmil"]
    mean_BPM = (BPM_15ch + BPM_lizmil) / 2
    axes[1].scatter(mean_BPM, RRI_diff, color="blue", label="BPM vs RRI Difference")
    axes[1].set_title("Bland-Altman Plot")
    axes[1].set_xlabel("mean_BPM")
    axes[1].set_ylabel("diff_RRI")
    axes[1].set_ylim([-1, 1])  # y軸の範囲を-1から1に設定
    axes[1].legend()
    # グラフ全体のタイトルとレイアウト調整
    fig.suptitle(f"patient5 corr and blanaltman plot", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # 保存と表示
    combined_path = (
        DATA_DIR + f"/RRI_plot/{patient_num}/{patient_num}_merge_corr_blanalt.png"
    )
    plt.savefig(combined_path)
    # plt.show()
    plt.close()
    dataframe.to_csv(DATA_DIR + f"/RRI_plot/{patient_num}/{patient_num}_merge_df.csv")


patient_num = "patient5"
# CSVファイルが保存されているフォルダのパス
folder_path = DATA_DIR + f"/RRI_plot/{patient_num}"

# 結合されたデータを格納するリスト
dataframes = []

# フォルダ内の全てのファイルを確認
for idx, file_name in enumerate(os.listdir(folder_path)):
    # ファイルがCSV形式であるか確認
    if file_name.endswith(".csv"):
        # ファイル名から数値部分を抽出 (例: 1corr0.91.csv -> 0.91)
        match = re.search(r"(?<!\d)(0\.\d{2}|1\.00)(?!\d)", file_name)
        if match:
            value = float(match.group())
            # 数値が0.85より大きい場合に採用
            if value > 0.85:
                # CSVファイルを読み込む
                file_path = os.path.join(folder_path, file_name)
                if idx == 0:
                    df = pd.read_csv(file_path)
                    # diff_15ch と diff_lizmil の差分の絶対値を計算
                    df["diff_diff"] = abs(df["diff_15ch"] - df["diff_lizmil"])

                    # nearest_idxごとに、diff_diffが最小の行を取得
                    df = df.loc[df.groupby("nearest_idx")["diff_diff"].idxmin()]
                    # df.to_csv(file_path)
                else:
                    df = pd.read_csv(file_path, header=0).iloc[:, :]
                    # diff_15ch と diff_lizmil の差分の絶対値を計算
                    df["diff_diff"] = abs(df["diff_15ch"] - df["diff_lizmil"])

                    # nearest_idxごとに、diff_diffが最小の行を取得
                    df = df.loc[df.groupby("nearest_idx")["diff_diff"].idxmin()]
                    # df.to_csv(file_path)
                dataframes.append(df)

# 全てのデータを結合
if dataframes:
    merged_df = pd.concat(dataframes, ignore_index=True)
    # 結合結果を保存
    create_corr_blanalt_plot(merged_df)
else:
    print("条件を満たすCSVファイルが見つかりませんでした。")
