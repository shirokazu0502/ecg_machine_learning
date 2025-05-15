import os
from re import search
import pandas as pd
import argparse
import sys
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sympy import plot

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_dir)
from config.settings import RAW_DATA_DIR, PROCESSED_DATA_DIR, DATASET_MADE_DATE
from config.name_dic import select_name_and_date

name, date = select_name_and_date()
peak_method = "cwt"
dataset_made_date = DATASET_MADE_DATE
dataset_input_path = (
    PROCESSED_DATA_DIR
    + "/pqrst_nkmodule_since{}_{}/{}_{}_0.8s/0/ch_1_base".format(
        dataset_made_date, peak_method, name, date
    )
)
# dataset_input_path = PROCESSED_DATA_DIR + "/synchro_data/patient5"
# dataset_input_path = RAW_DATA_DIR + "/takahashi_test/{}/{}_{}_0/0".format(
#     dataset_made_date, peak_method, name, date
# )
# dataset_number = input("何番目のdatasetが見たいですか？")
for dataset_number in range(20):
    file_path = dataset_input_path + f"/dataset_{str(dataset_number).zfill(3)}.csv"
    data = pd.read_csv(file_path)
    # プロットするカラムを選択（"Time"列を除く）
    columns_to_plot = data.columns[1:]  # "Time"列以外のすべての列
    num_columns = len(columns_to_plot)
    nrows = num_columns // 4 + 1
    ncols = 4
    # グラフの作成
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, num_columns * 2), sharex=True)
    dataset_made_date = DATASET_MADE_DATE
    dataset_output_path = PROCESSED_DATA_DIR + "/pqrst_nkmodule_since{}_{}".format(
        dataset_made_date, peak_method
    )
    # 各カラムをプロット
    for i in range(nrows):
        for j in range(ncols):
            print(columns_to_plot)
            print(i, j)
            if i * 4 + j >= num_columns:
                break
            column = columns_to_plot[i * 4 + j]
            ax = axes[i][j]
            ax.plot(data["Time"], data[column], label=column)
            ax.set_xlabel("Time")
            ax.set_ylabel(column)
            ax.legend(loc="upper right")

    # ラベルの設定とレイアウト調整
    plt.tight_layout()
    plt.show()
    plt.close()
    # 画像を保存
    output_path = "all_columns_plot.png"
    # plt.savefig(output_path)
