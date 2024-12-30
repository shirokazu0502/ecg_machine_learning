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
from config.settings import RAW_DATA_DIR, PROCESSED_DATA_DIR

file_path = (
    PROCESSED_DATA_DIR
    + "/pqrst_nkmodule_since1109_cwt/asano_0710_0.8s/0/dataset_000.csv"
)
data = pd.read_csv(file_path)
# プロットするカラムを選択（"Time"列を除く）
columns_to_plot = data.columns[1:]  # "Time"列以外のすべての列
num_columns = len(columns_to_plot)

# グラフの作成
fig, axes = plt.subplots(
    nrows=num_columns, ncols=1, figsize=(15, num_columns * 2), sharex=True
)

# 各カラムをプロット
for i, column in enumerate(columns_to_plot):
    ax = axes[i] if num_columns > 1 else axes
    ax.plot(data["Time"], data[column], label=column)
    ax.set_ylabel(column)
    ax.legend(loc="upper right")

# ラベルの設定とレイアウト調整
axes[-1].set_xlabel("Time")
plt.tight_layout()
plt.show()
plt.close()
# 画像を保存
output_path = "all_columns_plot.png"
# plt.savefig(output_path)
print(f"プロット画像が {output_path} に保存されました。")
