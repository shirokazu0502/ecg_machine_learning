import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.lines as mlines

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_dir)
from config.settings import RAW_DATA_DIR, PROCESSED_DATA_DIR, DATASET_MADE_DATE


def create_r_peak_scatter_plots_v1_subtractions():
    """
    V1からV2-V6を引いた差分をターゲットとし、入力チャネルとの比較グラフを
    1枚ずつ順番に表示する関数。
    """
    base_path = os.path.join(PROCESSED_DATA_DIR, "non_normalize")

    if not os.path.isdir(base_path):
        print(f"エラー: 指定されたディレクトリが存在しません: {base_path}")
        return

    try:
        subject_dirs = sorted(
            [
                d
                for d in os.listdir(base_path)
                if os.path.isdir(os.path.join(base_path, d))
            ]
        )
        if not subject_dirs:
            print(f"エラー: {base_path} 内に被験者のディレクトリが見つかりません。")
            return
    except FileNotFoundError:
        print(f"エラー: ディレクトリにアクセスできません: {base_path}")
        return

    print(f"被験者を検出しました: {subject_dirs}")

    # --- 最初のファイルから列名を確認 ---
    try:
        sample_file = os.path.join(
            base_path,
            subject_dirs[0],
            "0",
            "ch_16_base",
            "moving_ave_datasets",
            "dataset_000.csv",
        )
        sample_data = pd.read_csv(sample_file)
        all_columns = list(sample_data.columns)
    except Exception as e:
        print(f"エラー: サンプルファイルの読み込みに失敗しました: {e}")
        return

    x_columns = all_columns[1:16]
    # ★★★★★ 比較ターゲットをV1からの差分に拡張 ★★★★★
    y_columns = ["V1-V2", "V1-V3", "V1-V4", "V1-V5", "V1-V6"]
    required_original_cols = ["V1", "V2", "V3", "V4", "V5", "V6"]

    for col in required_original_cols:
        if col not in all_columns:
            print(f"エラー: 差分計算に必要な列 '{col}' がデータに存在しません。")
            return

    # --- 共通のスタイル設定 ---
    num_subjects = len(subject_dirs)
    subject_colors = plt.get_cmap("tab20")(np.linspace(0, 1, num_subjects))
    markers = [
        "o",
        "s",
        "D",
        "^",
        "v",
        "<",
        ">",
        "p",
        "h",
        "x",
        "+",
        "*",
        "|",
        "_",
        ".",
    ]

    # --- ターゲットチャネルごとにグラフを1枚ずつ作成 ---
    for y_col in y_columns:

        print(f"\n--- グラフを作成中: Input Channels vs {y_col} ---")

        fig, ax = plt.subplots(figsize=(14, 10))

        for i, subject in enumerate(subject_dirs):
            file_path = os.path.join(
                base_path,
                subject,
                "0",
                "ch_16_base",
                "moving_ave_datasets",
                "dataset_000.csv",
            )
            if not os.path.exists(file_path):
                continue
            try:
                data = pd.read_csv(file_path)
                if len(data) <= 150:
                    continue

                r_peak_point = data.iloc[150].copy()

                # ★★★★★ 文字列から差分を動的に計算 ★★★★★
                col1, col2 = y_col.split("-")
                y_val = r_peak_point[col1] - r_peak_point[col2]

                for j, x_col in enumerate(x_columns):
                    if x_col not in r_peak_point:
                        continue
                    x_val = r_peak_point[x_col]
                    marker = markers[j % len(markers)]
                    ax.scatter(
                        x_val,
                        y_val,
                        color=subject_colors[i],
                        marker=marker,
                        s=50,
                        edgecolors="k",
                        alpha=0.7,
                    )
            except Exception as e:
                print(f"エラー: {file_path} の処理中に問題が発生しました: {e}")

        # --- グラフの装飾 ---
        ax.set_title(f"R-wave Peak: Input Channels vs {y_col}", fontsize=16)
        ax.set_xlabel("Input Channel Value (cols 2–16)", fontsize=12)
        ax.set_ylabel(f"Channel Difference: {y_col}", fontsize=12)
        ax.grid(True)

        # --- 凡例の作成 ---
        subject_handles = [
            mlines.Line2D(
                [],
                [],
                color=subject_colors[i],
                marker="o",
                linestyle="None",
                markersize=8,
                label=subject,
            )
            for i, subject in enumerate(subject_dirs)
        ]
        channel_handles = [
            mlines.Line2D(
                [],
                [],
                color="gray",
                marker=markers[j % len(markers)],
                linestyle="None",
                markersize=8,
                label=col,
            )
            for j, col in enumerate(x_columns)
        ]

        # --- レイアウト調整と凡例表示 ---
        fig.subplots_adjust(right=0.78)
        legend1 = fig.legend(
            handles=subject_handles,
            title="Subjects",
            bbox_to_anchor=(0.8, 0.9),
            loc="upper left",
            title_fontsize="large",
            fontsize="medium",
        )
        fig.add_artist(legend1)
        fig.legend(
            handles=channel_handles,
            title="Input Channels",
            bbox_to_anchor=(0.8, 0.5),
            loc="center left",
            title_fontsize="large",
            fontsize="medium",
        )

        plt.show()


if __name__ == "__main__":
    create_r_peak_scatter_plots_v1_subtractions()
