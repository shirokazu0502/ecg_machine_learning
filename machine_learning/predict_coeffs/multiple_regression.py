import os
import sys
import pandas as pd
import glob
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import japanize_matplotlib  # 日本語表示のため
from pathlib import Path

base_dir = str(Path(__file__).resolve().parents[3])
sys.path.append(base_dir)
from config.settings import PROCESSED_DATA_DIR


def analyze_each_subject_individually():
    """
    【最終版】
    被験者ごとに独立した重回帰分析（R波ピーク周辺41点）を実行し、
    R2スコアをCSVに保存し、結果をプロットする。
    """
    # --- 1. ディレクトリと被験者を検索 ---
    base_path = os.path.join(PROCESSED_DATA_DIR, "15ch_arrange_direction")
    if not os.path.isdir(base_path):
        print(f"エラー: ベースディレクトリが存在しません: {base_path}")
        return
    try:
        subject_dirs = sorted(
            [
                d
                for d in os.listdir(base_path)
                if os.path.isdir(os.path.join(base_path, d))
            ]
        )
    except Exception as e:
        print(f"エラー: ディレクトリにアクセスできません: {e}")
        return

    print(
        f"以下の {len(subject_dirs)} 名の被験者について、個別に重回帰分析を実施します。\n"
    )

    # CSV保存用の結果を格納するリスト
    r2_results_list = []
    output_channels = None

    # --- 2. 被験者ごとにループ処理 ---
    for subject in subject_dirs:
        print(f"--- 被験者: {subject} の分析を開始 ---")

        search_path = os.path.join(
            base_path,
            subject,
            "0",
            "moving_ave_datasets",
            "dataset_*.csv",
        )
        file_list = sorted(glob.glob(search_path))

        if not file_list:
            print(f"  データファイルが見つからないため、スキップします。\n")
            continue

        subject_segments = []
        for file_path in file_list:
            try:
                data = pd.read_csv(file_path)
                if output_channels is None:
                    output_channels = (
                        ["A1", "A2", "A3"]
                        + [f"V{i}" for i in range(1, 7)]
                        + ["aVL", "aVF"]
                    )

                segment_start, segment_end = 130, 170
                data_segment = data.loc[segment_start:segment_end]

                if len(data_segment) == (segment_end - segment_start + 1):
                    subject_segments.append(data_segment)
            except Exception:
                continue

        if not subject_segments:
            print(f"  有効な心拍データが見つからないため、分析をスキップします。\n")
            continue

        subject_data = pd.concat(subject_segments, ignore_index=True)
        print(
            f"  {len(file_list)} 心拍 x 41点 = 計 {len(subject_data)} 点のデータを抽出しました。"
        )

        subject_data = subject_data.ffill().bfill().fillna(0)

        X = subject_data.iloc[:, 1:16]
        y = subject_data[output_channels]

        model = LinearRegression()
        model.fit(X, y)

        y_pred = model.predict(X)
        y_pred_df = pd.DataFrame(y_pred, columns=output_channels, index=y.index)

        print(f"  被験者 {subject} のR2スコア:")
        for ch_name in output_channels:
            score = r2_score(y[ch_name], y_pred_df[ch_name])
            print(f"    {ch_name}: {score:.4f}")
            r2_results_list.append(
                {"Subject": subject, "Channel": ch_name, "R2_Score": score}
            )

        # --- プロット処理 (散布図で全体の関係性を表示) ---
        print(f"  被験者 {subject} の結果をプロットします...")
        num_channels = len(output_channels)
        num_rows = 3
        num_cols = 4

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 12))
        axes = axes.flatten()

        for i, channel_to_plot in enumerate(output_channels):
            ax = axes[i]
            ax.scatter(
                y[channel_to_plot],
                y_pred_df[channel_to_plot],
                alpha=0.3,
                edgecolors="none",
                s=15,
            )

            lims = [
                min(y[channel_to_plot].min(), y_pred_df[channel_to_plot].min()),
                max(y[channel_to_plot].max(), y_pred_df[channel_to_plot].max()),
            ]
            padding = (lims[1] - lims[0]) * 0.05
            lims = [lims[0] - padding, lims[1] + padding]

            ax.plot(lims, lims, "r--", alpha=0.75, zorder=0)
            ax.set_xlim(lims)
            ax.set_ylim(lims)

            ax.set_title(channel_to_plot, fontsize=12)
            ax.set_xlabel("実際の電圧値", fontsize=9)
            ax.set_ylabel("予測された電圧値", fontsize=9)
            ax.grid(True)
            ax.set_aspect("equal", adjustable="box")

        for i in range(num_channels, len(axes)):
            axes[i].set_visible(False)

        fig.suptitle(
            f"被験者 {subject} の個別回帰分析結果（41点 x 全心拍）", fontsize=20
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

        print(f"--- 被験者: {subject} の分析を完了 ---\n")

    # --- 3. 全被験者のR2スコアをCSVに保存し、サマリーを表示 ---
    if r2_results_list:
        r2_df = pd.DataFrame(r2_results_list)
        pivot_df = r2_df.pivot_table(
            index="Subject", columns="Channel", values="R2_Score"
        ).reindex(columns=output_channels)
        mean_scores = pivot_df.mean()
        mean_scores.name = "Average"
        final_r2_df = pd.concat([pivot_df, mean_scores.to_frame().T])
        output_csv_path = "individual_segment_regression_r2_scores.csv"
        final_r2_df.to_csv(output_csv_path, encoding="utf-8-sig", float_format="%.4f")
        print(f"\n全被験者のR2スコア概要を '{output_csv_path}' に保存しました。")

        print("\n" + "=" * 50)
        print("---【R2スコア サマリー】全被験者を通した平均パフォーマンス ---")
        summary = (
            r2_df.groupby("Channel")["R2_Score"]
            .agg(["mean", "std"])
            .reindex(output_channels)
        )
        for ch_name, stats in summary.iterrows():
            print(f"  {ch_name:<5}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        print("=" * 50)


if __name__ == "__main__":
    analyze_each_subject_individually()
