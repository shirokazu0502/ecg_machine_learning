import argparse
import os
import pandas as pd
import numpy as np
from utils import (
    create_directory_if_not_exists,
    apply_filter,
    resample_dataframe,
    find_r_peaks,
    find_optimal_shift,
    get_pqrst_waves,
    plot_ecg_with_events
)
from settings import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SAMPLING_RATE
)

def load_and_prepare_data(filepath, expected_channels):
    """CSVを読み込み、チャンネル数を検証してデータフレームを返す"""
    if not os.path.exists(filepath):
        print(f"エラー: ファイルが見つかりません - {filepath}")
        return None
    df = pd.read_csv(filepath)
    # 元のスクリプトに基づき、不要な列を削除し、列名を整形
    if df.shape[1] >= 18 and expected_channels == 16:
        df = df.drop(columns=df.columns[16:])
        df.columns = [f'ch_{i+1}' for i in range(16)]
    elif df.shape[1] >= 12 and expected_channels == 12:
        # 12誘導データはヘッダーがあると想定
        pass
    else:
        print(f"エラー: {filepath} の列数が想定外です。({df.shape[1]}列)")
        return None
    return df

def main(args):
    """メイン処理"""
    create_directory_if_not_exists(args.output)

    # データの読み込み
    df_12ch = load_and_prepare_data(args.input_12ch, 12)
    df_16ch = load_and_prepare_data(args.input_16ch, 16)

    if df_12ch is None or df_16ch is None:
        return

    # 15chデータの作成 (ch1-15からch16を引く)
    df_15ch = pd.DataFrame()
    for i in range(15):
        df_15ch[f'ch_{i+1}'] = df_16ch[f'ch_{i+1}'] - df_16ch['ch_16']

    # フィルタリング
    df_12ch_filtered = apply_filter(df_12ch, args.rate_12ch, hpf_fp=2.0, hpf_fs=1.0)
    df_15ch_filtered = apply_filter(df_15ch, args.rate_16ch, hpf_fp=2.0, hpf_fs=1.0)

    # ピーク検出
    peaks_12ch = find_r_peaks(df_12ch_filtered[args.target_12ch], args.rate_12ch)
    peaks_15ch = find_r_peaks(df_15ch_filtered[args.target_16ch], args.rate_16ch)

    # 最適な時間オフセットを計算
    shift_samples, mse = find_optimal_shift(peaks_12ch, peaks_15ch, args.rate_12ch)
    print(f"計算された最適シフト: {shift_samples} サンプル (MSE: {mse:.4f})")

    # データフレームを同期
    if shift_samples > 0:
        df_16ch_synced = df_16ch.iloc[shift_samples:].reset_index(drop=True)
        df_12ch_synced = df_12ch.iloc[:len(df_16ch_synced)].reset_index(drop=True)
    else:
        df_12ch_synced = df_12ch.iloc[-shift_samples:].reset_index(drop=True)
        df_16ch_synced = df_16ch.iloc[:len(df_12ch_synced)].reset_index(drop=True)

    # リサンプリングして長さを揃える
    df_12ch_resampled = resample_dataframe(df_12ch_synced, args.rate_12ch, args.resample_rate)
    df_16ch_resampled = resample_dataframe(df_16ch_synced, args.rate_16ch, args.resample_rate)
    
    len_min = min(len(df_12ch_resampled), len(df_16ch_resampled))
    df_12ch_final = df_12ch_resampled.iloc[:len_min]
    df_16ch_final = df_16ch_resampled.iloc[:len_min]

    # 結合して1つのデータフレームに
    df_combined = pd.concat([df_12ch_final, df_16ch_final], axis=1)

    print("同期とリサンプリングが完了しました。")

    # 心拍の切り出し
    final_rate = args.resample_rate
    final_peaks = find_r_peaks(df_combined[args.target_12ch], final_rate)
    time_length_samples = int(args.heartbeat_seconds * final_rate)
    half_length = time_length_samples // 2

    for i, rpeak in enumerate(final_peaks):
        start_idx = rpeak - half_length
        end_idx = rpeak + half_length

        if start_idx < 0 or end_idx > len(df_combined):
            continue

        heartbeat_df = df_combined.iloc[start_idx:end_idx].copy()
        
        output_filename = f"dataset_{i:03d}.csv"
        output_path = os.path.join(args.output, output_filename)
        heartbeat_df.to_csv(output_path, index=False)

    print(f"{args.output} に {len(final_peaks)} 個の心拍データを保存しました。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ECGデータセット作成スクリプト（同期・リサンプリング機能付き）")
    parser.add_argument("--input_12ch", type=str, required=True, help="入力12誘導ECGのCSVパス")
    parser.add_argument("--input_16ch", type=str, required=True, help="入力16誘導センサーのCSVパス")
    parser.add_argument("-o", "--output", type=str, default=DEFAULT_OUTPUT_DIR, help="出力ディレクトリのパス")
    parser.add_argument("--rate_12ch", type=int, default=500, help="12誘導ECGのサンプリングレート")
    parser.add_argument("--rate_16ch", type=float, default=122.06, help="16誘導センサーのサンプリングレート")
    parser.add_argument("-rr", "--resample_rate", type=int, default=500, help="リサンプリング後のレート")
    parser.add_argument("--target_12ch", type=str, default="A2", help="同期に使用する12誘導のチャンネル名")
    parser.add_argument("--target_16ch", type=str, default="ch_1", help="同期に使用する15誘導のチャンネル名")
    parser.add_argument("-s", "--heartbeat_seconds", type=float, default=0.8, help="切り出す心拍の秒数")
    parser.add_argument("--debug_plot", action="store_true", help="デバッグ用のプロットを表示する")

    args = parser.parse_args()
    main(args)
