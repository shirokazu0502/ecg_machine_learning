import pandas as pd
import numpy as np
import neurokit2 as nk
import matplotlib.pyplot as plt
import os
import sys
import re
from glob import glob

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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


def extract_peaks(file_path, choice_flg, choice_column=None):
    df = pd.read_csv(file_path)
    if choice_flg:
        if not choice_column:
            choice_column = input(
                "以下の中から選んでください。A1, A2, V1, V2, V3, V4, V5, V6"
            )
    else:
        choice_column = "A2"
    ecg_signal = df[choice_column].copy().values
    ecg_signal = np.tile(ecg_signal, 10)
    # ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=500, method="neurokit")
    ecg_cleaned = ecg_signal
    rpeaks = nk.ecg_peaks(ecg_cleaned, 500)[1]["ECG_R_Peaks"]

    waves_peak_dict = nk.ecg_delineate(
        ecg_cleaned, rpeaks, sampling_rate=500, method="cwt", show=False
    )[1]

    extracted_data = {}
    for key, values in waves_peak_dict.items():
        if values is not None:
            closest_rpeak = min(
                rpeaks, key=lambda x: abs(x - values[0])
            )  # 検出できる波のリスト先頭が同様のR波とは限らないため
            valid_values = [v for v in values if not np.isnan(v)]
            extracted_data[key] = (
                valid_values[0] - closest_rpeak + 200 if valid_values else None
            )
    ecg_cleaned = ecg_cleaned[rpeaks[1] - 200 : rpeaks[1] + 200]
    extracted_data["ECG_R_Peaks"] = rpeaks[0] if len(rpeaks) > 0 else None
    return ecg_cleaned, extracted_data, choice_column


def calculate_dulation(index1, index2, rate):
    if index1 == None or index2 == None:
        return None
    else:
        return (index1 - index2) / rate


def plot_signal(waves_peak_dict, ecg_signal):
    fig = plt.figure(figsize=(24, 12))
    ax = plt.axes()
    ax.plot(ecg_signal)
    # ax.set_ylim(0.3, 0.8)
    color_dict = {
        "P_Onsets": "b",  # 青
        "P_Peaks": "deepskyblue",  # 緑
        "P_Offsets": "royalblue",  # 赤
        "Q_Peaks": "y",  # シアン
        "R_Peaks": "r",  # 青
        "R_Onsets": "darkred",  # マゼンタ
        "R_Offsets": "tomato",  # 黄
        "S_Peaks": "brown",  # 黒
        "T_Onsets": "g",  # 紫
        "T_Peaks": "limegreen",  # オレンジ
        "T_Offsets": "forestgreen",  # ブラウン
    }

    ecg_p_onsets = waves_peak_dict.get("ECG_P_Onsets")
    if ecg_p_onsets is not None:
        valid_ecg_p_onsets = np.array(ecg_p_onsets)[~np.isnan(ecg_p_onsets)].astype(int)
        ax.plot(
            valid_ecg_p_onsets,
            ecg_signal[valid_ecg_p_onsets],
            color_dict["P_Onsets"],
            label="P onset",
            marker="v",
            linestyle="None",
            alpha=0.7,
        )

    ecg_p_peaks = waves_peak_dict.get("ECG_P_Peaks")
    if ecg_p_peaks is not None:
        valid_ecg_p_peaks = np.array(ecg_p_peaks)[~np.isnan(ecg_p_peaks)].astype(int)
        ax.plot(
            valid_ecg_p_peaks,
            ecg_signal[valid_ecg_p_peaks],
            color_dict["P_Peaks"],
            label="P peaks",
            marker="o",
            linestyle="None",
            alpha=0.7,
        )
    # "ECG_S_Peaks"の処理

    ecg_p_offsets = waves_peak_dict.get("ECG_P_Offsets")
    if ecg_p_offsets is not None:
        valid_ecg_p_offsets = np.array(ecg_p_offsets)[~np.isnan(ecg_p_offsets)].astype(
            int
        )
        ax.plot(
            valid_ecg_p_offsets,
            ecg_signal[valid_ecg_p_offsets],
            color_dict["P_Offsets"],
            label="P Offsets",
            marker="^",
            linestyle="None",
            alpha=0.7,
        )

    ecg_q_peaks = waves_peak_dict.get("ECG_Q_Peaks")
    if ecg_q_peaks is not None:
        valid_ecg_q_peaks = np.array(ecg_q_peaks)[~np.isnan(ecg_q_peaks)].astype(int)
        ax.plot(
            valid_ecg_q_peaks,
            ecg_signal[valid_ecg_q_peaks],
            color_dict["Q_Peaks"],
            label="Q peaks",
            marker="o",
            linestyle="None",
            alpha=0.7,
        )
    # R波
    rpeaks = waves_peak_dict.get("ECG_R_Peaks")
    ax.plot(200, ecg_signal[rpeaks], "ro", label="R peaks", alpha=0.7)

    # ax.set_ylim(0.3, 0.8)
    ecg_s_peaks = waves_peak_dict.get("ECG_S_Peaks")
    if ecg_s_peaks is not None:
        valid_ecg_s_peaks = np.array(ecg_s_peaks)[~np.isnan(ecg_s_peaks)].astype(int)
        ax.plot(
            valid_ecg_s_peaks,
            ecg_signal[valid_ecg_s_peaks],
            color_dict["S_Peaks"],
            label="S peaks",
            marker="o",
            linestyle="None",
            alpha=0.7,
        )

    ecg_t_onsets = waves_peak_dict.get("ECG_T_Onsets")
    if ecg_t_onsets is not None:
        valid_ecg_t_onsets = np.array(ecg_t_onsets)[~np.isnan(ecg_t_onsets)].astype(int)
        ax.plot(
            valid_ecg_t_onsets,
            ecg_signal[valid_ecg_t_onsets],
            color_dict["T_Onsets"],
            label="T onset",
            marker="v",
            linestyle="None",
            alpha=0.7,
        )

    ecg_t_peaks = waves_peak_dict.get("ECG_T_Peaks")
    if ecg_t_peaks is not None:
        valid_ecg_t_peaks = np.array(ecg_t_peaks)[~np.isnan(ecg_t_peaks)].astype(int)
        ax.plot(
            valid_ecg_t_peaks,
            ecg_signal[valid_ecg_t_peaks],
            color_dict["T_Peaks"],
            label="T peaks",
            marker="o",
            linestyle="None",
            alpha=0.7,
        )

    ecg_t_offsets = waves_peak_dict.get("ECG_T_Offsets")
    if ecg_t_offsets is not None:
        valid_ecg_t_offsets = np.array(ecg_t_offsets)[~np.isnan(ecg_t_offsets)].astype(
            int
        )
        ax.plot(
            valid_ecg_t_offsets,
            ecg_signal[valid_ecg_t_offsets],
            color_dict["T_Offsets"],
            label="T Offsets",
            marker="^",
            linestyle="None",
            alpha=0.7,
        )
        # ax.set_title("{}_{}_{}".format(args.TARGET_NAME,args.TARGET_CHANNEL,signal_type))
        # クリックイベントで座標取得
    ax.legend()
    ax.grid(True)
    print("グラフ上のポイントをクリックしてください（Enterキーで終了）")
    selected_points = plt.ginput(0, timeout=0)  # 無制限にクリック可能
    plt.show()  # グラフを表示
    plt.close()
    # 取得した座標の表示と保存
    print("選択した座標:", selected_points)
    return selected_points


def process_and_detect_peaks(file_reconx, file_xo, column_choice_flg):
    ecg_cleaned_reconx, peaks_reconx, choice_column = extract_peaks(
        file_reconx, column_choice_flg
    )
    ecg_cleaned_xo, peaks_xo, choice_column = extract_peaks(
        file_xo, column_choice_flg, choice_column
    )
    # print(peaks_reconx)
    selected_points_reconx = plot_signal(peaks_reconx, ecg_cleaned_reconx)
    selected_points_xo = plot_signal(peaks_xo, ecg_cleaned_xo)
    try:
        use_flg = int(
            input(
                "再構成波形、元波形、の全てを入力する:3, 元波形の全てを入力する:2, そのまま誤差比較に使用する:1, 使用しない:0"
            )
        )
    except:
        use_flg = int(
            input(
                "整数を入力して!!元波系のP波を入力する:2, そのまま誤差比較に使用する:1, 使用しない:0"
            )
        )
    if use_flg == 0:
        return -1
    parameter_list = [
        "ECG_P_Onsets",
        "ECG_P_Peaks",
        "ECG_P_Offsets",
        "ECG_Q_Peaks",
        "ECG_S_Peaks",
        "ECG_T_Onsets",
        "ECG_T_Peaks",
        "ECG_T_Offsets",
    ]
    if use_flg == 2:
        # input_values = input("元波形の値を順番に入力、ない場合は0を入力").split()
        # for i, parameter in enumerate(parameter_list):
        #     if input_values[i] != "0":
        #         peaks_xo[parameter] = int(input_values[i])
        # x座標のみを取得し、整数値に変換
        input_values = [int(x) for x, y in selected_points_xo]
        for i, parameter in enumerate(parameter_list):
            peaks_xo[parameter] = int(input_values[i])
    elif use_flg == 3:
        # x座標のみを取得し、整数値に変換
        input_values_reconx = [int(x) for x, y in selected_points_reconx]
        for i, parameter in enumerate(parameter_list):
            peaks_reconx[parameter] = int(input_values_reconx[i])
        # x座標のみを取得し、整数値に変換
        input_values_xo = [int(x) for x, y in selected_points_xo]
        for i, parameter in enumerate(parameter_list):
            peaks_reconx[parameter] = int(input_values_xo[i])

    print(peaks_xo)

    if peaks_reconx["ECG_P_Peaks"] == None or peaks_xo["ECG_P_Peaks"] == None:
        return None  # スキップ
    peak_dic_reconx = {
        "p_peak_amplitude": ecg_cleaned_reconx[peaks_reconx.get("ECG_P_Peaks")],
        "p_duration": calculate_dulation(
            peaks_reconx.get("ECG_P_Offsets"), peaks_reconx.get("ECG_P_Onsets"), 500
        ),
        "r_peak_amplitude": ecg_cleaned_reconx[peaks_reconx.get("ECG_R_Peaks")],
        "qrs_duration": calculate_dulation(
            peaks_reconx.get("ECG_S_Peaks"), peaks_reconx.get("ECG_Q_Peaks"), 500
        ),
        "t_peak_amplitude": ecg_cleaned_reconx[peaks_reconx.get("ECG_T_Peaks")],
        "t_duration": calculate_dulation(
            peaks_reconx.get("ECG_T_Offsets"), peaks_reconx.get("ECG_T_Onsets"), 500
        ),
        "pq_duration": calculate_dulation(
            peaks_reconx.get("ECG_Q_Peaks"), peaks_reconx.get("ECG_P_Onsets"), 500
        ),
        "qr_duration": (calculate_dulation(200, peaks_reconx.get("ECG_Q_Peaks"), 500)),
    }

    peak_dic_xo = {
        "p_peak_amplitude": ecg_cleaned_xo[peaks_xo.get("ECG_P_Peaks")],
        "p_duration": calculate_dulation(
            peaks_xo.get("ECG_P_Offsets"), peaks_xo.get("ECG_P_Onsets"), 500
        ),
        "r_peak_amplitude": ecg_cleaned_xo[peaks_xo.get("ECG_R_Peaks")],
        "qrs_duration": calculate_dulation(
            peaks_xo.get("ECG_S_Peaks"), peaks_xo.get("ECG_Q_Peaks"), 500
        ),
        "t_peak_amplitude": ecg_cleaned_xo[peaks_xo.get("ECG_T_Peaks")],
        "t_duration": calculate_dulation(
            peaks_xo.get("ECG_T_Offsets"), peaks_xo.get("ECG_T_Onsets"), 500
        ),
        "pq_duration": calculate_dulation(
            peaks_xo.get("ECG_Q_Peaks"), peaks_xo.get("ECG_P_Onsets"), 500
        ),
        "qr_duration": (calculate_dulation(200, peaks_xo.get("ECG_Q_Peaks"), 500)),
    }

    differences = {
        key: abs(peak_dic_xo[key] - peak_dic_reconx[key])
        for key in peak_dic_xo.keys()
        if peak_dic_xo[key] is not None and peak_dic_reconx[key] is not None
    }
    return differences


def calculate_difference(data_type_list, folder_path):
    dataset_files = [
        os.path.join(folder_path, file)
        for file in os.listdir(folder_path)
        if file.startswith("dataset_") and file.endswith(".csv")
    ]

    all_differences = []

    for dataset_file in dataset_files:
        try:
            xo_dic, reconx_dic = {}, {}
            for data_type in data_type_list:
                extracted_peaks = process_and_detect_peaks(data_type, dataset_file)

                peak_dic = {}
                try:
                    peak_dic["p_peak"] = extracted_peaks["ECG_P_Peaks"]["Amplitude"]
                    peak_dic["p_duration"] = calculate_dulation(
                        extracted_peaks["ECG_P_Offsets"]["Index"],
                        extracted_peaks["ECG_P_Onsets"]["Index"],
                        500,
                    )
                    peak_dic["r_peak"] = extracted_peaks["ECG_R_Peaks"]["Amplitude"]

                    peak_dic["qrs_duration"] = calculate_dulation(
                        extracted_peaks["ECG_S_Peaks"]["Index"],
                        extracted_peaks["ECG_Q_Peaks"]["Index"],
                        500,
                    )
                    peak_dic["t_peak"] = extracted_peaks["ECG_T_Peaks"]["Amplitude"]
                    peak_dic["t_duration"] = calculate_dulation(
                        extracted_peaks["ECG_T_Offsets"]["Index"],
                        extracted_peaks["ECG_T_Onsets"]["Index"],
                        500,
                    )
                    peak_dic["pq_duration"] = calculate_dulation(
                        extracted_peaks["ECG_Q_Peaks"]["Index"],
                        extracted_peaks["ECG_P_Onsets"]["Index"],
                        500,
                    )
                    peak_dic["qr_duration"] = abs(
                        calculate_dulation(
                            extracted_peaks["ECG_R_Offsets"]["Index"],
                            extracted_peaks["ECG_Q_Peaks"]["Index"],
                            500,
                        )
                    )
                except TypeError:
                    print(f"Skipping {dataset_file} due to incomplete data.")
                    continue

                if data_type == "xo":
                    xo_dic = peak_dic
                else:
                    reconx_dic = peak_dic

            if xo_dic and reconx_dic:
                differences = {
                    key: abs(xo_dic[key] - reconx_dic[key]) for key in xo_dic.keys()
                }
                all_differences.append(differences)
                print("aaaaaaaaaa")
                print(all_differences)
            else:
                print(f"Skipping {dataset_file} due to missing peak dictionaries.")

        except Exception as e:
            print(f"Error processing {dataset_file}: {e}")

    # 平均を計算
    if all_differences:
        df = pd.DataFrame(all_differences)
        mean_differences = df.mean().to_dict()
        print("Mean differences:", mean_differences)

        # CSVに保存
        output_path = "mean_differences.csv"
        df.to_csv(output_path, index=False)
        print(f"差分データを {output_path} に保存しました。")
        return mean_differences
    else:
        print("No valid datasets found.")
        return None


# 実行
data_type_list = ["reconx", "xo"]
name, date = select_name_and_date()
folder_path = os.path.join(
    DATA_DIR, f"compare_recon_error/0.2_0.02_1.0/{name}/"
)  # 適切なフォルダを指定
print(folder_path)
files = glob(folder_path + "*.csv")
pattern = re.compile(r"dataset(\d+)_")

# ファイルを番号順にソート
file_pairs = {}
for file in files:
    match = pattern.search(file)
    if match:
        index = int(match.group(1))
        if "reconx" in file:
            file_pairs.setdefault(index, {})["reconx"] = file
        elif "xo" in file:
            file_pairs.setdefault(index, {})["xo"] = file

sorted_file_pairs = sorted(
    [(k, v) for k, v in file_pairs.items() if "reconx" in v and "xo" in v]
)
all_differences = []

column_choice_flg = int(input("波形ごとに誘導を選択しますか?:1 一つの誘導で固定:0"))
for _, pair in sorted_file_pairs:
    print(pair)
    differences = process_and_detect_peaks(
        pair["reconx"], pair["xo"], column_choice_flg
    )
    print(differences)
    # print(differences)
    if differences == -1:
        continue
    elif differences:
        all_differences.append(differences)

if all_differences:
    df = pd.DataFrame(all_differences)
    mean_differences = df.mean()
    print("Mean differences:")
    print(mean_differences)

    output_path = os.path.join(folder_path, f"{name}_mean_differences.csv")
    mean_differences.to_csv(output_path, index=True)
    # print(f"Mean differences saved to {output_path}")
else:
    print("No valid data pairs found.")

# mean_differences = calculate_difference(data_type_list, folder_path)
