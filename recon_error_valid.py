import pandas as pd
import numpy as np
import neurokit2 as nk
import matplotlib.pyplot as plt
import os
import sys

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


def process_and_detect_peaks(data_type):
    df_12ch = pd.read_csv(
        DATA_DIR
        + f"/compare_recon_error/patient4_1001_0.8s_0_dataset006_{data_type}.csv"
    )
    column = "A2"
    ecg_signal = df_12ch[column].copy().values
    ecg_signal = np.tile(ecg_signal, 10)
    # NaNの確認

    ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=500, method="neurokit")
    # ecg_cleaned = np.array(ecg_cleaned, dtype=np.float64)
    rpeaks = nk.ecg_peaks(ecg_signal, 500)[1]["ECG_R_Peaks"]  # R波の位置を取得
    # print("ecg_cleaned contains NaN:", np.isnan(ecg_cleaned).any())
    print(ecg_cleaned)
    # print("rpeaks contains NaN:", np.isnan(rpeaks).any())
    # ecg_signals, info = nk.ecg_process(ecg_cleaned, sampling_rate=500)
    # P波オンセットとT波オフセットの取得
    print(rpeaks)
    peak_method = "cwt"
    waves_peak_df, waves_peak_dict = nk.ecg_delineate(
        ecg_cleaned, rpeaks, sampling_rate=500, method="cwt", show=False
    )
    print(waves_peak_df)
    print(waves_peak_dict)
    extracted_data = {}
    for key, values in waves_peak_dict.items():
        for value in values:
            if value > 400 and value < 800:
                extracted_data[key] = value
    extracted_data["ECG_R_Peaks"] = 600
    # 各カラムの先頭要素を抽出
    for key, value in extracted_data.items():
        print(key, value)
        if pd.isna(value):  # int型の場合
            extracted_data[key] = {"Index": None, "Amplitude": None}
        else:
            print(value)
            extracted_data[key] = {
                "Index": value,
                "Amplitude": ecg_cleaned[value],
            }

    # 更新された辞書を確認
    print(extracted_data)
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

    if peak_method == "cwt":
        ecg_p_offsets = waves_peak_dict.get("ECG_P_Offsets")
        if ecg_p_offsets is not None:
            valid_ecg_p_offsets = np.array(ecg_p_offsets)[
                ~np.isnan(ecg_p_offsets)
            ].astype(int)
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
            valid_ecg_q_peaks = np.array(ecg_q_peaks)[~np.isnan(ecg_q_peaks)].astype(
                int
            )
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
    ax.plot(rpeaks, ecg_signal[rpeaks], "ro", label="R peaks", alpha=0.7)

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

    if peak_method == "cwt":
        ecg_t_onsets = waves_peak_dict.get("ECG_T_Onsets")
        if ecg_t_onsets is not None:
            valid_ecg_t_onsets = np.array(ecg_t_onsets)[~np.isnan(ecg_t_onsets)].astype(
                int
            )
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
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

    # ecg_signal=nk.ecg_clean(ecg_signal,sampling_rate=rate,method='neurokit')
    # df_12ch_cleaned[column]=ecg_signal

    return extracted_data


data_type_list = ["xo", "reconx"]
xo_dic = {}
reconx_dic = {}
for data_type in data_type_list:
    peak_dic = {}
    extracted_peaks = process_and_detect_peaks(data_type)

    peak_dic["p_peak"] = extracted_peaks["ECG_P_Peaks"]["Amplitude"]
    peak_dic["p_duration"] = (
        extracted_peaks["ECG_P_Offsets"]["Index"]
        - extracted_peaks["ECG_P_Onsets"]["Index"]
    )
    peak_dic["r_peak"] = extracted_peaks["ECG_R_Peaks"]["Amplitude"]
    peak_dic["qrs_duration"] = (
        extracted_peaks["ECG_S_Peaks"]["Index"]
        - extracted_peaks["ECG_Q_Peaks"]["Index"]
    )
    peak_dic["t_peak"] = extracted_peaks["ECG_P_Peaks"]["Amplitude"]
    peak_dic["t_duration"] = (
        extracted_peaks["ECG_T_Offsets"]["Index"]
        - extracted_peaks["ECG_T_Onsets"]["Index"]
    )
    peak_dic["pq_duration"] = (
        extracted_peaks["ECG_Q_Peaks"]["Index"]
        - extracted_peaks["ECG_P_Onsets"]["Index"]
    )
    peak_dic["qr_duration"] = (
        extracted_peaks["ECG_R_Offsets"]["Index"]
        - extracted_peaks["ECG_Q_Peaks"]["Index"]
    )
    print(peak_dic)
    if data_type == "xo":
        xo_dic = peak_dic
    else:
        reconx_dic = peak_dic

print(xo_dic, reconx_dic)
