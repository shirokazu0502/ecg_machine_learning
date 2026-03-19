import os
import time
import argparse
from Make_dataset_0120_16ch_synchro_per_heart import (
    ecg_clean_df_16ch,
    linear_interpolation_resample_All,
    peak_sc,
    peak_sc_16ch,
    peak_sc_plot,
    validate_integer_input,
    CSVReader_16ch,
    CSVReader_12ch,
    ecg_clean_df_12ch,
    PTwave_search3,
    HeartbeatCutter_prt,
    plot_and_select_all_points,
    write_text_file,
    create_directory_if_not_exists,
    calculate_moving_average,
    find_qrs_boundary,
    create_15ch_variations,
)
from glob import glob
from scipy import signal
from scipy.ndimage import uniform_filter1d, median_filter
import numpy as np
import pandas as pd
import neurokit2 as nk
from matplotlib import pyplot as plt
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

RATE = 500.00
RATE_12ch = 500.00
RATE_16ch = 122.06
TARGET_CHANNEL_12CH = "A2"
TARGET_CHANNEL_16ch = "ch_1"


def output_csv(file_path, file_name, data):
    dt = 1.0 / RATE
    time_tmp = np.arange(len(data)) * dt
    # time_data=pd.DataFrame(time,column="Time")
    time = pd.DataFrame()
    time["Time"] = time_tmp
    print(time)
    data = data.reset_index(drop=True)
    data_out = pd.concat([time, data], axis=1)
    print(data_out)
    data_out.to_csv(file_path + "/" + file_name, index=None)


def output_csv_eles(
    file_path,
    file_name,
    p_onset,
    t_offset,
    p_offset,
    t_onset,
    r_peak,
    p_peak,
    q_peak,
    s_peak,
    t_peak,
):
    data = {
        "p_onset": [p_onset],
        "t_offset": [t_offset],
        "p_offset": [p_offset],
        "t_onset": [t_onset],
        "r_peak": [r_peak],
        "p_peak": [p_peak],
        "q_peak": [q_peak],
        "s_peak": [s_peak],
        "t_peak": [t_peak],
    }
    data_out = pd.DataFrame(data)
    data_out.to_csv(file_path + "/" + file_name, index=None)


def normalize_data(df):
    """
    Finds the global maximum absolute value across all data and scales
    the entire dataframe to the [0, 1] range.
    """
    global_max_val = df.abs().max().max()
    if global_max_val > 0:
        normalized_df = 0.5 * (df / global_max_val) + 0.5
        return normalized_df
    return df  # Return original df if max is 0 to avoid division by zero


def clean_ecg_signal(
    ecg_signal,
    sampling_rate=500,
    bandpass_lowcut=0.05,
    bandpass_highcut=100,
    notch_freq=50,
    notch_Q=30,
):
    """
    ECG信号をバンドパスフィルタとノッチフィルタで前処理する関数

    Parameters:
    ----------
    ecg_signal : array-like
        入力する生のECG信号
    sampling_rate : int, optional
        サンプリング周波数（Hz）
    bandpass_lowcut : float, optional
        バンドパスフィルタの下限周波数（Hz）
    bandpass_highcut : float, optional
        バンドパスフィルタの上限周波数（Hz）
    notch_freq : float, optional
        ノッチフィルタの中心周波数（Hz）
    notch_Q : float, optional
        ノッチフィルタのQ値（フィルタの鋭さ）

    Returns:
    -------
    filtered_ecg : array-like
        フィルタ後のECG信号
    """

    # --- バンドパスフィルタ ---
    nyquist = sampling_rate / 2
    low = bandpass_lowcut / nyquist
    high = bandpass_highcut / nyquist

    b_bandpass, a_bandpass = signal.butter(N=4, Wn=[low, high], btype="band")
    filtered_ecg = signal.filtfilt(b_bandpass, a_bandpass, ecg_signal)

    # --- ノッチフィルタ ---
    w0 = notch_freq / nyquist
    b_notch, a_notch = signal.iirnotch(w0=w0, Q=notch_Q)
    filtered_ecg = signal.filtfilt(b_notch, a_notch, filtered_ecg)

    return filtered_ecg


def is_ascending(lst):
    """
    Checks if a list of points is strictly ascending and contains no invalid values.
    """
    if any(p is None or pd.isna(p) for p in lst):
        return False
    for i in range(len(lst) - 1):
        if lst[i] >= lst[i + 1]:
            return False
    return True


def find_p_element(arr, target):
    """
    Finds the element in arr that is less than or equal to target and is closest to it.
    Used for finding P-wave and Q-wave points before the R-peak.
    """
    # 1. 基本的なガード句
    if arr is None:
        return None

    # 2. 強制的に float 型の NumPy 配列に変換する
    # これにより、[int64, nan] が混在していても [float, nan] として統一され、エラーを防げます
    arr_np = np.array(arr, dtype=float)

    # 配列が空の場合は終了
    if arr_np.size == 0:
        return None

    # 3. NaN を除去 (float型なので安全に除去可能)
    arr_clean = arr_np[~np.isnan(arr_np)]

    # NaN除去後に空になった場合のガード
    if arr_clean.size == 0:
        return None

    # 4. 比較用に int 型にキャスト (ECGのインデックスは整数であるため)
    arr_int = arr_clean.astype(int)

    # 5. target 以下の候補を抽出
    candidates = arr_int[arr_int <= target]

    # 候補がない場合
    if candidates.size == 0:
        return None

    # 6. 最も近い値を返す
    # 時系列データ(昇順)であれば最後の要素 [-1] が最も target に近くなります。
    # 念のため max() を使うと順序関係なく最大値（最も近い値）が取れます。
    return candidates.max()


def find_t_element(arr, target):
    """
    Finds the element in arr that is greater than or equal to target and is closest to it.
    Used for finding S-wave and T-wave points after the R-peak.
    """
    """
    Finds the element in arr that is less than or equal to target and is closest to it.
    Used for finding P-wave and Q-wave points before the R-peak.
    """
    # 1. 基本的なガード句
    if arr is None:
        return None

    # 2. 強制的に float 型の NumPy 配列に変換する
    # これにより、[int64, nan] が混在していても [float, nan] として統一され、エラーを防げます
    arr_np = np.array(arr, dtype=float)

    # 配列が空の場合は終了
    if arr_np.size == 0:
        return None

    # 3. NaN を除去 (float型なので安全に除去可能)
    arr_clean = arr_np[~np.isnan(arr_np)]

    # NaN除去後に空になった場合のガード
    if arr_clean.size == 0:
        return None

    # 4. 比較用に int 型にキャスト (ECGのインデックスは整数であるため)
    arr_int = arr_clean.astype(int)

    # 5. target 以下の候補を抽出
    candidates = arr_int[arr_int >= target]

    # 候補がない場合
    if candidates.size == 0:
        return None

    # 6. 最も近い値を返す
    # 時系列データ(昇順)であれば最初の要素 [0] が最も target に近くなります。
    # 念のため min() を使うと順序関係なく最小値（最も近い値）が取れます。
    return candidates.min()


def find_valid_pqrst_set(waves_peak, rpeak, sampling_rate, time_length):
    """
    Finds and validates a complete set of 9 PQRST points for a single rpeak
    from the neurokit waves_peak dictionary.
    """
    # 1. Get candidate points using helper functions
    peak_dic = {}
    peak_dic["p_onset"] = find_p_element(waves_peak.get("ECG_P_Onsets"), rpeak)
    peak_dic["p_peak"] = find_p_element(waves_peak.get("ECG_P_Peaks"), rpeak)
    peak_dic["p_offset"] = find_p_element(waves_peak.get("ECG_P_Offsets"), rpeak)
    peak_dic["q_peak"] = find_p_element(waves_peak.get("ECG_Q_Peaks"), rpeak)
    peak_dic["s_peak"] = find_t_element(waves_peak.get("ECG_S_Peaks"), rpeak)
    peak_dic["t_onset"] = find_t_element(waves_peak.get("ECG_T_Onsets"), rpeak)
    peak_dic["t_peak"] = find_t_element(waves_peak.get("ECG_T_Peaks"), rpeak)
    peak_dic["t_offset"] = find_t_element(waves_peak.get("ECG_T_Offsets"), rpeak)

    if None in peak_dic.values():
        return None
    # 2. Assemble the full set
    start_idx = rpeak - 150
    print(f"p_onset: {peak_dic['p_onset']}")
    print(f"type of p_onset: {type(peak_dic['p_onset'])}")
    peak_dic["p_onset"] = peak_dic["p_onset"] - start_idx
    peak_dic["p_peak"] = peak_dic["p_peak"] - start_idx
    peak_dic["p_offset"] = peak_dic["p_offset"] - start_idx
    peak_dic["q_peak"] = peak_dic["q_peak"] - start_idx
    peak_dic["s_peak"] = peak_dic["s_peak"] - start_idx
    peak_dic["t_onset"] = peak_dic["t_onset"] - start_idx
    peak_dic["t_peak"] = peak_dic["t_peak"] - start_idx
    peak_dic["t_offset"] = peak_dic["t_offset"] - start_idx

    pqrst_points = [
        peak_dic["p_onset"],
        peak_dic["p_peak"],
        peak_dic["p_offset"],
        peak_dic["q_peak"],
        peak_dic["s_peak"],
        peak_dic["t_onset"],
        peak_dic["t_peak"],
        peak_dic["t_offset"],
    ]
    print(f"pqrst_points: {pqrst_points}")

    # 3. Validate the set
    # 3a. Check for chronological order (is_ascending handles None/NaN)
    if not is_ascending(pqrst_points):
        return None

    # もし一つでも0未満の値、もしくは400以上の値があれば無効
    if any(p < 0 or p >= 400 for p in pqrst_points):
        return None

    # R波ピークを挿入
    pqrst_points.insert(4, rpeak)  # R-peak is always at index 150 in the cut segment

    return pqrst_points


def main(args):
    dir_path = args.raw_datas_dir
    csv_reader_16ch = CSVReader_16ch(dir_path)
    print(dir_path)
    df_16ch = csv_reader_16ch.process_files()
    print(df_16ch)
    # cols = df_15ch.columns
    # df_15ch = pd.DataFrame()
    # for col in cols:
    #     df_15ch[col] = df_15ch[col] - df_15ch["ch_16"]
    # df_15ch = df_15ch.drop(columns=["ch_16"])
    df_16ch_cleaned = ecg_clean_df_16ch(df_16ch=df_16ch.copy(), rate=RATE_16ch)
    df_resample_16ch = linear_interpolation_resample_All(
        df=df_16ch_cleaned.copy(), sampling_rate=RATE_16ch, new_sampling_rate=RATE
    )
    df_16ch_cleaned = df_resample_16ch.copy()
    csv_reader_12ch = CSVReader_12ch(dir_path)
    df_12ch = csv_reader_12ch.process_files()
    df_12ch_cleaned = ecg_clean_df_12ch(df_12ch)
    if args.reverse == "off":
        sc_16ch = peak_sc_16ch(
            df_16ch_cleaned.copy(), RATE=RATE, TARGET=TARGET_CHANNEL_16ch
        )
        peak_sc_plot(df_16ch_cleaned.copy(), RATE=RATE, TARGET=TARGET_CHANNEL_16ch)
    else:
        df_16ch_reverse = df_16ch_cleaned.copy()
        df_16ch_reverse[TARGET_CHANNEL_16ch] = (-1) * df_16ch_cleaned.copy()[
            TARGET_CHANNEL_16ch
        ]
        df_16ch_cleaned = df_16ch_reverse.copy()
        sc_16ch = peak_sc_16ch(
            df_16ch_reverse.copy(), RATE=RATE, TARGET=TARGET_CHANNEL_16ch
        )
        peak_sc_plot(df_16ch_reverse.copy(), RATE=RATE, TARGET=TARGET_CHANNEL_16ch)
        # 先頭は削除
        sc_16ch = sc_16ch.drop(0)
    sc_12ch = peak_sc(df_12ch.copy(), RATE=RATE_12ch, TARGET=TARGET_CHANNEL_12CH)
    # print(sc_12ch)
    # input()
    peak_sc_plot(df_12ch.copy(), RATE=RATE_12ch, TARGET=TARGET_CHANNEL_12CH)

    # --- START OF PRE-COMPUTATION FOR AUTO-DELINEATION ---
    all_prt_points = None
    if args.manual_setting == "3":
        print(
            "--- Mode 3: Performing one-time automatic PQRST delineation on the full signal ---"
        )
        try:
            full_ecg_signal = df_12ch_cleaned[TARGET_CHANNEL_12CH].to_numpy()

            # 1. Detect R-peaks using the 'neurokit' method
            _, rpeaks_info = nk.ecg_peaks(
                full_ecg_signal, sampling_rate=RATE, method="neurokit"
            )

            # 2. Perform delineation using the 'peak' method to get all potential wave points
            _, waves_peak = nk.ecg_delineate(
                full_ecg_signal,
                rpeaks_info,
                sampling_rate=RATE,
                method="cwt",
            )

            all_prt_points = []

            # 3. For each R-peak, find and validate the complete PQRST set
            for rpeak in rpeaks_info["ECG_R_Peaks"]:
                valid_pqrst_set = find_valid_pqrst_set(
                    waves_peak, rpeak, RATE, args.time_range
                )
                print(f"valid_pqrst_set for rpeak {rpeak}: {valid_pqrst_set}")
                if valid_pqrst_set is not None:
                    all_prt_points.append(valid_pqrst_set)

            print(
                f"  > Successfully validated and retained {len(all_prt_points)} heartbeats."
            )

        except Exception as e:
            print(
                f"  > CRITICAL: One-time automatic delineation failed: {e}. Cannot proceed in mode 3."
            )
            # Set to None so the loop knows to skip
            all_prt_points = None
    # --- END OF PRE-COMPUTATION ---

    # --- START OF HEARTBEAT PROCESSING LOGIC ---
    if args.enable_averaging:
        # --- Averaging Mode ---
        print(
            f"--- Running in Averaging Mode (averaging {args.average_beats_num} beats) ---"
        )
        num_beats_total_16ch = len(sc_16ch[0])
        grouped_dataset_idx = 0
        print(args.average_beats_num)
        for group_start in range(0, num_beats_total_16ch, args.average_beats_num):
            group_end = group_start + args.average_beats_num
            if group_end > num_beats_total_16ch:
                print(
                    f"  > Not enough beats for a group of {group_start}group. Ending processing."
                )
                break

            print(
                f"\n--- Processing group {grouped_dataset_idx+1} (heartbeats {group_start+1}-{group_end}) ---"
            )

            # 1. Cut all 16ch heartbeats in the group
            beat_segments_16ch = []
            for i in range(group_start, group_end):
                center_16ch_idx = int(sc_16ch[0].iloc[i] * RATE)
                start_16ch_idx_loop = int(center_16ch_idx - 0.3 * RATE)
                end_16ch_idx_loop = int(center_16ch_idx + 0.5 * RATE)
                heartbeat_16ch_segment = df_16ch_cleaned.iloc[
                    start_16ch_idx_loop:end_16ch_idx_loop
                ]
                if len(heartbeat_16ch_segment) == 400:
                    beat_segments_16ch.append(heartbeat_16ch_segment.values)

            if not beat_segments_16ch:
                print("  > No valid 16ch segments found in this group. Skipping.")
                continue

            # 2. Average the 16ch waveforms
            stacked_beats_16ch = np.stack(beat_segments_16ch, axis=0)
            averaged_beat_16ch_values = np.mean(stacked_beats_16ch, axis=0)
            averaged_heartbeat_16ch = pd.DataFrame(
                averaged_beat_16ch_values, columns=df_16ch_cleaned.columns
            )

            # 3. Select the representative 12ch heartbeat (the first one in the group)
            representative_beat_index = grouped_dataset_idx
            if representative_beat_index >= len(sc_12ch[0]):
                print(
                    "  > Not enough 12ch R-peaks to match this group. Ending processing."
                )
                break
            representative_sc12 = sc_12ch[0].iloc[representative_beat_index]
            print(f"Representative beat index: {representative_beat_index}")
            # time.sleep(5)
            center_12ch_idx = int(representative_sc12 * RATE)
            start_12ch_idx = int(center_12ch_idx - 0.3 * RATE)
            end_12ch_idx = int(center_12ch_idx + 0.5 * RATE)
            representative_heartbeat_12ch = df_12ch_cleaned.iloc[
                start_12ch_idx:end_12ch_idx
            ]

            # 4. Merge and proceed
            heartbeat_16ch = averaged_heartbeat_16ch.reset_index(drop=True)
            heartbeat_12ch = representative_heartbeat_12ch.reset_index(drop=True)

            # --- PQRST Point Determination, Correction, and Saving (for the group) ---
            (
                p_onset,
                p_peak,
                p_offset,
                q_peak,
                r_peak,
                s_peak,
                t_onset,
                t_peak,
                t_offset,
            ) = (None,) * 9
            skip_heartbeat = False
            # (PQRST selection logic is applied to the representative 12-lead beat)
            if args.manual_setting == "1":
                window = int(0.4 * RATE)
                print(
                    f"  > Mode 1: Please select PQRST points for group {grouped_dataset_idx+1} on the plot..."
                )
                try:
                    absolute_points = plot_and_select_all_points(
                        df_12ch_cleaned,
                        rpeak=center_12ch_idx,
                        window=window,
                    )
                    (
                        p_onset_abs,
                        p_peak_abs,
                        p_offset_abs,
                        q_peak_abs,
                        r_peak,
                        s_peak_abs,
                        t_onset_abs,
                        t_peak_abs,
                        t_offset_abs,
                    ) = absolute_points
                    (
                        p_onset,
                        p_peak,
                        p_offset,
                        q_peak,
                        s_peak,
                        t_onset,
                        t_peak,
                        t_offset,
                    ) = [
                        p - start_12ch_idx
                        for p in [
                            p_onset_abs,
                            p_peak_abs,
                            p_offset_abs,
                            q_peak_abs,
                            s_peak_abs,
                            t_onset_abs,
                            t_peak_abs,
                            t_offset_abs,
                        ]
                    ]
                except Exception as e:
                    print(
                        f"  > Warning: Could not manually select points. Skipping group. Error: {e}"
                    )
                    skip_heartbeat = True
            elif args.manual_setting == "2":
                file_path = os.path.join(args.dataset_output_path, args.output_filepath)
                pt_filepath = os.path.join(
                    file_path, f"ponset_toffset_{str(grouped_dataset_idx).zfill(3)}.csv"
                )
                if os.path.exists(pt_filepath):
                    try:
                        df_pt = pd.read_csv(pt_filepath)
                        (
                            p_onset,
                            p_peak,
                            p_offset,
                            q_peak,
                            r_peak,
                            s_peak,
                            t_onset,
                            t_peak,
                            t_offset,
                        ) = (
                            df_pt["p_onset"][0],
                            df_pt["p_peak"][0],
                            df_pt["p_offset"][0],
                            df_pt["q_peak"][0],
                            df_pt["r_peak"][0],
                            df_pt["s_peak"][0],
                            df_pt["t_onset"][0],
                            df_pt["t_peak"][0],
                            df_pt["t_offset"][0],
                        )
                    except Exception as e:
                        skip_heartbeat = True
                else:
                    skip_heartbeat = True
            elif args.manual_setting == "3":
                print(f"all_prt_points: {all_prt_points}")
                print(f"representative_beat_index: {representative_beat_index}")
                if all_prt_points is not None and representative_beat_index < len(
                    all_prt_points
                ):
                    points = all_prt_points[representative_beat_index]
                    if any(pd.isna(p) for p in points):
                        print(
                            f"  > Warning: Incomplete (NaN) delineation for representative heartbeat {representative_beat_index}. Skipping group."
                        )
                        skip_heartbeat = True
                    else:
                        (
                            p_onset,
                            p_peak,
                            p_offset,
                            q_peak,
                            r_peak,
                            s_peak,
                            t_onset,
                            t_peak,
                            t_offset,
                        ) = points
                else:
                    print(
                        f"  > Warning: No pre-computed delineation available for representative heartbeat {representative_beat_index}. Skipping group."
                    )
                    skip_heartbeat = True

            if not skip_heartbeat:
                # --- Common Processing for the group ---
                file_path = os.path.join(args.dataset_output_path, args.output_filepath)
                create_directory_if_not_exists(file_path)
                # 12chと16chそれぞれで正規化処理
                heartbeat_16ch_normalized = normalize_data(heartbeat_16ch)
                heartbeat_12ch_normalized = normalize_data(heartbeat_12ch)
                merge_df = pd.concat(
                    [heartbeat_16ch_normalized, heartbeat_12ch_normalized], axis=1
                )
                data = merge_df.copy()
                for j, column in enumerate(data.columns):
                    # フィルタかける
                    data[column] = nk.ecg_clean(
                        data[column], sampling_rate=500, method="neurokit"
                    )
                    # pt_extendを実施
                    # インデックスp_onsetの値を取得
                    print(f"p_onset: {p_onset}, t_offset: {t_offset}")
                    value_at_p_onset = data.iloc[p_onset, j]
                    # インデックスt_offsetの値を取得
                    value_at_t_offset = data.iloc[t_offset, j]

                    # p_onsetより前の値を置き換える
                    data.iloc[:p_onset, j] = value_at_p_onset
                    # t_offsetより後の値を置き換える
                    data.iloc[t_offset:, j] = value_at_t_offset
                    signal = data[column].copy().values

                    # 0. PQ区間のノイズレベルを推定
                    pq_region = signal[p_offset:q_peak]
                    print("pq_region", pq_region)
                    noise_level = (
                        np.std(np.diff(pq_region)) if len(pq_region) > 1 else 0.1
                    )

                    # 1. QRS波の開始点と終了点を決定
                    # 'onset'の呼び出し (変更なし)
                    qrs_onset = find_qrs_boundary(
                        "onset", signal, p_offset, q_peak, noise_level
                    )

                    # 'offset'の呼び出し (r_peakの代わりにs_peakを渡す)
                    qrs_offset = find_qrs_boundary(
                        "offset", signal, s_peak, t_onset, noise_level
                    )

                    # 2. 基線として使用する4つの領域のデータ点を準備
                    print(p_onset, t_offset)
                    print(points)
                    indices_1 = np.arange(0, p_onset)
                    values_1 = signal[:p_onset]
                    # indices_2 = np.arange(p_offset, qrs_onset)
                    # values_2 = signal[p_offset:qrs_onset]
                    # indices_3 = np.arange(qrs_offset, t_onset)
                    # values_3 = signal[qrs_offset:t_onset]
                    indices_4 = np.arange(t_offset, len(signal))
                    values_4 = signal[t_offset:]

                    # 2つの領域をすべて結合
                    baseline_indices = np.concatenate([indices_1, indices_4])
                    baseline_values = np.concatenate([values_1, values_4])
                    print(len(signal))

                    # 3. 多項式フィッティングを実行
                    if len(baseline_indices) > 2 and len(baseline_values) == len(
                        baseline_indices
                    ):
                        poly_degree = 6
                        print("baseline_indices", len(baseline_indices))
                        print("baseline_values", len(baseline_values))
                        print("len(poly_degree)", poly_degree)
                        coeffs = np.polyfit(
                            baseline_indices, baseline_values, poly_degree
                        )
                        x_full = np.arange(len(signal))
                        baseline = np.polyval(coeffs, x_full)
                        corrected_signal = signal - baseline
                    else:
                        corrected_signal = signal

                    # 4. 結果をデータフレームに格納
                    data[column] = corrected_signal

                print(f"  > Saving dataset for group {grouped_dataset_idx + 1}...")
                output_csv(
                    file_name=f"dataset_{str(grouped_dataset_idx).zfill(3)}.csv",
                    file_path=file_path,
                    data=data,
                )
                output_csv_eles(
                    file_name=f"ponset_toffset_{str(grouped_dataset_idx).zfill(3)}.csv",
                    file_path=file_path,
                    p_onset=p_onset,
                    t_offset=t_offset,
                    p_offset=p_offset,
                    t_onset=t_onset,
                    r_peak=r_peak,
                    p_peak=p_peak,
                    q_peak=q_peak,
                    s_peak=s_peak,
                    t_peak=t_peak,
                )

            grouped_dataset_idx += 1
            if len(sc_12ch[0]) <= grouped_dataset_idx:
                print(
                    "  > No more 12ch beats available for selection. Ending processing."
                )
                break

    else:
        # --- Single-Beat Mode (Original Logic) ---
        print("--- Running in Single-Beat Mode ---")
        for i, (sc_16ch_peak, sc12) in enumerate(zip(sc_16ch[0], sc_12ch[0])):
            center_16ch_idx = int(sc_16ch_peak * RATE)
            start_16ch_idx = int(center_16ch_idx - 0.3 * RATE)
            end_16ch_idx = int(center_16ch_idx + 0.5 * RATE)
            heartbeat_16ch = df_16ch_cleaned.iloc[start_16ch_idx:end_16ch_idx]

            center_12ch_idx = int(sc12 * RATE)
            start_12ch_idx = int(center_12ch_idx - 0.3 * RATE)
            end_12ch_idx = int(center_12ch_idx + 0.5 * RATE)
            heartbeat_12ch = df_12ch_cleaned.iloc[start_12ch_idx:end_12ch_idx]

            heartbeat_16ch = heartbeat_16ch.reset_index(drop=True)
            heartbeat_12ch = heartbeat_12ch.reset_index(drop=True)

            p_onset, p_peak, p_offset, q_peak, s_peak, t_onset, t_peak, t_offset = (
                None,
            ) * 8
            skip_heartbeat = False

            if args.manual_setting == "1":
                window = int(0.4 * RATE)
                print(
                    f"  > Mode 1: Please select PQRST points for heartbeat {i+1} on the plot..."
                )
                try:
                    absolute_points = plot_and_select_all_points(
                        df_12ch_cleaned,
                        rpeak=center_12ch_idx,
                        window=window,
                    )
                    print(absolute_points)
                    (
                        p_onset_abs,
                        p_peak_abs,
                        p_offset_abs,
                        q_peak_abs,
                        r_peak,
                        s_peak_abs,
                        t_onset_abs,
                        t_peak_abs,
                        t_offset_abs,
                    ) = absolute_points
                    (
                        p_onset,
                        p_peak,
                        p_offset,
                        q_peak,
                        s_peak,
                        t_onset,
                        t_peak,
                        t_offset,
                    ) = [
                        p - start_12ch_idx
                        for p in [
                            p_onset_abs,
                            p_peak_abs,
                            p_offset_abs,
                            q_peak_abs,
                            s_peak_abs,
                            t_onset_abs,
                            t_peak_abs,
                            t_offset_abs,
                        ]
                    ]
                except Exception as e:
                    print(
                        f"  > Warning: Could not manually select points. Skipping heartbeat. Error: {e}"
                    )
                    skip_heartbeat = True
            elif args.manual_setting == "2":
                file_path = os.path.join(args.dataset_output_path, args.output_filepath)
                pt_filepath = os.path.join(
                    file_path, f"ponset_toffset_{str(i).zfill(3)}.csv"
                )
                if os.path.exists(pt_filepath):
                    try:
                        df_pt = pd.read_csv(pt_filepath)
                        (
                            p_onset,
                            p_peak,
                            p_offset,
                            q_peak,
                            r_peak,
                            s_peak,
                            t_onset,
                            t_peak,
                            t_offset,
                        ) = (
                            df_pt["p_onset"][0],
                            df_pt["p_peak"][0],
                            df_pt["p_offset"][0],
                            df_pt["q_peak"][0],
                            df_pt["r_peak"][0],
                            df_pt["s_peak"][0],
                            df_pt["t_onset"][0],
                            df_pt["t_peak"][0],
                            df_pt["t_offset"][0],
                        )
                    except Exception as e:
                        skip_heartbeat = True
                else:
                    skip_heartbeat = True
            elif args.manual_setting == "3":
                if all_prt_points is not None and i < len(all_prt_points):
                    points = all_prt_points[i]
                    # Check for NaN values from neurokit which indicate failed delineation for a point
                    if any(pd.isna(p) for p in points):
                        print(
                            f"  > Warning: Incomplete (NaN) delineation for heartbeat {i}. Skipping."
                        )
                        skip_heartbeat = True
                    else:
                        (
                            p_onset_abs,
                            p_peak_abs,
                            p_offset_abs,
                            q_peak_abs,
                            r_peak_abs,
                            s_peak_abs,
                            t_onset_abs,
                            t_peak_abs,
                            t_offset_abs,
                        ) = points
                        # Convert absolute indices to relative indices for the current heartbeat window
                        (
                            p_onset,
                            p_peak,
                            p_offset,
                            q_peak,
                            r_peak,
                            s_peak,
                            t_onset,
                            t_peak,
                            t_offset,
                        ) = [int(p) - start_12ch_idx for p in points]
                else:
                    print(
                        f"  > Warning: No pre-computed delineation available for heartbeat {i}. Skipping."
                    )
                    skip_heartbeat = True

            else:
                skip_heartbeat = True

            if not skip_heartbeat:
                # 12chと16chそれぞれで正規化処理
                heartbeat_16ch_normalized = normalize_data(heartbeat_16ch)
                heartbeat_12ch_normalized = normalize_data(heartbeat_12ch)
                merge_df = pd.concat(
                    [heartbeat_16ch_normalized, heartbeat_12ch_normalized], axis=1
                )
                data = merge_df.copy()
                file_path = os.path.join(args.dataset_output_path, args.output_filepath)
                create_directory_if_not_exists(file_path)
                for column in data.columns:
                    data[column] = nk.ecg_clean(
                        data[column], sampling_rate=RATE, method="neurokit"
                    )
                data = normalize_data(data)
                for j, column in enumerate(data.columns):
                    # フィルタかける
                    data[column] = nk.ecg_clean(
                        data[column], sampling_rate=500, method="neurokit"
                    )
                    # pt_extendを実施
                    # インデックスp_onsetの値を取得
                    value_at_p_onset = data.iloc[p_onset, j]
                    # インデックスt_offsetの値を取得
                    value_at_t_offset = data.iloc[t_offset, j]

                    # p_onsetより前の値を置き換える
                    data.iloc[:p_onset, j] = value_at_p_onset
                    # t_offsetより後の値を置き換える
                    data.iloc[t_offset:, j] = value_at_t_offset
                    signal = data[column].copy().values

                    # 0. PQ区間のノイズレベルを推定
                    pq_region = signal[p_offset:q_peak]
                    print("pq_region", pq_region)
                    noise_level = (
                        np.std(np.diff(pq_region)) if len(pq_region) > 1 else 0.1
                    )

                    # 1. QRS波の開始点と終了点を決定
                    # 'onset'の呼び出し (変更なし)
                    qrs_onset = find_qrs_boundary(
                        "onset", signal, p_offset, q_peak, noise_level
                    )

                    # 'offset'の呼び出し (r_peakの代わりにs_peakを渡す)
                    qrs_offset = find_qrs_boundary(
                        "offset", signal, s_peak, t_onset, noise_level
                    )

                    # 2. 基線として使用する4つの領域のデータ点を準備
                    print(p_onset, t_offset)
                    indices_1 = np.arange(0, p_onset)
                    values_1 = signal[:p_onset]
                    # indices_2 = np.arange(p_offset, qrs_onset)
                    # values_2 = signal[p_offset:qrs_onset]
                    # indices_3 = np.arange(qrs_offset, t_onset)
                    # values_3 = signal[qrs_offset:t_onset]
                    indices_4 = np.arange(t_offset, len(signal))
                    values_4 = signal[t_offset:]

                    # 2つの領域をすべて結合
                    baseline_indices = np.concatenate([indices_1, indices_4])
                    baseline_values = np.concatenate([values_1, values_4])
                    print(len(signal))

                    # 3. 多項式フィッティングを実行
                    if len(baseline_indices) > 2:
                        poly_degree = 6
                        print("baseline_indices", len(baseline_indices))
                        print("baseline_values", len(baseline_values))
                        print("len(poly_degree)", poly_degree)
                        coeffs = np.polyfit(
                            baseline_indices, baseline_values, poly_degree
                        )
                        x_full = np.arange(len(signal))
                        baseline = np.polyval(coeffs, x_full)
                        corrected_signal = signal - baseline
                    else:
                        corrected_signal = signal

                    # 4. 結果をデータフレームに格納
                    data[column] = corrected_signal

                output_csv(
                    file_name=f"dataset_{str(i).zfill(3)}.csv",
                    file_path=file_path,
                    data=data,
                )
                output_csv_eles(
                    file_name=f"ponset_toffset_{str(i).zfill(3)}.csv",
                    file_path=file_path,
                    p_onset=p_onset,
                    t_offset=t_offset,
                    p_offset=p_offset,
                    t_onset=t_onset,
                    r_peak=r_peak,
                    p_peak=p_peak,
                    q_peak=q_peak,
                    s_peak=s_peak,
                    t_peak=t_peak,
                )
    # --- END OF HEARTBEAT PROCESSING LOGIC ---
    # 移動平均を計算
    # 処理するCSVファイルの一覧を取得
    data_paths = sorted(
        glob(args.dataset_output_path + "/" + args.output_filepath + "/dataset_*.csv")
    )
    # pt_array_paths = sorted(
    #     glob(
    #         args.dataset_output_path
    #         + "/"
    #         + args.output_filepath
    #         + "/ponset_toffset_*.csv"
    #     )
    # )
    # pt_extend(data_paths, pt_array_paths)
    moving_ave_path = os.path.join(
        args.dataset_output_path, args.output_filepath, "moving_ave_datasets"
    )
    create_directory_if_not_exists(moving_ave_path)
    calculate_moving_average(data_paths, moving_ave_path, group_size=5)

    # 15chバリエーション作成
    create_15ch_variations(args)
    print("終了")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--dir_name", type=str, default='goto_0604/goto_0604_normal2')

    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--date", type=str, default="")
    parser.add_argument("--peak_method", type=str, default="")
    parser.add_argument("--pos", type=str, default="")
    parser.add_argument("--type", type=str, default="")
    parser.add_argument("--dir_name", type=str, default="")
    parser.add_argument("--png_path", type=str, default="")
    parser.add_argument("--output_filepath", type=str, default="")
    # parser.add_argument("--TARGET_CHANNEL_15ch", type=str, default='ch_1')
    parser.add_argument("--TARGET_CHANNEL_12CH", type=str, default="")
    # parser.add_argument("--cut_min_max_range", type=list, default=[0,10])
    parser.add_argument("--cut_min_max_range", type=list, default="")
    parser.add_argument("--time_range", type=float)
    parser.add_argument(
        "--reverse", type=str, default=""
    )  # onだと波形逆さまにしてピーク検出。これはシートセンサを逆向きに貼ったとき
    parser.add_argument("--project_path", type=str, default="")
    parser.add_argument("--raw_datas_dir", type=str, default="")
    parser.add_argument("--raw_datas_os", type=str, default="")
    parser.add_argument("--dataset_made_date", type=str, default="")
    parser.add_argument("--dataset_output_path", type=str, default="")
    parser.add_argument("--test_images_path", type=str, default="")
    parser.add_argument(
        "--manual_setting",
        type=str,
        default="3",
        help="PQRST detection mode: 1=Manual, 2=File, 3=Auto",
    )
    parser.add_argument(
        "--enable_averaging", type=bool, default=True, help="Enable averaging mode."
    )
    parser.add_argument(
        "--average_beats_num",
        type=int,
        default=2,
        help="Number of heartbeats to average.",
    )
    args = parser.parse_args()
    args.name, args.date = select_name_and_date()
    args.peak_method = (
        "cwt"  # neurokitのピーク検出アルゴリズムについてcwtかpeakがある。
    )
    args.pos = "center_4"
    args.type = ""
    args.dir_name = "{}/{}".format(args.name, args.type)
    args.png_path = ""
    args.time_range = 0.8
    args.output_filepath = "{}_{}_{}s/{}".format(
        args.name, args.date, str(args.time_range), args.pos
    )
    args.TARGET_CHANNEL_12CH = "A2"
    args.cut_min_max_range = [1.0, 100.0]
    args.reverse = "on"
    args.type = "{}_{}_{}".format(args.name, args.date, args.pos)
    args.dir_name = "{}/{}".format(args.name, args.type)
    # args.project_path='/home/cs28/share/goto/goto/ecg_project'
    # args.raw_datas_os=RAW_DATA_DIR
    # args.processed_datas_os=args.project_path+'/data/processed'
    # args.processed_datas_os=PROCESSED_DATA_DIR
    args.dataset_made_date = DATASET_MADE_DATE
    args.raw_datas_dir = RAW_DATA_DIR + "/takahashi_test/{}".format(args.dir_name)
    args.dataset_output_path = os.path.join(PROCESSED_DATA_DIR, "for_best_resample")
    main(args)
