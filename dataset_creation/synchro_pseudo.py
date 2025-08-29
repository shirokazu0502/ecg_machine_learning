import argparse
from Make_dataset_0120 import (
    ecg_clean_df_15ch,
    linear_interpolation_resample_All,
    peak_sc,
    peak_sc_15ch,
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
)
from glob import glob
from scipy import signal
from scipy.ndimage import uniform_filter1d, median_filter
import numpy as np
import pandas as pd
import neurokit2 as nk
import time
from config.settings import (
    DATA_DIR,
    BASE_DIR,
    PROCESSED_DATA_DIR,
    OUTPUT_DIR,
    RAW_DATA_DIR,
    TEST_DIR,
    RATE,
    RATE_12CH,
    RATE_15CH,
    TIME,
    DATASET_MADE_DATE,
)
from config.name_dic import select_name_and_date

RATE_12ch = 500.00
RATE_15CH = RATE_15CH
TARGET_CHANNEL_12CH = "A2"
TARGET_CHANNEL_15ch = "ch_1"
reverse = "on"
patient_number = "2"


def PT_wave_search(ecg_all):
    data_list = []
    # p波オンセット、T波オフセット手動設定
    points = plot_and_select_all_points(ecg_all, 150)
    # 患者の場合、波のピークが検出できない、switch求められないから全てをデータセットに
    (
        p_Onset_ele,
        p_Peaks_ele,
        p_Offset_ele,
        q_Peaks_ele,
        rpeak,
        s_Peaks_ele,
        t_Onset_ele,
        t_Peaks_ele,
        t_Offset_ele,
    ) = points
    data_list.append(
        [
            p_Onset_ele,
            rpeak,
            t_Offset_ele,
            p_Offset_ele,
            t_Onset_ele,
            p_Peaks_ele,
            q_Peaks_ele,
            s_Peaks_ele,
            t_Peaks_ele,
        ]
    )
    prt_array = np.array(data_list)
    return prt_array


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
    # もしディレクトリが存在しない場合は作成する
    create_directory_if_not_exists(file_path)
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
    # データとチャネルの次元を取得
    # 1. データ全体の絶対値の最大値を取得
    # .max()を2回適用して全要素から最大値を取得
    global_max_val = df.abs().max().max()

    # 2. 式を適用して正規化
    normalized_df = 0.5 * (df / global_max_val) + 0.5

    return normalized_df


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


def heart_fitting(
    data,
    p_onset,
    r_peak,
    t_offset,
    p_offset,
    t_onset,
    p_peak,
    q_peak,
    s_peak,
    t_peak,
):
    p_onset = p_onset
    r_peak = r_peak
    t_offset = t_offset
    p_offset = p_offset
    t_onset = t_onset
    p_peak = p_peak
    q_peak = q_peak
    s_peak = s_peak
    t_peak = t_peak

    for j, column in enumerate(data.columns):
        data[column] = nk.ecg_clean(data[column], sampling_rate=RATE, method="neurokit")

        # インデックスp_onsetの値を取得
        print("p_onset", p_onset)
        print("t_offset", t_offset)
        value_at_p_onset = data.iloc[p_onset, j]
        # インデックスt_offsetの値を取得
        value_at_t_offset = data.iloc[t_offset, j]
        print(value_at_p_onset, value_at_t_offset)

        # p_onsetより前の値を置き換える
        data.iloc[:p_onset, j] = float(value_at_p_onset)
        # t_offsetより後の値を置き換える
        data.iloc[t_offset:, j] = float(value_at_t_offset)
        signal = data[column].copy().values

        # 0. PQ区間のノイズレベルを推定
        pq_region = signal[p_offset:q_peak]
        print("pq_region", pq_region)
        noise_level = np.std(np.diff(pq_region)) if len(pq_region) > 1 else 0.1

        # 1. QRS波の開始点と終了点を決定
        # 'onset'の呼び出し (変更なし)
        qrs_onset = find_qrs_boundary("onset", signal, p_offset, q_peak, noise_level)

        # 'offset'の呼び出し (r_peakの代わりにs_peakを渡す)
        qrs_offset = find_qrs_boundary("offset", signal, s_peak, t_onset, noise_level)

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
            coeffs = np.polyfit(baseline_indices, baseline_values, poly_degree)
            x_full = np.arange(len(signal))
            baseline = np.polyval(coeffs, x_full)
            corrected_signal = signal - baseline
        else:
            corrected_signal = signal
        # # # グラフの作成
        # plt.figure(figsize=(12, 8))

        # # 1. 元の信号をプロット
        # plt.plot(signal, label="Original Signal", color="lightblue", zorder=1)

        # # 2. フィッティングに使った点を散布図でプロット
        # plt.scatter(
        #     baseline_indices,
        #     baseline_values,
        #     label="Baseline Points",
        #     color="red",
        #     s=10,
        #     zorder=3,
        # )

        # # 3. フィットさせた直線をプロット
        # plt.plot(
        #     baseline,
        #     label="Fitted Baseline (deg=1)",
        #     color="orange",
        #     linestyle="--",
        #     zorder=2,
        # )

        # # 4. 補正後の信号をプロット
        # plt.plot(
        #     corrected_signal,
        #     label="Corrected Signal",
        #     color="darkgreen",
        #     linewidth=2,
        #     zorder=4,
        # )

        # # グラフの体裁を整える
        # plt.axhline(0, color="gray", linestyle="-")  # ゼロの基準線
        # plt.title("Baseline Correction Debug Plot")
        # plt.legend()
        # plt.grid(True)
        # plt.show()
        # # corrected_signal = signal
        # 4. 結果をデータフレームに格納
        data[column] = corrected_signal
    return data


def calculate_time_averaged_peaks(
    df_15ch_cleaned, sc_15ch, window_size=20, pre_points=150, post_points=250
):
    """
    Calculate time-averaged data for each channel of 15-channel peaks.

    Parameters:
    df_15ch_cleaned : DataFrame
        Cleaned 15-channel data.
    sc_15ch : list or array
        Detected peak positions in seconds.
    window_size : int, optional
        Number of beats to average (default is 20).
    pre_points : int, optional
        Number of points before the peak to include (default is 150).
    post_points : int, optional
        Number of points after the peak to include (default is 250).

    Returns:
    averaged_data : list of DataFrames
        List of DataFrames, each containing time-averaged data for all channels.
    """
    averaged_data = []
    rate = RATE  # Sampling rate
    sc_15ch = sc_15ch.iloc[:, 0].tolist()

    for i in range(len(sc_15ch) - window_size + 1):
        channel_averages = []

        for channel in df_15ch_cleaned.columns:
            segment_sum = None

            for j in range(window_size):
                peak_idx = int(sc_15ch[i + j] * rate)
                start_idx = max(0, peak_idx - pre_points)
                end_idx = min(len(df_15ch_cleaned), peak_idx + post_points)

                segment = df_15ch_cleaned[channel].iloc[start_idx:end_idx].values

                if segment_sum is None:
                    segment_sum = np.zeros_like(segment)

                segment_sum += segment

            averaged_segment = segment_sum / window_size
            channel_averages.append(averaged_segment)

        # Combine all channel averages into a DataFrame
        averaged_df = pd.DataFrame(
            {
                f"ch_{idx + 1}": channel_averages[idx]
                for idx in range(len(channel_averages))
            }
        )
        averaged_data.append(averaged_df)

    return averaged_data


def main(args):
    dir_path = args.raw_datas_dir
    csv_reader_15ch = CSVReader_16ch(dir_path)
    print(dir_path)
    df_15ch = csv_reader_15ch.process_files()
    print(df_15ch)
    cols = df_15ch.columns
    # df_15ch = pd.DataFrame()
    for col in cols:
        df_15ch[col] = df_15ch[col] - df_15ch["ch_16"]
    df_15ch = df_15ch.drop(columns=["ch_16"])
    df_15ch_cleaned = ecg_clean_df_15ch(df_15ch=df_15ch.copy(), rate=RATE_15CH)
    df_resample_15ch = linear_interpolation_resample_All(
        df=df_15ch_cleaned.copy(), sampling_rate=RATE_15CH, new_sampling_rate=RATE
    )
    df_15ch_cleaned = df_resample_15ch.copy()
    csv_reader_12ch = CSVReader_12ch(dir_path)
    df_12ch = csv_reader_12ch.process_files()
    df_12ch_cleaned = ecg_clean_df_12ch(df_12ch)
    if reverse == "off":
        sc_15ch = peak_sc_15ch(
            df_15ch_cleaned.copy(), RATE=RATE, TARGET=TARGET_CHANNEL_15ch
        )
        peak_sc_plot(df_15ch_cleaned.copy(), RATE=RATE, TARGET=TARGET_CHANNEL_15ch)
    else:
        df_15ch_reverse = df_15ch_cleaned.copy()
        df_15ch_reverse[TARGET_CHANNEL_15ch] = (-1) * df_15ch_cleaned.copy()[
            TARGET_CHANNEL_15ch
        ]
        df_15ch_cleaned = df_15ch_reverse.copy()
        sc_15ch = peak_sc_15ch(
            df_15ch_reverse.copy(), RATE=RATE, TARGET=TARGET_CHANNEL_15ch
        )
        peak_sc_plot(df_15ch_reverse.copy(), RATE=RATE, TARGET=TARGET_CHANNEL_15ch)

    # 先頭と最後は削除(400データ取れるものを対象とするため)
    sc_15ch = sc_15ch.drop(0).reset_index(drop=True)
    sc_15ch = sc_15ch.drop(len(sc_15ch) - 1)

    # ピークを基準として20個のピークで時間平均を取る
    print("len(sc_15ch):", len(sc_15ch))
    averaged_15ch_datas = calculate_time_averaged_peaks(
        df_15ch_cleaned, sc_15ch, window_size=20, pre_points=150, post_points=250
    )

    sc_12ch = peak_sc(df_12ch.copy(), RATE=RATE_12ch, TARGET=TARGET_CHANNEL_12CH)
    # print(sc_12ch)
    # input()
    peak_sc_plot(df_12ch.copy(), RATE=RATE_12ch, TARGET=TARGET_CHANNEL_12CH)
    manual_setting = int(
        input(f"目視で手動設定を行う:1, ファイル読み込みで設定を行う:2\n")
    )
    # 15chと12chのデータ同期
    for i, (averaged_15ch_data, sc12) in enumerate(
        zip(averaged_15ch_datas, sc_12ch[0])
    ):

        center_12ch_idx = int(sc12 * RATE)
        start_12ch_idx = center_12ch_idx - int(0.375 * 0.8 * RATE)
        end_12ch_idx = center_12ch_idx + int(0.625 * 0.8 * RATE)
        heartbeat_12ch = df_12ch_cleaned.iloc[start_12ch_idx:end_12ch_idx]
        #     heartbeat_12ch.to_csv(
        #         PROCESSED_DATA_DIR + "/synchro_data/12ch_{}.csv".format(i, center_12ch_idx),
        #         header=False,
        #         index=False,
        #     )
        # 15chと12ch結合
        # heartbeat_15ch = heartbeat_15ch.reset_index(drop=True)
        heartbeat_15ch = averaged_15ch_data
        heartbeat_12ch = heartbeat_12ch.reset_index(drop=True)
        # 正規化実行
        heartbeat_15ch = normalize_data(heartbeat_15ch)
        heartbeat_12ch = normalize_data(heartbeat_12ch)
        print(heartbeat_15ch)
        print(heartbeat_12ch)
        merge_df = pd.concat([heartbeat_15ch, heartbeat_12ch], axis=1)
        print(merge_df)

        if len(heartbeat_12ch) < 0.8 * RATE:
            print("12chのデータが短いです。")
            break
        if manual_setting == 1:
            prt_eles = PT_wave_search(heartbeat_12ch)
        elif manual_setting == 2:
            p_to_t_filename = f"ponset_toffset_{i:03d}.csv"
            points_df = pd.read_csv(
                args.dataset_output_path + "/" + p_to_t_filename,
            )
            print(points_df)
            p_Onset_ele = points_df["p_onset"][0]
            p_Peaks_ele = points_df["p_peak"][0]
            p_Offset_ele = points_df["p_offset"][0]
            t_Onset_ele = points_df["t_onset"][0]
            t_Peaks_ele = points_df["t_peak"][0]
            t_Offset_ele = points_df["t_offset"][0]
            q_Peaks_ele = points_df["q_peak"][0]
            s_Peaks_ele = points_df["s_peak"][0]
            prt_eles = np.array(
                [
                    p_Onset_ele,
                    150,  # rpeakは不要
                    t_Offset_ele,
                    p_Offset_ele,
                    t_Onset_ele,
                    p_Peaks_ele,
                    q_Peaks_ele,
                    s_Peaks_ele,
                    t_Peaks_ele,
                ]
            )
        file_path = args.dataset_output_path
        data = merge_df
        print(prt_eles)

        p_onset = int(prt_eles[0])
        r_peak = int(prt_eles[1])
        t_offset = int(prt_eles[2])
        p_offset = int(prt_eles[3])
        t_onset = int(prt_eles[4])
        p_peak = int(prt_eles[5])
        q_peak = int(prt_eles[6])
        s_peak = int(prt_eles[7])
        t_peak = int(prt_eles[8])
        print("p_onset", p_onset)
        data = heart_fitting(
            data,
            p_onset=p_onset,
            r_peak=r_peak,
            t_offset=t_offset,
            p_offset=p_offset,
            t_onset=t_onset,
            p_peak=p_peak,
            q_peak=q_peak,
            s_peak=s_peak,
            t_peak=t_peak,
        )

        print("{}番目の心拍切り出し".format(i + 1))
        # print(data)
        file_name = "dataset_{}.csv".format(str(i).zfill(3))
        output_csv(file_name=file_name, file_path=file_path, data=data.copy())
        file_name_pt = "ponset_toffset_{}.csv".format(str(i).zfill(3))
        output_csv_eles(
            file_name=file_name_pt,
            file_path=file_path,
            p_onset=p_onset,
            r_peak=r_peak,
            t_offset=t_offset,
            p_offset=p_offset,
            t_onset=t_onset,
            p_peak=p_peak,
            q_peak=q_peak,
            s_peak=s_peak,
            t_peak=t_peak,
        )
    # 移動平均を計算
    # 処理するCSVファイルの一覧を取得
    data_paths = sorted(glob(args.dataset_output_path + "/dataset_*.csv"))
    # pt_array_paths = sorted(
    #     glob(
    #         args.dataset_output_path
    #         + "/"
    #         + args.output_filepath
    #         + "/ponset_toffset_*.csv"
    #     )
    # )
    # pt_extend(data_paths, pt_array_paths)
    moving_ave_path = args.dataset_output_path + "/moving_ave_datasets"
    create_directory_if_not_exists(moving_ave_path)
    calculate_moving_average(data_paths, moving_ave_path, group_size=5)

    print("終了")

    # prt_eles = PTwave_search3(
    #     ecg_A2=ecg_A2_np,
    #     header="A2",
    #     sampling_rate=RATE,
    #     args=args,
    #     time_length=args.time_range,
    #     method=args.peak_method,
    # )  # 1213からPQRST全部検出できるcwt方を使う。
    # heartbeat_cutter_prt = HeartbeatCutter_prt(
    #     con_data.copy(), time_length=args.time_range, prt_eles=prt_eles, args=args
    # )  # 切り出す秒数を指定する。
    # heartbeat_cutter_prt.cut_heartbeats(
    #     file_path=args.dataset_output_path + "/" + args.output_filepath,
    #     ch=TARGET_CHANNEL_15ch,
    #     cut_min_max_range=cut_min_max_range,
    #     args=args,
    # )


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
    args = parser.parse_args()
    args.name, args.date = select_name_and_date()
    args.peak_method = (
        "cwt"  # neurokitのピーク検出アルゴリズムについてcwtかpeakがある。
    )
    args.pos = "0"
    args.type = ""
    args.dir_name = "{}/{}".format(args.name, args.type)
    args.png_path = ""
    args.time_range = 0.8
    args.output_filepath = "{}_{}_{}s/{}".format(
        args.name, args.date, str(args.time_range), args.pos
    )
    args.TARGET_CHANNEL_12CH = "A2"
    args.cut_min_max_range = [1.0, 100.0]
    args.reverse = "off"
    args.type = "{}_{}_{}".format(args.name, args.date, args.pos)
    args.dir_name = "{}/{}".format(args.name, args.type)
    # args.project_path='/home/cs28/share/goto/goto/ecg_project'
    # args.raw_datas_os=RAW_DATA_DIR
    # args.processed_datas_os=args.project_path+'/data/processed'
    # args.processed_datas_os=PROCESSED_DATA_DIR
    args.dataset_made_date = DATASET_MADE_DATE
    args.raw_datas_dir = RAW_DATA_DIR + "/takahashi_test/{}".format(args.dir_name)
    args.dataset_output_path = (
        PROCESSED_DATA_DIR
        + "/15ch_arrange_direction/{}_{}_0.8s/0".format(args.name, args.date)
    )
    main(args)
