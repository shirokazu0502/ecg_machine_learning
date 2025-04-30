import argparse
from Make_dataset_0120 import (
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
    plot_and_select,
    write_text_file,
    create_directory_if_not_exists,
    calculate_moving_average,
)
from glob import glob
from scipy import signal
from scipy.ndimage import uniform_filter1d, median_filter
import numpy as np
import pandas as pd
import neurokit2 as nk
from config.settings import (
    DATA_DIR,
    BASE_DIR,
    PROCESSED_DATA_DIR,
    OUTPUT_DIR,
    RAW_DATA_DIR,
    TEST_DIR,
    RATE,
    RATE_16CH,
    TIME,
    DATASET_MADE_DATE,
)
from config.name_dic import select_name_and_date

RATE = 500.00
RATE_12ch = 500.00
RATE_16CH = RATE_16CH
TARGET_CHANNEL_12CH = "A2"
TARGET_CHANNEL_16ch = "ch_1"
reverse = "on"
patient_number = "2"


def PT_wave_search(ecg_all):
    data_list = []
    # p波オンセット、T波オフセット手動設定
    p_Onset_ele, t_Offset_ele = plot_and_select(ecg_all, 200)
    # 患者の場合、波のピークが検出できない、switch求められないから全てをデータセットに
    p_Offset_ele = None
    t_Onset_ele = None
    p_Peaks_ele = None
    t_Peaks_ele = None
    s_Peaks_ele = None
    q_Peaks_ele = None
    data_list.append(
        [
            p_Onset_ele,
            200,
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
    data_out.to_csv(file_path + "/" + file_name, index=None)


def output_csv_eles(
    file_path,
    file_name,
    p_onset,
    t_offset,
    p_offset,
    t_onset,
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
        "p_peak": [p_peak],
        "q_peak": [q_peak],
        "s_peak": [s_peak],
        "t_peak": [t_peak],
    }
    data_out = pd.DataFrame(data)
    data_out.to_csv(file_path + "/" + file_name, index=None)


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


def main(args):
    dir_path = args.raw_datas_dir
    csv_reader_16ch = CSVReader_16ch(dir_path)
    print(dir_path)
    df_16ch = csv_reader_16ch.process_files()
    print(df_16ch)
    # cols = df_16ch.columns
    # df_16ch = pd.DataFrame()
    # for col in cols:
    #     df_16ch[col] = df_16ch[col] - df_16ch["ch_16"]
    # df_16ch = df_16ch.drop(columns=["ch_16"])
    df_16ch_cleaned = ecg_clean_df_16ch(df_16ch=df_16ch.copy(), rate=RATE_16CH)
    df_resample_16ch = linear_interpolation_resample_All(
        df=df_16ch_cleaned.copy(), sampling_rate=RATE_16CH, new_sampling_rate=RATE
    )
    df_16ch_cleaned = df_resample_16ch.copy()
    csv_reader_12ch = CSVReader_12ch(dir_path)
    df_12ch = csv_reader_12ch.process_files()
    df_12ch_cleaned = ecg_clean_df_12ch(df_12ch)
    if reverse == "off":
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
    # 16chと12chのデータ同期
    for i, (sc_16ch, sc12) in enumerate(zip(sc_16ch[0], sc_12ch[0])):
        center_16ch_idx = int(sc_16ch * RATE)
        start_16ch_idx = int(center_16ch_idx - 0.4 * RATE)
        end_16ch_idx = int(center_16ch_idx + 0.4 * RATE)
        heartbeat_16ch = df_16ch_cleaned.iloc[start_16ch_idx:end_16ch_idx]
        print(start_16ch_idx)
        #     heartbeat_16ch.to_csv(
        #         PROCESSED_DATA_DIR + "/synchro_data/16ch_{}.csv".format(i, center_16ch_idx),
        #         header=False,
        #         index=False,
        #     )
        center_12ch_idx = int(sc12 * RATE)
        start_12ch_idx = int(center_12ch_idx - 0.4 * RATE)
        end_12ch_idx = int(center_12ch_idx + 0.4 * RATE)
        heartbeat_12ch = df_12ch_cleaned.iloc[start_12ch_idx:end_12ch_idx]
        #     heartbeat_12ch.to_csv(
        #         PROCESSED_DATA_DIR + "/synchro_data/12ch_{}.csv".format(i, center_12ch_idx),
        #         header=False,
        #         index=False,
        #     )
        # 16chと12ch結合
        heartbeat_16ch = heartbeat_16ch.reset_index(drop=True)
        heartbeat_12ch = heartbeat_12ch.reset_index(drop=True)
        merge_df = pd.concat([heartbeat_16ch, heartbeat_12ch], axis=1)
        print(merge_df)

        # ecg_A2 = merge_df["A2"]
        # print(ecg_A2)
        # ecg_A2_np = ecg_A2.to_numpy().T
        # # return 0
        # # prt_eles=PTwave_search(ecg_A2=ecg_A2_np,header="A2",sampling_rate=RATE,args=args,time_length=0.7)
        # prt_eles = PTwave_search3(
        #     ecg_A2=ecg_A2_np,
        #     header="A2",
        #     sampling_rate=RATE,
        #     args=args,
        #     time_length=args.time_range,
        #     method=args.peak_method,
        # )  # 1213からPQRST全部検出できるcwt方を使う。
        # heartbeat_cutter_prt = HeartbeatCutter_prt(print
        #     con_data.copy(), time_length=args.time_range, prt_eles=prt_eles, args=args
        # )  # 切り出す秒数を指定する。
        # heartbeat_cutter_prt.cut_heartbeats(
        #     file_path=args.dataset_output_path + "/" + args.output_filepath,
        #     ch=TARGET_CHANNEL_16ch,
        #     cut_min_max_range=cut_min_max_range,
        #     args=args,
        # )print
        prt_eles = PT_wave_search(heartbeat_12ch)
        file_path = args.dataset_output_path
        data = merge_df
        p_onset = int(prt_eles[:, 0])
        t_offset = int(prt_eles[:, 2])
        p_offset = None
        t_onset = None
        p_peak = None
        q_peak = None
        s_peak = None
        t_peak = None
        for j, column in enumerate(data.columns):
            # フィルタかける
            data[column] = clean_ecg_signal(data[column])
            # pt_extendを実施
            # インデックスp_onsetの値を取得
            print(data)
            value_at_p_onset = data.iloc[p_onset, j]
            # インデックスt_offsetの値を取得
            value_at_t_offset = data.iloc[t_offset, j]
            # p_onsetより前の値を置き換える
            data.iloc[:p_onset, j] = value_at_p_onset
            # t_offsetより後の値を置き換える
            data.iloc[t_offset:, j] = value_at_t_offset
            signal = data[column].copy().values
            # 基線を計算（信号の移動中央値を基線とする）
            window_size = 200
            baseline = median_filter(signal, size=window_size)
            # 基線補正（信号全体から基線を引く）
            signal = signal - baseline
            data[column] = signal
        print("{}番目の心拍切り出し".format(i + 1))
        # print(data)
        file_name = "dataset_{}.csv".format(str(i).zfill(3))
        output_csv(file_name=file_name, file_path=file_path, data=data.copy())
        file_name_pt = "ponset_toffset_{}.csv".format(str(i).zfill(3))
        output_csv_eles(
            file_name=file_name_pt,
            file_path=file_path,
            p_onset=p_onset,
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
    #     ch=TARGET_CHANNEL_16ch,
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
    # parser.add_argument("--TARGET_CHANNEL_16ch", type=str, default='ch_1')
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
    args.reverse = "on"
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
        + "/pqrst_nkmodule_since{}_{}/{}_{}_0.8s/0".format(
            args.dataset_made_date, args.peak_method, args.name, args.date
        )
    )
    main(args)
