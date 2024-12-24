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
)
import pandas as pd
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


def cut_heartbeats(self, file_path, ch, cut_min_max_range, args):
    center_idxs = self.prt_eles[:, 1]
    p_indexs_onsets = self.prt_eles[:, 0]
    t_indexs_offsets = self.prt_eles[:, 2]
    p_indexs_offsets = self.prt_eles[:, 3]
    t_indexs_onsets = self.prt_eles[:, 4]
    p_indexs_peaks = self.prt_eles[:, 5]
    q_indexs_peaks = self.prt_eles[:, 6]
    s_indexs_peaks = self.prt_eles[:, 7]
    t_indexs_peaks = self.prt_eles[:, 8]
    create_directory(file_path)
    data = []
    datas = []
    data.append(ch)
    data.append(cut_min_max_range[0])
    data.append(cut_min_max_range[1])
    write_text_file(data, file_path + "/" + "TARGET_CHANNEL.txt")
    print(file_path)

    pt_info = []
    for i, center_idx in enumerate(center_idxs):
        data = self.con_data[center_idx - self.range : center_idx + self.range].copy()

        p_onset = (
            p_indexs_onsets[i] - center_idx + self.range
        )  # prt_ele[0]はponsetの座標
        t_offset = (
            t_indexs_offsets[i] - center_idx + self.range
        )  # prt_ele[2]はtoffsetの座標
        p_offset = (
            p_indexs_offsets[i] - center_idx + self.range
        )  # prt_ele[0]はponsetの座標
        t_onset = (
            t_indexs_onsets[i] - center_idx + self.range
        )  # prt_ele[2]はtoffsetの座標
        p_peak = (
            p_indexs_peaks[i] - center_idx + self.range
        )  # prt_ele[2]はtoffsetの座標
        q_peak = (
            q_indexs_peaks[i] - center_idx + self.range
        )  # prt_ele[2]はtoffsetの座標
        s_peak = (
            s_indexs_peaks[i] - center_idx + self.range
        )  # prt_ele[2]はtoffsetの座標
        t_peak = (
            t_indexs_peaks[i] - center_idx + self.range
        )  # prt_ele[2]はtoffsetの座標

        # plot_heartbeats_sotoume(data.copy(),i,p_onset,t_offset)
        # plot_heartbeats(data.copy(),i)
        print("{}番目の心拍切り出し".format(i + 1))
        # print(data)
        file_name = "dataset_{}.csv".format(str(i).zfill(3))
        self.output_csv(file_name=file_name, file_path=file_path, data=data.copy())
        file_name_pt = "ponset_toffsett_{}.csv".format(str(i).zfill(3))
        # self.output_csv_ponset_toffset(file_name=file_name_pt,file_path=file_path,p_onset=p_onset,t_offset=t_offset,p_offset=p_offset,t_onset=t_onset)
        self.output_csv_eles(
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
        pt_time = (t_offset - p_onset) / RATE
        t_onset_time = (t_onset) / RATE
        t_offset_time = (t_offset) / RATE
        r_offset = 210
        r_t_index = (t_onset - r_offset) / RATE
        L_weight = 1.3  # 水増しで伸ばす最大の倍率
        check_value = (
            (400 - t_offset) / (t_onset - 210) / (L_weight - 1)
        )  # ST部を引き延ばす水増しをしても大丈夫か確かめる指標。1以上でOK

        pt_info_temp = [
            self.name,
            self.pos,
            str(i).zfill(3),
            pt_time,
            t_onset_time,
            t_offset_time,
            r_t_index,
            check_value,
            L_weight,
        ]
        pt_info.append(pt_info_temp)
    # input(pt_info)
    data_num_info = [[self.name, self.pos, len(center_idxs)]]
    # append_to_csv(filename="Dataset/pqrst2/pt_time_all_{}s.csv".format(str(self.time_length)),data=pt_info)
    # dataset_num_to_csv(filename="Dataset/pqrst2/dataset_num_{}s.csv".format(str(self.time_length)),data=data_num_info)
    append_to_csv(
        filename=args.dataset_output_path
        + "/pt_time_all_{}s.csv".format(str(self.time_length)),
        data=pt_info,
    )
    dataset_num_to_csv(
        filename=args.dataset_output_path
        + "/dataset_num_{}s.csv".format(str(self.time_length)),
        data=data_num_info,
    )


def main(args):
    RATE = 500.00
    RATE_12ch = 500.00
    RATE_15CH = 122.06
    TARGET_CHANNEL_12CH = "A2"
    TARGET_CHANNEL_15CH = validate_integer_input()
    reverse = "on"

    dir_path = RAW_DATA_DIR + "/extract_patient_data/patient5/"
    csv_reader_16ch = CSVReader_16ch(dir_path)
    df_16ch = csv_reader_16ch.process_files()
    print(df_16ch)
    cols = df_16ch.columns
    df_15ch = pd.DataFrame()
    for col in cols:
        df_15ch[col] = df_16ch[col] - df_16ch["ch_16"]
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
            df_15ch_cleaned.copy(), RATE=RATE, TARGET=TARGET_CHANNEL_15CH
        )
        peak_sc_plot(df_15ch_cleaned.copy(), RATE=RATE, TARGET=TARGET_CHANNEL_15CH)
    else:
        df_15ch_reverse = df_15ch_cleaned.copy()
        df_15ch_reverse[TARGET_CHANNEL_15CH] = (-1) * df_15ch_cleaned.copy()[
            TARGET_CHANNEL_15CH
        ]
        df_15ch_cleaned = df_15ch_reverse.copy()
        sc_15ch = peak_sc_15ch(
            df_15ch_reverse.copy(), RATE=RATE, TARGET=TARGET_CHANNEL_15CH
        )
        peak_sc_plot(df_15ch_reverse.copy(), RATE=RATE, TARGET=TARGET_CHANNEL_15CH)
    print(df_15ch_cleaned)
    sc_12ch = peak_sc(df_12ch.copy(), RATE=RATE_12ch, TARGET=TARGET_CHANNEL_12CH)
    # print(sc_12ch)
    # input()
    peak_sc_plot(df_12ch.copy(), RATE=RATE_12ch, TARGET=TARGET_CHANNEL_12CH)
    # 15chと12chのデータ同期
    for i, (sc15, sc12) in enumerate(zip(sc_15ch[0], sc_12ch[0])):
        center_15ch_idx = int(sc15 * RATE)
        start_15ch_idx = int(center_15ch_idx - 0.4 * RATE)
        end_15ch_idx = int(center_15ch_idx + 0.4 * RATE)
        heartbeat_15ch = df_15ch_cleaned.iloc[start_15ch_idx:end_15ch_idx]
        #     heartbeat_15ch.to_csv(
        #         PROCESSED_DATA_DIR + "/synchro_data/15ch_{}.csv".format(i, center_15ch_idx),
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
        # 15chと12ch結合
        heartbeat_15ch = heartbeat_15ch.reset_index(drop=True)
        heartbeat_12ch = heartbeat_12ch.reset_index(drop=True)
        merge_df = pd.concat([heartbeat_15ch, heartbeat_12ch], axis=1)
        print(merge_df)

        ecg_A2 = merge_df["A2"]
        print(ecg_A2)
        ecg_A2_np = ecg_A2.to_numpy().T
        # return 0
        # prt_eles=PTwave_search(ecg_A2=ecg_A2_np,header="A2",sampling_rate=RATE,args=args,time_length=0.7)
        prt_eles = PTwave_search3(
            ecg_A2=ecg_A2_np,
            header="A2",
            sampling_rate=RATE,
            args=args,
            time_length=args.time_range,
            method=args.peak_method,
        )  # 1213からPQRST全部検出できるcwt方を使う。
        heartbeat_cutter_prt = HeartbeatCutter_prt(
            con_data.copy(), time_length=args.time_range, prt_eles=prt_eles, args=args
        )  # 切り出す秒数を指定する。
        heartbeat_cutter_prt.cut_heartbeats(
            file_path=args.dataset_output_path + "/" + args.output_filepath,
            ch=TARGET_CHANNEL_15CH,
            cut_min_max_range=cut_min_max_range,
            args=args,
        )

        merge_df.to_csv(
            PROCESSED_DATA_DIR + "/synchro_data/patient5/dataset_{}.csv".format(i),
            index=None,
        )

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
    #     ch=TARGET_CHANNEL_15CH,
    #     cut_min_max_range=cut_min_max_range,
    #     args=args,
    # )
