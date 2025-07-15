import pandas as pd
import numpy as np
import neurokit2 as nk
import matplotlib.pyplot as plt

RATE = 500.00
RATE_12ch = 500.00
RATE_15CH = 122.06


def validate_integer_input():
    try:
        value = int(input("整数を入力してください（1から15までの範囲）: "))
        if value < 0 or value > 16:
            raise ValueError("入力された整数は範囲外です。")
        else:
            ch = "ch_" + str(value)
            return ch
    except ValueError as e:
        print("エラー:", e)
        return None


def ecg_clean_df_12ch(df_12ch, rate=RATE):
    ecg_signal = df_12ch.copy()["A2"]
    # cleand_signal=nk.ecg_clean(ecg_signal,sampling_rate=500,method="neurokit")
    # print(cleaned_signal)
    # print(type(cleaned_signal))
    # plt.plot(cleaned_signal)
    print(type(df_12ch))
    plt.plot(df_12ch["A2"])
    plt.title("org")
    plt.close()
    plt.cla()
    # plt.show()
    df_12ch_cleaned = pd.DataFrame()
    for i, column in enumerate(df_12ch.columns):
        # df0_mul.plot()
        # df1=df.iloc[:,i]
        ecg_signal = df_12ch[column].copy().values
        ecg_signal = nk.ecg_clean(ecg_signal, sampling_rate=rate, method="neurokit")
        df_12ch_cleaned[column] = ecg_signal
        # print(df[column])
        # print("")

    # print(type(df_12ch_cleaned))
    # plt.plot(df_12ch_cleaned["A2"])
    # plt.title("cleaned")
    # plt.show()
    return df_12ch_cleaned


def ecg_clean_df_15ch(df_15ch, rate):
    ecg_signal = df_15ch.copy()["ch_1"]
    print(type(df_15ch))
    df_15ch_cleaned = pd.DataFrame()
    for i, column in enumerate(df_15ch.columns):
        ecg_signal = df_15ch[column].copy().values
        ecg_signal = nk.ecg_clean(ecg_signal, sampling_rate=rate, method="neurokit")
        df_15ch_cleaned[column] = ecg_signal

    fig = plt.figure(num=None, figsize=(12, 5), dpi=100, facecolor="w", edgecolor="k")
    axis_line_width = 2.0
    tick_label_size = 18
    # 最初のグラフ（8プロット）
    plot_time = np.arange(len(df_15ch)) / RATE_15CH
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(plot_time, df_15ch["ch_1"], label="ch_1")
    ax1.plot(plot_time, df_15ch_cleaned["ch_1"], label="filtered ch1")
    # ax1.legend(fontsize=12, ncol=1)
    ax1.legend(loc="upper right", fontsize=18, ncol=1, bbox_to_anchor=(1, 1))
    ax1.tick_params(labelsize=tick_label_size, direction="in")
    plt.xlim(3.8, 8)
    plt.ylim(-100, 200)
    for axis in ["top", "bottom", "left", "right"]:
        ax1.spines[axis].set_linewidth(axis_line_width)
    # ax2 = fig.add_subplot(2, 1, 2)
    # ax2.plot(df_15ch_cleaned["ch_1"],label='filtered ch1')
    # ax2.legend(loc='center left', fontsize=12, ncol=1, bbox_to_anchor=(1, 0.5))
    # ax2.tick_params(labelsize=tick_label_size,direction='in')
    # # print(type(df_15ch_cleaned))
    # plt.plot(df_15ch_cleaned["ch_1"])
    # plt.title("cleaned")
    print()
    plt.savefig("taniguchi_filter.svg")
    plt.tight_layout()
    plt.show()
    return df_15ch_cleaned


class AutoIntegerFileHandler:
    def __init__(self, filename):
        self.filename = filename

    def check_file(self):
        path = self.filename
        if os.path.exists(path):
            if os.path.isfile(path):
                print(f"The path '{path}' exists and it is a file.")
                if input("ok? y or n") == "y":
                    return True
                # return True
        else:
            print(f"The path '{path}' does not exist.")
        return False

    def input_integer(self, RATE, cut_time):
        time = cut_time
        print("同期する時間は{}(s)".format(time))
        integer = int(RATE * time)
        print("integer={}".format(integer))
        return integer

    def write_integer(
        self, RATE, cut_time, target_15ch, reverse, target_12ch, cut_min_max_range
    ):
        integer = self.input_integer(RATE, cut_time)
        # with open(self.filename, 'w') as file:
        #    file.write(str(integer)+'\n')
        # #    file.write("TARGET_CHANNEL_15ch="+str(self.ch))
        #    file.write(str(target_15ch)+'\n')
        #    file.write(str(target_12ch)+'\n')
        #    file.write(str(cut_min_max_range[0])+'\n')
        #    file.write(str(cut_min_max_range[1])+'\n')

        data = {
            "INDEX": str(integer),
            "TARGET_CH_16ch": str(target_15ch),
            "REVERSE": reverse,  # ピーク検出するときにTARGET_15chの波形を反転させるかどうかを決める。
            "TARGET_CH_12ch": str(target_12ch),
            "START_TIME": str(
                cut_min_max_range[0]
            ),  # 書いてるだけで別に同期ファイルがある場合は使わないデータ
            "END_TIME": str(
                cut_min_max_range[1]
            ),  # 書いてるだけで別に同期ファイルがある場合は使わないデータ
        }

        column_order = [
            "INDEX",
            "TARGET_CH_16ch",
            "REVERSE",
            "TARGET_CH_12ch",
            "START_TIME",
            "END_TIME",
        ]
        with open(self.filename, mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=column_order)

            # カラム名を書き込む
            writer.writeheader()

            # データを書き込む
            writer.writerow(data)

    def read_integer(self):
        # CSVファイルを読み込む
        with open(self.filename, "r") as file:
            reader = csv.DictReader(file)

            # カラム名を取得
            columns = reader.fieldnames
            print(columns)
            data_list = []
            data_dict = {}

            # データを読み込み、表示
            for i, row in enumerate(reader):
                print(row)
                data_dict = row

                # データを辞書型に格納
            # data_dict = {row["INDEX"]: row for row in data_list}
            print(data_dict)

            # print("CSVファイルの内容を辞書に格納しました。")
            # input("posseeeeeeee")
            return (
                int(data_dict["INDEX"]),
                data_dict["TARGET_CH_16ch"],
                data_dict["REVERSE"],
                data_dict["TARGET_CH_12ch"],
            )


def linear_interpolation_resample_All(df, sampling_rate, new_sampling_rate):
    # 時系列データの時間情報を正規化
    df_new = pd.DataFrame(columns=df.columns)
    dt = 1.0 / sampling_rate
    time = np.arange(len(df)) * dt
    time_normalized = (time - time[0]) / (time[-1] - time[0])

    for i in range(len(df.columns)):
        data = df[df.columns[i]].copy()
        data = data.to_numpy()

        # 線形補間関数を作成
        interpolator = interp1d(time_normalized, data)

        # 新しい時間情報を生成
        new_time_normalized = np.linspace(
            0, 1, int((time[-1] - time[0]) * new_sampling_rate)
        )

        # 線形補間によるリサンプリング
        new_data = interpolator(new_time_normalized)

        # 新しい時間情報を元のスケールに戻す
        new_time = new_time_normalized * (time[-1] - time[0]) + time[0]
        df_new[df.columns[i]] = new_data
        print("{}_線形補間リサンプリング完了".format(df.columns[i]))
    print("old_datalength={}".format(len(df)))
    print("new_datalength={}".format(len(df_new)))

    return df_new


class ArrayComparator:
    def __init__(self, sc_15ch, sc_12ch, cut_min_max_range):
        self.sc_12ch = sc_12ch
        self.sc_15ch = sc_15ch
        self.cut_min_max_range = cut_min_max_range

    def cul_diff(self):
        time_12ch = self.sc_12ch[0][1:].to_numpy()
        time_15ch = self.sc_15ch[0][1:].to_numpy()
        diff_12ch = np.diff(self.sc_12ch[0])
        diff_15ch = np.diff(self.sc_15ch[0])
        return time_12ch, time_15ch, diff_12ch, diff_15ch

    def peak_diff_plot(self):
        time1, time2, diff1, diff2 = self.cul_diff()
        print(diff1)
        print(diff2)
        # データ1のプロット
        plt.plot(time1, diff1, label="12ch", color="r")
        plt.scatter(time1, diff1, label="12ch", color="r")
        # データ2のプロット
        plt.plot(time2, diff2, label="15ch", color="b")
        plt.scatter(time2, diff2, label="15ch", color="b")

        # グラフのタイトルと凡例
        plt.title("compare of peak time diff")
        plt.legend()

        # 軸ラベルの設定
        plt.xlabel("time(s)")
        plt.ylabel("diff(s)")

        # グラフの表示
        plt.show()
        plt.close()

    def find_best_cut_time(self):
        cut_min_max_range = self.cut_min_max_range
        min_mse = float("inf")  # 初期値として最大値を設定
        best_index = 0
        # target=-15#後ろから3つを基準に平均二乗誤差でマッチするインデックスを探す。
        time1, time2, diff_12ch, diff_15ch = self.cul_diff()
        # target=-len(diff_12ch)
        target = 0
        # target=5#後ろから3つを基準に平均二乗誤差でマッチするインデックスを探す。
        # diff_12ch=diff_12ch[target:]
        large_size = len(diff_15ch)
        small_size = len(diff_12ch)

        for i in range(large_size - small_size + 1):
            if (
                time2[i] - time1[target] < cut_min_max_range[0]
                or time2[i] - time1[target] > cut_min_max_range[1]
            ):  # 始めの4.0秒は使わない
                print("continue " + str(i))
                continue

            current_subset = diff_15ch[i : i + small_size]
            mse = np.mean((current_subset - diff_12ch) ** 2)

            if mse < min_mse:
                min_mse = mse
                best_index = i
        print("12chの最初のピークのtime={}".format(time1[target]))
        print("15chの対応するピークのtime={}".format(time2[best_index]))
        cut_time = time2[best_index] - time1[target]
        print("差分={}".format(cut_time))
        return cut_time, min_mse

    def peak_diff_plot_move(self, cut_time):
        cut_time = self.find_best_cut_time()
        time1, time2, diff1, diff2 = self.cul_diff()
        time1_v2 = time1 + cut_time
        print(diff1)
        print(diff2)
        # データ1のプロット
        plt.plot(time1, diff1, label="12ch", color="r")
        plt.scatter(time1, diff1, label="12ch", color="r")
        # データ2のプロット
        plt.plot(time2, diff2, label="15ch", color="b")
        plt.scatter(time2, diff2, label="15ch", color="b")

        # データ1のプロットのcut_time分平行移動
        plt.plot(time1_v2, diff1, label="12ch_move", color="g")
        plt.scatter(time1_v2, diff1, label="12ch_move", color="g")
        # グラフのタイトルと凡例
        plt.title("compare of peak time diff")
        plt.legend()

        # 軸ラベルの設定
        plt.xlabel("time(s)")
        plt.ylabel("diff(s)")

        # グラフの表示
        plt.show()
        plt.close()


# 1. CSVデータを読み込む
df = pd.read_csv(
    "../data/raw/extract_patient_data/patient2_12ch_20000.csv"
)  # CSVファイルのパスを指定してください
RATE = 125.56  # サンプリングレートを指定してください（例：1000Hz）
TARGET = "A1"  # CSVファイルのECGデータのカラム名を指定してください

# 2. ECG信号のクリーンアップ
ecg_signal = nk.ecg_clean(df[TARGET], sampling_rate=RATE)

# 3. ピークを検出
_, rpeaks = nk.ecg_peaks(ecg_signal, sampling_rate=RATE)

# 4. ピーク間隔の計算
rpeak_indices = rpeaks["ECG_R_Peaks"]
peak_times = rpeak_indices / RATE  # ピークのタイムスタンプ（秒）

# ピーク間隔（RR間隔）を計算
rr_intervals = np.diff(peak_times)

# 5. ピーク間隔の平均値を計算
mean_rr_interval = np.mean(rr_intervals)

# 結果を表示
print(f"ピーク間隔の平均値: {mean_rr_interval:.4f} 秒")

# オプション: 心電図信号とピークをプロット
plt.figure(figsize=(10, 4))
plt.plot(df[TARGET], label="ECG Signal")
plt.plot(rpeak_indices, df[TARGET][rpeak_indices], "ro", label="R-peaks")
plt.xlabel("Samples")
plt.ylabel("ECG")
plt.legend()
plt.show()


def main(args):
    # TARGET_CHANNEL_15CH=args.TARGET_CHANNEL_15CH
    TARGET_CHANNEL_12CH = args.TARGET_CHANNEL_12CH
    cut_min_max_range = args.cut_min_max_range
    # ファイル読み込み
    # dir_path = "./0_packetloss_data/"+args.dir_name
    # dir_path = "./0_packetloss_data_{}/".format(DATASET_MADE_DATE)+args.dir_name
    # dir_path = args.dataset_dir
    dir_path = args.raw_datas_dir
    df_16ch = pd.read_csv(dir_path + "db_patient_16ch_data_20000.csv", header=None)
    df_16ch = df_16ch.drop(columns=[16])
    df_16ch = df_16ch.rename(columns=lambda x: "ch_" + str(x + 1))
    print(df_16ch)
    cols = df_16ch.columns
    df_15ch = pd.DataFrame()
    for col in cols:
        df_15ch[col] = df_16ch[col] - df_16ch["ch_16"]
    df_15ch = df_15ch.drop(columns=["ch_16"])

    df_12ch = pd.read_csv(dir_path + "KT_4794335_lizmil_10000.csv")
    df_12ch_cleaned = ecg_clean_df_12ch(df_12ch)
    # input("")
    # 同期用インデックスファイルを読み込みと書き込み
    handler = AutoIntegerFileHandler(dir_path + "/同期インデックス_nkmodule.txt")
    # 同期する前
    # if(DEBUG_PLOT==True):
    #     Plot_15ch=MultiPlotter(df_15ch.copy(),RATE=RATE_15CH)
    #     Plot_15ch.plot_all_channels(xmin=0,xmax=10,ylim=0)
    #     # Plot_15ch.multi_plot(xmin=0,xmax=100,ylim=0)
    #     # Plot_16ch=MultiPlotter(df_16ch.copy(),RATE=RATE_15CH)
    #     # Plot_16ch.multi_plot(xmin=0,xmax=100,ylim=0)
    #     plt.show()
    #     plt.close()
    # input()

    if handler.check_file() == False:  # 同期するためのファイルが存在していないとき。
        # df_15ch_pf = multi_pf(df_15ch.copy(),fp=2.0,fs=1.0)
        reverse = args.reverse
        print("TARGET_CHNNEL_15chは")
        TARGET_CHANNEL_15CH = validate_integer_input()
        # df_15ch_pf = hpf_lpf(df_15ch.copy(),HPF_fp=2.0,HPF_fs=1.0,LPF_fp=0,LPF_fs=0,RATE=RATE_15CH)
        # df_15ch_pf = hpf_lpf(df_15ch.copy(),HPF_fp=HPF_FP,HPF_fs=HPF_FS,LPF_fp=0,LPF_fs=0,RATE=RATE_15CH)
        df_15ch_pf = ecg_clean_df_15ch(df_15ch=df_15ch.copy(), rate=RATE_15CH)
        df_resample_15ch = linear_interpolation_resample_All(
            df=df_15ch_pf.copy(), sampling_rate=RATE_15CH, new_sampling_rate=RATE
        )
        df_15ch_pf = df_resample_15ch.copy()
        if reverse == "off":
            sc_15ch = peak_sc_15ch(
                df_15ch_pf.copy(), RATE=RATE, TARGET=TARGET_CHANNEL_15CH
            )
            peak_sc_plot(df_15ch_pf.copy(), RATE=RATE, TARGET=TARGET_CHANNEL_15CH)

        # 心電図のサンプリングレートが500になってない場合
        df_12ch_pf = ecg_clean_df_12ch(df_12ch=df_12ch.copy(), rate=125.56)
        df_resample_12ch = linear_interpolation_resample_All(
            df=df_12ch_pf.copy(), sampling_rate=125.56, new_sampling_rate=RATE
        )
        df_12ch_pf = df_resample_12ch.copy()
        sc_12ch = peak_sc(df_12ch_pf.copy(), RATE=RATE_12ch, TARGET=TARGET_CHANNEL_12CH)
        # print(sc_12ch)
        # input()
        peak_sc_plot(df_12ch_pf.copy(), RATE=RATE_12ch, TARGET=TARGET_CHANNEL_12CH)
        print("reverse=={}".format(reverse))
        # peak_sc_plot(df_15ch_pf.copy(),RATE=RATE_15CH,TARGET=TARGET_CHANNEL_15CH)
        print(sc_15ch)

        comparator = ArrayComparator(
            sc_15ch=sc_15ch, sc_12ch=sc_12ch, cut_min_max_range=cut_min_max_range
        )
        comparator.peak_diff_plot()
        cut_time = comparator.find_best_cut_time()
        comparator.peak_diff_plot_move(cut_time)
        # comparator.find_best_cut_time()
        # peak_diff_plot(sc_12ch,sc_15ch)
        Plot_15ch_pf = MultiPlotter(df_15ch_pf, RATE=RATE)
        Plot_15ch_pf.multi_plot(xmin=0, xmax=100, ylim=0)
        Plot_15ch_pf.multi_plot_15ch_with_sc(xmin=0, xmax=20, ylim=0, sc=sc_15ch)
        plt.show()
        plt.close()
        print(int(cut_time * RATE))
        if input("write_to_CSV OK? y or n") == "y":
            handler.write_integer(
                RATE=RATE,
                cut_time=cut_time,
                target_15ch=TARGET_CHANNEL_15CH,
                reverse=reverse,
                target_12ch=TARGET_CHANNEL_12CH,
                cut_min_max_range=cut_min_max_range,
            )

        else:
            return 0

    else:  # 同期するファイルが存在しているとき。
        # df_15ch_pf = hpf_lpf(df_15ch.copy(),HPF_fp=HPF_FP,HPF_fs=HPF_FS,LPF_fp=0,LPF_fs=0,RATE=RATE_15CH)
        # df_15ch_pf = multi_pf(df_15ch.copy(),fp=0.2,fs=0.1)
        # df_15ch_pf = df_15ch.copy()
        print("ファイルが存在します。")
        # return 0
    df_15ch_pf = ecg_clean_df_15ch(df_15ch=df_15ch.copy(), rate=RATE_15CH)
    df_resample_15ch = linear_interpolation_resample_All(
        df=df_15ch_pf.copy(), sampling_rate=RATE_15CH, new_sampling_rate=RATE
    )

    syn_index, TARGET_CHANNEL_15CH, reverse, TARGET_CHANNEL_12CH = (
        handler.read_integer()
    )  # 同期するインデックス
    print(syn_index)
