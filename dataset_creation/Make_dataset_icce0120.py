import argparse
import sys
import os
import codecs
import struct
import binascii
from pylab import *
import matplotlib.cm as cm
import pandas as pd
from scipy import signal
### グラフ表示が要らない場合はFalse
# DEBUG_PLOT = True
DEBUG_PLOT = False
###
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import neurokit2 as nk
import warnings
import csv
HPF_FP=2.0
HPF_FS=1.0
DATASET_MADE_DATE="icce0120" #packet_loss_data_{}の部分
# DATASET_MADE_DATE="icce0116" #packet_loss_data_{}の部分
RATE_12ch=500
RATE_15ch=122.06
RATE=500
TIME=24  #記録時間は24秒または10秒

def create_text_file(file_path, content):
    """
    指定されたパスにテキストファイルを作成し、内容を書き込む関数。

    Parameters:
    - file_path: 作成するテキストファイルのパス
    - content: ファイルに書き込む内容（テキスト）

    Returns:
    - なし
    """
    with open(file_path, 'w') as file:
        file.write(content)

def append_to_csv(filename, data):
    # ファイルが存在しない場合は新しいファイルを作成
    if not os.path.exists(filename):
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Name", "pos", "Number", "P-T","T_Onset","T_offset","R_offset-T_Onset","if check_value >1 ok)","L_weight"])

    # データをCSVファイルに追記
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        for data_row in data:
            writer.writerow(data_row)

    print("データが追記されました。")

def dataset_num_to_csv(filename, data):
    # ファイルが存在しない場合は新しいファイルを作成
    if not os.path.exists(filename):
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Name", "pos", "Num"])

    # データをCSVファイルに追記
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        for data_row in data:
            writer.writerow(data_row)

    print("データが追記されました。")

def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
def data_plot_after_splitting2(ecg_list: list, doppler_list: list, npeaks: int,target_name:str, label_list: list,
                              sampling_rate: float=500, figtitle: str='title', savefig: bool=True, figpath: str='./plot_target', fontsize: int=15) -> None:

    """心拍を分割したものをまとめてプロットする関数。

    1行目：ECGの時系列波形
    2行目：Dopplerの時系列波形
    3行目：DopplerのCWTスペクトログラム
    4行目：ECG 時系列
    5行目：Doppler 時系列
    6行目：Doppler CWT
          ・
          ・
          ・
    これを繰り返す。

    Parameters
    ----------
    ecg_list : list
        ECGの時系列波形のリスト(要素数は心拍数となる)
    doppler_list : list
        Dopplerの時系列波形のリスト(要素数は心拍数となる)
    doppler_cwt_list : list
        DopplerのCWTスペクトログラムのリスト(要素数は心拍数となる)
    freqs_list : list
        CWTをプロットするときの縦軸に必要な、周波数情報のリスト
    sequence_num_list : list
        データ取得時、BLE通信でデータが欠落していないか確かめるために出力していた連番のリスト
        現在は使っていないので無視してよい
        (適当に空のリスト[[], [], ...(心拍数の数だけ続ける)]をいれるか、そもそもこの関数から消すか)
    npeaks : int
        心拍数
    sampling_rate : float
        サンプリングレート
    figtitle : str
        出力する画像の上部に記載するタイトル（fig.suptitle()）
    savefig: bool=False
        Trueにしたら画像を後述するfigpathに保存する
    figpath: str='./title.png'
        画像出力先パス
    fontsize: int=20
        画像のフォントサイズ


    Returns
    -------
    None

    """
    if npeaks <= 10: # 心拍数が10を超えたら次の行に移るようにする
        nrow = 3
        ncol = npeaks
    else:
        ncol = 10
        nrow = -(-npeaks // ncol) * 3 # 切り上げ

    fig = plt.figure(figsize=(18,5*nrow/3))

    # 時系列波形
    for peak_idx in range(npeaks):
        ecg = ecg_list[peak_idx]
        doppler = doppler_list[peak_idx]

        N = len(ecg[0]) # サンプル点数
        time_array = np.arange(0,N) / sampling_rate # グラフ横軸（時間）

        # ECG
        ax1 = fig.add_subplot(nrow,ncol,peak_idx+(peak_idx//ncol)*2*ncol+1)
        for i in range(12):
            ax1.plot(time_array, ecg[i][:])
        ax1.set_title(label_list[peak_idx],fontsize=5)
        # data drop idx：連番(sequence_number)は0-255が連なっていることが期待されるが、そうなっていないものを抽出する。

        # doppler(time)
        ax2 = fig.add_subplot(nrow,ncol,peak_idx+(peak_idx//ncol)*2*ncol+ncol+1, sharex=ax1)
        # ax2.set_ylim([-100,100])
        # print(len(doppler))
        for i in range(15):
            # ax2.plot(time_array, doppler[:][i])
            ax2.plot(time_array, doppler[i][:])
        # ax2.set_xticks(np.linspace(0, 2, 3),fontsize=5)
        # ax2.set_xticks(np.linspace(0, 2, 5), minor=True)
        # x軸共有のため、上二つのラベルは非表示にする
        plt.setp(ax1.get_xticklabels(), visible=False)
        plt.setp(ax2.get_xticklabels(), visible=False)

        # 右と上の枠線を消すcf)https://qiita.com/irs/items/fe909442be057f0efb48
        for ax in [ax1, ax2]:
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_linewidth(5)
            ax.spines['bottom'].set_linewidth(5)
            ax.xaxis.set_tick_params(direction='in',bottom=True,top=False,left=False,right=False,length=10,width=5)
            ax.yaxis.set_tick_params(direction='in',bottom=True,top=False,left=False,right=False,length=5,width=5)

    fig.suptitle(figtitle, fontsize=fontsize)
    fig.tight_layout(rect=[0,0,1,0.96]) # rect指定の順番は左下原点で(left,bottom,right,top). suptitle+tight_layout組み合わせる場合は注意
    fig.patch.set_facecolor('white') # 背景色を白にする

    plt.show()
    # if savefig:
        # fig.savefig(figpath+'/'+target_name+'.png') # 背景を透明にしたければtransparent=Trueオプションを付ければ良いっぽい

    plt.close()
def peak_search(data_frame, sampling_rate):
# peak search for ecg """
    peak_times = []
    peak_vals = []
    temp_max = [-1, -9999]
    temp_min = [-1, 9999]
    max_search_flag = True
    # max_ratio = 0.6
    # max_ratio = 0.4
    max_ratio = 0.8
    shift_rate = sampling_rate

    shift_min = int(0.45 * sampling_rate)
    shift_max = int(0.8 * sampling_rate)
    first_skip = int(0.1 * shift_rate)
    finish_search = int(0.45* shift_rate)
    for idx, val in enumerate(data_frame):
        if (idx < first_skip):
            continue
        if (max_search_flag or (idx - temp_max[0] > shift_min)) and val >= temp_max[1] or (idx - temp_max[0] > shift_max):
            temp_max = [idx, val]
            max_search_flag = True
        if val < temp_min[1]:
            temp_min = [idx, val]
        if max_search_flag and (idx - temp_max[0] > finish_search):
            peak_times.append(temp_max[0])
            peak_vals.append(temp_max[1])
            temp_max[1] -= (temp_max[1] - temp_min[1]) * (1.0 - max_ratio)
            temp_min = [None, 999]
            max_search_flag = False
    return peak_times, peak_vals

def peak_search_nk_15ch(df_target,RATE):
    print("safe")
    ecg_signal=df_target.copy().to_numpy().T
    # ecg_signal=nk.ecg_clean(ecg_signal,sampling_rate=RATE,method='neurokit')
    print(ecg_signal)
    _, rpeaks = nk.ecg_peaks(ecg_signal, RATE)
    print(rpeaks['ECG_R_Peaks'])
    vals=ecg_signal[rpeaks['ECG_R_Peaks']]
    return rpeaks['ECG_R_Peaks'],vals
def peak_search_nk(df_target,RATE):
    print("safe")
    ecg_signal=df_target.copy().to_numpy().T
    # ecg_signal=nk.ecg_clean(ecg_signal,sampling_rate=RATE,method='neurokit')
    print(ecg_signal)
    _, rpeaks = nk.ecg_peaks(ecg_signal, RATE)
    print(rpeaks['ECG_R_Peaks'])
    vals=ecg_signal[rpeaks['ECG_R_Peaks']]
    return rpeaks['ECG_R_Peaks'],vals

def peak_sc_15ch(dataframe,RATE,TARGET):
    times,val=peak_search_nk_15ch(dataframe[TARGET],RATE)
    dt=1.0/RATE
    N=len(dataframe)
    time_np=np.array(times)
    time1=time_np*dt
    sc=pd.DataFrame(index=[])
    sc[0]=time1
    sc[1]=val
    #print(sc)
    return sc
def peak_sc(dataframe,RATE,TARGET):
    times,val=peak_search_nk(dataframe[TARGET],RATE)
    dt=1.0/RATE
    N=len(dataframe)
    time_np=np.array(times)
    time1=time_np*dt
    sc=pd.DataFrame(index=[])
    sc[0]=time1
    sc[1]=val
    #print(sc)
    return sc

# def peak_sc(dataframe,RATE,TARGET):
#     times,val=peak_search(dataframe[TARGET],RATE)
#     dt=1.0/RATE
#     N=len(dataframe)
#     time_np=np.array(times)
#     time1=time_np*dt
#     sc=pd.DataFrame(index=[])
#     sc[0]=time1
#     sc[1]=val
#     #print(sc)
#     return sc

def peak_sc_plot(dataframe,RATE,TARGET):
    # times,val=peak_search(dataframe[TARGET],RATE)
    times,val=peak_search_nk(dataframe[TARGET],RATE)
    dt=1.0/RATE
    N=len(dataframe)
    time_np=np.array(times)
    time1=time_np*dt
    sc=pd.DataFrame(index=[])
    sc[0]=time1
    sc[1]=val
    plt.scatter(x=time1,y=val,color='red')
    time=np.arange(len(dataframe))*dt
    plt.plot(time,dataframe[TARGET])
    plt.title(TARGET)
    #print(sc)
    plt.show()
    plt.close()
    # print(sc)
    # input()
    return sc

def lpf(sampling_rate, fp, fs, x):
    """ low pass filter """
    # fp = 0.5                          # 通過域端周波数[Hz]
    # fs = 0.1                          # 阻止域端周波数[Hz]
    gpass = 1                        # 通過域最大損失量[dB]
    gstop = 20                       # 阻止域最小減衰量[dB]
    norm_pass = fp / (sampling_rate / 2)
    norm_stop = fs / (sampling_rate / 2)
    N, Wn = signal.cheb2ord(wp=norm_pass, ws=norm_stop, gpass=gpass, gstop=gstop, analog=0)
    b, a = signal.cheby2(N, gstop, Wn, "low")
    z = signal.lfilter(b, a, x)
    # return b, a, z
    return z

def hpf(sampling_rate, fp, fs,x):
    """ high pass filter """
    #fp = 0.5                          # 通過域端周波数[Hz]
    #fs = 0.1                          # 阻止域端周波数[Hz]
    gpass = 1                       # 通過域最大損失量[dB]
    gstop = 20                      # 阻止域最小減衰量[dB]
    norm_pass = fp / (sampling_rate / 2)
    norm_stop = fs / (sampling_rate / 2)
    N, Wn = signal.cheb2ord(wp=norm_pass, ws=norm_stop, gpass=gpass, gstop=gstop, analog=0)
    b, a = signal.cheby2(N, gstop, Wn, "high")
    z = signal.lfilter(b, a, x)
    #return b, a, z
    return z

def hpf_lpf(df,HPF_fp,HPF_fs,LPF_fp,LPF_fs,RATE):
    N=len(df)
    #drop_idx=[15,16]
    dt=1.0/RATE
    t_mul=np.arange(N)*dt
    #print(dff)
    #plt.figure()
    for i,column in enumerate(df.columns):
        #df0_mul.plot()
        # df1=df.iloc[:,i]
        df1=df[column].copy().values
        # print(df[column])
        # z1_mul=hpf(RATE,2.0,1.0,df1)
        # z1_mul=hpf(RATE,2.0,1.0,df1)
        df1_temp=df1
        if(LPF_fp!=0 and LPF_fs!=0):
            df1_temp=lpf(RATE,LPF_fp,LPF_fs,df1_temp)

        df1_temp=hpf(RATE,HPF_fp,HPF_fs,df1_temp)
        # z1_mul=hpf(RATE,fp,fs,df1)
        #z1_mul=hpf(RATE,0.3,0.1,df1)
        df[column]=df1_temp
        # print(df[column])
        # print("")
    return df

def multi_pf(df,fp,fs):
    N=len(df)
    #drop_idx=[15,16]
    RATE=RATE_15ch
    dt=1.0/RATE
    t_mul=np.arange(N)*dt
    #print(dff)
    #plt.figure()
    for i,column in enumerate(df.columns):
        #df0_mul.plot()
        # df1=df.iloc[:,i]
        df1=df[column].copy().values
        # print(df[column])
        # z1_mul=hpf(RATE,2.0,1.0,df1)
        z1_mul=hpf(RATE,fp,fs,df1)
        #z1_mul=hpf(RATE,0.3,0.1,df1)
        df[column]=z1_mul
        # print(df[column])
        # print("")
    return df
# def multi_plot(xmin, xmax,ylim, df):
#     print(len(df))
#     XLIM0, XLIM1 = xmin, xmax
#     sample_rate=122
#     dt=1/sample_rate
#     plot_time = np.arange(len(df))*dt
#     # YLIM = 2 ** 15
#     YLIM =ylim
#     lines_sound = []
#     fig = plt.figure(num=None, figsize=(8, 6), dpi=100, facecolor='w', edgecolor='k')

#     ax = fig.add_subplot(4, 1, 1)
#     for i in range(0, 4):
#         temp_line, = ax.plot(plot_time, df[df.columns[i]], linewidth=0.5, linestyle="-", label=df.columns[i])
#         lines_sound.append(temp_line)
#     if(YLIM!=0):
#         plt.ylim(-1 * YLIM, YLIM)
#     plt.xlim(XLIM0, XLIM1)
#     plt.legend(loc='upper right')

#     ax = plt.subplot(4, 1, 2)
#     for i in range(4, 8):
#         temp_line, = ax.plot(plot_time, df[df.columns[i]], linewidth=0.5, linestyle="-", label=df.columns[i])
#         lines_sound.append(temp_line)
#     if(YLIM!=0):
#         plt.ylim(-1 * YLIM, YLIM)
#     plt.xlim(XLIM0, XLIM1)
#     plt.legend(loc='upper right')

#     ax = plt.subplot(4, 1, 3)
#     for i in range(8, 12):
#         temp_line, = ax.plot(plot_time, df[df.columns[i]], linewidth=0.5, linestyle="-", label=df.columns[i])
#         lines_sound.append(temp_line)
#     plt.xlim(XLIM0, XLIM1)
#     if(YLIM!=0):
#         plt.ylim(-1 * YLIM, YLIM)
#     plt.legend(loc='upper right')

#     ax = plt.subplot(4, 1, 4)
#     for i in range(12, 15):
#         temp_line, = ax.plot(plot_time, df[df.columns[i]], linewidth=0.5, linestyle="-", label=df.columns[i])
#         lines_sound.append(temp_line)
#     ax.set_xlabel("t(s)")
#     plt.xlim(XLIM0, XLIM1)
#     if(YLIM!=0):
#         plt.ylim(-1 * YLIM, YLIM)
#     plt.legend(loc='upper right')

#     plt.tight_layout()
#     # plt.show()

#     return 0.0
def linear_interpolation_resample_All(df,sampling_rate,new_sampling_rate):
        # 時系列データの時間情報を正規化
    df_new=pd.DataFrame(columns=df.columns)
    dt=1.0/sampling_rate
    time=np.arange(len(df))*dt
    time_normalized = (time - time[0]) / (time[-1] - time[0])

    for i in range(len(df.columns)):
        data=df[df.columns[i]].copy()
        data=data.to_numpy()

        # 線形補間関数を作成
        interpolator = interp1d(time_normalized, data)

        # 新しい時間情報を生成
        new_time_normalized = np.linspace(0, 1, int((time[-1] - time[0]) * new_sampling_rate))

        # 線形補間によるリサンプリング
        new_data = interpolator(new_time_normalized)

        # 新しい時間情報を元のスケールに戻す
        new_time = new_time_normalized * (time[-1] - time[0]) + time[0]
        df_new[df.columns[i]]=new_data
        print("{}_線形補間リサンプリング完了".format(df.columns[i]))
    print("old_datalength={}".format(len(df)))
    print("new_datalength={}".format(len(df_new)))

    return df_new

class ArrayComparator:
    def __init__(self, sc_15ch, sc_12ch,cut_min_max_range):
        self.sc_12ch=sc_12ch
        self.sc_15ch=sc_15ch
        self.cut_min_max_range=cut_min_max_range
    def cul_diff(self):
        time_12ch=self.sc_12ch[0][1:].to_numpy()
        time_15ch=self.sc_15ch[0][1:].to_numpy()
        diff_12ch=np.diff(self.sc_12ch[0])
        diff_15ch=np.diff(self.sc_15ch[0])
        return time_12ch,time_15ch,diff_12ch,diff_15ch

    def peak_diff_plot(self):
        time1,time2,diff1,diff2=self.cul_diff()
        print(diff1)
        print(diff2)
        # データ1のプロット
        plt.plot(time1, diff1, label='12ch',color="r")
        plt.scatter(time1, diff1, label='12ch',color="r")
        # データ2のプロット
        plt.plot(time2, diff2, label='15ch',color="b")
        plt.scatter(time2, diff2, label='15ch',color="b")

        # グラフのタイトルと凡例
        plt.title('compare of peak time diff')
        plt.legend()

        # 軸ラベルの設定
        plt.xlabel('time(s)')
        plt.ylabel('diff(s)')

        # グラフの表示
        plt.show()
        plt.close()

    def find_best_cut_time(self):
        cut_min_max_range=self.cut_min_max_range
        min_mse = float('inf')  # 初期値として最大値を設定
        best_index = 0
        # target=-15#後ろから3つを基準に平均二乗誤差でマッチするインデックスを探す。
        time1,time2,diff_12ch,diff_15ch=self.cul_diff()
        # target=-len(diff_12ch)
        target=0
        # target=5#後ろから3つを基準に平均二乗誤差でマッチするインデックスを探す。
        # diff_12ch=diff_12ch[target:]
        large_size=len(diff_15ch)
        small_size=len(diff_12ch)

        for i in range(large_size - small_size + 1):
            if(time2[i]-time1[target]<cut_min_max_range[0] or time2[i]-time1[target]>cut_min_max_range[1]):#始めの4.0秒は使わない
                print("continue "+str(i))
                continue

            current_subset =diff_15ch[i:i+small_size]
            mse = np.mean((current_subset - diff_12ch) ** 2)

            if mse < min_mse:
                min_mse = mse
                best_index = i
        print("12chの最初のピークのtime={}".format(time1[target]))
        print("15chの対応するピークのtime={}".format(time2[best_index]))
        cut_time=time2[best_index]-time1[target]
        print("差分={}".format(cut_time))
        return cut_time

    def peak_diff_plot_move(self,cut_time):
        cut_time=self.find_best_cut_time()
        time1,time2,diff1,diff2=self.cul_diff()
        time1_v2=time1+cut_time
        print(diff1)
        print(diff2)
        # データ1のプロット
        plt.plot(time1, diff1, label='12ch',color="r")
        plt.scatter(time1, diff1, label='12ch',color="r")
        # データ2のプロット
        plt.plot(time2, diff2, label='15ch',color="b")
        plt.scatter(time2, diff2, label='15ch',color="b")

        # データ1のプロットのcut_time分平行移動
        plt.plot(time1_v2, diff1, label='12ch_move',color="g")
        plt.scatter(time1_v2, diff1, label='12ch_move',color="g")
        # グラフのタイトルと凡例
        plt.title('compare of peak time diff')
        plt.legend()

        # 軸ラベルの設定
        plt.xlabel('time(s)')
        plt.ylabel('diff(s)')

        # グラフの表示
        plt.show()
        plt.close()

class MultiPlotter_both:
    def __init__(self,df12,df15,RATE12,RATE15):
        self.df12 = df12
        self.df15 = df15
        self.RATE12=RATE12
        self.RATE15=RATE15

    def multi_plot_12ch_15ch_with_sc(self, xmin, xmax, ylim,sc,ch,png_path):
        print(len(self.df12))
        line_width=1.0
        XLIM0, XLIM1 = xmin, xmax
        sample_rate = self.RATE12
        dt = 1 / sample_rate
        plot_time = np.arange(len(self.df12)) * dt
        YLIM = 0
        lines_sound = []
        fig = plt.figure(num=None, figsize=(8, 6), dpi=100, facecolor='w', edgecolor='k')

        ax = fig.add_subplot(4, 1, 1)
        plt.suptitle("target_ch_15={},png_path={}".format(ch,png_path))
        for i in range(0, 6):
            temp_line, = ax.plot(plot_time, self.df12[self.df12.columns[i]], linewidth=0.5, linestyle="-",
                                 label=self.df12.columns[i])
            lines_sound.append(temp_line)
        for j in range(len(sc[0])):
            # ax.axvline(x=sc[0][j],color='red',linewidth=-0.1)
            # ax.axvline(x=sc[0][j],color='black',linewidth=0.5,linestyle='--')
            ax.axvline(x=sc[0][j],color='black',linewidth=line_width,linestyle='--')
        if (YLIM != 0):
            plt.ylim(-1 * YLIM, YLIM)
        plt.xlim(XLIM0, XLIM1)
        # plt.legend(loc='upper right',fontsize=5)
        plt.legend(loc='center left',fontsize=10,ncol=2,bbox_to_anchor=(1.,.5))

        ax = plt.subplot(4, 1, 2)
        for i in range(6, 12):
            temp_line, = ax.plot(plot_time, self.df12[self.df12.columns[i]], linewidth=0.5, linestyle="-",
                                 label=self.df12.columns[i])
            lines_sound.append(temp_line)
        for j in range(len(sc[0])):
            # ax.axvline(x=sc[0][j],color='red',linewidth=-0.1)
            ax.axvline(x=sc[0][j],color='black',linewidth=line_width,linestyle='--')
        if (YLIM != 0):
            plt.ylim(-1 * YLIM, YLIM)
        plt.xlim(XLIM0, XLIM1)
        # plt.legend(loc='upper right',fontsize=5)
        plt.legend(loc='center left',fontsize=10,ncol=2,bbox_to_anchor=(1.,.5))

        sample_rate = self.RATE15
        dt = 1 / sample_rate
        plot_time = np.arange(len(self.df15)) * dt
        YLIM = ylim
        lines_sound = []

        ax = plt.subplot(4, 1, 3)
        for i in range(0, 8):
            temp_line, = ax.plot(plot_time, self.df15[self.df15.columns[i]], linewidth=0.5, linestyle="-",
                                 label=self.df15.columns[i])
            lines_sound.append(temp_line)
        for j in range(len(sc[0])):
            # ax.axvline(x=sc[0][j],color='red',linewidth=-0.1)
            # ax.axvline(x=sc[0][j],color='black',linewidth=0.5,linestyle='--')
            ax.axvline(x=sc[0][j],color='black',linewidth=line_width,linestyle='--')

        if (YLIM != 0):
            plt.ylim(-1 * YLIM, YLIM)
        plt.xlim(XLIM0, XLIM1)
        # plt.legend(loc='upper right',fontsize=10,ncol=2)
        plt.legend(loc='center left',fontsize=10,ncol=2,bbox_to_anchor=(1.,.5))

        ax = plt.subplot(4, 1, 4)
        for i in range(8, 15):
            temp_line, = ax.plot(plot_time, self.df15[self.df15.columns[i]], linewidth=0.5, linestyle="-",
                                 label=self.df15.columns[i])
            lines_sound.append(temp_line)
        for j in range(len(sc[0])):
            # ax.axvline(x=sc[0][j],color='red',linewidth=-0.1)
            # ax.axvline(x=sc[0][j],color='black',linewidth=0.5,linestyle='--')
            ax.axvline(x=sc[0][j],color='black',linewidth=line_width,linestyle='--')
        if (YLIM != 0):
            plt.ylim(-1 * YLIM, YLIM)
        plt.xlim(XLIM0, XLIM1)
        # plt.legend(loc='upper right')
        plt.legend(loc='center left',fontsize=10,ncol=2,bbox_to_anchor=(1.,.5))


        ax.set_xlabel("t(s)")
        plt.xlim(XLIM0, XLIM1)
        if (YLIM != 0):
            plt.ylim(-1 * YLIM, YLIM)
        # plt.legend(loc='upper right',fontsize=5)
        plt.legend(loc='center left',fontsize=10,ncol=2,bbox_to_anchor=(1.,.5))

        plt.tight_layout()
        plt.savefig(png_path)
        # plt.show()

class MultiPlotter:
    def __init__(self, df,RATE):
        self.df = df
        self.RATE=RATE
    def multi_plot(self,xmin,xmax,ylim):
        if(len(self.df.columns)==15):
            self.multi_plot_15ch(xmin,xmax,ylim)
        if(len(self.df.columns)==12):
            self.multi_plot_12ch(xmin,xmax,ylim)
        if(len(self.df.columns)==16):
            self.multi_plot_16ch(xmin,xmax,ylim)

    def plot_all_channels(self, xmin, xmax, ylim):
        print(len(self.df))
        XLIM0, XLIM1 = xmin, xmax
        sample_rate = self.RATE
        dt = 1 / sample_rate
        plot_time = np.arange(len(self.df)) * dt
        YLIM = ylim
        lines_sound = []
        fig, ax = plt.subplots(figsize=(8, 6), dpi=100, facecolor='w', edgecolor='k')
        plt.rcParams["font.family"] = "Arial"   # 使用するフォント
        # plt.rcParams["font.size"] = 20
        for i in range(15):
            ax.plot(plot_time, self.df[self.df.columns[i]], linewidth=2.0, linestyle="-",
                    label=self.df.columns[i])

        ax.set_xlim(XLIM0, XLIM1)
        if YLIM != 0:
            ax.set_ylim(-YLIM, YLIM)
        ax.legend(loc='upper right')
        # ax.set_xlabel("t(s)")
        plt.legend(fontsize=20,ncol=2)
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        plt.xlabel("t(s)",fontsize=30)
        plt.tight_layout()
        # plt.show()

        return 0.0

    def multi_plot_12ch(self, xmin, xmax, ylim):
        print(len(self.df))
        XLIM0, XLIM1 = xmin, xmax
        sample_rate = self.RATE
        dt = 1 / sample_rate
        plot_time = np.arange(len(self.df)) * dt
        YLIM = ylim
        lines_sound = []
        fig = plt.figure(num=None, figsize=(8, 6), dpi=100, facecolor='w', edgecolor='k')

        ax = fig.add_subplot(4, 1, 1)
        for i in range(0, 3):
            temp_line, = ax.plot(plot_time, self.df[self.df.columns[i]], linewidth=0.5, linestyle="-",
                                 label=self.df.columns[i])
            lines_sound.append(temp_line)
        if (YLIM != 0):
            plt.ylim(-1 * YLIM, YLIM)
        plt.xlim(XLIM0, XLIM1)
        plt.legend(loc='upper right')

        ax = plt.subplot(4, 1, 2)
        for i in range(3, 6):
            temp_line, = ax.plot(plot_time, self.df[self.df.columns[i]], linewidth=0.5, linestyle="-",
                                 label=self.df.columns[i])
            lines_sound.append(temp_line)
        if (YLIM != 0):
            plt.ylim(-1 * YLIM, YLIM)
        plt.xlim(XLIM0, XLIM1)
        plt.legend(loc='upper right')

        ax = plt.subplot(4, 1, 3)
        for i in range(6, 9):
            temp_line, = ax.plot(plot_time, self.df[self.df.columns[i]], linewidth=0.5, linestyle="-",
                                 label=self.df.columns[i])
            lines_sound.append(temp_line)
        plt.xlim(XLIM0, XLIM1)
        if (YLIM != 0):
            plt.ylim(-1 * YLIM, YLIM)
        plt.legend(loc='upper right')

        ax = plt.subplot(4, 1, 4)
        for i in range(9,12):
            temp_line, = ax.plot(plot_time, self.df[self.df.columns[i]], linewidth=0.5, linestyle="-",
                                 label=self.df.columns[i])
            lines_sound.append(temp_line)
        ax.set_xlabel("t(s)")
        plt.xlim(XLIM0, XLIM1)
        if (YLIM != 0):
            plt.ylim(-1 * YLIM, YLIM)
        plt.legend(loc='upper right')

        plt.tight_layout()
        # plt.show()

        return 0.0


    def multi_plot_12ch_with_sc(self, xmin, xmax, ylim,sc):
        print(len(self.df))
        XLIM0, XLIM1 = xmin, xmax
        sample_rate = self.RATE
        dt = 1 / sample_rate
        plot_time = np.arange(len(self.df)) * dt
        YLIM = ylim
        lines_sound = []
        fig = plt.figure(num=None, figsize=(8, 6), dpi=100, facecolor='w', edgecolor='k')

        ax = fig.add_subplot(4, 1, 1)
        for i in range(0, 3):
            temp_line, = ax.plot(plot_time, self.df[self.df.columns[i]], linewidth=0.5, linestyle="-",
                                 label=self.df.columns[i])
            lines_sound.append(temp_line)
        for j in range(len(sc[0])):
            # ax.axvline(x=sc[0][j],color='red',linewidth=-0.1)
            ax.axvline(x=sc[0][j],color='black',linewidth=0.5,linestyle='--')
        if (YLIM != 0):
            plt.ylim(-1 * YLIM, YLIM)
        plt.xlim(XLIM0, XLIM1)
        plt.legend(loc='upper right')

        ax = plt.subplot(4, 1, 2)
        for i in range(3, 6):
            temp_line, = ax.plot(plot_time, self.df[self.df.columns[i]], linewidth=0.5, linestyle="-",
                                 label=self.df.columns[i])
            lines_sound.append(temp_line)
        for j in range(len(sc[0])):
            # ax.axvline(x=sc[0][j],color='red',linewidth=-0.1)
            ax.axvline(x=sc[0][j],color='black',linewidth=0.5,linestyle='--')
        if (YLIM != 0):
            plt.ylim(-1 * YLIM, YLIM)
        plt.xlim(XLIM0, XLIM1)
        plt.legend(loc='upper right')

        ax = plt.subplot(4, 1, 3)
        for i in range(6, 9):
            temp_line, = ax.plot(plot_time, self.df[self.df.columns[i]], linewidth=0.5, linestyle="-",
                                 label=self.df.columns[i])
            lines_sound.append(temp_line)
        for j in range(len(sc[0])):
            # ax.axvline(x=sc[0][j],color='red',linewidth=-0.1)
            ax.axvline(x=sc[0][j],color='black',linewidth=0.5,linestyle='--')
        plt.xlim(XLIM0, XLIM1)
        if (YLIM != 0):
            plt.ylim(-1 * YLIM, YLIM)
        plt.legend(loc='upper right')

        ax = plt.subplot(4, 1, 4)
        for i in range(9,12):
            temp_line, = ax.plot(plot_time, self.df[self.df.columns[i]], linewidth=0.5, linestyle="-",
                                 label=self.df.columns[i])
            lines_sound.append(temp_line)
        for j in range(len(sc[0])):
            # ax.axvline(x=sc[0][j],color='red',linewidth=-0.1)
            ax.axvline(x=sc[0][j],color='black',linewidth=0.5,linestyle='--')
        ax.set_xlabel("t(s)")
        plt.xlim(XLIM0, XLIM1)
        if (YLIM != 0):
            plt.ylim(-1 * YLIM, YLIM)
        plt.legend(loc='upper right')

        plt.tight_layout()
        # plt.show()

        return 0.0
    def multi_plot_15ch_with_sc(self, xmin, xmax, ylim,sc):
        print(len(self.df))
        XLIM0, XLIM1 = xmin, xmax
        sample_rate = self.RATE
        dt = 1 / sample_rate
        plot_time = np.arange(len(self.df)) * dt
        YLIM = ylim
        lines_sound = []
        fig = plt.figure(num=None, figsize=(8, 6), dpi=100, facecolor='w', edgecolor='k')

        ax = fig.add_subplot(4, 1, 1)
        for i in range(0, 4):
            temp_line, = ax.plot(plot_time, self.df[self.df.columns[i]], linewidth=0.5, linestyle="-",
                                 label=self.df.columns[i])
            lines_sound.append(temp_line)
        for j in range(len(sc[0])):
            # ax.axvline(x=sc[0][j],color='red',linewidth=-0.1)
            ax.axvline(x=sc[0][j],color='black',linewidth=0.5,linestyle='--')

        if (YLIM != 0):
            plt.ylim(-1 * YLIM, YLIM)
        plt.xlim(XLIM0, XLIM1)
        plt.legend(loc='upper right')

        ax = plt.subplot(4, 1, 2)
        for i in range(4, 8):
            temp_line, = ax.plot(plot_time, self.df[self.df.columns[i]], linewidth=0.5, linestyle="-",
                                 label=self.df.columns[i])
            lines_sound.append(temp_line)
        for j in range(len(sc[0])):
            # ax.axvline(x=sc[0][j],color='red',linewidth=-0.1)
            ax.axvline(x=sc[0][j],color='black',linewidth=0.5,linestyle='--')
        if (YLIM != 0):
            plt.ylim(-1 * YLIM, YLIM)
        plt.xlim(XLIM0, XLIM1)
        plt.legend(loc='upper right')

        ax = plt.subplot(4, 1, 3)
        for i in range(8, 12):
            temp_line, = ax.plot(plot_time, self.df[self.df.columns[i]], linewidth=0.5, linestyle="-",
                                 label=self.df.columns[i])
            lines_sound.append(temp_line)
        for j in range(len(sc[0])):
            # ax.axvline(x=sc[0][j],color='red',linewidth=-0.1)
            ax.axvline(x=sc[0][j],color='black',linewidth=0.5,linestyle='--')
        plt.xlim(XLIM0, XLIM1)
        if (YLIM != 0):
            plt.ylim(-1 * YLIM, YLIM)
        plt.legend(loc='upper right')

        ax = plt.subplot(4, 1, 4)
        for i in range(12, 15):
            temp_line, = ax.plot(plot_time, self.df[self.df.columns[i]], linewidth=0.5, linestyle="-",
                                 label=self.df.columns[i])
            lines_sound.append(temp_line)
        for j in range(len(sc[0])):
            # ax.axvline(x=sc[0][j],color='red',linewidth=-0.1)
            ax.axvline(x=sc[0][j],color='black',linewidth=0.5,linestyle='--')
        ax.set_xlabel("t(s)")
        plt.xlim(XLIM0, XLIM1)
        if (YLIM != 0):
            plt.ylim(-1 * YLIM, YLIM)
        plt.legend(loc='upper right')

        plt.tight_layout()
        # plt.show()
        return 0.0
    def multi_plot_16ch(self, xmin, xmax, ylim):
        print(len(self.df))
        XLIM0, XLIM1 = xmin, xmax
        sample_rate = self.RATE
        dt = 1 / sample_rate
        plot_time = np.arange(len(self.df)) * dt
        YLIM = ylim
        lines_sound = []
        fig = plt.figure(num=None, figsize=(8, 6), dpi=100, facecolor='w', edgecolor='k')

        ax = fig.add_subplot(4, 1, 1)
        for i in range(0, 4):
            temp_line, = ax.plot(plot_time, self.df[self.df.columns[i]], linewidth=0.5, linestyle="-",
                                 label=self.df.columns[i])
            lines_sound.append(temp_line)
        if (YLIM != 0):
            plt.ylim(-1 * YLIM, YLIM)
        plt.xlim(XLIM0, XLIM1)
        plt.legend(loc='upper right')

        ax = plt.subplot(4, 1, 2)
        for i in range(4, 8):
            temp_line, = ax.plot(plot_time, self.df[self.df.columns[i]], linewidth=0.5, linestyle="-",
                                 label=self.df.columns[i])
            lines_sound.append(temp_line)
        if (YLIM != 0):
            plt.ylim(-1 * YLIM, YLIM)
        plt.xlim(XLIM0, XLIM1)
        plt.legend(loc='upper right')

        ax = plt.subplot(4, 1, 3)
        for i in range(8, 12):
            temp_line, = ax.plot(plot_time, self.df[self.df.columns[i]], linewidth=0.5, linestyle="-",
                                 label=self.df.columns[i])
            lines_sound.append(temp_line)
        plt.xlim(XLIM0, XLIM1)
        if (YLIM != 0):
            plt.ylim(-1 * YLIM, YLIM)
        plt.legend(loc='upper right')

        ax = plt.subplot(4, 1, 4)
        for i in range(12, 16):
            temp_line, = ax.plot(plot_time, self.df[self.df.columns[i]], linewidth=0.5, linestyle="-",
                                 label=self.df.columns[i])
            lines_sound.append(temp_line)
        ax.set_xlabel("t(s)")
        plt.xlim(XLIM0, XLIM1)
        if (YLIM != 0):
            plt.ylim(-1 * YLIM, YLIM)
        plt.legend(loc='upper right')

        plt.tight_layout()
        # plt.show()
        return 0.0

    def multi_plot_15ch(self, xmin, xmax, ylim):
        print(len(self.df))
        XLIM0, XLIM1 = xmin, xmax
        sample_rate = self.RATE
        dt = 1 / sample_rate
        plot_time = np.arange(len(self.df)) * dt
        YLIM = ylim
        lines_sound = []
        fig = plt.figure(num=None, figsize=(8, 6), dpi=100, facecolor='w', edgecolor='k')

        ax = fig.add_subplot(4, 1, 1)
        for i in range(0, 4):
            temp_line, = ax.plot(plot_time, self.df[self.df.columns[i]], linewidth=0.5, linestyle="-",
                                 label=self.df.columns[i])
            lines_sound.append(temp_line)
        if (YLIM != 0):
            plt.ylim(-1 * YLIM, YLIM)
        plt.xlim(XLIM0, XLIM1)
        plt.legend(loc='upper right')

        ax = plt.subplot(4, 1, 2)
        for i in range(4, 8):
            temp_line, = ax.plot(plot_time, self.df[self.df.columns[i]], linewidth=0.5, linestyle="-",
                                 label=self.df.columns[i])
            lines_sound.append(temp_line)
        if (YLIM != 0):
            plt.ylim(-1 * YLIM, YLIM)
        plt.xlim(XLIM0, XLIM1)
        plt.legend(loc='upper right')

        ax = plt.subplot(4, 1, 3)
        for i in range(8, 12):
            temp_line, = ax.plot(plot_time, self.df[self.df.columns[i]], linewidth=0.5, linestyle="-",
                                 label=self.df.columns[i])
            lines_sound.append(temp_line)
        plt.xlim(XLIM0, XLIM1)
        if (YLIM != 0):
            plt.ylim(-1 * YLIM, YLIM)
        plt.legend(loc='upper right')

        ax = plt.subplot(4, 1, 4)
        for i in range(12, 15):
            temp_line, = ax.plot(plot_time, self.df[self.df.columns[i]], linewidth=0.5, linestyle="-",
                                 label=self.df.columns[i])
            lines_sound.append(temp_line)
        ax.set_xlabel("t(s)")
        plt.xlim(XLIM0, XLIM1)
        if (YLIM != 0):
            plt.ylim(-1 * YLIM, YLIM)
        plt.legend(loc='upper right')

        plt.tight_layout()
        # plt.show()
        return 0.0

class CSVReader_12ch:
    def __init__(self, directory):
        self.directory = directory

    def search_files(self):
        files_found = []
        for filename in os.listdir(self.directory):
            if filename.startswith("12ch") and filename.endswith(".csv"):
                files_found.append(filename)
        return files_found

    def read_csv_file(self, filename):
        file_path = os.path.join(self.directory, filename)
        df = pd.read_csv(file_path)
        print(f"ファイル {filename} を読み込みました。")
        # 読み込んだデータフレームの操作などを行う
        # ...
        print(df)
        return df

    def header_make(self,df):
        df=df.drop(columns=[15,16])
        df = df.rename(columns=lambda x: 'ch_' + str(x+1))
        return df

    def process_files(self):
        files_found = self.search_files()
        if len(files_found) > 0:
            df=self.read_csv_file(files_found[0])
            # df=self.header_make(df)
            print(df)
        else:
            print("指定した条件のCSVファイルは存在しません。")
            print("12ch")
            exit()
        return df

class CSVReader_16ch:
    def __init__(self, directory):
        self.directory = directory

    def search_files(self):
        files_found = []
        for filename in os.listdir(self.directory):
            if filename.startswith("db") and filename.endswith(".csv"):
                files_found.append(filename)
        return files_found

    def read_csv_file(self, filename):
        file_path = os.path.join(self.directory, filename)
        df = pd.read_csv(file_path,header=None)
        print(f"ファイル {filename} を読み込みました。")
        # 読み込んだデータフレームの操作などを行う
        # ...
        print(df)
        return df

    def header_make(self,df):
        df=df.drop(columns=[16,17])
        df = df.rename(columns=lambda x: 'ch_' + str(x+1))
        return df

    def process_files(self):
        files_found = self.search_files()
        if len(files_found) > 0:
            df=self.read_csv_file(files_found[0])
            df=self.header_make(df)
            print(df)
        else:
            print("指定した条件のCSVファイルは存在しません。")
            print("15ch")
            exit()
        return df

# class CSVReader_15ch:
#     def __init__(self, directory):
#         self.directory = directory

#     def search_files(self):
#         files_found = []
#         for filename in os.listdir(self.directory):
#             if filename.startswith("db") and filename.endswith(".csv"):
#                 files_found.append(filename)
#         return files_found

#     def read_csv_file(self, filename):
#         file_path = os.path.join(self.directory, filename)
#         df = pd.read_csv(file_path,header=None)
#         print(f"ファイル {filename} を読み込みました。")
#         # 読み込んだデータフレームの操作などを行う
#         # ...
#         print(df)
#         return df

#     def header_make(self,df):
#         df=df.drop(columns=[15,16])
#         df = df.rename(columns=lambda x: 'ch_' + str(x+1))
#         return df

#     def process_files(self):
#         files_found = self.search_files()
#         if len(files_found) > 0:
#             df=self.read_csv_file(files_found[0])
#             df=self.header_make(df)
#             print(df)
#         else:
#             print("指定した条件のCSVファイルは存在しません。")
#             print("15ch")
#             exit()
#         return df

class AutoIntegerFileHandler:
    def __init__(self, filename):
       self.filename = filename
    def check_file(self):
        path=self.filename
        if os.path.exists(path):
            if os.path.isfile(path):
                print(f"The path '{path}' exists and it is a file.")
                if(input("ok? y or n")=="y"):
                    return True
                # return True
        else:
            print(f"The path '{path}' does not exist.")
        return False

    def input_integer(self,RATE,cut_time):
        time=cut_time
        print("同期する時間は{}(s)".format(time))
        integer=int(RATE*time)
        print("integer={}".format(integer))
        return integer

    def write_integer(self,RATE,cut_time,target_15ch,reverse,target_12ch,cut_min_max_range):
        integer=self.input_integer(RATE,cut_time)
        # with open(self.filename, 'w') as file:
        #    file.write(str(integer)+'\n')
        # #    file.write("TARGET_CHANNEL_15ch="+str(self.ch))
        #    file.write(str(target_15ch)+'\n')
        #    file.write(str(target_12ch)+'\n')
        #    file.write(str(cut_min_max_range[0])+'\n')
        #    file.write(str(cut_min_max_range[1])+'\n')

        data = {
            "INDEX": str(integer),
            "TARGET_CH_15ch":str(target_15ch),
            "REVERSE": reverse,#ピーク検出するときにTARGET_15chの波形を反転させるかどうかを決める。
            "TARGET_CH_12ch":str(target_12ch),
            "START_TIME":str(cut_min_max_range[0]) ,#書いてるだけで別に同期ファイルがある場合は使わないデータ
            "END_TIME":str(cut_min_max_range[1]) , #書いてるだけで別に同期ファイルがある場合は使わないデータ
        }

        column_order = ["INDEX", "TARGET_CH_15ch", "REVERSE", "TARGET_CH_12ch", "START_TIME", "END_TIME"]
        with open(self.filename, mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=column_order)

            # カラム名を書き込む
            writer.writeheader()

            # データを書き込む
            writer.writerow(data)

    def read_integer(self):
        # CSVファイルを読み込む
        with open(self.filename, 'r') as file:
            reader = csv.DictReader(file)

            # カラム名を取得
            columns = reader.fieldnames
            print(columns)
            data_list=[]
            data_dict={}

            # データを読み込み、表示
            for i ,row in enumerate(reader):
                print(row)
                data_dict=row

                       # データを辞書型に格納
            # data_dict = {row["INDEX"]: row for row in data_list}
            print(data_dict)

            # print("CSVファイルの内容を辞書に格納しました。")
            # input("posseeeeeeee")
            return int(data_dict["INDEX"]),data_dict["TARGET_CH_15ch"],data_dict["REVERSE"],data_dict["TARGET_CH_12ch"]


class IntegerFileHandler:
    def __init__(self, filename):
       self.filename = filename
    def check_file(self):
        path=self.filename
        if os.path.exists(path):
            if os.path.isfile(path):
                print(f"The path '{path}' exists and it is a file.")
                if(input("ok? y or n")=="y"):
                   return True
        else:
            print(f"The path '{path}' does not exist.")
        return False

    def input_integer(self,RATE):
        time=float(input("同期する時間を入力してください"))
        integer=int(RATE*time)
        print("integer={}".format(integer))
        return integer

    def write_integer(self,RATE):
       integer=self.input_integer(RATE)
       with open(self.filename, 'w') as file:
           file.write(str(integer))
    def read_integer(self):
       with open(self.filename, 'r') as file:
           content = file.read()
           print(content)
           try:
               value = int(content)
               return value
           except ValueError:
               print("Error: The file does not contain a valid integer.")
               return None

def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print("ディレクトリ {} を作成しました。".format(directory_path))
    else:
        print("ディレクトリ {} は既に存在しています。".format(directory_path))

def write_text_file(data, file_path):
    with open(file_path, 'w') as file:
        for line in data:
            file.write(str(line) + '\n')

class HeartbeatCutterRandom:
    def __init__(self, con_data,time_length):
        self.con_data = con_data
        self.range=int(time_length*RATE/2)
    def output_csv(self,file_path,file_name,data):
        dt=1.0/RATE
        time_tmp=np.arange(len(data))*dt
        # time_data=pd.DataFrame(time,column="Time")
        time = pd.DataFrame()
        time['Time']=time_tmp
        print(time)
        data=data.reset_index(drop=True)
        data_out=pd.concat([time,data],axis=1)
        print(data_out)
        data_out.to_csv(file_path+'/'+file_name,index=None)

    def cut_heartbeats(self, center_idxs,file_path,ch,cut_min_max_range):
        create_directory(file_path)
        data=[]
        data.append(ch)
        data.append(cut_min_max_range[0])
        data.append(cut_min_max_range[1])
        write_text_file(data,file_path+"/"+"TARGET_CHANNEL.txt")

        for i, center_idx in enumerate(center_idxs):
            data = self.con_data[center_idx - self.range:center_idx + self.range].copy()
            print("{}番目の心拍切り出し".format(i + 1))
            # print(data)
            file_name='dataset_{}.csv'.format(str(i).zfill(3))
            self.output_csv(file_name=file_name,file_path=file_path,data=data.copy())
# サンプルデータ
class HeartbeatCutter_prt:
    def __init__(self, con_data,time_length,prt_eles,args):
        self.con_data = con_data
        self.range=int(time_length*RATE/2)
        self.time_length=time_length
        self.prt_eles=prt_eles
        self.name=args.name
        self.pos=args.pos

    def output_csv_eles(self,file_path,file_name,p_onset,t_offset,p_offset,t_onset,p_peak,q_peak,s_peak,t_peak):
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
        data_out=pd.DataFrame(data)
        data_out.to_csv(file_path+'/'+file_name,index=None)
    def output_csv_ponset_toffset(self,file_path,file_name,p_onset,t_offset,p_offset,t_onset):
        data = {
            "p_onset": [p_onset],
            "t_offset": [t_offset],
            "p_offset": [p_offset],
            "t_onset": [t_onset]
        }
        data_out=pd.DataFrame(data)
        data_out.to_csv(file_path+'/'+file_name,index=None)

    def output_csv(self,file_path,file_name,data):
        dt=1.0/RATE
        time_tmp=np.arange(len(data))*dt
        # time_data=pd.DataFrame(time,column="Time")
        time = pd.DataFrame()
        time['Time']=time_tmp
        print(time)
        data=data.reset_index(drop=True)
        data_out=pd.concat([time,data],axis=1)
        print(data_out)
        data_out.to_csv(file_path+'/'+file_name,index=None)

    def cut_heartbeats(self,file_path,ch,cut_min_max_range,args):
        center_idxs=self.prt_eles[:,1]
        p_indexs_onsets=self.prt_eles[:,0]
        t_indexs_offsets=self.prt_eles[:,2]
        p_indexs_offsets=self.prt_eles[:,3]
        t_indexs_onsets=self.prt_eles[:,4]
        p_indexs_peaks=self.prt_eles[:,5]
        q_indexs_peaks=self.prt_eles[:,6]
        s_indexs_peaks=self.prt_eles[:,7]
        t_indexs_peaks=self.prt_eles[:,8]
        create_directory(file_path)
        data=[]
        datas=[]
        data.append(ch)
        data.append(cut_min_max_range[0])
        data.append(cut_min_max_range[1])
        write_text_file(data,file_path+"/"+"TARGET_CHANNEL.txt")
        print(file_path)

        pt_info=[]
        for i, center_idx in enumerate(center_idxs):
            data = self.con_data[center_idx - self.range:center_idx + self.range].copy()

            p_onset=p_indexs_onsets[i]- center_idx + self.range#prt_ele[0]はponsetの座標
            t_offset=t_indexs_offsets[i]- center_idx + self.range#prt_ele[2]はtoffsetの座標
            p_offset=p_indexs_offsets[i]- center_idx + self.range#prt_ele[0]はponsetの座標
            t_onset=t_indexs_onsets[i]- center_idx + self.range#prt_ele[2]はtoffsetの座標
            p_peak=p_indexs_peaks[i]- center_idx + self.range#prt_ele[2]はtoffsetの座標
            q_peak=q_indexs_peaks[i]- center_idx + self.range#prt_ele[2]はtoffsetの座標
            s_peak=s_indexs_peaks[i]- center_idx + self.range#prt_ele[2]はtoffsetの座標
            t_peak=t_indexs_peaks[i]- center_idx + self.range#prt_ele[2]はtoffsetの座標

            print("{}番目の心拍切り出し".format(i + 1))
            # print(data)
            file_name='dataset_{}.csv'.format(str(i).zfill(3))
            self.output_csv(file_name=file_name,file_path=file_path,data=data.copy())
            file_name_pt='ponset_toffsett_{}.csv'.format(str(i).zfill(3))
            # self.output_csv_ponset_toffset(file_name=file_name_pt,file_path=file_path,p_onset=p_onset,t_offset=t_offset,p_offset=p_offset,t_onset=t_onset)
            self.output_csv_eles(file_name=file_name_pt,file_path=file_path,p_onset=p_onset,t_offset=t_offset,p_offset=p_offset,t_onset=t_onset,p_peak=p_peak,q_peak=q_peak,s_peak=s_peak,t_peak=t_peak)
            pt_time=(t_offset-p_onset)/RATE
            t_onset_time=(t_onset)/RATE
            t_offset_time=(t_offset)/RATE
            r_offset=210
            r_t_index=(t_onset-r_offset)/RATE
            L_weight=1.3#水増しで伸ばす最大の倍率
            check_value=(400-t_offset)/(t_onset-210)/(L_weight-1)#ST部を引き延ばす水増しをしても大丈夫か確かめる指標。1以上でOK

            pt_info_temp=[self.name,self.pos,str(i).zfill(3),pt_time,t_onset_time,t_offset_time,r_t_index,check_value,L_weight]
            pt_info.append(pt_info_temp)
        # input(pt_info)
        data_num_info=[[self.name,self.pos,len(center_idxs)]]
        # append_to_csv(filename="Dataset/pqrst2/pt_time_all_{}s.csv".format(str(self.time_length)),data=pt_info)
        # dataset_num_to_csv(filename="Dataset/pqrst2/dataset_num_{}s.csv".format(str(self.time_length)),data=data_num_info)
        append_to_csv(filename="Dataset/pqrst_nkmodule_since{}_{}/pt_time_all_{}s.csv".format(DATASET_MADE_DATE,args.peak_method,str(self.time_length)),data=pt_info)
        dataset_num_to_csv(filename="Dataset/pqrst_nkmodule_since{}_{}/dataset_num_{}s.csv".format(DATASET_MADE_DATE,args.peak_method,str(self.time_length)),data=data_num_info)


class HeartbeatCutter:
    def __init__(self, con_data,time_length):
        self.con_data = con_data
        self.range=int(time_length*RATE/2)
    def output_csv(self,file_path,file_name,data):
        dt=1.0/RATE
        time_tmp=np.arange(len(data))*dt
        # time_data=pd.DataFrame(time,column="Time")
        time = pd.DataFrame()
        time['Time']=time_tmp
        print(time)
        data=data.reset_index(drop=True)
        data_out=pd.concat([time,data],axis=1)
        print(data_out)
        data_out.to_csv(file_path+'/'+file_name,index=None)

    def cut_heartbeats(self, center_idxs,file_path,ch,cut_min_max_range):
        create_directory(file_path)
        data=[]
        datas=[]
        data.append(ch)
        data.append(cut_min_max_range[0])
        data.append(cut_min_max_range[1])
        write_text_file(data,file_path+"/"+"TARGET_CHANNEL.txt")

        for i, center_idx in enumerate(center_idxs):
            data = self.con_data[center_idx - self.range:center_idx + self.range].copy()
            print("{}番目の心拍切り出し".format(i + 1))
            # print(data)
            file_name='dataset_{}.csv'.format(str(i).zfill(3))
            self.output_csv(file_name=file_name,file_path=file_path,data=data.copy())
            # df_data=data[:].copy()
            # np_data=df_data.value.T
            # datas.append(np_data)
            # labels='dataset_{}'.format(i)
        # x_list=np.concatenate(datas).tolist()
        # data_plot_after_splitting2(y_list,x_list,len(y_list),target_name=str(ch),figtitle="All_data",label_list=label_2)

def find_start_index(sc,time_length):
    print(sc)
    data_time_length=time_length
    idxs=[]
    times=sc.tolist()
    # print(times)
    for i in range(len(sc)):
        if(times[i]>(data_time_length/2.0)):
            if(times[i]<TIME-data_time_length/2.0):
                idxs.append(i)
    # print(idxs)
    make_data_start_indexs=[]
    for i in range(len(idxs)):
        make_data_start_indexs.append(int(times[idxs[i]]*RATE))
    # print(make_data_start_indexs)
    return make_data_start_indexs#整数の配列

def validate_integer_input():
    try:
        value = int(input("整数を入力してください（1から15までの範囲）: "))
        if value < 0 or value > 16:
            raise ValueError("入力された整数は範囲外です。")
        else:
            ch="ch_"+str(value)
            return ch
    except ValueError as e:
        print("エラー:", e)
        return None

def PQRST_plot_one(ecg,sampling_rate,header):
    ecg_signal=ecg
    ecg_signal=nk.ecg_clean(ecg_signal,sampling_rate=500,method='neurokit')
    print(ecg_signal)
    _, rpeaks = nk.ecg_peaks(ecg_signal, sampling_rate)
    _, waves_peak = nk.ecg_delineate(ecg_signal,
                                     rpeaks,
                                     sampling_rate=sampling_rate,
                                     method="peak",
                                     show=True,
                                     show_type='all')
    plt.title(args.name+'_'+args.pos+"_12ch_"+header)
    compare_path='./0_packetloss_data/pqrst'
    create_directory_if_not_exists(compare_path)
    # plt.savefig(compare_path+'/12ch_A2_'+args.type+'.png')
    plt.savefig(compare_path+'/12ch_'+header+'_'+args.type+'.png')
    # print(waves_peak)
    # input()
    plt.show()
    if(DEBUG_PLOT==True):
        plt.show()
    plt.close()

def PQRST_plot(ecg,sampling_rate,headers):
    for i in range(ecg.shape[0]):
        ecg_signal=ecg[i]
        print(ecg_signal)
        _, rpeaks = nk.ecg_peaks(ecg_signal, sampling_rate)
        _, waves_peak = nk.ecg_delineate(ecg_signal,
                                         rpeaks,
                                         sampling_rate=sampling_rate,
                                         method="peak",
                                         show=True,
                                         show_type='all')
        plt.title(headers[i])
        plt.show()

def PQRST_plot_grid_15ch(ecg_list, sampling_rate, headers,args):
    """
    4x3のグリッドに12個のECG信号のPQRST波をプロットする関数

    Parameters:
        ecg_list (numpy.ndarray): プロットするECG信号の2次元配列（各行が1つのECG信号を表す）
        sampling_rate (int): サンプリングレート（Hz）
        headers (list of strings): 各プロットのタイトルのリスト（要素は文字列）

    Raises:
        ValueError: ecg_listの行数が12でない場合

    Returns:
        None
    """
    if ecg_list.shape[0] != 15 or len(headers) != 15:
        raise ValueError("ecg_list must contain exactly 15 rows, and headers must contain exactly 15 elements.")

    fig, axes = plt.subplots(5, 3, figsize=(15, 12))

    for i, (ecg_signal, title) in enumerate(zip(ecg_list, headers)):
        ecg_signal=nk.ecg_clean(ecg_signal,sampling_rate=500,method='neurokit')
        # print(ecg_signal)
        rpeaks = nk.ecg_peaks(ecg_signal, sampling_rate)[1]['ECG_R_Peaks']  # R波の位置を取得
        waves_peak_df, waves_peak_dict = nk.ecg_delineate(ecg_signal,
                                                         rpeaks,
                                                         sampling_rate=sampling_rate,
                                                         method="peak",
                                                         show=False,
                                                         show_type='all')

        ax = axes[i // 3, i % 3]  # サブプロットのインデックスを計算
        ax.plot(ecg_signal)
        ax.plot(rpeaks, ecg_signal[rpeaks], "ro", label="R peaks")

        # NaNを含まない部分を取り出してからプロット（整数型に変換）
        # input(waves_peak_dict)
        ecg_p_peaks = waves_peak_dict.get("ECG_P_Peaks")
        if ecg_p_peaks is not None:
            valid_ecg_p_peaks = np.array(ecg_p_peaks)[~np.isnan(ecg_p_peaks)].astype(int)
            ax.plot(valid_ecg_p_peaks, ecg_signal[valid_ecg_p_peaks], "bo", label="P peaks")

        ecg_q_peaks = waves_peak_dict.get("ECG_Q_Peaks")
        if ecg_q_peaks is not None:
            valid_ecg_q_peaks = np.array(ecg_q_peaks)[~np.isnan(ecg_q_peaks)].astype(int)
            ax.plot(valid_ecg_q_peaks, ecg_signal[valid_ecg_q_peaks], "yo", label="Q peaks")

        # ecg_r_peaks = waves_peak_dict.get("ECG_R_Peaks")
        # if ecg_r_peaks is not None:
        #     valid_ecg_r_peaks = np.array(ecg_r_peaks)[~np.isnan(ecg_r_peaks)].astype(int)
        #     ax.plot(valid_ecg_r_peaks, ecg_signal[valid_ecg_r_peaks], "ro", label="R peaks")

        ecg_s_peaks = waves_peak_dict.get("ECG_S_Peaks")
        if ecg_s_peaks is not None:
            valid_ecg_s_peaks = np.array(ecg_s_peaks)[~np.isnan(ecg_s_peaks)].astype(int)
            ax.plot(valid_ecg_s_peaks, ecg_signal[valid_ecg_s_peaks], "go", label="S peaks")

        ecg_t_peaks = waves_peak_dict.get("ECG_T_Peaks")
        if ecg_t_peaks is not None:
            valid_ecg_t_peaks = np.array(ecg_t_peaks)[~np.isnan(ecg_t_peaks)].astype(int)
            ax.plot(valid_ecg_t_peaks, ecg_signal[valid_ecg_t_peaks], "mo", label="T peaks")

        # ax.set_title(title)
        ax.set_title("ch{}".format(i))
        # ax.legend()
        ax.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=1)
        ax.grid(True)

    plt.suptitle(args.name+'_'+args.pos+"_15ch")
    plt.tight_layout()
    compare_path='./0_packetloss_data/pqrst'
    create_directory_if_not_exists(compare_path)
    # plt.savefig(compare_path+'/15ch'+args.type+'.png')
    if(DEBUG_PLOT==True):
        plt.show()
    # plt.show()
    plt.close()
def PQRST_plot_grid(ecg_list, sampling_rate, headers,args):
    """
    4x3のグリッドに12個のECG信号のPQRST波をプロットする関数

    Parameters:
        ecg_list (numpy.ndarray): プロットするECG信号の2次元配列（各行が1つのECG信号を表す）
        sampling_rate (int): サンプリングレート（Hz）
        headers (list of strings): 各プロットのタイトルのリスト（要素は文字列）

    Raises:
        ValueError: ecg_listの行数が12でない場合

    Returns:
        None
    """
    if ecg_list.shape[0] != 12 or len(headers) != 12:
        raise ValueError("ecg_list must contain exactly 12 rows, and headers must contain exactly 12 elements.")

    fig, axes = plt.subplots(4, 3, figsize=(15, 12))

    for i, (ecg_signal, title) in enumerate(zip(ecg_list, headers)):
        ecg_signal=nk.ecg_clean(ecg_signal,sampling_rate=500,method='neurokit')
        # print(ecg_signal)
        rpeaks = nk.ecg_peaks(ecg_signal, sampling_rate)[1]['ECG_R_Peaks']  # R波の位置を取得
        waves_peak_df, waves_peak_dict = nk.ecg_delineate(ecg_signal,
                                                         rpeaks,
                                                         sampling_rate=sampling_rate,
                                                         method="peak",
                                                         show=False)


        ax = axes[i // 3, i % 3]  # サブプロットのインデックスを計算
        ax.plot(ecg_signal)
        ax.plot(rpeaks, ecg_signal[rpeaks], "ro", label="R peaks")

        # NaNを含まない部分を取り出してからプロット（整数型に変換）
        # input(waves_peak_dict)
        ecg_p_peaks = waves_peak_dict.get("ECG_P_Peaks")
        if ecg_p_peaks is not None:
            valid_ecg_p_peaks = np.array(ecg_p_peaks)[~np.isnan(ecg_p_peaks)].astype(int)
            ax.plot(valid_ecg_p_peaks, ecg_signal[valid_ecg_p_peaks], "bo", label="P peaks")

        ecg_q_peaks = waves_peak_dict.get("ECG_Q_Peaks")
        if ecg_q_peaks is not None:
            valid_ecg_q_peaks = np.array(ecg_q_peaks)[~np.isnan(ecg_q_peaks)].astype(int)
            ax.plot(valid_ecg_q_peaks, ecg_signal[valid_ecg_q_peaks], "yo", label="Q peaks")

        # ecg_r_peaks = waves_peak_dict.get("ECG_R_Peaks")
        # if ecg_r_peaks is not None:
        #     valid_ecg_r_peaks = np.array(ecg_r_peaks)[~np.isnan(ecg_r_peaks)].astype(int)
        #     ax.plot(valid_ecg_r_peaks, ecg_signal[valid_ecg_r_peaks], "ro", label="R peaks")

        ecg_s_peaks = waves_peak_dict.get("ECG_S_Peaks")
        if ecg_s_peaks is not None:
            valid_ecg_s_peaks = np.array(ecg_s_peaks)[~np.isnan(ecg_s_peaks)].astype(int)
            ax.plot(valid_ecg_s_peaks, ecg_signal[valid_ecg_s_peaks], "go", label="S peaks")

        ecg_t_peaks = waves_peak_dict.get("ECG_T_Peaks")
        if ecg_t_peaks is not None:
            valid_ecg_t_peaks = np.array(ecg_t_peaks)[~np.isnan(ecg_t_peaks)].astype(int)
            ax.plot(valid_ecg_t_peaks, ecg_signal[valid_ecg_t_peaks], "mo", label="T peaks")

        ax.set_title(title)
        ax.legend()
        ax.grid(True)
    plt.suptitle(args.name+'_'+args.pos+"_12ch")
    plt.tight_layout()
    compare_path='./0_packetloss_data/pqrst'
    create_directory_if_not_exists(compare_path)
    plt.savefig(compare_path+'/12ch'+args.type+'.png')
    if(DEBUG_PLOT==True):
        plt.show()
    plt.show()
    plt.close()

def find_p_element(arr, target):#ターゲットのR波のインデックスと最も近いP並みのOnsetを探す.
    # 配列が空の場合はNoneを返す
    if arr.size==0:
        return None

    # 初期値として最初の要素を仮の最も近い要素として設定
    closest_element = arr[0]

    # 配列をループして最も近い要素を見つける
    for num in arr:
        # 指定した数以下かつ、現在の要素が仮の最も近い要素よりも近い場合
        if num <= target and abs(num - target) < abs(closest_element - target):
            closest_element = num

    return closest_element

def find_t_element(arr, target):#ターゲットのR波のインデックスと最も近いT波のOffsetを探す.
    # 配列が空の場合はNoneを返す
    if arr.size == 0:
        return None

    # 初期値として最初の要素を仮の最も近い要素として設定
    closest_element = arr[0]

    # 配列をループして最も近い要素を見つける
    for num in arr:
        # 指定した数以上かつ、現在の要素が仮の最も近い要素よりも近い場合
        if num >= target and abs(num - target) < abs(closest_element - target):
            closest_element = num

    return closest_element

def find_s_element(arr, target):#ターゲットのR波のインデックスと最も近いT波のOffsetを探す.
    # 配列が空の場合はNoneを返す
    if arr.size == 0:
        return None

    # 初期値として最初の要素を仮の最も近い要素として設定
    closest_element = arr[0]

    # 配列をループして最も近い要素を見つける
    for num in arr:
        # 指定した数以上かつ、現在の要素が仮の最も近い要素よりも近い場合
        if num >= target and abs(num - target) < abs(closest_element - target):
            closest_element = num

    return closest_element

def is_all_elements_integer(array_2d):
    # ndarrayのすべての要素が整数か確認
    if not np.issubdtype(array_2d.dtype, np.integer):
        warnings.warn("ndarray contains non-integer elements.", UserWarning)
        return False
    return True

def PTwave_search(ecg_A2, sampling_rate, header,args,time_length):
    peak_method=args.peak_method
    ecg_signal=nk.ecg_clean(ecg_A2,sampling_rate=sampling_rate,method='neurokit')
    print(ecg_signal)
    rpeaks = nk.ecg_peaks(ecg_signal, sampling_rate)[1]['ECG_R_Peaks']  # R波の位置を取得
    waves_peak_df, waves_peak_dict = nk.ecg_delineate(ecg_signal,
                                                     rpeaks,
                                                     sampling_rate=sampling_rate,
                                                     method="peak",
                                                     show=False)
    # NaNを含まない部分を取り出してからプロット（整数型に変換）
    # input(waves_peak_dict)
    # fig=plt.figure(figsize=(12,12))
    ax=plt.axes()
    ax.plot(ecg_signal)
    ax.plot(rpeaks, ecg_signal[rpeaks], "ro", label="R peaks")

    ecg_p_onsets = waves_peak_dict.get("ECG_P_Onsets")
    if ecg_p_onsets is not None:
        valid_ecg_p_onsets = np.array(ecg_p_onsets)[~np.isnan(ecg_p_onsets)].astype(int)
        ax.plot(valid_ecg_p_onsets, ecg_signal[valid_ecg_p_onsets], "bo", label="P onset")


    ecg_t_offsets = waves_peak_dict.get("ECG_T_Offsets")
    if ecg_t_offsets is not None:
        valid_ecg_t_offsets = np.array(ecg_t_offsets)[~np.isnan(ecg_t_offsets)].astype(int)
        ax.plot(valid_ecg_t_offsets, ecg_signal[valid_ecg_t_offsets], "mo", label="T Offsets")
    ax.set_title(args.name+'_'+args.pos+"_12ch_A2")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    compare_path=os.path.join(args.file_path,"..","..")
    create_directory_if_not_exists(compare_path)
    plt.savefig(compare_path+"/12ch"+args.type+'.png')
    # plt.show()
    if(DEBUG_PLOT==True):
        plt.show()
    plt.close()
    print(valid_ecg_t_offsets)
    print(valid_ecg_p_onsets)
    print("len(r)_{},len(p)_{},len(t)_{}".format(len(rpeaks),len(valid_ecg_p_onsets),len(valid_ecg_t_offsets)))
    data_list=[]
    for rpeak in rpeaks:
        #2秒間きりだすために500Hz×1秒=500インデックス前後に存在するpqrだけを使う。
        if(rpeak>int(0.5*time_length*sampling_rate) and rpeak<12000-int(0.5*time_length*sampling_rate)):
            p_ele=find_p_element(valid_ecg_p_onsets,rpeak)
            t_ele=find_t_element(valid_ecg_t_offsets,rpeak)
            print(rpeak-p_ele)
            print(t_ele-rpeak)
            if(0< rpeak-p_ele<int(0.5*time_length*sampling_rate) and 0 < t_ele-rpeak<int(0.5*time_length*sampling_rate)):
                data_list.append([p_ele,rpeak,t_ele])
                print(rpeak - p_ele)
                # input("")

    prt_array=np.array(data_list)
    print(prt_array)
    print(len(prt_array))
    is_all_elements_integer(prt_array)
    return prt_array

def is_ascending(lst):
    # リストが5つの要素を持っていることを確認
    print(lst)
    # if len(lst) != 5:
    #     return False

    # リストの要素を順番に比較
    for i in range(4):
        if lst[i] > lst[i + 1]:
            return False
    return True

def PTwave_search3(ecg_A2, sampling_rate, header,args,time_length,method):#P_Onset,T_Offset,P_Offset,T_Onsetを返す関数
    peak_method=args.peak_method
    ecg_signal=nk.ecg_clean(ecg_A2,sampling_rate=sampling_rate,method='neurokit')
    print(ecg_signal)
    rpeaks = nk.ecg_peaks(ecg_signal, sampling_rate)[1]['ECG_R_Peaks']  # R波の位置を取得
    waves_peak_df, waves_peak_dict = nk.ecg_delineate(ecg_signal,
                                                     rpeaks,
                                                     sampling_rate=sampling_rate,
                                                     method=method,
                                                     show=False)
    # NaNを含まない部分を取り出してからプロット（整数型に変換）
    # input(waves_peak_dict)
    # fig=plt.figure(figsize=(12,12))
    # ax=plt.axes()
    # ax.plot(ecg_signal)
    # ax.plot(rpeaks, ecg_signal[rpeaks], "ro", label="R peaks")


    for key in waves_peak_dict.keys():
        print(key)
    plt.close()
    fig=plt.figure(figsize=(24,12))
    ax=plt.axes()
    ax.plot(ecg_signal)
    # ax.set_ylim(0.3, 0.8)
    color_dict = {
        "P_Onsets": 'b',    # 青
        "P_Peaks": 'deepskyblue',     # 緑
        "P_Offsets": 'royalblue',   # 赤
        "Q_Peaks": 'y',     # シアン
        "R_Peaks": 'r',    # 青
        "R_Onsets": 'darkred',    # マゼンタ
        "R_Offsets": 'tomato',   # 黄
        "S_Peaks": 'brown',     # 黒
        "T_Onsets": 'g',  # 紫
        "T_Peaks": 'limegreen',  # オレンジ
        "T_Offsets": 'forestgreen'   # ブラウン
    }

    ecg_p_onsets = waves_peak_dict.get("ECG_P_Onsets")
    if ecg_p_onsets is not None:
        valid_ecg_p_onsets = np.array(ecg_p_onsets)[~np.isnan(ecg_p_onsets)].astype(int)
        ax.plot(valid_ecg_p_onsets, ecg_signal[valid_ecg_p_onsets],color_dict["P_Onsets"], label="P onset",marker='v',linestyle='None', alpha=0.7)

    ecg_p_peaks = waves_peak_dict.get("ECG_P_Peaks")
    if ecg_p_peaks is not None:
        valid_ecg_p_peaks = np.array(ecg_p_peaks)[~np.isnan(ecg_p_peaks)].astype(int)
        ax.plot(valid_ecg_p_peaks, ecg_signal[valid_ecg_p_peaks], color_dict["P_Peaks"], label="P peaks",marker='o',linestyle='None', alpha=0.7)
    # "ECG_S_Peaks"の処理

    if(peak_method=='cwt'):
        ecg_p_offsets = waves_peak_dict.get("ECG_P_Offsets")
        if ecg_p_offsets is not None:
            valid_ecg_p_offsets = np.array(ecg_p_offsets)[~np.isnan(ecg_p_offsets)].astype(int)
            ax.plot(valid_ecg_p_offsets, ecg_signal[valid_ecg_p_offsets], color_dict["P_Offsets"], label="P Offsets",marker='^',linestyle='None', alpha=0.7)

        ecg_q_peaks = waves_peak_dict.get("ECG_Q_Peaks")
        if ecg_q_peaks is not None:
            valid_ecg_q_peaks = np.array(ecg_q_peaks)[~np.isnan(ecg_q_peaks)].astype(int)
            ax.plot(valid_ecg_q_peaks, ecg_signal[valid_ecg_q_peaks], color_dict["Q_Peaks"], label="Q peaks",marker='o',linestyle='None', alpha=0.7)
#R波
    ax.plot(rpeaks, ecg_signal[rpeaks], "ro", label="R peaks", alpha=0.7)

    # ax.set_ylim(0.3, 0.8)
    ecg_s_peaks = waves_peak_dict.get("ECG_S_Peaks")
    if ecg_s_peaks is not None:
        valid_ecg_s_peaks = np.array(ecg_s_peaks)[~np.isnan(ecg_s_peaks)].astype(int)
        ax.plot(valid_ecg_s_peaks, ecg_signal[valid_ecg_s_peaks], color_dict["S_Peaks"], label="S peaks",marker='o',linestyle='None', alpha=0.7)

    if(peak_method=='cwt'):
        ecg_t_onsets = waves_peak_dict.get("ECG_T_Onsets")
        if ecg_t_onsets is not None:
            valid_ecg_t_onsets = np.array(ecg_t_onsets)[~np.isnan(ecg_t_onsets)].astype(int)
            ax.plot(valid_ecg_t_onsets, ecg_signal[valid_ecg_t_onsets], color_dict["T_Onsets"], label="T onset",marker='v',linestyle='None', alpha=0.7)

    ecg_t_peaks = waves_peak_dict.get("ECG_T_Peaks")
    if ecg_t_peaks is not None:
        valid_ecg_t_peaks = np.array(ecg_t_peaks)[~np.isnan(ecg_t_peaks)].astype(int)
        ax.plot(valid_ecg_t_peaks, ecg_signal[valid_ecg_t_peaks], color_dict["T_Peaks"], label="T peaks",marker='o',linestyle='None', alpha=0.7)


    ecg_t_offsets = waves_peak_dict.get("ECG_T_Offsets")
    if ecg_t_offsets is not None:
        valid_ecg_t_offsets = np.array(ecg_t_offsets)[~np.isnan(ecg_t_offsets)].astype(int)
        ax.plot(valid_ecg_t_offsets, ecg_signal[valid_ecg_t_offsets], color_dict["T_Offsets"], label="T Offsets",marker='^',linestyle='None', alpha=0.7)
    ax.set_title(args.name+'_'+args.pos+"_12ch_A2")
    # ax.set_title("{}_{}_{}".format(args.TARGET_NAME,args.TARGET_CHANNEL,signal_type))
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    compare_path='./0_packetloss_data_{}/T_Offsets'.format(DATASET_MADE_DATE)
    create_directory_if_not_exists(compare_path)
    plt.savefig(compare_path+"/12ch"+args.type+'.png')
    if(DEBUG_PLOT==True):
        plt.show()
    plt.close()
    print(valid_ecg_t_offsets)
    print(valid_ecg_p_onsets)
    print("len(r)_{},len(p)_{},len(t)_{}".format(len(rpeaks),len(valid_ecg_p_onsets),len(valid_ecg_t_offsets)))
    data_list=[]
    print(rpeaks)
    rpeak_num=len(rpeaks)
    for i in range(rpeak_num):
        rpeak=rpeaks[i]
        if(i==0):
            rpeak_before=0
        else:
            rpeak_before=rpeaks[i-1]

        if(i==rpeak_num-1):
            rpeak_next=10000000000000
        else:
            rpeak_next=rpeaks[i+1]

        #2秒間きりだすために500Hz×0.8秒=500インデックス前後に存在するpqrだけを使う。
        if(rpeak>int(0.5*time_length*sampling_rate) and rpeak<12000-int(0.5*time_length*sampling_rate)):
            p_Onset_ele=find_p_element(valid_ecg_p_onsets,rpeak)
            p_Offset_ele=find_p_element(valid_ecg_p_offsets,rpeak)
            t_Offset_ele=find_t_element(valid_ecg_t_offsets,rpeak)
            t_Onset_ele=find_t_element(valid_ecg_t_onsets,rpeak)
            p_Peaks_ele=find_p_element(valid_ecg_p_peaks,rpeak)
            t_Peaks_ele=find_t_element(valid_ecg_t_peaks,rpeak)
            s_Peaks_ele=find_s_element(valid_ecg_s_peaks,rpeak)
            q_Peaks_ele=find_p_element(valid_ecg_q_peaks,rpeak)
            # print(p_Onset_ele)
            # print(p_Peaks_ele)
            # print(p_Offset_ele)
            # print("p_ele")
            # print(rpeak)
            # print("r_ele")
            # print(s_Peaks_ele)
            # print("s_ele")
            # print(t_Onset_ele)
            # print(t_Peaks_ele)
            # print(t_Offset_ele)
            ele_list=[p_Onset_ele,p_Peaks_ele,p_Offset_ele,q_Peaks_ele,rpeak,s_Peaks_ele,t_Onset_ele,t_Peaks_ele,t_Offset_ele]
            switch=is_ascending(lst=ele_list)
            print(switch)
            # input("")
            if(switch==True and rpeak-int(time_length*sampling_rate*0.5)<p_Onset_ele<rpeak and rpeak<t_Offset_ele<rpeak+int(time_length*sampling_rate*0.5)):
                data_list.append([p_Onset_ele,rpeak,t_Offset_ele,p_Offset_ele,t_Onset_ele,p_Peaks_ele,q_Peaks_ele,s_Peaks_ele,t_Peaks_ele])
                print(rpeak - p_Onset_ele)

            # if(rpeak-int(time_length*sampling_rate*0.5)<p_Onset_ele<rpeak and\
            #     rpeak<t_Offset_ele<rpeak+int(time_length*sampling_rate*0.5) and\
            #           rpeak<t_Onset_ele<rpeak+t_Offset_ele and\
            #             p_Onset_ele<p_Offset_ele<rpeak):#P波オンセット、P波
            #     # data_list.append([p_Onset_ele,rpeak,t_Offset_ele,p_Offset_ele,t_Onset_ele])
            #     before=[p_Onset_ele,rpeak,t_Offset_ele,p_Offset_ele,t_Onset_ele]
            #     print(before)
            #     # print(rpeak - p_Onset_ele)
            #     input("")

    prt_array=np.array(data_list)
    # print(prt_array)
    print("データセットにできる心拍の数は{}".format(len(prt_array)))
    # input("")
    is_all_elements_integer(prt_array)
    return prt_array


def PTwave_plot(ecg_list, sampling_rate, headers,args):
    if ecg_list.shape[0] != 12 or len(headers) != 12:
        raise ValueError("ecg_list must contain exactly 12 rows, and headers must contain exactly 12 elements.")

    fig, axes = plt.subplots(4, 3, figsize=(15, 12))

    for i, (ecg_signal, title) in enumerate(zip(ecg_list, headers)):
        ecg_signal=nk.ecg_clean(ecg_signal,sampling_rate=500,method='neurokit')
        # print(ecg_signal)
        rpeaks = nk.ecg_peaks(ecg_signal, sampling_rate)[1]['ECG_R_Peaks']  # R波の位置を取得
        waves_peak_df, waves_peak_dict = nk.ecg_delineate(ecg_signal,
                                                         rpeaks,
                                                         sampling_rate=sampling_rate,
                                                         method="peak",
                                                         show=False)
        # print(waves_peak_dict)
        # input()


        ax = axes[i // 3, i % 3]  # サブプロットのインデックスを計算
        ax.plot(ecg_signal)
        ax.plot(rpeaks, ecg_signal[rpeaks], "ro", label="R peaks")

        # NaNを含まない部分を取り出してからプロット（整数型に変換）
        # input(waves_peak_dict)

        ecg_p_onsets = waves_peak_dict.get("ECG_P_Onsets")
        if ecg_p_onsets is not None:
            valid_ecg_p_onsets = np.array(ecg_p_onsets)[~np.isnan(ecg_p_onsets)].astype(int)
            ax.plot(valid_ecg_p_onsets, ecg_signal[valid_ecg_p_onsets], "bo", label="P onset")


        ecg_t_offsets = waves_peak_dict.get("ECG_T_Offsets")
        if ecg_t_offsets is not None:
            valid_ecg_t_offsets = np.array(ecg_t_offsets)[~np.isnan(ecg_t_offsets)].astype(int)
            ax.plot(valid_ecg_t_offsets, ecg_signal[valid_ecg_t_offsets], "mo", label="T Offsets")

        ax.set_title(title)
        ax.legend()
        ax.grid(True)
    plt.suptitle(args.name+'_'+args.pos+"_12ch")
    plt.tight_layout()
    compare_path='./0_packetloss_data/pqrst'
    create_directory_if_not_exists(compare_path)
    # plt.savefig(compare_path+'/12ch'+args.type+'.png')
    if(DEBUG_PLOT==True):
        plt.show()
    plt.show()
    plt.close()


def ecg_clean_df_12ch(df_12ch,rate=RATE):
    ecg_signal=df_12ch.copy()["A1"]
    # cleand_signal=nk.ecg_clean(ecg_signal,sampling_rate=500,method="neurokit")
    # print(cleaned_signal)
    # print(type(cleaned_signal))
    # plt.plot(cleaned_signal)
    print(type(df_12ch))
    plt.plot(df_12ch["A2"])
    plt.title("org")
    # plt.show()
    df_12ch_cleaned=pd.DataFrame()
    for i,column in enumerate(df_12ch.columns):
        #df0_mul.plot()
        # df1=df.iloc[:,i]
        ecg_signal=df_12ch[column].copy().values
        ecg_signal=nk.ecg_clean(ecg_signal,sampling_rate=rate,method='neurokit')
        df_12ch_cleaned[column]=ecg_signal
        # print(df[column])
        # print("")

    print(type(df_12ch_cleaned))
    plt.plot(df_12ch_cleaned["A2"])
    plt.title("cleaned")
    plt.show()
    return df_12ch_cleaned
def ecg_clean_df_15ch(df_15ch,rate):
    ecg_signal=df_15ch.copy()["ch_1"]
    # cleand_signal=nk.ecg_clean(ecg_signal,sampling_rate=500,method="neurokit")
    # print(cleaned_signal)
    # print(type(cleaned_signal))
    # plt.plot(cleaned_signal)
    print(type(df_15ch))
    plt.plot(df_15ch["ch_1"])
    plt.title("org")
    # plt.show()
    df_15ch_cleaned=pd.DataFrame()
    for i,column in enumerate(df_15ch.columns):
        #df0_mul.plot()
        # df1=df.iloc[:,i]
        ecg_signal=df_15ch[column].copy().values
        ecg_signal=nk.ecg_clean(ecg_signal,sampling_rate=rate,method='neurokit')
        df_15ch_cleaned[column]=ecg_signal
        # print(df[column])
        # print("")

    print(type(df_15ch_cleaned))
    plt.plot(df_15ch_cleaned["ch_1"])
    plt.title("cleaned")
    plt.show()
    return df_15ch_cleaned




def main(args):
    # TARGET_CHANNEL_15CH=args.TARGET_CHANNEL_15CH
    TARGET_CHANNEL_12CH=args.TARGET_CHANNEL_12CH
    cut_min_max_range=args.cut_min_max_range
    # ファイル読み込み
    # dir_path = "./0_packetloss_data/"+args.dir_name
    dir_path = "./0_packetloss_data_{}/".format(DATASET_MADE_DATE)+args.dir_name
    csv_reader_16ch = CSVReader_16ch(dir_path)
    df_16ch=csv_reader_16ch.process_files()
    print(df_16ch)
    cols=df_16ch.columns
    df_15ch=pd.DataFrame()
    for col in cols:
        df_15ch[col]=df_16ch[col]-df_16ch['ch_16']
    df_15ch=df_15ch.drop(columns=['ch_16'])

    csv_reader_12ch = CSVReader_12ch(dir_path)
    df_12ch=csv_reader_12ch.process_files()
    df_12ch_cleaned=ecg_clean_df_12ch(df_12ch)
    # input("")
    #同期用インデックスファイルを読み込みと書き込み
    handler = AutoIntegerFileHandler(dir_path+'/同期インデックス_nkmodule.txt')
    #同期する前
    # if(DEBUG_PLOT==True):
    #     Plot_15ch=MultiPlotter(df_15ch.copy(),RATE=RATE_15ch)
    #     Plot_15ch.plot_all_channels(xmin=0,xmax=10,ylim=0)
    #     # Plot_15ch.multi_plot(xmin=0,xmax=100,ylim=0)
    #     # Plot_16ch=MultiPlotter(df_16ch.copy(),RATE=RATE_15ch)
    #     # Plot_16ch.multi_plot(xmin=0,xmax=100,ylim=0)
    #     plt.show()
    #     plt.close()
    # input()


    if(handler.check_file()==False):#同期するためのファイルが存在していないとき。
        # df_15ch_pf = multi_pf(df_15ch.copy(),fp=2.0,fs=1.0)
        reverse=args.reverse
        print("TARGET_CHNNEL_15chは")
        TARGET_CHANNEL_15CH=validate_integer_input()
        # df_15ch_pf = hpf_lpf(df_15ch.copy(),HPF_fp=2.0,HPF_fs=1.0,LPF_fp=0,LPF_fs=0,RATE=RATE_15ch)
        # df_15ch_pf = hpf_lpf(df_15ch.copy(),HPF_fp=HPF_FP,HPF_fs=HPF_FS,LPF_fp=0,LPF_fs=0,RATE=RATE_15ch)
        df_15ch_pf = ecg_clean_df_15ch(df_15ch=df_15ch.copy(),rate=RATE_15ch)
        df_resample_15ch=linear_interpolation_resample_All(df=df_15ch_pf.copy(),sampling_rate=RATE_15ch,new_sampling_rate=RATE)
        df_15ch_pf=df_resample_15ch.copy()
        # Plot_15ch_pf=MultiPlotter(df_15ch_pf.copy(),RATE=RATE_15ch)
        # Plot_15ch_pf.multi_plot(xmin=0,xmax=10,ylim=0)
        # df_15ch_reverse=df_15ch_pf.copy()
        # df_15ch_reverse[TARGET_CHANNEL_15CH]=(-1)*df_15ch_pf.copy()[TARGET_CHANNEL_15CH]
        # Plot_15ch_reverse=MultiPlotter(df_15ch_reverse.copy(),RATE=RATE_15ch)
        # Plot_15ch_reverse.multi_plot(xmin=0,xmax=10,ylim=0)
        # plt.show()
        # plt.close()
        # df_12ch_pf = hpf_lpf(df_12ch.copy(),HPF_fp=2.0,HPF_fs=1.0,LPF_fp=55.0,LPF_fs=60.0,RATE=RATE_12ch)
        # sc_15ch=peak_sc(df_15ch_pf[syn_index:syn_index+20*RATE_15ch].copy(),RATE=RATE_15ch,TARGET="ch_4")
        if(reverse=='off'):
            sc_15ch=peak_sc_15ch(df_15ch_pf.copy(),RATE=RATE,TARGET=TARGET_CHANNEL_15CH)
            peak_sc_plot(df_15ch_pf.copy(),RATE=RATE,TARGET=TARGET_CHANNEL_15CH)
        # else:
        #     df_15ch_reverse=df_15ch_pf.copy()
        #     df_15ch_reverse[TARGET_CHANNEL_15CH]=(-1)*df_15ch_pf.copy()[TARGET_CHANNEL_15CH]
        #     sc_15ch=peak_sc(df_15ch_reverse.copy(),RATE=RATE_15ch,TARGET=TARGET_CHANNEL_15CH)
        #     peak_sc_plot(df_15ch_reverse.copy(),RATE=RATE_15ch,TARGET=TARGET_CHANNEL_15CH)

        sc_12ch=peak_sc(df_12ch.copy(),RATE=RATE_12ch,TARGET=TARGET_CHANNEL_12CH)
        # print(sc_12ch)
        # input()
        peak_sc_plot(df_12ch.copy(),RATE=RATE_12ch,TARGET=TARGET_CHANNEL_12CH)
        print('reverse=={}'.format(reverse))
        # peak_sc_plot(df_15ch_pf.copy(),RATE=RATE_15ch,TARGET=TARGET_CHANNEL_15CH)
        print(sc_15ch)

        comparator=ArrayComparator(sc_15ch=sc_15ch,sc_12ch=sc_12ch,cut_min_max_range=cut_min_max_range)
        comparator.peak_diff_plot()
        cut_time=comparator.find_best_cut_time()
        comparator.peak_diff_plot_move(cut_time)
        # comparator.find_best_cut_time()
        # peak_diff_plot(sc_12ch,sc_15ch)
        Plot_15ch_pf=MultiPlotter(df_15ch_pf,RATE=RATE)
        Plot_15ch_pf.multi_plot(xmin=0,xmax=100,ylim=0)
        Plot_15ch_pf.multi_plot_15ch_with_sc(xmin=0,xmax=20,ylim=0,sc=sc_15ch)
        plt.show()
        plt.close()
        print(int(cut_time*RATE))
        if(input("write_to_CSV OK? y or n")=="y"):
            handler.write_integer(RATE=RATE,cut_time=cut_time,target_15ch=TARGET_CHANNEL_15CH,reverse=reverse,target_12ch=TARGET_CHANNEL_12CH,cut_min_max_range=cut_min_max_range)

        else:
            return 0

    else:#同期するファイルが存在しているとき。
        # df_15ch_pf = hpf_lpf(df_15ch.copy(),HPF_fp=HPF_FP,HPF_fs=HPF_FS,LPF_fp=0,LPF_fs=0,RATE=RATE_15ch)
        # df_15ch_pf = multi_pf(df_15ch.copy(),fp=0.2,fs=0.1)
        # df_15ch_pf = df_15ch.copy()
        print("ファイルが存在します。")
        # return 0
    df_15ch_pf = ecg_clean_df_15ch(df_15ch=df_15ch.copy(),rate=RATE_15ch)
    df_resample_15ch=linear_interpolation_resample_All(df=df_15ch_pf.copy(),sampling_rate=RATE_15ch,new_sampling_rate=RATE)

    syn_index,TARGET_CHANNEL_15CH,reverse,TARGET_CHANNEL_12CH = handler.read_integer()#同期するインデックス
    print(syn_index)
    # if(target_ch!=TARGET_CHANNEL_15ch):
    #     print("target_ch!=args.TARGE_CHSNNEL_15ch")
    #     print("target_ch"+target_ch)
    #     return 0
    # input("15ch_pf")
    if(DEBUG_PLOT==True):
        if(reverse=='off'):
            sc_15ch_pf=peak_sc_15ch(df_resample_15ch[syn_index:].copy(),RATE=RATE_15ch,TARGET=TARGET_CHANNEL_15CH)

        sc_12ch=peak_sc(df_12ch.copy(),RATE=RATE_12ch,TARGET=TARGET_CHANNEL_12CH)

        # Plot_15ch_pf=MultiPlotter(df_15ch_pf[syn_index:].copy(),RATE=RATE_15ch)
        # Plot_15ch_pf.multi_plot(xmin=0,xmax=TIME,ylim=10000)
        # Plot_15ch_pf.multi_plot_15ch_with_sc(xmin=0,xmax=10,ylim=10000,sc=peak_sc(df_15ch_pf[syn_index:].copy(),RATE=RATE_15ch,TARGET=TARGET_CHANNEL_15CH))
        # Plot_12ch=MultiPlotter(df_12ch,RATE=RATE)
        # Plot_12ch.multi_plot(xmin=0,xmax=TIME,ylim=0)
        # Plot_12ch.multi_plot_12ch_with_sc(xmin=0,xmax=10,ylim=0,sc=peak_sc(df_15ch_pf[syn_index:].copy(),RATE=RATE_15ch,TARGET=TARGET_CHANNEL_15CH))
        Plot_15ch_pf=MultiPlotter(df_resample_15ch.copy(),RATE=RATE)
        Plot_15ch_pf.multi_plot(xmin=0,xmax=10,ylim=0)
        Plot_15ch_pf.plot_all_channels(xmin=0,xmax=8,ylim=0)

        Plot_12ch=MultiPlotter_both(df12=df_12ch_cleaned,df15=df_resample_15ch[syn_index:].copy(),RATE12=RATE_12ch,RATE15=RATE)#cleanされた12chにする。
        # Plot_12ch.multi_plot_12ch_15ch_with_sc(xmin=0,xmax=TIME,ylim=0,sc=sc_15ch_pf,ch=TARGET_CHANNEL_15CH,png_path=args.png_path+'15chsc')
        # Plot_12ch.multi_plot_12ch_15ch_with_sc(xmin=0,xmax=TIME,ylim=0,sc=sc_12ch,ch=TARGET_CHANNEL_15CH,png_path=args.png_path+'12chsc')
        # Plot_12ch.multi_plot_12ch_15ch_with_sc(xmin=0,xmax=TIME,ylim=0,sc=sc_15ch_pf,ch=TARGET_CHANNEL_15CH,png_path=args.png_path+'15chsc')
        Plot_12ch.multi_plot_12ch_15ch_with_sc(xmin=0,xmax=5,ylim=0,sc=sc_12ch,ch=TARGET_CHANNEL_15CH,png_path=args.png_path+'12chsc')
        plt.show()
        plt.close()
        # input()

    # print(df_12ch)
    print(syn_index)
    print(df_resample_15ch)

    # df_syn_resample_15ch=df_resample_15ch[syn_index:].copy()
    df_syn_resample_15ch=df_resample_15ch[syn_index:].copy().reset_index(drop=True)
    df_syn_resample_15ch_24s=df_syn_resample_15ch[:TIME*RATE]
    print(df_syn_resample_15ch_24s)
    plt.plot(df_syn_resample_15ch_24s["ch_1"])
    plt.plot(df_12ch_cleaned["A2"])
    plt.show()


    con_data=pd.concat([df_syn_resample_15ch_24s,df_12ch_cleaned],axis=1)#df_12ch_cleanedを用いる。
    con_data_dir="Dataset/pqrst_nkmodule_since{}_{}/".format(DATASET_MADE_DATE,args.peak_method)+args.output_filepath
    create_directory_if_not_exists(con_data_dir)

    con_data.to_csv(con_data_dir+'/condata_24s.csv',index=None)

    ecg_A2=con_data["A2"]
    print(ecg_A2)
    ecg_A2_np=ecg_A2.to_numpy().T
    # return 0
    # prt_eles=PTwave_search(ecg_A2=ecg_A2_np,header="A2",sampling_rate=RATE,args=args,time_length=0.7)
    prt_eles=PTwave_search3(ecg_A2=ecg_A2_np,header="A2",sampling_rate=RATE,args=args,time_length=args.time_range,method=args.peak_method)#1213からPQRST全部検出できるcwt方を使う。
    heartbeat_cutter_prt=HeartbeatCutter_prt(con_data.copy(),time_length=args.time_range,prt_eles=prt_eles,args=args)#切り出す秒数を指定する。
    heartbeat_cutter_prt.cut_heartbeats(file_path="Dataset/pqrst_nkmodule_since{}_{}/".format(DATASET_MADE_DATE,args.peak_method)+args.output_filepath,ch=TARGET_CHANNEL_15CH,cut_min_max_range=cut_min_max_range,args=args)

    con_data_np=con_data.to_numpy().T
    headers=con_data.columns
    print(headers)
    print(con_data_np.shape)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--dir_name", type=str, default='goto_0604/goto_0604_normal2')


    # parser.add_argument("--name", type=str, default='goto')
    # parser.add_argument("--date", type=str, default='1219')
    # parser.add_argument("--name", type=str, default='taniguchi')
    # parser.add_argument("--date", type=str, default='1107')
    # parser.add_argument("--name", type=str, default='yoshikura')
    # parser.add_argument("--date", type=str, default='1130')
    # parser.add_argument("--name", type=str, default='kawai')
    # parser.add_argument("--date", type=str, default='1115')
    # parser.add_argument("--name", type=str, default='matumoto')
    # parser.add_argument("--date", type=str, default='1128')
    # parser.add_argument("--name", type=str, default='takahashi')
    # parser.add_argument("--date", type=str, default='1102')


    # parser.add_argument("--name", type=str, default='asano')
    # parser.add_argument("--date", type=str, default='1201')
    # parser.add_argument("--name", type=str, default='takamatu')
    # parser.add_argument("--date", type=str, default='1130')

    # parser.add_argument("--name", type=str, default='togo')
    # parser.add_argument("--date", type=str, default='1107')


    # parser.add_argument("--name", type=str, default='takahashiJr')
    # parser.add_argument("--date", type=str, default='1106')


    # parser.add_argument("--name", type=str, default='sato')
    # parser.add_argument("--date", type=str, default='1115')

    # parser.add_argument("--name", type=str, default='takahashi')
    # parser.add_argument("--date", type=str, default='1220')
    parser.add_argument("--peak_method", type=str, default='cwt')
    parser.add_argument("--pos", type=str, default='')
    parser.add_argument("--type", type=str, default='')
    # parser.add_argument("--dir_name", type=str, default='goto_0621/goto_0621_72')
    # parser.add_argument("--dir_name", type=str, default='goto_0621/goto_0621_72')
    parser.add_argument("--dir_name", type=str, default='')
    parser.add_argument("--png_path", type=str, default='')
    parser.add_argument("--output_filepath", type=str, default='')
    # parser.add_argument("--TARGET_CHANNEL_15CH", type=str, default='ch_1')
    parser.add_argument("--TARGET_CHANNEL_12CH", type=str, default='A2')
    parser.add_argument("--cut_min_max_range", type=list, default=[0,10])
    # parser.add_argument("--cut_min_max_range", type=list, default=[3,8])
    parser.add_argument("--time_range", type=float, default=0.8)
    parser.add_argument("--reverse", type=str, default='off')
    # parser.add_argument("--reverse", type=str, default='on')
    args = parser.parse_args()
    # args.output_filepath='{}_{}_2s/{}'.format(args.name,args.date,args.pos)
    # args.type='{}_{}_{}'.format(args.name,args.date,args.pos)
    # args.dir_name='{}_{}/{}'.format(args.name,args.date,args.type)
    # main(args)
    # args.name="mori"
    # args.date="0712"
    # args.name="sato"
    # args.date="0710"
    # args.date="0719"
    # args.name="matumoto"
    # args.date="0718"
    # args.date="0623"
    # args.name="asano"
    # # args.date="0710"
    # # args.date="0719"
    args.name="gosha"
    args.date="0712"
    # args.date="0726"
    # args.name="taniguchi"
    # args.date="0712"
    # args.date="0726"
    args.pos='0'
    # args.pos='right1'
    # args.pos='72'
    # args.pos='144'
    # args.pos='216'
    # args.pos='288'
    # args.pos='right'
    # args.pos='left'


    # args.pos='0'
    # args.pos='right1'
    # args.pos='72'
    # args.pos='144'
    # args.pos='216'
    # args.pos='288'
    # args.pos='right'
    # args.pos='left'

    args.output_filepath='{}_{}_{}s/{}'.format(args.name,args.date,str(args.time_range),args.pos)

    args.type='{}_{}_{}'.format(args.name,args.date,args.pos)
    args.dir_name='{}/{}'.format(args.name,args.type)
    # args.png_path='./0_packet_loss_data/{}/{}_compare'.format(args.name,args.type)
    main(args)
    print(args.name+'_'+args.pos)

    # poses=["left","right"]
    # # poses=["right"]

    # for pos in poses:
    #     args.pos=pos
    #     # args.output_filepath='{}_{}_2s/{}_{}_2s_{}'.format(args.name, args.date,args.name, args.date, args.pos)
    #     args.output_filepath='{}_{}_2s/{}'.format(args.name, args.date,args.name, args.date, args.pos)
    #     args.type='{}_{}_{}'.format(args.name,args.date,args.pos)
    #     args.dir_name='{}/{}'.format(args.name,args.type)
    #     args.png_path='./data/{}/{}_compare.png'.format(args.name,args.type)
    #     main(args)


    # for pos in range(0,5):
    #     args.pos=str(pos*72)
    #     args.output_filepath='{}_{}_2s/{}'.format(args.name,args.date,args.pos)
    #     args.type='{}_{}_{}'.format(args.name,args.date,args.pos)
    #     args.dir_name='{}/{}'.format(args.name,args.type)
    #     args.png_path='./data/{}/{}_compare.png'.format(args.name,args.type)
    #     main(args)