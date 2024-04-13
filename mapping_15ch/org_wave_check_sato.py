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
DEBUG_PLOT = True
# DEBUG_PLOT = False
###
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import neurokit2 as nk
import warnings
import csv
from collections import Counter
import seaborn as sns
import matplotlib.animation as animation
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_dir)
from config.settings import MAPPING_DATA_DIR, RATE_15CH, OUTPUT_DIR
HPF_FP=2.0
HPF_FS=1.0
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

def peak_search_nk(df_target,RATE):
    print("safe")
    ecg_signal=df_target.copy().to_numpy().T
    ecg_signal=nk.ecg_clean(ecg_signal,sampling_rate=RATE,method='neurokit')
    print(ecg_signal)
    _, rpeaks = nk.ecg_peaks(ecg_signal, RATE)
    print(rpeaks['ECG_R_Peaks'])
    vals=ecg_signal[rpeaks['ECG_R_Peaks']]
    return rpeaks['ECG_R_Peaks'],vals

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
        for i in range(len(self.df.columns)):
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
            print("{}番目の心拍切り出し".format(i + 1))
            # print(data)
            file_name='dataset_{}.csv'.format(str(i).zfill(3))
            self.output_csv(file_name=file_name,file_path=file_path,data=data.copy())
            file_name_pt='ponset_toffsett_{}.csv'.format(str(i).zfill(3))
            self.output_csv_ponset_toffset(file_name=file_name_pt,file_path=file_path,p_onset=p_onset,t_offset=t_offset,p_offset=p_offset,t_onset=t_onset)
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

def PTwave_search4(ecg_A2, sampling_rate, header,args,time_length,method):#P_Onset,T_Offset,P_Offset,T_Onsetを返す関数
    peak_method=args.peak_method
    ecg_signal=nk.ecg_clean(ecg_A2,sampling_rate=sampling_rate,method='neurokit')
    print(ecg_signal)
    rpeaks = nk.ecg_peaks(ecg_signal, sampling_rate)[1]['ECG_R_Peaks']  # R波の位置を取得
    print(rpeaks)
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
    # plt.close()
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

#R波
    ax.plot(rpeaks, ecg_signal[rpeaks], "ro", label="R peaks", alpha=0.7)

    # ax.set_ylim(0.3, 0.8)
    ax.set_title(args.name+'_'+args.pos+'_'+args.target_ch)
    # ax.set_title("{}_{}_{}".format(args.TARGET_NAME,args.TARGET_CHANNEL,signal_type))
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    # compare_path='./0_packetloss_data_{}/T_Offsets'.format(DATASET_MADE_DATE)
    # create_directory_if_not_exists(compare_path)
    # plt.savefig(compare_path+"/ch_"+args.pos+'.png')
    # if(DEBUG_PLOT==True):
        # plt.show()
    plt.close()
    plt.cla()
    return rpeaks


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
            # print(rpeak-p_ele)
            # print(t_ele-rpeak)
            # print("rpeak_before={}".format(rpeak_before))
            # print("p_ele={}".format(p_ele))
            # print("t_ele={}".format(t_ele))
            # print("rpeak_next={}".format(rpeak_next))

            # if(0< rpeak-p_ele<int(0.5*time_length*sampling_rate) and 0 < t_ele-rpeak<int(0.5*time_length*sampling_rate)):
            # if(rpeak_before<p_ele<rpeak and rpeak<t_ele<rpeak_next):
            if(rpeak-int(time_length*sampling_rate*0.5)<p_Onset_ele<rpeak and\
                rpeak<t_Offset_ele<rpeak+int(time_length*sampling_rate*0.5) and\
                      rpeak<t_Onset_ele<rpeak+t_Offset_ele and\
                        p_Onset_ele<p_Offset_ele<rpeak):#P波オンセット、P波
                data_list.append([p_Onset_ele,rpeak,t_Offset_ele,p_Offset_ele,t_Onset_ele])
                print(rpeak - p_Onset_ele)
                # input("")

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
    # plt.show()
    plt.close()
    plt.cla()
    return df_12ch_cleaned
def ecg_clean_df_15ch(df_15ch,rate):
    ecg_signal=df_15ch.copy()["ch_1"]
    # cleand_signal=nk.ecg_clean(ecg_signal,sampling_rate=500,method="neurokit")
    # print(cleaned_signal)
    # print(type(cleaned_signal))
    # plt.plot(cleaned_signal)
    print(type(df_15ch))
    # plt.plot(df_15ch["ch_1"])
    # plt.title("org")
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
    # plt.plot(df_15ch_cleaned["ch_1"])
    # plt.title("cleaned")
    # plt.show()
    # plt.close()
    # plt.cla()
    return df_15ch_cleaned

def ecg_clean_df_16ch(df_15ch,rate):
    ecg_signal=df_15ch.copy()["ch_1"]
    # cleand_signal=nk.ecg_clean(ecg_signal,sampling_rate=500,method="neurokit")
    # print(cleaned_signal)
    # print(type(cleaned_signal))
    # plt.plot(cleaned_signal)
    print(type(df_15ch))
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

    # plt.plot(df_15ch["ch_1"])
    # plt.title("org")
    # print(type(df_15ch_cleaned))
    # plt.plot(df_15ch_cleaned["ch_1"])
    # plt.title("cleaned")
    # plt.show()
    # plt.close()
    # plt.cla()
    return df_15ch_cleaned

def average_columns(dfs, column):
    # 指定されたカラムが存在するすべてのデータフレームの該当カラムの平均を計算
    print(df[column] for df in dfs if column in df.columns)
    sum_df = sum(df[column] for df in dfs if column in df.columns)
    count = sum(column in df.columns for df in dfs)
    print(count)
    new_df=sum_df/count
    return new_df

# def average_columns(dfs, column):
#     # 指定されたカラムが存在するすべてのデータフレームの該当カラムの合計を計算
#     sum_df = sum(np.fromiter((df[column] for df in dfs if column in df.columns), dtype=float))
#     count = sum(column in df.columns for df in dfs)
#     return sum_df / count
# def average_columns(dfs, column):
#     # 各データフレームでのカラムの合計を計算
#     sum_df = sum(df[column].sum() for df in dfs if column in df.columns)
#     # カラムが存在するデータフレームの数をカウント
#     count = sum(column in df.columns for df in dfs)
#     # 合計をカウントで割って平均を計算
#     return sum_df / count if count > 0 else 0
# common_columns=[]

def update_org(frame):
    ax3.clear()
    # frame番目のデータでヒートマップを作成

    sns.heatmap(org_data[:, :, frame], ax=ax3, cmap='coolwarm', annot=True, cbar=False)
    # sns.heatmap(np_reshape[:, :, frame], ax=ax, cmap='coolwarm', annot=True, cbar=False,vmax=0,vmin=-2)
    # sns.heatmap(np_reshape[:, :, frame], ax=ax, cmap='coolwarm', annot=True, cbar=False)
    # sns.heatmap(np_reshape[:, :, frame], ax=ax, cmap='rainbow', annot=True, cbar=False,vmax=0.7,vmin=0.3)
    # sns_heatmap = sns.heatmap(np_reshape[:, :, frame], ax=ax, cmap='coolwarm', annot=False,vmax=1,vmin=0)
# カラーバーの設定
    # cbar = sns_heatmap.collections[0].colorbar
    # cbar.ax.clear()
    print(frame)

    # cbar.ax.set_position([0.15, 0.7, 0.7, 0.02])
    # plt.title(f'Heatmap Frame: {frame}')
    plt.title(f'time(second): {frame/RATE_15ch}')
    plt.xlabel('Column')
    plt.ylabel('Row')

def update_log(frame):
    ax.clear()
    # frame番目のデータでヒートマップを作成

    # sns.heatmap(np_reshape[:, :, frame], ax=ax, cmap='coolwarm', annot=True, cbar=False,vmax=max_log_value,vmin=second_smallest)
    # sns.heatmap(np_reshape[:, :, frame], ax=ax, cmap='coolwarm', annot=True, cbar=False,vmax=0,vmin=-2)
    # sns.heatmap(np_reshape[:, :, frame], ax=ax, cmap='coolwarm', annot=True, cbar=False)
    # sns.heatmap(np_reshape[:, :, frame], ax=ax, cmap='rainbow', annot=True, cbar=False,vmax=0.7,vmin=0.3)
    sns_heatmap = sns.heatmap(np_reshape[:, :, frame], ax=ax, cmap='coolwarm', annot=True,vmax=1,vmin=0,cbar=False)
# カラーバーの設定
    # cbar = sns_heatmap.collections[0].colorbar
    # cbar.ax.clear()
    print(frame)

    # cbar.ax.set_position([0.15, 0.7, 0.7, 0.02])
    # plt.title(f'Heatmap Frame: {frame}')
    plt.title(f'time(second): {frame/RATE_15ch}')
    plt.xlabel('Column')
    plt.ylabel('Row')

def update_wave_log(frame):
    ax4.clear()
    # print(frame)
    # frame番目のデータでヒートマップを作成
    # sns.heatmap(np_reshape[0, 0, frame], ax=ax, cmap='coolwarm', annot=False, cbar=False)
    # cbar = ax.collections[0].colorbar
    # cbar.ax.set_position([0.15, 0.7, 0.7, 0.02])
    # plt.plot(np.arange(np_reshape.shape[2])/RATE_15ch,np_reshape[0, 0, :],color="black")
    for i in range(13):
        for j in range(10):
            plt.plot(np.arange(np_reshape.shape[2])/RATE_15ch,np_reshape[i, j, :],color="black")
            # plt.plot(np.arange(normalized_datk.shape[2])/RATE_15ch,normalized_data[i, j, :],color="black")
    # plt.plot(np.arange(np_reshape.shape[2])/RATE_15ch,np_reshape[12, 9, :],color="black")
    # plt.plot((np_reshape.shape[2])/RATE_15ch*frame,np_reshape[0, 0, frame])
    ax4.axvline((1.0/RATE_15ch)*frame,color="r")
    plt.title(f'time(second): {frame/RATE_15ch}')
    plt.xlabel('Column')
    plt.ylabel('Row')

def update_wave_org(frame):
    ax4.clear()
    # print(frame)
    # frame番目のデータでヒートマップを作成
    # sns.heatmap(np_reshape[0, 0, frame], ax=ax, cmap='coolwarm', annot=False, cbar=False)
    # cbar = ax.collections[0].colorbar
    # cbar.ax.set_position([0.15, 0.7, 0.7, 0.02])
    plt.plot(np.arange(np_reshape.shape[2])/RATE_15ch,np_reshape[0, 0, :])
    # for i in range(13):
    #     for j in range(10):
    #         plt.plot(np.arange(org_data.shape[2])/RATE_15ch,org_data[i, j, :],color="black")
    #         # plt.plot(np.arange(normalized_data.shape[2])/RATE_15ch,normalized_data[i, j, :],color="black")
    # plt.plot(np.arange(np_reshape.shape[2])/RATE_15ch,np_reshape[12, 9, :],color="black")
    # plt.plot((np_reshape.shape[2])/RATE_15ch*frame,np_reshape[0, 0, frame])
    ax4.axvline((1.0/RATE_15ch)*frame,color="r")
    plt.title(f'time(second): {frame/RATE_15ch}')
    plt.xlabel('Column')
    plt.ylabel('Row')


def main(args):
    # TARGET_CHANNEL_15CH=args.TARGET_CHANNEL_15CH
    TARGET_CHANNEL_12CH=args.TARGET_CHANNEL_12CH
    cut_min_max_range=args.cut_min_max_range
    # ファイル読み込み
    # dir_path = "./0_packetloss_data/"+args.dir_name
    dir_path = "{}/{}".format(MAPPING_DATA_DIR,args.dir_name)
    csv_reader_16ch = CSVReader_16ch(dir_path)
    df_16ch=csv_reader_16ch.process_files()
    print(df_16ch)
    cols=df_16ch.columns
    df_15ch=pd.DataFrame()
    for col in cols:
        df_15ch[col]=df_16ch[col]-df_16ch['ch_16']
    df_15ch=df_15ch.drop(columns=['ch_16'])


    # csv_reader_12ch = CSVReader_12ch(dir_path)
    # df_12ch=csv_reader_12ch.process_files()
    # df_12ch_cleaned=ecg_clean_df_12ch(df_12ch)
    # input("")
    #同期用インデックスファイルを読み込みと書き込み
    handler = AutoIntegerFileHandler(dir_path+'/同期インデックス_nkmodule.txt')
    #同期する前
    # if(DEBUG_PLOT==True):
    #     Plot_15ch=MultiPlotter(df_15ch.copy(),RATE=RATE_15ch)
    #     Plot_15ch.multi_plot(xmin=0,xmax=100,ylim=0)
    args.output_filepath='{}_{}_{}s/{}'.format(args.name,args.date,str(args.time_range),args.pos)

    args.type='{}_{}_{}'.format(args.name,args.date,args.pos)
    args.dir_name='{}/{}'.format(args.name,args.type)
    #     Plot_16ch=MultiPlotter(df_16ch.copy(),RATE=RATE_15ch)
    #     Plot_16ch.multi_plot(xmin=0,xmax=100,ylim=0)
    #     plt.show()
    #     plt.close()
    # input()


    df_15ch_pf = ecg_clean_df_15ch(df_15ch=df_15ch.copy(),rate=RATE_15ch)
    # df_15ch_pf["ch1"]
    ecg_A2_np=df_15ch.copy()[args.target_ch].to_numpy().T
    print(ecg_A2_np.shape)
    rpeaks=PTwave_search4(ecg_A2=ecg_A2_np,header="ch_1",sampling_rate=RATE_15ch,args=args,time_length=args.time_range,method=args.peak_method)#1213からPQRST全部検出できるcwt方を使う。
    if(rpeaks[0]-int(RATE_15ch*0.4)<=0):
        rpeaks=rpeaks[1:]

    df_keep=df_15ch_pf.copy()[rpeaks[0]-int(RATE_15ch*0.4):rpeaks[0]+int(RATE_15ch*0.4)].reset_index(drop=True)
    # print(df_keep)
    # input("")
    if(len(rpeaks)<10):
        print("10拍未満!!!!!!")
        input("")

    for i in range(1,10):
        # print(i)
        # print(df_15ch[rpeaks[i]-int(RATE_15ch*0.4):rpeaks[i]+int(RATE_15ch*0.4)].reset_index(drop=True))
        df_keep=df_keep.copy()+(df_15ch_pf.copy()[rpeaks[i]-int(RATE_15ch*0.4):rpeaks[i]+int(RATE_15ch*0.4)].reset_index(drop=True))
        print(df_keep)
        # input("")
    print(df_keep)
    df_keep=df_keep/10.0
    return df_keep
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


    parser.add_argument("--name", type=str, default='sato')
    # parser.add_argument("--name", type=str, default='goto')
    parser.add_argument("--date", type=str, default='1229')

    # parser.add_argument("--name", type=str, default='takahashi')
    # parser.add_argument("--date", type=str, default='1230')

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
    # parser.add_argument("--cut_min_max_range", type=list, default=[0,10])
    parser.add_argument("--cut_min_max_range", type=list, default=[5,10])
    parser.add_argument("--time_range", type=float, default=0.8)
    parser.add_argument("--reverse", type=str, default='off')
    parser.add_argument("--target_ch", type=str, default='ch1')
    # parser.add_argument("--reverse", type=str, default='on')
    args = parser.parse_args()
    # args.output_filepath='{}_{}_2s/{}'.format(args.name,args.date,args.pos)
    # args.type='{}_{}_{}'.format(args.name,args.date,args.pos)
    # args.dir_name='{}_{}/{}'.format(args.name,args.date,args.type)
    # main(args)
    # args.pos='0'
    args.pos='1'
    # args.pos='2'
    # args.pos='3'
    # args.pos='4'
    # args.pos='5'
    # args.pos='5'
    # args.pos='grand'
    # args.pos='normal'
    # args.pos='right1'
    # args.pos='right2'
    # args.pos='left1'
    # args.pos='left2'
    # args.output_filepath='{}_{}_{}s/{}'.format(args.name,args.date,str(args.time_range),args.pos)

    # args.type='{}_{}_{}'.format(args.name,args.date,args.pos)
    # args.dir_name='{}/{}'.format(args.name,args.type)
    list_dfs=[]
    target_ch_list=[4,1,4,1,4,4,3,1,4,3,4,4]#佐藤
    # target_ch_list=[4,1,4,1,4,4,3,1,4,3,4,4]#高橋
    for j in range(12):
        args.pos=str(j+1)
        args.output_filepath='{}_{}_{}s/{}'.format(args.name,args.date,str(args.time_range),args.pos)

        args.type='{}_{}_{}'.format(args.name,args.date,args.pos)
        args.dir_name='{}/{}'.format(args.name,args.type)
        args.target_ch="ch_{}".format(str(target_ch_list[j]))
        # args.target_ch="ch_{}".format(str(4))
        df_keep=main(args)
        df_keep["ch_16"]=0.0
        list_dfs.append(df_keep)
    print(len(list_dfs))
    list_df_yoko=[]
    for i in range(4):
            df_3=list_dfs[i*3+2]
            df_2=list_dfs[i*3+1]
            df_1=list_dfs[i*3]
            print(df_2)
            print(df_3["ch_4"])
            df_2=df_2.add(df_3["ch_4"], axis='index')
            df_1=df_1.add(df_2["ch_4"], axis='index')
            print(df_2)
            list_df_yoko.append(df_1)
            list_df_yoko.append(df_2)
            list_df_yoko.append(df_3)
            print(i)
            # input("")
    df_keep_1=list_df_yoko[0]
    df_keep_2=list_df_yoko[1]
    df_keep_3=list_df_yoko[2]
    df_keep_4=list_df_yoko[3]
    df_keep_5=list_df_yoko[4]
    df_keep_6=list_df_yoko[5]
    df_keep_7=list_df_yoko[6]
    df_keep_8=list_df_yoko[7]
    df_keep_9=list_df_yoko[8]
    df_keep_10=list_df_yoko[9]
    df_keep_11=list_df_yoko[10]
    df_keep_12=list_df_yoko[11]

    df_keep_7=df_keep_7.add(df_keep_12["ch_13"], axis='index')
    df_keep_8=df_keep_8.add(df_keep_12["ch_13"], axis='index')
    df_keep_9=df_keep_9.add(df_keep_12["ch_13"], axis='index')

    df_keep_4=df_keep_4.add(df_keep_9["ch_13"], axis='index')
    df_keep_5=df_keep_5.add(df_keep_9["ch_13"], axis='index')
    df_keep_6=df_keep_6.add(df_keep_9["ch_13"], axis='index')

    df_keep_1=df_keep_1.add(df_keep_6["ch_13"], axis='index')
    df_keep_2=df_keep_2.add(df_keep_6["ch_13"], axis='index')
    df_keep_3=df_keep_3.add(df_keep_6["ch_13"], axis='index')
    new_columns_list=[]
    for j in range(12):
        # new_columns_list.append({f'ch_{i+1}': "pos_{}_".format(j+1)+str(i//4+1)+'_'+str(i%4+1) for i in range(15)})
        new_columns_list.append({f'ch_{i+1}': str(i//4+1+(j%3)*3)+'_'+str(i%4+1+(j//3)*3) for i in range(16)})
    df_renamed_1 = df_keep_1.rename(columns=new_columns_list[0])
    df_renamed_2 = df_keep_2.rename(columns=new_columns_list[1])
    df_renamed_3 = df_keep_3.rename(columns=new_columns_list[2])
    df_renamed_4 = df_keep_4.rename(columns=new_columns_list[3])
    df_renamed_5 = df_keep_5.rename(columns=new_columns_list[4])
    df_renamed_6 = df_keep_6.rename(columns=new_columns_list[5])
    df_renamed_7 = df_keep_7.rename(columns=new_columns_list[6])
    df_renamed_8 = df_keep_8.rename(columns=new_columns_list[7])
    df_renamed_9 = df_keep_9.rename(columns=new_columns_list[8])
    df_renamed_10 = df_keep_10.rename(columns=new_columns_list[9])
    df_renamed_11 = df_keep_11.rename(columns=new_columns_list[10])
    df_renamed_12 = df_keep_12.rename(columns=new_columns_list[11])

    # df_renamed_1 = df_keep_1.rename(columns={col: 'pos1_' + col for col in df_keep_1.columns})
    # df_renamed_2 = df_keep_2.rename(columns={col: 'pos2_' + col for col in df_keep_2.columns})
    # df_renamed_3 = df_keep_3.rename(columns={col: 'pos3_' + col for col in df_keep_3.columns})
    # df_renamed_4 = df_keep_4.rename(columns={col: 'pos4_' + col for col in df_keep_4.columns})
    # df_renamed_5 = df_keep_5.rename(columns={col: 'pos5_' + col for col in df_keep_5.columns})
    # df_renamed_6 = df_keep_6.rename(columns={col: 'pos6_' + col for col in df_keep_6.columns})
    # df_renamed_7 = df_keep_7.rename(columns={col: 'pos7_' + col for col in df_keep_7.columns})
    # df_renamed_8 = df_keep_8.rename(columns={col: 'pos8_' + col for col in df_keep_8.columns})
    # df_renamed_9 = df_keep_9.rename(columns={col: 'pos9_' + col for col in df_keep_9.columns})
    # df_renamed_10 = df_keep_10.rename(columns={col: 'pos10_' + col for col in df_keep_10.columns})
    # df_renamed_11 = df_keep_11.rename(columns={col: 'pos11_' + col for col in df_keep_11.columns})
    # df_renamed_12 = df_keep_12.rename(columns={col: 'pos12_' + col for col in df_keep_12.columns})
    list_final=[df_renamed_1,df_renamed_2,df_renamed_3,df_renamed_4,df_renamed_5,df_renamed_6,df_renamed_7,df_renamed_8,df_renamed_9,df_renamed_10,df_renamed_11,df_renamed_12]
    # 全てのデータフレームのカラムを1つのリストに集める
    all_columns = list(df_renamed_1.columns) +list(df_renamed_2.columns)+list(df_renamed_3.columns)+list(df_renamed_4.columns)+list(df_renamed_5.columns)+list(df_renamed_6.columns)+list(df_renamed_7.columns)+list(df_renamed_8.columns)+list(df_renamed_9.columns)+list(df_renamed_10.columns)+list(df_renamed_11.columns)+list(df_renamed_12.columns)
    # カラムの出現回数をカウント
    column_count = Counter(all_columns)
    # 2回以上出現するカラムを見つける
    common_columns = [column for column, count in column_count.items() if count >= 2]
    print(common_columns)
    new_ser_list=[]
    for common_column in common_columns:
        print(common_column)
        # print(average_columns(list_final,column=common_column))
        new_ser=average_columns(list_final,column=common_column)
        new_ser.name=common_column
        new_ser_list.append(new_ser)
    print(len(new_ser_list))
    print(new_ser_list[0])
    # input("")
    df_new_commons=pd.DataFrame(new_ser_list)
    df_new_commons=df_new_commons.transpose()
    print(df_new_commons)
    # list_all=list_final+new_df_list

    # df_all=pd.concat(list_all,axis=1)
    df_final=pd.concat(list_final,axis=1)
    # print(df_final["7_13"])
    # input()
    df_final_drop=df_final.drop(columns=common_columns)
    # df_all=df_all.drop(columns=common_column)
    df_all=pd.concat([df_final_drop,df_new_commons],axis=1)
    # columns_to_mean=[["1_4_1","2_1_1"],
    #                  ["1_4_2","2_2_1"],
    #                  ["1_4_3","2_3_1"],
    #                  ["2_4_3","3_1_1"],
    #                  ["2_4_3","3_2_1"],
    #                  ["2_4_3","3_3_1"],

    #            E\      ]
    sort_columns=[]
    for i in range(10):
        for j in range(13):
            column_name_sort="{}_{}".format(str(i+1),str(j+1))
            sort_columns.append(column_name_sort)

    # df_all=df_all.sort_index(axis=1)
    # print(df_all["7_13"])
    df_all = df_all.reindex(columns=sort_columns)
    print(df_all)
    # print(df_all["4_4"])
    # print(df_all)
    df_all.to_csv("{}_all.csv".format(args.name))
    np_all=df_all.copy().to_numpy()
    print(np_all.shape)
    np_reshape=np.zeros((10,13,96))
    for i in range(10):
        for j in range(13):
            np_reshape[i,j,:]=df_all["{}_{}".format(str(i+1),str(j+1))].to_numpy()
    print(np_reshape)
    print(np_reshape[0][0])
    np_reshape=np_reshape.transpose(1, 0, 2)

    print(np_reshape)
    print(np_reshape[12, 0, :][np.newaxis, np.newaxis, :])
    # input()：ｗ
    np_reshape=np_reshape-np_reshape[0,0, :][np.newaxis, np.newaxis, :]#
    org_data=np_reshape.copy()
    # result = A - A[1, 1, :][np.newaxis, np.newaxis, :]
    # print(org_data)
    # input()





    max_value=np_reshape.max()
    min_value=np_reshape.min()
    normalized_data = (np_reshape - min_value) / (max_value - min_value)
    np_reshape=normalized_data.copy()


    # # print(type(min_value))
    # # print(type(np_reshape))
    # # print(np_reshape)
    # # np_reshape=np_reshape+abs(min_value)
    # # print(np_reshape)
    # # input()
    # # # np_reshape=normalized_data

    # # log_data = np.where(np_reshape != 0, np.log(np_reshape), 1e-10)
    # # np_reshape=log_data
    # log_data=np.log(np_reshape)
    # max_log_value=log_data.max()
    # min_log_value=log_data.min()
    # # print(min_log_value)
    # flattened_array = log_data.flatten()
    # # 配列を並べ替え
    # sorted_array = np.sort(flattened_array)
    # # 2番目に小さい要素を取得
    # second_smallest = sorted_array[1]
    # np_reshape=log_data.copy()
    # print(max_log_value)
    # print(second_smallest)
    # print(min_log_value)


    plt.figure(figsize=(10, 13))  # ヒートマップのサイズを設定
    # sns.heatmap(np_reshape[:,:,0], annot=True, cmap='coolwarm',vmax=1,vmin=0)  # annot=Trueは各セルの値を表示
    # sns.heatmap(np_reshape[:, :, 0],  cmap='coolwarm', annot=True, cbar=True,vmax=max_log_value,vmin=second_smallest)
    # sns.heatmap(org_data[:, :, 47],  cmap='coolwarm', annot=True, cbar=True,vmax=1,vmin=0)
    sns.heatmap(np_reshape[:, :, 48],  cmap='coolwarm', annot=False, cbar=True,vmax=1,vmin=0)
    plt.title('10x13 Heatmap_org')  # ヒートマップのタイトル
    plt.show()
    # sns.heatmap(np_reshape[:, :, 49],  cmap='coolwarm', annot=True, cbar=True,vmax=1,vmin=0)
    # plt.title('10x13 Heatmap_org')  # ヒートマップのタイトル
    # plt.show()
    print(np_reshape)

    fig, ax = plt.subplots(figsize=(10, 13))
    # アニメーションの作成
    ani = animation.FuncAnimation(fig, update_log, frames=range(np_reshape.shape[2]), interval=100)
    ani.save('heatmap_animation.mp4', writer='ffmpeg', fps=10)
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    ani_4 = animation.FuncAnimation(fig4, update_wave_org, frames=range(np_reshape.shape[2]), interval=100)
    # アニメーションの保存 (ffmpegが必要)
    ani_4.save('wave.mp4', writer='ffmpeg', fps=10)


    # fig3, ax3 = plt.subplots(figsize=(10, 13))
    # # アニメーションの作成
    # ani3 = animation.FuncAnimation(fig, update_org, frames=range(np_reshape.shape[2]), interval=100)
    # ani3.save('heatmap_animation_org.mp4', writer='ffmpeg', fps=10)
    # fig2, ax2 = plt.subplots(figsize=(8, 6))
    # ani_2 = animation.FuncAnimation(fig2, update_wave_org, frames=range(np_reshape.shape[2]), interval=100)
    # # アニメーションの保存 (ffmpegが必要)
    # ani_2.save('wave_org.mp4', writer='ffmpeg', fps=10)













    # plt.show()
    # args.pos='72'
    # args.pos='144'
    # args.pos='216'
    # args.pos='288'
    # args.pos='right'
    # args.pos='left'
    # for i in range(1,13):
    #     args.pos=str(i)
    #     args.output_filepath='{}_{}_{}s/{}'.format(args.name,args.date,str(args.time_range),args.pos)

    #     args.type='{}_{}_{}'.format(args.name,args.date,args.pos)
    #     args.dir_name='{}/{}'.format(args.name,args.type)
    #     # args.png_path='./0_packet_loss_data/{}/{}_compare'.format(args.name,args.type)
    #     main(args)
    #     print(args.name+'_'+args.pos)

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