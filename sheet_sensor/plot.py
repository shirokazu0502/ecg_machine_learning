import pandas as pd
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

def plot_sc_all(dataframe,sampling_rate,png_path):
    df=dataframe
    RATE=sampling_rate
    fig, axes = plt.subplots(nrows=15, ncols=1, figsize=(80, 80))
    # サブプロットにデータをプロットする
    for i, ax in enumerate(axes.flatten()):
        if(i<15):
            TARGET=df.columns[i]
            # グラフのタイトルを設定
            ax.set_title(df.columns[i])
            times,val=peak_search(df[TARGET],RATE)
            dt=1.0/RATE
            # N=len(dataframe)
            time_np=np.array(times)
            time1=time_np*dt
            sc=pd.DataFrame(index=[])
            sc[0]=time1
            sc[1]=val
            ax.scatter(x=time1,y=val,color='red')
            time=np.arange(len(df))*dt
            ax.plot(time,df[TARGET])

    # グラフの間隔を調整
    fig.tight_layout()

    # グラフを表示
    if(png_path!=None):
        plt.savefig(png_path+'all_sc.png')
    else:
        # plt.show()
        print("")
    plt.close()
    plt.cla()

def peak_sc_plot(dataframe,RATE,TARGET):
    times,val=peak_search(dataframe[TARGET],RATE)
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
    #print(sc)
    # plt.show()
    plt.close()
    plt.cla()
    return sc

def peak_search(data_frame, sampling_rate):
# peak search for ecg """
    peak_times = []
    peak_vals = []
    temp_max = [-1, -9999]
    temp_min = [-1, 9999]
    max_search_flag = True
    max_ratio = 0.6
    # max_ratio = 0.4
    # max_ratio = 0.8
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
def peak_sc(dataframe,RATE,TARGET):
    times,val=peak_search(dataframe[TARGET],RATE)
    dt=1.0/RATE
    N=len(dataframe)
    time_np=np.array(times)
    time1=time_np*dt
    sc=pd.DataFrame(index=[])
    sc[0]=time1
    sc[1]=val
    #print(sc)
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
class CSVReader_16ch:
    def __init__(self,directory, file_name):
        self.directory = directory
        self.file_name = file_name

    # def search_files(self):
    #     files_found = []
    #     for filename in os.listdir(self.directory):
    #         if filename.startswith("db") and filename.endswith(".csv"):
    #             files_found.append(filename)
    #     return files_found

    def read_csv_file(self, filename):
        file_path = os.path.join(self.directory, filename)
        print(file_path)
        df = pd.read_csv(file_path,header=None)
        print(f"ファイル {filename} を読み込みました。")
        # 読み込んだデータフレームの操作などを行う
        # ...
        # print(df)
        return df

    def header_make(self,df):
        df=df.drop(columns=[16,17])
        df = df.rename(columns=lambda x: 'ch_' + str(x+1))
        return df

    def process_files(self):
        file_name=self.file_name
        df=self.read_csv_file(file_name)
        df=self.header_make(df)

        print(df)
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

class MultiPlotter:
    def __init__(self, df,RATE):
        self.df = df
        self.RATE=RATE
    def multi_plot(self,xmin,xmax,ylim):
        if(len(self.df.columns)==16):
            self.multi_plot_16ch(xmin,xmax,ylim)
        if(len(self.df.columns)==15):
            self.multi_plot_15ch(xmin,xmax,ylim)
        if(len(self.df.columns)==12):
            self.multi_plot_12ch(xmin,xmax,ylim)
        if(len(self.df.columns)>15):
            self.multi_plot_16ch(xmin,xmax,ylim)


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
        plt.close()
        plt.cla()
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
        plt.close()
        plt.cla()
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
        plt.close()
        plt.cla()
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
class plot_wave:
    def __init__(self,df,RATE):
        self.df=df
        self.RATE=RATE

    def plot_all_channels(self, xmin, xmax, ylim,png_path=None):
        if(len(self.df.columns)==15):
            self.plot_all_channels_15ch(xmin,xmax,ylim,png_path)
        if(len(self.df.columns)==16):
            self.plot_all_channels_16ch(xmin,xmax,ylim,png_path)

    def plot_all_channels_16ch(self, xmin, xmax, ylim,png_path):
        colors =["r"
                ,"r"
                ,"r"
                ,"r"
                ,"b"
                ,"b"
                ,"b"
                ,"b"
                ,"g"
                ,"g"
                ,"g"
                ,"g"
                ,"y"
                ,"y"
                ,"y"
                ,"y"
        ]
        # print(len(self.df))
        XLIM0, XLIM1 = xmin, xmax
        sample_rate = self.RATE
        dt = 1 / sample_rate
        plot_time = np.arange(len(self.df)) * dt
        YLIM = ylim
        lines_sound = []
        fig, ax = plt.subplots(figsize=(12, 6), dpi=100, facecolor='w', edgecolor='k')
        plt.rcParams["font.family"] = "Arial"   # 使用するフォント
        # plt.rcParams["font.size"] = 20
        for i in range(len(self.df.columns)):
            ax.plot(plot_time, self.df[self.df.columns[i]], linewidth=2.0, linestyle="-",
                    label=self.df.columns[i],color=colors[i])

        ax.set_xlim(XLIM0, XLIM1)
        if YLIM != 0:
            ax.set_ylim(-YLIM, YLIM)
        ax.legend(loc='upper right')
        # ax.set_xlabel("t(s)")
        # plt.legend(fontsize=15,ncol=2)
        plt.legend(fontsize=8,ncol=2)
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        plt.xlabel("t(s)",fontsize=30)
        plt.tight_layout()
        # plt.savefig("")
        if(png_path!=None):
            plt.savefig(png_path+'_all_channels.png')
            plt.show()
        else:
            plt.show()
        plt.close()
        plt.cla()
        return 0.0
    def plot_all_channels_15ch(self, xmin, xmax, ylim,png_path):
        colors =["r"
                ,"r"
                ,"r"
                ,"r"
                ,"b"
                ,"b"
                ,"b"
                ,"b"
                ,"g"
                ,"g"
                ,"g"
                ,"g"
                ,"y"
                ,"y"
                ,"y"
        ]
        # print(len(self.df))
        XLIM0, XLIM1 = xmin, xmax
        sample_rate = self.RATE
        dt = 1 / sample_rate
        plot_time = np.arange(len(self.df)) * dt
        YLIM = ylim
        lines_sound = []
        fig, ax = plt.subplots(figsize=(12, 6), dpi=100, facecolor='w', edgecolor='k')
        plt.rcParams["font.family"] = "Arial"   # 使用するフォント
        # plt.rcParams["font.size"] = 20
        for i in range(15):
            ax.plot(plot_time, self.df[self.df.columns[i]], linewidth=2.0, linestyle="-",
                    label=self.df.columns[i],color=colors[i])

        ax.set_xlim(XLIM0, XLIM1)
        if YLIM != 0:
            ax.set_ylim(-YLIM, YLIM)
        ax.legend(loc='upper right')
        # ax.set_xlabel("t(s)")
        # plt.legend(fontsize=15,ncol=2)
        plt.legend(fontsize=8,ncol=2)
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        plt.xlabel("t(s)",fontsize=30)
        plt.tight_layout()
        # plt.savefig("")
        # plt.show()
        if(png_path!=None):
            plt.savefig(png_path+'_all_channels_org.png')
            plt.show()
        else:
            plt.show()
        plt.close()
        plt.cla()
        return 0.0


def multi_pf(df,fp,fs,RATE):
    N=len(df)
    #drop_idx=[15,16]
    RATE=RATE
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

def check_directory_existence(directory_path):
    if os.path.exists(directory_path):
        message = "ディレクトリが存在します"
    else:
        message = "ディレクトリが存在しません"
    print(message)

def main(latest_file,dir_path,png_path=None):
    RATE=122.06
    # dir_path = "./"
    # dir_path = "../"
    csv_reader_16ch = CSVReader_16ch(directory=dir_path,file_name=latest_file)
    df_16ch=csv_reader_16ch.process_files()
    # print(df_16ch)
    cols=df_16ch.columns
    # print(df_16ch)
    df_all_ch_15ch=pd.DataFrame()#15chのときか、16chのとき（16chの平均値を基準として各チャネルの差分）
    df_all_ch=pd.DataFrame()#15chのときか、16chのとき（16chの平均値を基準として各チャネルの差分）

    #15chのとき　ch16を基準とする。
    for col in cols:
        df_all_ch_15ch[col]=df_16ch[col]-df_16ch['ch_16']
    print(df_all_ch_15ch)
    df_all_ch_15ch=df_all_ch_15ch.drop(columns=['ch_16'])
    df_all_ch_pf_15ch = multi_pf(df_all_ch_15ch.copy(),fp=0.2,fs=0.1,RATE=RATE)
    plotter_15ch=plot_wave(df_all_ch_pf_15ch,RATE=RATE)
    plotter_15ch.plot_all_channels(xmax=2,xmin=0,ylim=0,png_path=png_path)
    # #16chのとき
    # df_mean=df_16ch.mean(axis=1)
    # for col in cols:
    #     df_all_ch[col]=df_16ch[col].copy()-df_mean
    # # print(df_all_ch)
    # # df_all_ch_pf = multi_pf(df_all_ch.copy(),fp=0.2,fs=0.1,RATE=RATE)
    # # df_all_ch_pf = hpf_lpf(df_all_ch.copy(),HPF_fp=0.2,HPF_fs=0.1,LPF_fp=20,LPF_fs=30,RATE=RATE)
    # df_all_ch_pf = hpf_lpf(df_all_ch.copy(),HPF_fp=0.2,HPF_fs=0.1,LPF_fp=0,LPF_fs=0,RATE=RATE)
    # # plot_sc_all(dataframe=df_all_ch_pf,sampling_rate=RATE,png_path=png_path)
    # plotter=plot_wave(df_all_ch_pf,RATE=RATE)
    # plotter.plot_all_channels(xmax=2,xmin=0,ylim=0,png_path=png_path)

    return 0


    print(dir_path)
    csv_reader_16ch = CSVReader_16ch(dir_path)
    df_16ch=csv_reader_16ch.process_files()
    print(df_16ch)
    cols=df_16ch.columns
    df_15ch=pd.DataFrame()
    for col in cols:
        df_15ch[col]=df_16ch[col]-df_16ch['ch_16']
    df_15ch=df_15ch.drop(columns=['ch_16'])
    print(df_15ch)
    df_15ch_pf = hpf_lpf(df_15ch.copy(),HPF_fp=2.0,HPF_fs=1.0,LPF_fp=0,LPF_fs=0,RATE=122)
    if(DEBUG_PLOT==True):
        Plot_16ch=MultiPlotter(df_16ch.copy(),RATE=122)
        Plot_16ch.multi_plot(xmin=45,xmax=55,ylim=1000)
        plt.show()
        Plot_15ch=MultiPlotter(df_15ch.copy(),RATE=122)
        Plot_15ch.multi_plot(xmin=45,xmax=55,ylim=1000)
        # plt.show()
        Plot_15ch_pf=MultiPlotter(df_15ch_pf.copy(),RATE=122)
        Plot_15ch_pf.multi_plot(xmin=45,xmax=55,ylim=1000)
        plt.show()
# check_directory_existence("./../data/")
# check_directory_existence("./../0_packetloss_data/asano/asano_0710_0/db20230710_172443.csv")
# main(latest_file="../0_packetloss_data/asano/asano_0710_0/db20230710_172443.csv")
# main(latest_file="../0_packetloss_data/matumoto/matumoto_0623_right/db20230623_203113.csv")
# main(latest_file="test/db20230710_172443.csv")
# path="../data_since10_10/takahashi/takahashi_1102_0/"
# path="../data_since10_10/takahashi/takahashi_1102_right1/"
# path="../data_since10_10/takahashi/takahashi_1102_right2/"
# path="../data_since10_10/takahashi/takahashi_1102_left1/"
# file="db20231102_170450.csv"



# path="../data_since10_10/kawai/kawai_1115_0/"
# file="db20231115_142948.csv"
# main(latest_file=path+file,png_path=path+'test')
#