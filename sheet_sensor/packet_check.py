import pandas as pd
import os
import argparse
import sys
import codecs
import struct
import binascii
from pylab import *
import matplotlib.cm as cm
import numpy as np
from scipy import signal
import neurokit2 as nk
def peak_search_nk(df_target,RATE):
    # print("safe")
    ecg_signal=df_target.copy().to_numpy().T
    ecg_signal=nk.ecg_clean(ecg_signal,sampling_rate=RATE,method='neurokit')
    # print(ecg_signal)
    _, rpeaks = nk.ecg_peaks(ecg_signal, RATE)
    # print(rpeaks['ECG_R_Peaks'])
    vals=ecg_signal[rpeaks['ECG_R_Peaks']]
    return rpeaks['ECG_R_Peaks'],vals

def plot_sc_all(dataframe,sampling_rate,png_path,packet_loss_indexes,packet_loss_values):
    indexes=packet_loss_indexes
    df=dataframe
    RATE=sampling_rate
    print("packet_loss={}".format(len(indexes)))
    plt.clf()
    plt.close()

    fig, axes = plt.subplots(nrows=15, ncols=1, figsize=(10, 80))
    # サブプロットにデータをプロットする
    # plt.title("packet_loss={}".format(len(indexes)))
    for i, ax in enumerate(axes.flatten()):
        # if(i==0):

        if(i<15):
            TARGET=df.columns[i]
            # グラフのタイトルを設定
            ax.set_title(df.columns[i])
            # times,val=peak_search(df[TARGET],RATE)
            times,val=peak_search_nk(df[TARGET],RATE)
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
            if(len(indexes)!=0):
                for j,index in enumerate(indexes):
                    line_width=packet_loss_values[j]*5.0
                    ax.axvline(x=index*dt,color='blue',linewidth=line_width,linestyle='--')
    # グラフの間隔を調整
    # fig.tight_layout()

    # グラフを表示
    # print(png_path)
    # input("")
    plt.savefig(png_path)
    plt.show()
    plt.clf()
    plt.close()

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
class MultiPlotter:
    def __init__(self, df,RATE):
        self.df = df
        self.RATE=RATE
    def multi_plot(self,xmin,xmax,ylim):
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

class CSVReader_latentfile:
    def __init__(self, dir_path,latentfile):
        self.directory = dir_path
        self.latentfile = latentfile


    def read_csv_file(self, filename):
        file_path = os.path.join(self.directory, filename)
        # print(file_path)
        df = pd.read_csv(file_path,header=None)
        print(f"ファイル {filename} を読み込みました。")
        # 読み込んだデータフレームの操作などを行う
        # print(df)
        return df

    def header_make(self,df):
        df=df.drop(columns=[17])
        df = df.rename(columns=lambda x: 'ch_' + str(x+1))
        df = df.rename(columns={'ch_17':'packet'})
        return df

    def process_files(self):
        df=self.read_csv_file(self.latentfile)
        df=self.header_make(df)
        # print(df)
        return df

class CSVReader:
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
        # print(df)
        return df

    def header_make(self,df):
        df=df.drop(columns=[17])
        df = df.rename(columns=lambda x: 'ch_' + str(x+1))
        df = df.rename(columns={'ch_17':'packet'})
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

# def get_indexes(array):
    # return np.where(array > 1)[0]
class PacketLossCount:
    def __init__(self,df):
        self.array=df["packet"].to_numpy()

    def differents(self):
        array2=self.array[::6]
        diff=array2[1:]-array2[:-1]
        return diff

    def get_values_indexes(self,diff):
        array=diff
        array=np.where((array < 1),array+256,array)
        array=array-1
        indexes=np.where((array != 0))[0]
        values=array[indexes]
        return values,indexes
    def DisplayPacketLoss(self,packet_loss_sum):
        print("パケットロスの合計={}".format(packet_loss_sum))

    def process(self):
        diff=self.differents()
        packet_loss_values,indexes=self.get_values_indexes(diff)
        packet_loss_sum=packet_loss_values.sum()
        self.DisplayPacketLoss(packet_loss_sum)
        return packet_loss_sum,packet_loss_values,indexes

def get_values_indexes(array):
    array=np.where((array < 1),array+256,array)
    print(array)
    array=array-1
    print(array)
    indexes=np.where((array != 0))[0]
    values=array[indexes]
    return values,indexes

def main(args):
    RATE=122.06
    # pd.set_option('display.max_rows', 10000)
    csv_reader=CSVReader(args.dir_name)
    df=csv_reader.process_files()
    #パケットロスの計算
    # array=df['packet'].to_numpy()
    # array2=array[::6]
    # diff=array2[1:]-array2[:-1]
    # print(diff)
    # print(args.pos)
    # packet_loss_values,indexes=get_values_indexes(diff)
    # print(indexes)
    # print("パケットロスの合計")
    # packet_loss_sum=packet_loss_values.sum()
    #パケットロスの計算
    packetloss_counter=PacketLossCount(df)
    packet_loss_sum,packet_loss_values,indexes=packetloss_counter.process()

    print(len(indexes))
    indexes=6*indexes
    print(indexes)
    df_16ch=df.drop(columns=["packet"])
    cols=df_16ch.columns
    df_15ch=pd.DataFrame()
    for col in cols:
        df_15ch[col]=df_16ch[col]-df_16ch['ch_16']
    df_15ch=df_15ch.drop(columns=['ch_16'])
    # print(df_15ch)
    df_15ch_pf = hpf_lpf(df_15ch.copy(),HPF_fp=2.0,HPF_fs=1.0,LPF_fp=0,LPF_fs=0,RATE=RATE)

    # print(df_15ch_pf)
    # print(df["packet"][indexes[0]:])
    # Plot_15ch=MultiPlotter(df_15ch_pf.copy(),RATE=122)
    # Plot_15ch.multi_plot(xmin=0,xmax=40,ylim=0)
    # plt.show()

    png_path=args.png_path+'{}_packet_loss={}.png'.format(args.type,packet_loss_sum)
    print(type(indexes))
    plot_sc_all(dataframe=df_15ch_pf,sampling_rate=RATE,png_path=png_path,packet_loss_indexes=indexes,packet_loss_values=packet_loss_values)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_name", type=str, default='')
    parser.add_argument("--name", type=str, default='')
    parser.add_argument("--type", type=str, default='')
    parser.add_argument("--date", type=str, default='')
    parser.add_argument("--pos", type=str, default='')
    args = parser.parse_args()

    args.name=input("name")
    args.pos=input("pos")
    args.type='{}_{}_{}'.format(args.name,args.date,args.pos)
    # args.png_path='./data/packet_loss_check/'
    # args.dir_name='./data/{}/{}/'.format(args.name,args.type)
    args.png_path='./data/packet_loss_check/'
    args.dir_name='./data/{}/{}/'.format(args.name,args.type)
    print(args.dir_name)
    main(args)


    # poses=["left","right"]
    # # # poses=["right"]

    # for pos in poses:
    #     args.pos=pos
    #     args.type='{}_{}_{}'.format(args.name,args.date,args.pos)
    #     # args.png_path='./data/{}/'.format(args.name,args.type)
    #     args.png_path='./data/packet_loss_check/'
    #     args.dir_name='./data/{}/{}/'.format(args.name,args.type)
    #     print(args.dir_name)
    #     main(args)


    # for pos in range(0,5):
    #     args.pos=str(pos*72)
    #     args.type='{}_{}_{}'.format(args.name,args.date,args.pos)
    #     # args.png_path='./data/{}/'.format(args.name,args.type)
    #     args.png_path='./data/packet_loss_check/'
    #     args.dir_name='./data/{}/{}/'.format(args.name,args.type)
    #     main(args)
