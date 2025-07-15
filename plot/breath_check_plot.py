import os
from re import search
import pandas as pd
import argparse
import sys
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sympy import plot
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_dir)
from config.settings import RAW_DATA_DIR

class CSVReader_1ch:
    def __init__(self, directory):
        self.directory = directory

    def search_files(self):
        files_found = []
        for filename in os.listdir(self.directory):
            if filename.endswith("resp.csv"):
                files_found.append(filename)
        return files_found

    def read_csv_file(self, filename):
        file_path = os.path.join(self.directory, filename)
        df = pd.read_csv(file_path,header=None)
        print(f"ファイル {filename} を読み込みました。")
        print("{}の行数は{}、列数は{}です。".format(filename, len(df), len(df.columns)))
        # 読み込んだデータフレームの操作などを行う
        # ...
        print(df)
        return df
    
    def process_files(self):
        files_found = self.search_files()
        print(files_found)
        df = pd.DataFrame()
        if len(files_found) > 0:
            df=self.read_csv_file(files_found[0])
            # df=self.header_make(df)
            print(df)
        else:
            print("指定した条件のCSVファイルは存在しません。")
            print("12ch")
        return df
    
    # CSVデータ特定の箇所のデータを抽出する
    def exctract_csv_range(self, df, start_index, end_index):
        df = df[start_index:end_index]
        return df

    def find_peaks(self, df):
        peaks, _ = find_peaks(df[0], height=0)
        return peaks

    # 呼吸成分を0から30の間に正規化する
    def normalize_breath(self, df):
        df[0] = (df[0] - df[0].min()) / (df[0].max() - df[0].min()) * 100
        return df

# データの読み込み
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
        print("{}の行数は{}、列数は{}です。".format(filename, len(df), len(df.columns)))
        # 読み込んだデータフレームの操作などを行う
        # ...
        return df
    
    def process_files(self):
        files_found = self.search_files()
        if len(files_found) > 0:
            df=self.read_csv_file(files_found[0])
        else:
            print("指定した条件のCSVファイルは存在しません。")
            print("12ch")
            exit()
        return df
    
    # CSVデータ特定の箇所のデータを抽出する
    def exctract_csv_range(self, df, start_index, end_index):
        df = df[start_index:end_index]
        return df

    def find_peaks(self, df):
        peaks, _ = find_peaks(df[0], height=0)
        return peaks

    

class DataPlotter:
    def __init__(self, df_16ch, df_1ch):
        self.df_16ch = df_16ch
        self.df_1ch = df_1ch
        # self.peaks = peaks

    def plot_16ch_data(self):
        plt.plot(self.df_16ch[0], label="16ch")
        plt.legend()
        plt.show()
        
    
    def plot_12ch_data(self, column):
        print(column)
        print(len(self.df_12ch[0:column]))
        if column == 0:
            plt.plot(self.df_12ch[0], label="12ch")
        else:
            plt.plot(self.df_12ch["A2"], label="12ch")
        plt.legend()
        plt.show()
    
    # 同時にプロットする
    def multi_plot(self):
        plt.plot(self.df_16ch['time'], self.df_16ch[0], label="16ch")
        plt.plot(self.df_1ch['time'], self.df_1ch[0], label="1ch")
        plt.legend()
        plt.show()

def main(args):
    breath_datas_dir = args.breath_datas_dir
    csv_reader_1ch = CSVReader_1ch(breath_datas_dir)
    df_1ch = csv_reader_1ch.process_files()
    df_1ch['time'] = df_1ch.index/100.00
    df_1ch = csv_reader_1ch.normalize_breath(df_1ch)
    csv_reader_16ch = CSVReader_16ch(breath_datas_dir)
    df_16ch = csv_reader_16ch.process_files()
    df_16ch['time'] = df_16ch.index/122.06
    data_plotter = DataPlotter(df_16ch, df_1ch)
    data_plotter.multi_plot()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_dir", type=str, default="")
    args = parser.parse_args()
    args.breath_datas_dir=RAW_DATA_DIR+"/sheet_sensor_csvdatas/takahashi_breath_check/takahashi_breath_check_0610_/通常呼吸"
    main(args)