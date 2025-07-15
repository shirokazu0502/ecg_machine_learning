import os
import pandas as pd
import argparse
import sys
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sympy import plot
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_dir)
from config.settings import RAW_DATA_DIR

class CSVReader_12ch:
    def __init__(self, directory):
        self.directory = directory

    def search_files(self):
        files_found = []
        for filename in os.listdir(self.directory):
            if (filename.startswith("KT") or filename.startswith("12ch")) and filename.endswith(".csv"):
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
        if len(files_found) > 0:
            df=self.read_csv_file(files_found[0])
            # df=self.header_make(df)
            print(df)
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
        print(df)
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
    def __init__(self, df_16ch, df_12ch):
        self.df_16ch = df_16ch
        self.df_12ch = df_12ch
        # self.peaks = peaks

    def plot_16ch_data(self):
        plt.plot(self.df_16ch[0], label="16ch")
        # plt.plot(self.peaks, self.df_16ch[0][self.peaks], "x", label="peak")
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

def main(args):
    patient_datas_dir = args.patient_datas_dir
    csv_reader_16ch = CSVReader_16ch(patient_datas_dir)
    df_16ch = csv_reader_16ch.process_files()
    csv_reader_12ch = CSVReader_12ch(patient_datas_dir)
    df_12ch = csv_reader_12ch.process_files()
    df_16ch = csv_reader_16ch.exctract_csv_range(df_16ch, 753000, 755000)
    df_12ch = csv_reader_12ch.exctract_csv_range(df_12ch, 753000, 755000)
    data_plotter = DataPlotter(df_16ch, df_12ch)
    data_plotter.plot_16ch_data()
    print(df_12ch.columns)
    data_plotter.plot_12ch_data(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_dir", type=str, default="")
    args = parser.parse_args()
    args.patient_datas_dir=RAW_DATA_DIR+"/patient_data/Af001_KT_4794335"
    main(args)