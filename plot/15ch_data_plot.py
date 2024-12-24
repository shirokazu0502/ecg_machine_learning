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


class CSVReader_12ch:
    def __init__(self, directory):
        self.directory = directory

    def search_files(self):
        files_found = []
        for filename in os.listdir(self.directory):
            if (
                filename.startswith("KT") or filename.startswith("12ch")
            ) and filename.endswith(".csv"):
                files_found.append(filename)
        return files_found

    def read_csv_file(self, filename):
        file_path = os.path.join(self.directory, filename)
        df = pd.read_csv(file_path, header=None)
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
            df = self.read_csv_file(files_found[0])
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
        df = pd.read_csv(file_path, header=None)
        print(f"ファイル {filename} を読み込みました。")
        print("{}の行数は{}、列数は{}です。".format(filename, len(df), len(df.columns)))
        # 読み込んだデータフレームの操作などを行う
        # ...
        return df

    def process_files(self):
        files_found = self.search_files()
        if len(files_found) > 0:
            df = self.read_csv_file(files_found[0])
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
        plt.savefig("16ch_patient3.svg", format="svg")
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
    normal_datas_dir = args.normal_datas_dir
    for dir_path in [patient_datas_dir, normal_datas_dir]:
        csv_reader_16ch = CSVReader_16ch(dir_path)
        df_16ch = csv_reader_16ch.process_files()
        df_16ch = df_16ch.drop(17, axis=1)
        print(df_16ch)
        csv_reader_12ch = CSVReader_12ch(dir_path)
        df_12ch = csv_reader_12ch.process_files()
        search_list = [[5, 1, -4, -5, 0, -1, -2, -3, 1, 2, -1, 1, 4, 3, -3, -1, 220]]
        # search_list=[[17430, 16952, 16799, 16748, 15324, 14738, 16094, 15870, 15265, 14769, 13534, 13868, 12025, 11946, 13721, 13161, 11]]
        # search_list=[[3, 6, 14, 10, 24, 11, 9, 11, 15, 13, 1, 4, 3, 9, 1, 3, 105]]
        search_df = pd.DataFrame(search_list)
        # データフレームの形状を取得します
        main_shape = df_16ch.shape
        row_shape = search_df.shape
        print(main_shape)
        print(row_shape)
        # target_df = df_16ch[(df_16ch[16]==8) & (df_16ch[0]==13476) & (df_16ch[1]==13524) & (df_16ch[2]==12513) ]
        # 検索範囲を決定します
        # max_row_index = main_shape[0] - row_shape[0]
        # for i in range(max_row_index + 2):
        #     # メインのデータフレームの一部を抽出します
        #     temp_df = df_16ch.iloc[i:i + row_shape[0]]
        #     print(temp_df)
        #     # 行のリストと一致するかどうかをチェックします
        #     if temp_df.reset_index(drop=True).equals(search_df):
        #         print("{}で見つかりました".format(i))
        #         return i
        # print(target_df)
        # print("Not found")
        if dir_path == patient_datas_dir:
            # #     df_12ch = csv_reader_12ch.exctract_csv_range(df_12ch, 753000, 755000)
            #     for i in range(0, len(df_16ch), 100000):
            #         extract_df_16ch = csv_reader_16ch.exctract_csv_range(df_16ch, i, i+100000)
            data_plotter = DataPlotter(df_16ch, df_12ch)
            data_plotter.plot_16ch_data()

        # peaks = csv_reader_16ch.find_peaks(df_16ch)
        if dir_path == patient_datas_dir:
            plot_column = 0
        else:
            plot_column = 1
            # 1行目をカラム名として設定
            df_12ch.columns = df_12ch.iloc[0]
            # 1行目を削除
            df_12ch = df_12ch.drop(0)
            print(df_12ch["A2"])
            # 　整数型に変更
            df_12ch = df_12ch.astype(float)
        print(df_12ch.columns)
        # data_plotter.plot_12ch_data(plot_column)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_dir", type=str, default="")
    args = parser.parse_args()
    args.patient_datas_dir = RAW_DATA_DIR + "/patient_data/Af003_KR_3894797"
    args.normal_datas_dir = RAW_DATA_DIR + "/takahashi_test/taniguchi/taniguchi_1107_0"
    main(args)
