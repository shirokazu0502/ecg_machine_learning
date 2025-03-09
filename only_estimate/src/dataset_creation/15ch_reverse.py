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
import time
import numpy as np
from glob import glob
from scipy.signal import detrend, butter, filtfilt
from scipy.ndimage import uniform_filter1d, median_filter

### グラフ表示が要らない場合はFalse
DEBUG_PLOT = True
# DEBUG_PLOT = False
###
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import neurokit2 as nk
import warnings
import csv
from matplotlib.ticker import MultipleLocator

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_dir)
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
from config.name_dic import select_name_and_date


def reverse_signal(csv_paths):
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        df.loc[:, df.columns.str.contains("ch")] *= -1
        df.to_csv(csv_path, index=False)


def main(args):
    csv_paths = sorted(
        glob(args.dataset_output_path + "/" + args.output_filepath + "/dataset_*.csv")
    )
    monving_ave_csv_pahts = sorted(
        glob(
            args.dataset_output_path
            + "/"
            + args.output_filepath
            + "/moving_ave_datasets/dataset_*.csv"
        )
    )
    reverse_signal(csv_paths)
    reverse_signal(monving_ave_csv_pahts)
    print("終了")


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
    # parser.add_argument("--TARGET_CHANNEL_15CH", type=str, default='ch_1')
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
    # args.name='goto'#yoshikura takahashi matumoto
    # args.date='1219'
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
    args.cut_min_max_range = [1.0, 50.0]
    args.reverse = "off"
    args.type = "{}_{}_{}".format(args.name, args.date, args.pos)
    args.dir_name = "{}/{}".format(args.name, args.type)
    # args.project_path='/home/cs28/share/goto/goto/ecg_project'
    # args.raw_datas_os=RAW_DATA_DIR
    # args.processed_datas_os=args.project_path+'/data/processed'
    # args.processed_datas_os=PROCESSED_DATA_DIR
    args.dataset_made_date = DATASET_MADE_DATE
    args.raw_datas_dir = RAW_DATA_DIR + "/takahashi_test/{}".format(args.dir_name)
    args.dataset_output_path = PROCESSED_DATA_DIR + "/pqrst_nkmodule_since{}_{}".format(
        args.dataset_made_date, args.peak_method
    )
    args.test_images_path = TEST_DIR + "/raw_datas_test"
    main(args)
