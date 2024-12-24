from heapq import merge
import pandas as pd
import os
import pandas as pd
import argparse
import sys
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sympy import plot

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
from config.settings import RAW_DATA_DIR, OUTPUT_DIR

process_flg = 0
if process_flg == 0:
    patient_datas_dir = RAW_DATA_DIR + "/extract_patient_data/patient5"
    # df = pd.read_csv(patient_datas_dir+"/db20240513_180746.csv",header=None)
    df = pd.read_csv(
        patient_datas_dir + "/db20240822_171309_overnight.csv", header=None
    )
    # あるデータフレームの範囲を抽出する
    # # 5000刻みで3つfor文で作成する
    # for i in range(3):
    #     extract_df = df[5000*i:5000*(i+1)]
    #     extract_df.to_csv(RAW_DATA_DIR+'/extract_patient_data/db_patient_16ch_data{}_{}_{}.csv'.format(i+1, 5000*i, 5000*(i+1)), header=False, index=False)
    #     print("データを抽出しました。")
    extract_df = df[0:20000]
    extract_df.to_csv(
        RAW_DATA_DIR
        + "/extract_patient_data/patient3/db_patient3_16ch_data_0s_19s.csv",
        header=False,
        index=False,
    )
elif process_flg == 1:
    merge_df = pd.DataFrame(columns=["A1", "A2", "V1", "V2", "V3", "V4", "V5", "V6"])
    patient_output_dir = (
        OUTPUT_DIR
        + "/patient_data0427/PRTweight_1.0_0.001_1.0_augumentation=/Waveforms/patient"
    )
    # 10個のCSVをつなげる
    for i in range(10):
        df = pd.read_csv(
            patient_output_dir
            + "/patient_0427_0.8s_0_dataset0{:02}_reconx.csv".format(i),
            header=0,
        )
        merge_df = pd.concat([merge_df, df])
    merge_df.to_csv(
        patient_output_dir + "/patient_merge.csv", header=False, index=False
    )
else:
    print("i")
