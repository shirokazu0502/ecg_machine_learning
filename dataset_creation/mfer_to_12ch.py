import argparse
import sys
import os
import codecs
import struct
import binascii
from pylab import *
import matplotlib.cm as cm
import pandas as pd

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
    RATE_16CH,
    TIME,
    DATASET_MADE_DATE,
)

from config.name_dic import select_name_and_date

### グラフ表示が要らない場合はFalse
DEBUG_PLOT = True
###


def search_files(dir):
    files_found = []
    for filename in os.listdir(dir):
        if filename.startswith("") and filename.endswith(".mwf"):
            files_found.append(filename)
    return files_found


def process_files(dir):
    files_found = search_files(dir)
    if len(files_found) > 0:
        file_name = files_found[0]
        print("file_name={}".format(file_name))
    else:
        print("指定した条件のファイルは存在しません。")
        exit()
    return file_name


def convert_8_to_12ch(args, df_data):
    output_file_path = args.raw_datas_dir + "/12ch_" + args.file_name + ".csv"
    output_file_path2 = "./data/12ch_" + args.file_name + ".csv"
    df2 = df_data
    df2["A3"] = df2["A2"] - df2["A1"]
    # df2["aVR"]=-(df2["A1"]-df2["A2"]/2.0)#間違い
    df2["aVR"] = -(
        df2["A1"] + df2["A2"] / 2.0
    )  # これで変換しなおしてデータを作り直す。
    df2["aVL"] = df2["A1"] - df2["A3"] / 2.0
    df2["aVF"] = df2["A2"] + df2["A3"] / 2.0
    print(df2)
    new_order = [
        "A1",
        "A2",
        "A3",
        "aVR",
        "aVL",
        "aVF",
        "V1",
        "V2",
        "V3",
        "V4",
        "V5",
        "V6",
    ]
    df2 = df2[new_order]
    dt = 1.0 / 500.0
    t = np.arange(len(df2)) * dt

    # プロットの設定とデータのプロット
    fig, axis = plt.subplots(len(df2.columns), 1)
    color_list = ["b", "g", "r", "c", "m", "y", "k", "w"]

    plt.suptitle(args.type)
    print(df2)

    for p, num in enumerate(df2.columns):
        axis[p].plot(t, df2[num], label=num, color=color_list[p % 4])
        fig.legend()

    # ラベルや凡例の表示
    plt.xlabel("Time")
    plt.ylabel("Value")
    # グリッドの表示（任意）
    plt.grid(True)
    # プロットの表示
    plt.tight_layout()
    png_path = args.test_images_path_12lead + "12ch_" + args.type + ".png"
    plt.savefig(png_path)
    # plt.show()
    plt.close()

    df2.to_csv(output_file_path, index=None)
    # df2.to_csv(output_file_path2,index=None)


def convert_mfer_to_8ch(args):
    data = []
    if args.time_length == 10:
        flag = "1e840001"
    if args.time_length == 24:
        flag = "1e840002"
    # flag1='1E'

    with open(args.raw_datas_dir + "/" + args.file_name, "rb") as f:
        offset = 2000
        f.seek(0)
        idx = 0
        while True:
            bytes = f.read(4)
            # print(offset)
            flag1 = str(bytes.hex()) == flag
            # print(flag1)
            # if(offset>2291):
            # print(offset)
            # input()
            if bytes:
                data = struct.unpack(">i", bytes)[0]
                # print(bytes.hex())
                if flag1 == True:
                    # print(offset)
                    idx = offset
                    break
                    # print(f"{data=}")
                # print("idx={}".format(offset))
                # data = struct.unpack('>h', bytes)[0]
                # print(f"{data=}")
                # break

            if not data:
                break
            offset += 1
            f.seek(offset)

    # with open(sys.argv[1], 'rb') as f:
    with open(args.raw_datas_dir + "/" + args.file_name, "rb") as f:
        ###データ取得
        data_list = []
        df_data = pd.DataFrame()
        time_list = []
        counter = 0
        sampling_rate = 500  # Hz
        dt = 1.0 / sampling_rate
        time_length = args.time_length
        print("\n  データ読み込み中")
        f.seek(idx + 6)
        channel_list = ["A1", "A2", "V1", "V2", "V3", "V4", "V5", "V6"]
        for ch in channel_list:
            for _ in range(time_length * sampling_rate):
                bytes = f.read(2)
                # print(bytes.hex())
                if bytes:
                    data = struct.unpack("<h", bytes)[0]
                    # data = struct.unpack('>h', bytes)[0]
                    # print(f"{data=}")
                    data_list.append(data)
                    time_list.append(counter * sampling_rate)
                    counter += 1
            df_data[ch] = data_list
            data_list = []
        print(df_data)
        # input()
    return df_data


def is_directory_exists(directory_path):
    """
    指定されたディレクトリが存在するかどうかを確認します。

    :param directory_path: 確認したいディレクトリのパス
    :return: ディレクトリが存在する場合はTrue、存在しない場合はFalseを返します。
    """
    return os.path.exists(directory_path) and os.path.isdir(directory_path)


def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def main(args):
    file_name = process_files(dir=args.raw_datas_dir)
    args.file_name = file_name
    df_data = convert_mfer_to_8ch(args)
    print(df_data)
    convert_8_to_12ch(args, df_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, default="")
    parser.add_argument("--file_name", type=str, default="")
    parser.add_argument(
        "--time_length", type=int, default=24
    )  # 24 or 10を選ぶ。若干バイナリの読み方変わる。BzとMFER format　ツールで見比べる。
    # parser.add_argument("--name", type=str, default='taniguchi')
    # parser.add_argument("--date", type=str, default='1107')
    parser.add_argument("--type", type=str, default="")
    # parser.add_argument("--os_dir", type=str, default='')
    parser.add_argument("--raw_datas_dir", type=str, default="")
    parser.add_argument("--test_images_path_12lead", type=str, default="")
    args = parser.parse_args()

    # args.test_images_path_12lead='./data/{}/'.format(args.name,args.name,args.date)
    # args.test_images_path_12lead='./0_packetloss_data/{}/'.format(args.name)
    # args.test_images_path_12lead='./0_packetloss_data_1115/{}/'.format(args.name)
    # args.name='goto'#yoshikura takahashi matumoto
    # args.date='0418'
    args.name, args.date = select_name_and_date()

    args.pos = "0"  # 0なら胸,1なら腹,2なら腰
    args.type = "{}_{}_{}".format(args.name, args.date, args.pos)
    args.dir_name = "{}/{}".format(args.name, args.type)
    # args.test_images_path_12lead='./0_packetloss_data_{}/{}/'.format(DATASET_MADE_DATE,args.name)
    args.test_images_path_12lead = (
        TEST_DIR + "/raw_datas_test" + "/" + DATASET_MADE_DATE + "/12lead_raw/"
    )
    args.raw_datas_dir = RAW_DATA_DIR + "/takahashi_test/{}".format(args.dir_name)
    create_directory_if_not_exists(args.test_images_path_12lead)
    # args.test_images_path_12lead='./data/{}/{}_{}/'.format(args.name,args.name,args.date)
    # args.test_images_path_12lead='./data/{}_{}/'.format(args.name,args.date)

    # pos='0'
    # args.raw_datas_dir='{}{}'.format(args.test_images_path_12lead,args.type)
    print(args.raw_datas_dir)
    print(args.raw_datas_dir)

    main(args)

    #

    # for i in range(5):
    #     pos=str(72*i)
    #     args.type='{}_{}_{}'.format(args.name,args.date,pos)
    #     args.raw_datas_dir='{}{}'.format(args.test_images_path_12lead,args.type)
    #     if is_directory_exists(args.raw_datas_dir):
    #         print(f"{args.raw_datas_dir} は存在します。")
    #         main(args)
    #     else:
    #         # print("存在しません")
    #         print(f"{args.raw_datas_dir} は存在しません。")
    #     # print(args.raw_datas_dir)
    #     # main(args)

    # dir_names=["left","right"]
    # for i in range(len(dir_names)):
    #     args.type='{}_{}_{}'.format(args.name,args.date,dir_names[i])
    #     args.raw_datas_dir='{}{}'.format(args.test_images_path_12lead,args.type)
    #     # print(args.raw_datas_dir)
    #     # main(args)
    #     if is_directory_exists(args.raw_datas_dir):
    #         print(f"{args.raw_datas_dir} は存在します。")
    #         main(args)
    #     else:
    #         # print("存在しません")
    #         print(f"{args.raw_datas_dir} は存在しません。")
