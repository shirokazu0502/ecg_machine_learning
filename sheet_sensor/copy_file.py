import os
import glob
import shutil
from datetime import datetime
import plot
import packet_check
import argparse
import sys
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_dir)
from config.settings import RAW_DATA_CSV_DIR

def get_data():
    # 現在の日付を取得
    current_date = datetime.now().date()
    # 月と日を4桁の形式で表示（0埋め）
    formatted_date = current_date.strftime("%m%d")
    # 4桁の形式で表示された日付の値
    print(formatted_date)
    return formatted_date

def main():
    print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    print(RAW_DATA_CSV_DIR)
    file_list = glob.glob(os.path.join(RAW_DATA_CSV_DIR, '*.csv'))
    latest_file = max(file_list, key=os.path.getmtime)
    latest_file_name = os.path.basename(latest_file)

    # dir_path = "./"
    csv_reader=packet_check.CSVReader_latentfile(dir_path=RAW_DATA_CSV_DIR,latentfile=latest_file_name)
    df=csv_reader.process_files()

    print(latest_file_name)

    packetloss_counter=packet_check.PacketLossCount(df)
    packet_loss_sum,packet_loss_values,indexes=packetloss_counter.process()

    plot.main(latest_file_name,RAW_DATA_CSV_DIR)

    a=input("copy? y/n")
    if(a!="y"):
        return 0
    name = input("nameを入力してください: ")


    how = input("計測条件を入力してください: ")
    month_day=get_data()
    # folder_path = "../data_since10_10/"+name+"/"+name+'_'+month_day+'_'+how
    COPY_PATH = RAW_DATA_CSV_DIR+"/"+name+"/"+name+'_'+month_day+'_'+how
    if not os.path.exists(COPY_PATH):
        os.makedirs(COPY_PATH)
        print("フォルダを作成しました。")
        print(COPY_PATH)
    else:
        print("指定されたフォルダは既に存在します。")

    shutil.copy(latest_file, COPY_PATH)
    plot.main(latest_file_name,png_path=COPY_PATH)

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--dir_name", type=str, default='')
    # parser.add_argument("--name", type=str, default='')
    # parser.add_argument("--type", type=str, default='')
    # parser.add_argument("--date", type=str, default=month_day)
    # parser.add_argument("--pos", type=str, default='')
    # args = parser.parse_args()

    # args.name=name
    # args.pos=how
    # args.type='{}_{}_{}'.format(args.name,args.date,args.pos)
    # # args.png_path='../data/packet_loss_check/'
    # # args.dir_name='../data/{}/{}/'.format(args.name,args.type)
    # args.png_path='../data_since10_10/packet_loss_check/'
    # args.dir_name='../data_since10_10/{}/{}/'.format(args.name,args.type)
    # print(args.dir_name)
    # packet_check.main(args)
main()
