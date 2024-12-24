import serial
import serial.tools.list_ports
import time
import sys
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

################################
OUTPUT_FILENAME = datetime.now().strftime("db%Y%m%d_%H%M%S") + "_resp.csv" #出力ファイル名

CHUNK = 100 # number of data points to read at a time
RATE = 100 # time resolution of the recording device (Hz)
PLOT_LEN = 12 #PLOT_LEN * CHUNK / RATE = second
# TIME = 30 # measurement time (s)
################################

#符号付n進数を変換するときに使います。
def two_comp(val, bits):
    if(val & (1 << (bits -1))) != 0:
        val = val - (1 << bits)
    return val

if __name__ == '__main__':

    #グラフ初期化
    fig = plt.figure(num=None, figsize=(8, 6), dpi=100, facecolor='w', edgecolor='k')
    # plot_bufferACCx = np.zeros(CHUNK * PLOT_LEN)
    # plot_bufferACCy = np.zeros(CHUNK * PLOT_LEN)
    # plot_bufferACCz = np.zeros(CHUNK * PLOT_LEN)
    # plot_bufferGYROx = np.zeros(CHUNK * PLOT_LEN)
    # plot_bufferGYROy = np.zeros(CHUNK * PLOT_LEN)
    # plot_bufferGYROz = np.zeros(CHUNK * PLOT_LEN)
    plot_bufferTEMP0 = np.zeros(CHUNK * PLOT_LEN)
    # plot_bufferTEMP1 = np.zeros(CHUNK * PLOT_LEN)
    plot_time = np.arange(CHUNK * PLOT_LEN) / RATE

    # ax = plt.subplot(3, 1, 1)
    # lines_ACCx, = ax.plot(plot_time, plot_bufferACCx, color="blue", linewidth=0.5, linestyle="-", label="ACCx")
    # lines_ACCy, = ax.plot(plot_time, plot_bufferACCy, color="red", linewidth=0.5, linestyle="-", label="ACCy")
    # lines_ACCz, = ax.plot(plot_time, plot_bufferACCz, color="green", linewidth=0.5, linestyle="-", label="ACCz")
    # plt.ylim(-1 * (2 ** 15), (2 ** 15))
    # plt.legend(loc='lower right')

    # ax = plt.subplot(3, 1, 2)
    # lines_GYROx, = ax.plot(plot_time, plot_bufferGYROx, color="blue", linewidth=0.5, linestyle="-", label="GYROx")
    # lines_GYROy, = ax.plot(plot_time, plot_bufferGYROy, color="red", linewidth=0.5, linestyle="-", label="GYROy")
    # lines_GYROz, = ax.plot(plot_time, plot_bufferGYROz, color="green", linewidth=0.5, linestyle="-", label="GYROz")
    # plt.ylim(-1 * (2 ** 15), (2 ** 15))
    # plt.legend(loc='lower right')

    # ax = plt.subplot(3, 1, 3)
    # lines_TEMP0, = ax.plot(plot_time, plot_bufferTEMP0, color="blue", linewidth=0.5, linestyle="-", label="respiration")
    # lines_TEMP1, = ax.plot(plot_time, plot_bufferTEMP1, color="red", linewidth=0.5, linestyle="-", label="respiration")
    # plt.ylim(-1 * (2 ** 12), (2 ** 12))
    # plt.legend(loc='lower right')

    ax = plt.subplot(1, 1, 1)
    lines_TEMP0, = ax.plot(plot_time, plot_bufferTEMP0, color="blue", linewidth=1, linestyle="-", label="respiration")
    # lines_TEMP1, = ax.plot(plot_time, plot_bufferTEMP1, color="red", linewidth=0.5, linestyle="-", label="respiration")
    plt.xlabel('Time(s)',{'fontsize':15})
    plt.ylabel('Output',{'fontsize':15})
    plt.ylim(-1 * (2 ** 0), (2 ** 12))
    plt.tick_params(labelsize=15) #メモリの数字の大きさ
    plt.legend(loc='lower right' , fontsize=15)

    plt.tight_layout()
    #plt.show()
    plt.pause(.01)


    #シリアル通信設定
    #パソコン用
    ser = serial.Serial()
    #ser.baudrate = 38400
    ser.baudrate = 115200
#    serial_port = 1
#    serial_port = 2
    for idx in range(len(list(serial.tools.list_ports.comports()))):
        print(list(serial.tools.list_ports.comports())[idx])
        if "nRF" in (list(serial.tools.list_ports.comports())[idx][1]):
            PORT_NUM = idx
            print("detect^")
    PORT_NUM = 1
    ser.port = list(serial.tools.list_ports.comports())[PORT_NUM][0]
    print(list(serial.tools.list_ports.comports())[PORT_NUM][0])
    ser.open()
    #Raspberry pi用（↑の書き方でだめなときはこちらを使う）
    #ser = serial.Serial('/dev/ttyUSB0', 38400)

    time.sleep(1)

    output_file = open(OUTPUT_FILENAME, 'w')


    flag_check = 0
    count_data = 0
    count_time = 0
    data_bufferTEMP0 = np.zeros(CHUNK)
    # data_bufferTEMP1 = np.zeros(CHUNK)
    # data_bufferGYROx = np.zeros(CHUNK)
    # data_bufferGYROy = np.zeros(CHUNK)
    # data_bufferGYROz = np.zeros(CHUNK)
    # data_bufferACCx = np.zeros(CHUNK)
    # data_bufferACCy = np.zeros(CHUNK)
    # data_bufferACCz = np.zeros(CHUNK)
    PACK_DATA_NUM = 13
    ONE_DATA_LEN = 8 * 2
    while True:
        read_data = ser.readline()
        try:
            split_data = (read_data.decode('utf-8')[:]).rstrip('\r\n').split(':')[1].split(',')
            # print(split_data)
            if len(split_data) >= 10:
                packet_number = int(split_data[ONE_DATA_LEN * PACK_DATA_NUM], 16)
                for i in range(PACK_DATA_NUM):
                    respiration0 = two_comp((int(split_data[i * ONE_DATA_LEN + 1], 16) << 8) + int(split_data[i * ONE_DATA_LEN], 16), 16)
                    # print(i,respiration0)
                    # respiration1 = two_comp((int(split_data[i * ONE_DATA_LEN + 3], 16) << 8) + int(split_data[i * ONE_DATA_LEN + 2], 16), 16)
                    # gyro_x = two_comp((int(split_data[i * ONE_DATA_LEN + 5], 16) << 8) + int(split_data[i * ONE_DATA_LEN + 4], 16), 16)
                    # gyro_y = two_comp((int(split_data[i * ONE_DATA_LEN + 7], 16) << 8) + int(split_data[i * ONE_DATA_LEN + 6], 16), 16)
                    # gyro_z = two_comp((int(split_data[i * ONE_DATA_LEN + 9], 16) << 8) + int(split_data[i * ONE_DATA_LEN + 8], 16), 16)
                    # acc_x = two_comp((int(split_data[i * ONE_DATA_LEN + 11], 16) << 8) + int(split_data[i * ONE_DATA_LEN + 10], 16), 16)
                    # acc_y = two_comp((int(split_data[i * ONE_DATA_LEN + 13], 16) << 8) + int(split_data[i * ONE_DATA_LEN + 12], 16), 16)
                    # acc_z = two_comp((int(split_data[i * ONE_DATA_LEN + 15], 16) << 8) + int(split_data[i * ONE_DATA_LEN + 14], 16), 16)
                    # output_file.write(f"{respiration0},{respiration1},{gyro_x},{gyro_y},{gyro_z},{acc_x},{acc_y},{acc_z},{packet_number},\n")
                    output_file.write(f"{respiration0},{packet_number},\n")
                    data_bufferTEMP0[count_data] = respiration0
                    # data_bufferTEMP1[count_data] = respiration1
                    # data_bufferGYROx[count_data] = gyro_x
                    # data_bufferGYROy[count_data] = gyro_y
                    # data_bufferGYROz[count_data] = gyro_z
                    # data_bufferACCx[count_data] = acc_x
                    # data_bufferACCy[count_data] = acc_y
                    # data_bufferACCz[count_data] = acc_z
                    count_data += 1
                    count_time += 1
                    if count_data == CHUNK:
                        plot_bufferTEMP0 = np.hstack([plot_bufferTEMP0[CHUNK:CHUNK * PLOT_LEN], data_bufferTEMP0])
                        # plot_bufferTEMP1 = np.hstack([plot_bufferTEMP1[CHUNK:CHUNK * PLOT_LEN], data_bufferTEMP1])
                        # plot_bufferGYROx = np.hstack([plot_bufferGYROx[CHUNK:CHUNK * PLOT_LEN], data_bufferGYROx])
                        # plot_bufferGYROy = np.hstack([plot_bufferGYROy[CHUNK:CHUNK * PLOT_LEN], data_bufferGYROy])
                        # plot_bufferGYROz = np.hstack([plot_bufferGYROz[CHUNK:CHUNK * PLOT_LEN], data_bufferGYROz])
                        # plot_bufferACCx = np.hstack([plot_bufferACCx[CHUNK:CHUNK * PLOT_LEN], data_bufferACCx])
                        # plot_bufferACCy = np.hstack([plot_bufferACCy[CHUNK:CHUNK * PLOT_LEN], data_bufferACCy])
                        # plot_bufferACCz = np.hstack([plot_bufferACCz[CHUNK:CHUNK * PLOT_LEN], data_bufferACCz])

                        lines_TEMP0.set_data(plot_time , plot_bufferTEMP0)
                        # lines_TEMP1.set_data(plot_time , plot_bufferTEMP1)
                        # lines_GYROx.set_data(plot_time , plot_bufferGYROx)
                        # lines_GYROy.set_data(plot_time , plot_bufferGYROy)
                        # lines_GYROz.set_data(plot_time , plot_bufferGYROz)
                        # lines_ACCx.set_data(plot_time , plot_bufferACCx)
                        # lines_ACCy.set_data(plot_time , plot_bufferACCy)
                        # lines_ACCz.set_data(plot_time , plot_bufferACCz)

                        count_data = 0
                        # plt.pause(.01)
                        fig.canvas.draw()
                        fig.canvas.flush_events()

                        # if count_time / RATE == TIME:
                        #     exit()

        except UnicodeDecodeError: #エラーでなければtry-except処理無くても良いかも
            print("UnicodeDecodeError -> retry...\n")
    output_file.close()
