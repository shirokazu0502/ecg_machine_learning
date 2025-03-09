import serial
import serial.tools.list_ports
import time
import sys
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os
from scipy import signal

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_dir)
print(base_dir)
from config.settings import RAW_DATA_CSV_DIR


################################
OUTPUT_FILENAME = datetime.now().strftime("db%Y%m%d_%H%M%S") + ".csv"  # 出力ファイル名
# OUTPUT_FILENAME = RAW_DATA_CSV_DIR+'/'+datetime.now().strftime("db%Y%m%d_%H%M%S") + ".csv" #出力ファイル名
CHUNK = 180  # number of data points to read at a time
# RATE = 44100 # time resolution of the recording device (Hz)
################################
PORT_NUM = 0
PACKET_LEN = 30  # CHUNK should be > this value
# PACKET_LEN = 90# CHUNK should be > this value

# ADC_BIT = 24
ADC_BIT = 16

if ADC_BIT == 24:
    SPI_DLEN = 4  # 24bit
    BIT_WORD = 6  # 24bit
    RATE = 122
    PLOT_LEN = 5  # PLOT_LEN * CHUNK / RATE = second
    YLIM = 2**23
    # YLIM = 2 ** 15
    SER_DLEN = 181  # 15ch * 4set * 3byte + 1
else:
    SPI_DLEN = 6  # 16bit
    BIT_WORD = 4  # 16bit
    # RATE = 243 # time resolution of the recording device (Hz)
    RATE = 122  # time resolution of the recording device (Hz)
    PLOT_LEN = 10  # PLOT_LEN * CHUNK / RATE = second
    YLIM = 2**9
    # YLIM = 2 ** 15
    SER_DLEN = 193  # 16ch * 6set * 2byte + 1
    # SER_DLEN = 211 # 15ch * 7set * 2byte + 1


def init_hpf(sampling_rate, fp, fs):
    """high pass filter"""
    # fp = 0.5                          # 通過域端周波数[Hz]
    # fs = 0.1                          # 阻止域端周波数[Hz]
    gpass = 1  # 通過域最大損失量[dB]
    gstop = 60  # 阻止域最小減衰量[dB]
    norm_pass = fp / (sampling_rate / 2)
    norm_stop = fs / (sampling_rate / 2)
    N, Wn = signal.cheb2ord(
        wp=norm_pass, ws=norm_stop, gpass=gpass, gstop=gstop, analog=0
    )
    b, a = signal.cheby2(N, gstop, Wn, "high")
    z = signal.lfilter_zi(b, a)
    return b, a, z


def init_lpf(sampling_rate, fp, fs):
    """low pass filter"""
    # fp = 30                          # 通過域端周波数[Hz]
    # fs = 50                         # 阻止域端周波数[Hz]
    gpass = 1  # 通過域最大損失量[dB]
    gstop = 45  # 阻止域最小減衰量[dB]
    norm_pass = fp / (sampling_rate / 2)
    norm_stop = fs / (sampling_rate / 2)
    N, Wn = signal.cheb2ord(
        wp=norm_pass, ws=norm_stop, gpass=gpass, gstop=gstop, analog=0
    )
    b, a = signal.cheby2(N, gstop, Wn, "low")
    z = signal.lfilter_zi(b, a)
    return b, a, z


# 符号付n進数を変換するときに使います。
def two_comp(val, bits):
    if (val & (1 << (bits - 1))) != 0:
        val = val - (1 << bits)
    return val


if __name__ == "__main__":
    NUM_CH = 16
    # グラフ初期化
    # plot_buffer1 = np.zeros(CHUNK * PLOT_LEN)

    lpf_b, lpf_a, lpf_zi = init_lpf(RATE, 50, 60)
    hpf_b, hpf_a, hpf_zi = init_hpf(RATE, 0.3, 0.1)

    lpf_zi_list = []
    hpf_zi_list = []
    plot_buffer = []
    for _ in range(NUM_CH):
        lpf_zi_list.append(np.copy(lpf_zi))
        hpf_zi_list.append(np.copy(hpf_zi))
        plot_buffer.append(np.zeros(CHUNK * PLOT_LEN))

    plot_time = np.arange(CHUNK * PLOT_LEN) / RATE
    lines_sound = []

    fig = plt.figure(num=None, figsize=(8, 6), dpi=100, facecolor="w", edgecolor="k")
    ax = plt.subplot(4, 1, 1)
    (temp_line,) = ax.plot(
        plot_time, plot_buffer[0], color="blue", linewidth=0.5, linestyle="-", label="0"
    )
    lines_sound.append(temp_line)
    (temp_line,) = ax.plot(
        plot_time, plot_buffer[1], color="red", linewidth=0.5, linestyle="-", label="1"
    )
    lines_sound.append(temp_line)
    (temp_line,) = ax.plot(
        plot_time,
        plot_buffer[2],
        color="green",
        linewidth=0.5,
        linestyle="-",
        label="2",
    )
    lines_sound.append(temp_line)
    (temp_line,) = ax.plot(
        plot_time,
        plot_buffer[3],
        color="black",
        linewidth=0.5,
        linestyle="-",
        label="3",
    )
    lines_sound.append(temp_line)
    plt.ylim(-1 * YLIM, YLIM)
    plt.legend(loc="lower right")

    ax = plt.subplot(4, 1, 2)
    (temp_line,) = ax.plot(
        plot_time, plot_buffer[4], color="blue", linewidth=0.5, linestyle="-", label="4"
    )
    lines_sound.append(temp_line)
    (temp_line,) = ax.plot(
        plot_time, plot_buffer[5], color="red", linewidth=0.5, linestyle="-", label="5"
    )
    lines_sound.append(temp_line)
    (temp_line,) = ax.plot(
        plot_time,
        plot_buffer[6],
        color="green",
        linewidth=0.5,
        linestyle="-",
        label="6",
    )
    lines_sound.append(temp_line)
    (temp_line,) = ax.plot(
        plot_time,
        plot_buffer[7],
        color="black",
        linewidth=0.5,
        linestyle="-",
        label="7",
    )
    lines_sound.append(temp_line)
    plt.ylim(-1 * YLIM, YLIM)
    plt.legend(loc="lower right")

    ax = plt.subplot(4, 1, 3)
    (temp_line,) = ax.plot(
        plot_time, plot_buffer[8], color="blue", linewidth=0.5, linestyle="-", label="8"
    )
    lines_sound.append(temp_line)
    (temp_line,) = ax.plot(
        plot_time, plot_buffer[9], color="red", linewidth=0.5, linestyle="-", label="9"
    )
    lines_sound.append(temp_line)
    (temp_line,) = ax.plot(
        plot_time,
        plot_buffer[10],
        color="green",
        linewidth=0.5,
        linestyle="-",
        label="10",
    )
    lines_sound.append(temp_line)
    (temp_line,) = ax.plot(
        plot_time,
        plot_buffer[11],
        color="black",
        linewidth=0.5,
        linestyle="-",
        label="11",
    )
    lines_sound.append(temp_line)
    plt.ylim(-1 * YLIM, YLIM)
    plt.legend(loc="lower right")

    ax = plt.subplot(4, 1, 4)
    (temp_line,) = ax.plot(
        plot_time,
        plot_buffer[12],
        color="blue",
        linewidth=0.5,
        linestyle="-",
        label="12",
    )
    lines_sound.append(temp_line)
    (temp_line,) = ax.plot(
        plot_time,
        plot_buffer[13],
        color="red",
        linewidth=0.5,
        linestyle="-",
        label="13",
    )
    lines_sound.append(temp_line)
    (temp_line,) = ax.plot(
        plot_time,
        plot_buffer[14],
        color="green",
        linewidth=0.5,
        linestyle="-",
        label="14",
    )
    lines_sound.append(temp_line)
    (temp_line,) = ax.plot(
        plot_time,
        plot_buffer[15],
        color="black",
        linewidth=0.5,
        linestyle="-",
        label="15",
    )
    lines_sound.append(temp_line)
    plt.ylim(-1 * YLIM, YLIM)
    plt.legend(loc="lower right")

    plt.tight_layout()
    # plt.show()
    plt.pause(0.01)

    # シリアル通信設定
    # パソコン用
    ser = serial.Serial()
    # ser.baudrate = 38400
    ser.baudrate = 115200
    #    serial_port = 1
    #    serial_port = 2
    PORT_NUM = 4
    for idx in range(len(list(serial.tools.list_ports.comports()))):
        print(f"{list(serial.tools.list_ports.comports())[idx][1]}だお")
        if "USB" in (list(serial.tools.list_ports.comports())[idx][1]):
            PORT_NUM = idx
            print("detect^")
        print(idx)
    ser.port = list(serial.tools.list_ports.comports())[PORT_NUM][0]
    print(list(serial.tools.list_ports.comports())[PORT_NUM][0])
    print("kokodayo")
    ser.open()
    print(ser)
    # Raspberry pi用（↑の書き方でだめなときはこちらを使う）
    # ser = serial.Serial('/dev/ttyUSB0', 38400)

    time.sleep(1)

    output_file = open(os.path.join(RAW_DATA_CSV_DIR, OUTPUT_FILENAME), "w")

    flag_check = 0
    count_data = 0
    # data_buffer1 = np.zeros(CHUNK)
    data_buffer = []
    for _ in range(NUM_CH):
        data_buffer.append(np.zeros(CHUNK))
    val = np.zeros(NUM_CH)

    MVLEN = 10
    mvavg_buf = np.zeros(MVLEN)
    mvavg_val = 0
    mvavg_idx = 0

    while True:
        try:
            read_data = ser.readline().decode("utf-8")[:].split("\r")
            for split_data in read_data:
                temp = 0
                rec_vals = []
                for idx, c in enumerate(split_data):
                    val_c = ord(c) - 48
                    if idx % 4 == 0:
                        temp = (val_c << 2) & 0xFC
                    elif idx % 4 == 1:
                        rec_vals.append(temp + ((val_c >> 4) & 0x03))
                        temp = (val_c << 4) & 0xF0
                    elif idx % 4 == 2:
                        rec_vals.append(temp + ((val_c >> 2) & 0x0F))
                        temp = (val_c << 6) & 0xC0
                    else:
                        rec_vals.append(temp + val_c)
                print(len(rec_vals))
                # print(rec_vals)
                if len(rec_vals) < SER_DLEN:
                    print("less")
                    print(rec_vals)
                    break
                if len(rec_vals) > SER_DLEN:
                    print("large")
                    print(rec_vals)
                # if len(split_data) < SER_DLEN:
                #     print(split_data)
                #     break
                # if len(split_data) > SER_DLEN:
                #     print(split_data)

                # print(split_data)
                # print(split_data[4 * BIT_WORD * 15 + 15 * BIT_WORD:])
                for i in range(SPI_DLEN):
                    for ch in range(NUM_CH):
                        # startidx = i * BIT_WORD * 15 + ch * BIT_WORD
                        # val[ch] = two_comp(int(split_data[startidx:startidx + BIT_WORD], 16), ADC_BIT)
                        if ADC_BIT == 24:
                            startidx = i * 3 * NUM_CH + ch * 3
                            val[ch] = two_comp(
                                (rec_vals[0 + startidx] << 16)
                                + (rec_vals[1 + startidx] << 8)
                                + rec_vals[2 + startidx],
                                ADC_BIT,
                            )
                        else:
                            startidx = i * 2 * NUM_CH + ch * 2
                            val[ch] = two_comp(
                                (rec_vals[0 + startidx] << 8) + rec_vals[1 + startidx],
                                ADC_BIT,
                            )
                        output_file.write(f"{int(val[ch])},")
                        data_buffer[ch][count_data] = val[ch]
                    if ADC_BIT == 24:
                        startidx = (SPI_DLEN - 1) * 3 * NUM_CH + NUM_CH * 3
                    else:
                        startidx = (SPI_DLEN - 1) * 2 * NUM_CH + NUM_CH * 2
                    seq = rec_vals[startidx]
                    output_file.write(f"{seq},\n")
                    # mvavg_val += ch1_val
                    # mvavg_val -= mvavg_buf[mvavg_idx]
                    # mvavg_buf[mvavg_idx] = ch1_val
                    # mvavg_idx = (mvavg_idx + 1) % MVLEN
                    # data_buffer2[count_data] = mvavg_val / MVLEN
                    # data_buffer2[count_data] = ch2_val
                    count_data += 1
                    if count_data == CHUNK:
                        for ch in range(NUM_CH):
                            lpf_out, lpf_zi_ = signal.lfilter(
                                lpf_b, lpf_a, data_buffer[ch], zi=lpf_zi_list[ch]
                            )
                            lpf_zi_list[ch] = lpf_zi_
                            hpf_out, hpf_zi_ = signal.lfilter(
                                hpf_b, hpf_a, lpf_out, zi=hpf_zi_list[ch]
                            )
                            hpf_zi_list[ch] = hpf_zi_
                            plot_buffer[ch] = np.hstack(
                                [plot_buffer[ch][CHUNK : CHUNK * PLOT_LEN], lpf_out]
                            )
                            # plot_buffer[ch] = np.hstack([plot_buffer[ch][CHUNK:CHUNK * PLOT_LEN], hpf_out])
                            # plot_buffer[ch] = np.hstack([plot_buffer[ch][CHUNK:CHUNK * PLOT_LEN], data_buffer[ch]])
                            lines_sound[ch].set_data(plot_time, plot_buffer[ch])
                        count_data = 0
                        # plt.pause(.01)
                        fig.canvas.draw()
                        fig.canvas.flush_events()

        except UnicodeDecodeError:  # エラーでなければtry-except処理無くても良いかも
            print("UnicodeDecodeError -> retry...\n")
    output_file.close()
