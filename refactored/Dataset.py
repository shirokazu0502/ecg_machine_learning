import re
from re import A
from tkinter import W
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset
import random
import time
import gc
from scipy.interpolate import interp1d
import neurokit2 as nk

from settings import (
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


def replace_slash_with_underscore(input_string):
    print(input_string.replace("/", "_"))
    return input_string.replace("/", "_")


def Train_Test_person_datas(dirnames, target_name):
    train_list = []
    test_list = []
    for string in dirnames:
        pattern = r"(\w+)_\w+_\w+_\w+"
        match = re.search(pattern, string)

        if match:
            last_name = match.group(1)
            if last_name != target_name:
                train_list.append(string)
            else:
                test_list.append(string)
    return train_list, test_list


def Train_Test_person_datas2(dirnames, target_name):
    train_list = []
    test_list = []
    for string in dirnames:
        pattern = r"(\w+)_\w+_\w+"
        match = re.search(pattern, string)

        if match:
            last_name = match.group(1)
            if last_name != target_name:
                train_list.append(string)
            else:
                test_list.append(string)
    return train_list, test_list


def get_directory_names(directory_path):
    directory_names = []

    for entry in os.scandir(directory_path):
        if entry.is_dir():
            directory_names.append(entry.name)

    return directory_names


def get_directory_names_all(directory_path):
    directory_names = []

    for entry in os.scandir(directory_path):
        if entry.is_dir():
            for entry_in in os.scandir(os.path.join(directory_path, entry.name)):
                if entry_in.is_dir():
                    dir_name = os.path.join(entry.name, entry_in.name)
                    directory_names.append(dir_name)

    return directory_names


class NormalizeMinorMax_bat(object):
    def __call__(self, in_data):
        size = in_data.shape[0]
        normalized_data = torch.zeros_like(in_data)
        for i in range(size):
            time_data = in_data[i]
            max_val = torch.max(time_data)
            min_val = torch.min(time_data)
            val = max(abs(max_val), abs(min_val))
            normalized_data_tmp = 0.5 * (time_data / val) + 0.5
            normalized_data[i, :, :] = normalized_data_tmp
        return normalized_data


class NormalizeMinMax_bat(object):
    def __call__(self, in_data):
        size = in_data.shape[0]
        normalized_data = torch.zeros_like(in_data)
        for i in range(size):
            time_data = in_data[i]
            max_val = torch.max(time_data)
            min_val = torch.min(time_data)
            normalized_data_tmp = (time_data - min_val) / (max_val - min_val)
            normalized_data[i, :, :] = normalized_data_tmp
        return normalized_data


class NormalizeMinMax(object):
    def __call__(self, in_data):
        size = in_data.shape[0]
        normalized_data = torch.zeros_like(in_data)
        for i in range(size):
            time_data = in_data[i]
            for ch in range(in_data.shape[1]):
                ch_data = time_data[ch, :]
                max_val = torch.max(ch_data)
                min_val = torch.min(ch_data)
                normalized_ch = (ch_data - min_val) / (max_val - min_val)
                normalized_data[i, ch, :] = normalized_ch
        return normalized_data


class NormalizeTimeSeries(object):
    def __call__(self, in_data):
        size = in_data.shape[0]
        for i in range(size):
            time_data = in_data[i]
            mean = torch.mean(time_data)
            std = torch.std(time_data)
            normalized_time_data = (time_data - mean) / std
            in_data[i] = normalized_time_data
        return in_data


def Normalize(in_data):
    size = in_data.shape[0]
    for i in range(size):
        time_data = in_data[i]
        mean = torch.mean(time_data)
        std = torch.std(time_data)
        normalized_time_data = (time_data - mean) / std
        in_data[i] = normalized_time_data
    return in_data


class random_slide2(object):
    def __call__(self, data, random_number):
        slide_data = torch.zeros_like(data[:, :750])
        slide_data = data[:, random_number : random_number + 750]
        return slide_data


class random_slide(object):
    def __call__(self, data, random_numbers):
        slide_data = torch.zeros_like(data[:, :, :750])
        for i in range(len(random_numbers)):
            slide_data[i] = data[i, :, random_numbers[i] : random_numbers[i] + 750]
        return slide_data


class Original_Compose(object):
    def __call__(self, data, random_number):
        Normalize = NormalizeTimeSeries()
        data = Normalize(data)
        random_slider = random_slide()
        data = random_slider(data, random_number)
        return data


class MyDataset(TensorDataset):
    def __init__(self, in_data, out_data, name, transform=None, transform2=None):
        self.in_data = in_data
        self.out_data = out_data
        self.data_num = len(in_data)
        self.name = name
        self.transform = transform
        self.transform2 = transform2

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        if self.transform:
            mul_data = self.transform(self.in_data)[idx]
            ecg_data = self.transform(self.out_data)[idx]
            name = self.name[idx]
        else:
            mul_data = self.in_data[idx]
            ecg_data = self.out_data[idx]
            name = self.name[idx]

        return mul_data, ecg_data, name


class MyDataset_15ch_only(TensorDataset):
    def __init__(self, in_data, name, pt_index, transform=None):
        self.in_data = in_data
        self.data_num = len(in_data)
        self.name = name
        self.transform = transform
        self.pt_index = pt_index

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        mul_data = self.in_data[idx]
        name = self.name[idx]
        pt_index = self.pt_index[idx]

        return mul_data, name, pt_index


class MyDataset5(TensorDataset):
    def __init__(self, in_data, out_data, name, pt_index, transform=None):
        self.in_data = in_data
        self.out_data = out_data
        self.data_num = len(in_data)
        self.name = name
        self.transform = transform
        self.pt_index = pt_index

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        mul_data = self.in_data[idx]
        ecg_data = self.out_data[idx]
        name = self.name[idx]
        pt_index = self.pt_index[idx]

        return mul_data, ecg_data, name, pt_index


class MyDataset4(TensorDataset):
    def __init__(self, in_data, out_data, name, pt_index, transform=None):
        self.in_data = in_data
        self.out_data = out_data
        self.data_num = len(in_data)
        self.name = name
        self.transform = transform
        self.pt_index = pt_index

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        random_number = torch.randint(low=0, high=251, size=(1,))
        mul_data = self.in_data[idx]
        ecg_data = self.out_data[idx]
        mul_data = self.transform(mul_data, random_number)
        ecg_data = self.transform(ecg_data, random_number)
        name = self.name[idx]
        pt_index = self.pt_index[idx] - random_number.detach().numpy().copy()

        return mul_data, ecg_data, name, pt_index


class MyDataset3(TensorDataset):
    def __init__(self, in_data, out_data, name, transform=None):
        self.in_data = in_data
        self.out_data = out_data
        self.data_num = len(in_data)
        self.name = name
        self.transform = transform

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        batch_size = self.in_data.shape[0]
        random_numbers = torch.randint(low=0, high=251, size=(batch_size,))
        mul_data = self.transform(self.in_data, random_numbers)[idx]
        ecg_data = self.transform(self.out_data, random_numbers)[idx]
        name = self.name[idx]

        return mul_data, ecg_data, name


class MyDataset2(TensorDataset):
    def __init__(self, in_data, out_data, name, transform=None, transform2=None):
        self.in_data = in_data
        self.out_data = out_data
        self.data_num = len(in_data)
        self.name = name
        self.transform = transform
        self.transform2 = transform2

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        if self.transform:
            if self.transform2:
                batch_size = self.in_data.shape[0]
                random_numbers = torch.randint(low=0, high=251, size=(batch_size,))
                mul_data = self.transform(self.in_data)
                ecg_data = self.transform(self.out_data)
                mul_data = self.transform2(mul_data, random_numbers)[idx]
                ecg_data = self.transform2(ecg_data, random_numbers)[idx]
            else:
                mul_data = self.transform(self.in_data[:, :, 125:875])[idx]
                ecg_data = self.transform(self.out_data[:, :, 125:875])[idx]
        else:
            if self.transform2:
                batch_size = self.in_data.shape[0]
                random_numbers = torch.randint(low=0, high=251, size=(batch_size,))
                mul_data = self.transform2(self.in_data, random_numbers)[idx]
                ecg_data = self.transform2(self.out_data, random_numbers)[idx]
            else:
                mul_data = self.in_data[:, :, 125:875][idx]
                ecg_data = self.out_data[:, :, 125:875][idx]

        name = self.name[idx]

        return mul_data, ecg_data, name


class MyDataset_for_estimate(TensorDataset):
    def __init__(self, in_data, out_data, name, transform=None):
        self.in_data = in_data
        self.out_data = out_data
        self.data_num = len(in_data)
        self.name = name
        self.transform = transform

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        pgv_data = self.in_data[idx]
        name = self.name[idx]

        return pgv_data, name


def noise_make(mean, scale, datanum, ch_num):
    rnd = np.random.normal(loc=mean, scale=scale, size=datanum * ch_num)
    rnd = rnd.reshape(-1, datanum, ch_num)
    return rnd


def create_noise_data(PGV_torch, mean, scale, datanum, ch_num):
    noise = noise_make(mean, scale, datanum, ch_num)
    PGV_noise = PGV_torch + noise
    return PGV_noise


def min_max_2(x):
    x = x.to("cpu").detach().numpy().copy()
    num = x.shape[0]
    for i in range(num):
        min_val = x[i].min(axis=None, keepdims=True)
        max_val = x[i].max(axis=None, keepdims=True)
        if (max_val - min_val) != 0:
            a = max(abs(max_val), abs(min_val))
            x[i] = x[i] / (2.0 * a) + 0.5
    x = torch.FloatTensor(x)
    return x


def normalize_tensor_data(tensor):
    global_max_val = torch.max(torch.abs(tensor))
    normalized_data = 0.5 * (tensor / global_max_val) + 0.5
    return normalized_data


def pt_extend(tensor, pt_array):
    num_data = tensor.size(0)
    data_length = tensor.size(2)
    new_data = tensor
    for i in range(num_data):
        time_data = tensor[i]
        pwave = time_data[:, pt_array[0]]
        twave = time_data[:, pt_array[1]]
        for j in range(pt_array[0]):
            new_data[i, :, j] = pwave
        for j in range(data_length - pt_array[1]):
            new_data[i, :, pt_array[1] + j] = twave
    return new_data


def linear_interpolation_All(extation_range_ECG, extation_range_PGV, extation_rate):
    length = extation_range_ECG.shape[1]
    x = np.arange(length)
    new_x = np.linspace(0, length - 1, int((length) * extation_rate))
    ECG_shape = (extation_range_ECG.shape[0], len(new_x))
    new_tensor_ECG = torch.zeros(ECG_shape, dtype=torch.float32)
    PGV_shape = (extation_range_PGV.shape[0], len(new_x))
    new_tensor_PGV = torch.zeros(PGV_shape, dtype=torch.float32)
    for i in range(extation_range_ECG.shape[0]):
        data = extation_range_ECG[i, :].numpy().copy()
        interpolator = interp1d(x, data)
        new_data = interpolator(new_x)
        new_data_tensor_ECG = torch.tensor(new_data)
        new_tensor_ECG[i] = new_data_tensor_ECG
    for i in range(extation_range_PGV.shape[0]):
        data = extation_range_PGV[i, :].numpy().copy()
        interpolator = interp1d(x, data)
        new_data = interpolator(new_x)
        new_data_tensor = torch.tensor(new_data)
        new_tensor_PGV[i] = new_data_tensor
    return new_tensor_ECG, new_tensor_PGV


def get_unique_filename(base_filename, extension):
    counter = 1
    unique_filename = base_filename + extension
    while os.path.exists(unique_filename):
        unique_filename = f"{base_filename}_{counter}{extension}"
        counter += 1
    return unique_filename


def Dataset_setup_8ch_pt_augmentation(
    TARGET_NAME,
    transform_type,
    Dataset_name,
    dataset_num,
    DataAugumentation,
    ave_data_flg,
    datalength=400,
    num_channels=15,  # New argument
):
    ecg_ch_num = 8
    PGV_train_set = []
    ECG_train_set = []
    label_train_set = []
    pt_train_set = []
    PGV_test_set = []
    ECG_test_set = []
    label_test_set = []
    pt_test_set = []
    directory_path = os.path.join(PROCESSED_DATA_DIR, Dataset_name)
    dir_names = get_directory_names_all(directory_path)
    Train_list, Test_list = Train_Test_person_datas2(dir_names, target_name=TARGET_NAME)

    # The 12-lead ECG columns to be dropped to get the input data
    ecg_12_lead_cols = ["A1", "A2", "V1", "V2", "V3", "V4", "V5", "V6"]

    if ave_data_flg == 1:
        ave_path = "moving_ave_datasets"
    else:
        ave_path = ""
    base_channels = [""]

    for j in range(len(Train_list)):
        for base_ch in base_channels:
            path_to_dataset = os.path.join(directory_path, Train_list[j], base_ch)
            for i in range(dataset_num):
                path = os.path.join(
                    path_to_dataset, ave_path, "dataset_{}.csv".format(str(i).zfill(3))
                )
                pt_path = os.path.join(
                    path_to_dataset, "ponset_toffset_{}.csv".format(str(i).zfill(3))
                )

                if not os.path.isfile(path) or not os.path.isfile(pt_path):
                    pass
                else:
                    label_name = replace_slash_with_underscore(
                        Train_list[j]
                    ) + "_dataset{}".format(str(i).zfill(3))
                    # Load full data, header is now at row 0
                    data = pd.read_csv(path, header=0)
                    df_pt = pd.read_csv(pt_path, header=None, skiprows=1)
                    pt_array = np.array(df_pt.iloc[0], dtype=int)

                    # Select input and output columns by index
                    data_mul = data.iloc[:, 2:17]
                    output_cols = ["A1", "A2", "V1", "V2", "V3", "V4", "V5", "V6"]
                    data_ecg = data[output_cols]
                    PGV_train = torch.FloatTensor(data_mul.T.values)
                    PGV_train = PGV_train.reshape(-1, num_channels, datalength)
                    PGV_train = normalize_tensor_data(PGV_train)
                    PGV_train_set.append(PGV_train)

                    ECG_train = torch.FloatTensor(data_ecg.T.values)
                    ECG_train = ECG_train.reshape(-1, ecg_ch_num, datalength)
                    ECG_train = normalize_tensor_data(ECG_train)
                    ECG_train_set.append(ECG_train)
                    label_train_set.append(label_name)
                    pt_train_set.append(pt_array)

    for j in range(len(Test_list)):
        for base_ch in base_channels:
            path_to_dataset = os.path.join(directory_path, Test_list[j], base_ch)
            for i in range(dataset_num):
                path = os.path.join(
                    path_to_dataset, ave_path, "dataset_{}.csv".format(str(i).zfill(3))
                )
                pt_path = os.path.join(
                    path_to_dataset, "ponset_toffset_{}.csv".format(str(i).zfill(3))
                )
                if not os.path.isfile(path):
                    pass
                else:
                    label_name = replace_slash_with_underscore(
                        Test_list[j]
                    ) + "_dataset{}".format(str(i).zfill(3))
                    data = pd.read_csv(path, header=0)
                    df_pt = pd.read_csv(pt_path, header=None, skiprows=1)
                    pt_array = np.array(df_pt.iloc[0], dtype=int)

                    # Select input and output columns by index
                    data_mul = data.iloc[:, 2:17]
                    output_cols = ["A1", "A2", "V1", "V2", "V3", "V4", "V5", "V6"]
                    data_ecg = data[output_cols]

                    PGV_test = torch.FloatTensor(data_mul.T.values)
                    PGV_test = PGV_test.reshape(-1, num_channels, datalength)
                    PGV_test = normalize_tensor_data(PGV_test)
                    PGV_test_set.append(PGV_test)

                    ECG_test = torch.FloatTensor(data_ecg.T.values)
                    ECG_test = ECG_test.reshape(-1, ecg_ch_num, datalength)
                    ECG_test = normalize_tensor_data(ECG_test)
                    ECG_test_set.append(ECG_test)
                    label_test_set.append(label_name)
                    pt_test_set.append(pt_array)

    PGV_train_set = torch.cat(PGV_train_set, dim=0)
    ECG_train_set = torch.cat(ECG_train_set, dim=0)
    PGV_test_set = torch.cat(PGV_test_set, dim=0)
    ECG_test_set = torch.cat(ECG_test_set, dim=0)

    if transform_type == "random":
        train_dataset = MyDataset4(
            PGV_train_set,
            ECG_train_set,
            label_train_set,
            transform=random_slide2(),
            pt_index=pt_train_set,
        )
        test_dataset = MyDataset4(
            PGV_test_set,
            ECG_test_set,
            label_test_set,
            transform=random_slide2(),
            pt_index=pt_test_set,
        )
    elif transform_type == "normal":
        train_dataset = MyDataset5(
            PGV_train_set,
            ECG_train_set,
            label_train_set,
            transform="",
            pt_index=pt_train_set,
        )
        test_dataset = MyDataset5(
            PGV_test_set,
            ECG_test_set,
            label_test_set,
            transform="",
            pt_index=pt_test_set,
        )
    elif transform_type == "abnormal":
        train_dataset = MyDataset5(
            PGV_train_set,
            ECG_train_set,
            label_train_set,
            transform="",
            pt_index=pt_train_set,
        )
        test_dataset = MyDataset5(
            PGV_test_set,
            ECG_test_set,
            label_test_set,
            transform="",
            pt_index=pt_test_set,
        )
    else:
        train_dataset = MyDataset2(PGV_train_set, ECG_train_set, label_train_set)
        test_dataset = MyDataset2(PGV_test_set, ECG_test_set, label_test_set)

    return train_dataset, test_dataset
