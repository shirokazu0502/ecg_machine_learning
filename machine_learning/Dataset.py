import re
import glob
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
import math

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
import utils  # Import the refactored utils

# Define the base output directory for aggregated data (must match rereference_by_central_4ch.py)
MANIFEST_FILE_NAME = "aggregated_patients.txt"


def get_center_aggregated_patients(Dataset_name):
    """
    Safely reads the manifest file to get a list of patient names
    whose 'center_*' data has been aggregated.
    Returns an empty list if the manifest file is not found.
    """
    BASE_AGGREGATED_OUTPUT_DIR = os.path.join(PROCESSED_DATA_DIR, Dataset_name)
    manifest_path = os.path.join(BASE_AGGREGATED_OUTPUT_DIR, MANIFEST_FILE_NAME)
    aggregated_patients = []
    try:
        with open(manifest_path, "r") as f:
            for line in f:
                patient_name = line.strip()
                if patient_name:
                    aggregated_patients.append(patient_name)
    except FileNotFoundError:
        print(
            f"  > INFO: Manifest file '{manifest_path}' not found. No special subsetting will be applied."
        )
    except Exception as e:
        print(
            f"  > ERROR reading manifest file '{manifest_path}': {e}. No special subsetting will be applied."
        )
    return aggregated_patients


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
        # Reconstruct the base name by taking all parts except the last two (date and suffix)
        base_name = "_".join(string.split("_")[:-2])

        if base_name != target_name:
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


def normalize_tensor_data(tensor):
    global_max_val = torch.max(torch.abs(tensor))
    if global_max_val > 0:
        normalized_data = 0.5 * (tensor / global_max_val) + 0.5
        return normalized_data
    return tensor


def Dataset_setup_8ch_pt_augmentation(
    TARGET_NAME,
    Dataset_name,
    dataset_num,
    DataAugmentation,
    ave_data_flg,
    datalength=400,
    num_channels=15,
):
    ecg_ch_num = 8
    PGV_train_set, ECG_train_set, label_train_set, pt_train_set = [], [], [], []
    PGV_test_set, ECG_test_set, label_test_set, pt_test_set = [], [], [], []

    directory_path = os.path.join(PROCESSED_DATA_DIR, Dataset_name)
    dir_names = get_directory_names_all(directory_path)
    Train_list, Test_list = Train_Test_person_datas2(dir_names, target_name=TARGET_NAME)

    output_cols = ["A1", "A2", "V1", "V2", "V3", "V4", "V5", "V6"]
    ave_path = "moving_ave_datasets" if ave_data_flg == 1 else ""
    base_channels = [""]

    # --- Augmentation Control ---
    augmentations_to_apply = DataAugmentation.split(",") if DataAugmentation else []
    AUGMENTATION_MAP = {
        "pq_warp": utils.make_pq_extension_datas,
        "st_warp": utils.make_st_extension_datas,
        "p_height": utils.make_p_height_extation,
        "t_height": utils.make_t_height_extation,
    }
    # --- End Augmentation Control ---

    for j in range(len(Train_list)):
        for base_ch in base_channels:
            path_to_dataset = os.path.join(directory_path, Train_list[j], base_ch)
            for i in range(dataset_num):
                path = os.path.join(
                    path_to_dataset, ave_path, "dataset_{}.csv".format(str(i).zfill(3))
                )
                pt_path = os.path.join(
                    path_to_dataset,
                    ave_path,
                    "ponset_toffset_{}.csv".format(str(i).zfill(3)),
                )

                if not (os.path.isfile(path) and os.path.isfile(pt_path)):
                    continue

                label_name = (
                    f"{Train_list[j].replace('/', '_')}_dataset{str(i).zfill(3)}"
                )
                data = pd.read_csv(path, header=0)
                df_pt = pd.read_csv(pt_path, header=None, skiprows=1)
                pt_array = np.array(df_pt.iloc[0], dtype=int)

                input_start_col = 2
                input_end_col = input_start_col + num_channels
                data_mul = data.iloc[:, input_start_col:input_end_col]
                data_ecg = data[output_cols]

                PGV_train = torch.FloatTensor(data_mul.T.values).reshape(
                    -1, num_channels, datalength
                )
                ECG_train = torch.FloatTensor(data_ecg.T.values).reshape(
                    -1, ecg_ch_num, datalength
                )

                # Append original data
                PGV_train_set.append(normalize_tensor_data(PGV_train))
                ECG_train_set.append(normalize_tensor_data(ECG_train))
                label_train_set.append(label_name)
                pt_train_set.append(pt_array)

                # --- Apply Augmentations ---
                if not augmentations_to_apply:
                    continue

                for aug_key in augmentations_to_apply:
                    if aug_key not in AUGMENTATION_MAP:
                        continue

                    aug_func = AUGMENTATION_MAP[aug_key]

                    # Rates can be customized or randomized here
                    extation_rates = [0.8, 1.2]

                    for rate in extation_rates:
                        try:
                            (
                                aug_ECG,
                                aug_PGV,
                                aug_label,
                                aug_pt,
                            ) = aug_func(
                                PGV_train[0],
                                ECG_train[0],
                                pt_array,
                                label_name,
                                extation_rate=rate,
                            )

                            # Ensure augmented data has the correct shape and is normalized
                            aug_PGV = aug_PGV.view(1, num_channels, datalength)
                            aug_ECG = aug_ECG.view(1, ecg_ch_num, datalength)

                            PGV_train_set.append(normalize_tensor_data(aug_PGV))
                            ECG_train_set.append(normalize_tensor_data(aug_ECG))
                            label_train_set.append(aug_label)
                            pt_train_set.append(aug_pt)

                        except Exception as e:
                            print(
                                f"Warning: Augmentation '{aug_key}' failed for {label_name} with rate {rate}. Error: {e}"
                            )

    # Process Test set (no augmentation)
    aggregated_patients = get_center_aggregated_patients(Dataset_name)
    for j in range(len(Test_list)):
        for base_ch in base_channels:
            # Construct the base path to where the dataset files are located, including ave_path
            path_to_dataset_base = os.path.join(
                directory_path, Test_list[j], base_ch, ave_path
            )

            # First, find all available dataset files for this base_ch
            all_test_files = sorted(
                glob.glob(os.path.join(path_to_dataset_base, "dataset_*.csv"))
            )

            files_to_process = all_test_files  # Default to all files

            # Check if this test subject is an aggregated patient
            is_target_aggregated = False
            for agg_patient_full_name in aggregated_patients:
                if Test_list[j].startswith(
                    agg_patient_full_name
                ):  # Check if the full name starts with the short Test_list[j] name
                    is_target_aggregated = True
                    break

            if is_target_aggregated:
                total_files = len(all_test_files)
                # start_index = 0  # center_1を使う場合
                # end_index = math.floor(total_files / 4)
                # start_index = math.floor(total_files / 4) # center_2を使う場合
                # end_index = math.floor(total_files / 2)
                # start_index = math.floor(total_files / 2)  # center_3を使う場合
                # end_index = math.floor(3 * total_files / 4)
                start_index = math.floor(3 * total_files / 4)  # center_4を使う場合
                end_index = math.floor(total_files)

                print(
                    f"  > INFO: Test subject '{Test_list[j]}' is an aggregated patient. Applying evaluation subsetting."
                )
                print(
                    f"  > INFO: Subsetting Test_list for evaluation from {total_files} to {end_index - start_index} files (indices {start_index} to {end_index-1})."
                )
                files_to_process = all_test_files[start_index:end_index]

            # Now, iterate over the (potentially sliced) list of file paths
            for path in files_to_process:
                # Reconstruct the pt_path based on the dataset file path
                file_basename = os.path.basename(path)  # e.g., dataset_000.csv
                file_index_str = file_basename.split("_")[1].split(".")[0]  # e.g., 000

                pt_path = os.path.join(
                    path_to_dataset_base,  # Use the base path
                    f"ponset_toffset_{file_index_str}.csv",
                )

                if not (os.path.isfile(path) and os.path.isfile(pt_path)):
                    print(
                        f"  > WARNING: Missing corresponding pt_path for {os.path.basename(path)}. Skipping."
                    )
                    continue

                label_name = f"{Test_list[j].replace('/', '_')}_dataset{file_index_str}"
                data = pd.read_csv(path, header=0)
                df_pt = pd.read_csv(pt_path, header=None, skiprows=1)
                pt_array = np.array(df_pt.iloc[0], dtype=int)

                input_start_col = 2
                input_end_col = input_start_col + num_channels
                data_mul = data.iloc[:, input_start_col:input_end_col]
                data_ecg = data[output_cols]

                PGV_test = torch.FloatTensor(data_mul.T.values).reshape(
                    -1, num_channels, datalength
                )
                ECG_test = torch.FloatTensor(data_ecg.T.values).reshape(
                    -1, ecg_ch_num, datalength
                )

                PGV_test_set.append(normalize_tensor_data(PGV_test))
                ECG_test_set.append(normalize_tensor_data(ECG_test))
                label_test_set.append(label_name)
                pt_test_set.append(pt_array)

    if not PGV_train_set:
        raise FileNotFoundError(f"No training data found for any target.")
    PGV_train_set = torch.cat(PGV_train_set, dim=0)
    ECG_train_set = torch.cat(ECG_train_set, dim=0)

    if not PGV_test_set:
        raise FileNotFoundError(f"No test data found for TARGET_NAME: {TARGET_NAME}")
    PGV_test_set = torch.cat(PGV_test_set, dim=0)
    ECG_test_set = torch.cat(ECG_test_set, dim=0)

    train_dataset = MyDataset5(
        PGV_train_set,
        ECG_train_set,
        label_train_set,
        transform=None,
        pt_index=pt_train_set,
    )
    test_dataset = MyDataset5(
        PGV_test_set,
        ECG_test_set,
        label_test_set,
        transform=None,
        pt_index=pt_test_set,
    )

    return train_dataset, test_dataset
